"""Recon-GRPO: train the reasoning model M_r with action-reconstruction rewards.

For each target turn (c, a*), the policy samples rationalizations r ~ M_r(c, a*). A frozen
action model reconstructs a_i = M_a(c, r_i), and the reward is the LM-judge alignment of a_i
with a* minus a penalty when r_i copies a* verbatim. See recon/grpo/README.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM

from recon.grpo.data import (
    TURN_FILTERS,
    collect_split_session_ids,
    enumerate_turn_jobs,
    load_json,
    load_tokenizer,
    select_sessions,
    split_session_keys_by_original_roots,
    subsample_jobs,
)
from recon.grpo.examples import build_examples
from recon.grpo.prompt_format import load_prompt_config
from recon.grpo.rewards import DimJudgeReward, SanityReward, _SanityJudge


def _build_model(base_model: str) -> AutoModelForCausalLM:
    token = os.getenv("HF_TOKEN")
    ws = (os.environ.get("WORLD_SIZE") or "1").strip()
    try:
        distributed_training = int(ws) > 1
    except ValueError:
        distributed_training = False
    if distributed_training:
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=token,
        )
    return AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )


def _configure_model(model, tokenizer, args):
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if args.full_finetune:
        return model
    from peft import LoraConfig, PeftModel, get_peft_model

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.lora_adapter:
        return PeftModel.from_pretrained(model, args.lora_adapter, is_trainable=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _save_final_model(output_dir: str | Path, trainer, tokenizer) -> None:
    output_dir = Path(output_dir)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


def _resolve_resume(value: Optional[str]):
    if value is None or value == "":
        return None
    if value.lower() in ("latest", "true"):
        return True
    return value


def _public_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recon-GRPO training for one target user")

    # Data
    p.add_argument("--data_path", required=True)
    p.add_argument("--split_path", default=None)
    p.add_argument("--split_names", nargs="*", default=None)
    p.add_argument("--proportion", type=float, required=True)
    p.add_argument("--mode", choices=["start", "end", "random"], required=True)
    p.add_argument("--max_sessions", type=int, default=None)
    p.add_argument("--num_val_sessions", type=int, default=0)
    p.add_argument("--max_train_sessions", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument(
        "--turn_filters",
        nargs="*",
        default=None,
        help=f"Turn filter names to skip. Available: {sorted(TURN_FILTERS)}",
    )
    p.add_argument("--max_context_length", type=int, default=0)
    p.add_argument("--build_examples_num_workers", type=int, default=0)

    # Model
    p.add_argument("--base_model", required=True)
    p.add_argument("--base_tokenizer", default=None)
    p.add_argument("--domain_config", required=True, help="Prompt config, e.g. configs/example.json")
    p.add_argument("--target_alias", default=None, help="Replace target author name in context with this alias")
    p.add_argument("--lora_adapter", default=None)
    p.add_argument("--full_finetune", action="store_true")
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--run_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_completion_length", type=int, required=True)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--num_generations_eval", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=float, required=True)
    p.add_argument("--num_evals_per_epoch", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per_device_train_batch_size", type=int, required=True)
    p.add_argument("--per_device_eval_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, required=True)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, required=True)
    p.add_argument("--lr_scheduler_type", required=True)
    p.add_argument("--logging_steps", type=int, required=True)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", default=None)

    # Judges / action generator (litellm)
    p.add_argument("--sanity_judge_model", required=True)
    p.add_argument("--sanity_judge_api_base", default=None)
    p.add_argument("--action_generator_model", required=True)
    p.add_argument("--action_generator_api_base", default=None)
    p.add_argument("--dim_judge_model", required=True)
    p.add_argument("--dim_judge_api_base", default=None)
    p.add_argument("--dim_judge_max_retries", type=int, default=3)
    p.add_argument("--judge_concurrency", type=int, default=8)
    p.add_argument("--action_temperature", type=float, default=0.7)
    p.add_argument("--sanity_judge_temperature", type=float, default=0.0)
    p.add_argument("--dim_judge_temperature", type=float, default=0.0)
    p.add_argument("--sanity_failure_reward", type=float, default=-1.0)
    p.add_argument("--reward_weight_sanity", type=float, default=1.0)
    p.add_argument("--reward_weight_dim_judge", type=float, default=1.0)
    p.add_argument(
        "--multi_objective_aggregation",
        choices=["sum_then_normalize", "normalize_then_sum"],
        default="sum_then_normalize",
    )

    # Logging
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--log_completions", action="store_true")

    return p.parse_args()


def run_training(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    is_rank0 = (os.environ.get("RANK") or "0").strip() == "0"
    if is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(_public_cfg(args), indent=2) + "\n", encoding="utf-8"
        )

    print("=" * 70)
    print("  RECON-GRPO")
    print("=" * 70)
    for k, v in sorted(_public_cfg(args).items()):
        print(f"  {k:30s} = {v}")
    print("=" * 70)

    sessions = load_json(args.data_path)
    prompt_config = load_prompt_config(args.domain_config)
    if args.target_alias:
        from dataclasses import replace as _replace
        prompt_config = _replace(prompt_config, target_alias=args.target_alias)

    if args.split_path is not None:
        split = load_json(args.split_path)
        split_names = args.split_names or sorted(split.keys())
        candidate_ids = collect_split_session_ids(split, split_names) & set(sessions.keys())
    else:
        candidate_ids = set(sessions.keys())
    if not candidate_ids:
        raise ValueError("No sessions after split filtering")

    selected_ids = select_sessions(
        list(candidate_ids),
        proportion=args.proportion,
        mode=args.mode,
        max_sessions=args.max_sessions,
        seed=args.seed,
    )
    if not selected_ids:
        raise ValueError("No sessions remain after proportion/mode/max_sessions")

    train_ids, val_ids = split_session_keys_by_original_roots(
        selected_ids,
        seed=args.seed,
        num_val_sessions=args.num_val_sessions,
        max_train_sessions=args.max_train_sessions,
    )
    if not train_ids:
        raise ValueError("No train sessions after val split")

    print(f"  train_sessions = {len(train_ids)}, val_sessions = {len(val_ids)}")

    tokenizer = load_tokenizer(args.base_tokenizer, args.base_model)

    rng_train = random.Random(args.seed)
    rng_val = random.Random((args.seed + 2_654_435_761) & 0xFFFFFFFF)

    train_jobs_full = enumerate_turn_jobs(sessions, set(train_ids), args.turn_filters)
    if not train_jobs_full:
        raise ValueError("No qualifying train turns (target author / turn filters).")
    train_jobs = subsample_jobs(train_jobs_full, args.max_train_samples, rng_train)
    train_examples = build_examples(
        sessions,
        tokenizer,
        max_context_length=args.max_context_length,
        allowed_session_ids=set(train_ids),
        turn_filter_names=args.turn_filters,
        turn_jobs=train_jobs,
        num_workers=args.build_examples_num_workers,
        parallel_base_tokenizer=args.base_tokenizer,
        parallel_base_model=args.base_model,
        prompt_config=prompt_config,
    )
    if val_ids:
        val_jobs_full = enumerate_turn_jobs(sessions, set(val_ids), args.turn_filters)
        val_jobs = subsample_jobs(val_jobs_full, args.max_val_samples, rng_val)
        val_examples = build_examples(
            sessions,
            tokenizer,
            max_context_length=args.max_context_length,
            allowed_session_ids=set(val_ids),
            turn_filter_names=args.turn_filters,
            turn_jobs=val_jobs,
            num_workers=args.build_examples_num_workers,
            parallel_base_tokenizer=args.base_tokenizer,
            parallel_base_model=args.base_model,
            prompt_config=prompt_config,
        )
    else:
        val_examples = []
    if not train_examples:
        raise ValueError("No training examples after turn filtering")

    print(f"  train_examples = {len(train_examples)}, val_examples = {len(val_examples)}")

    model = _build_model(args.base_model)
    model = _configure_model(model, tokenizer, args)
    if not args.full_finetune and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()

    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    sanity_judge = _SanityJudge(
        model=args.sanity_judge_model,
        temperature=args.sanity_judge_temperature,
        max_retries=args.dim_judge_max_retries,
        concurrency=args.judge_concurrency,
        api_base=args.sanity_judge_api_base or None,
    )

    if args.reward_weight_dim_judge != 0 and args.reward_weight_sanity == 0:
        raise ValueError(
            "reward_weight_sanity must be non-zero when reward_weight_dim_judge is non-zero "
            "(SanityReward must run first to populate the shared sanity cache)"
        )

    reward_funcs: list[Any] = []
    reward_weights: list[float] = []
    # SanityReward must come first: it populates the sanity cache that DimJudgeReward reads.
    if args.reward_weight_sanity != 0:
        reward_funcs.append(
            SanityReward(
                sanity_judge=sanity_judge,
                sanity_failure_reward=args.sanity_failure_reward,
            )
        )
        reward_weights.append(args.reward_weight_sanity)
    if args.reward_weight_dim_judge != 0:
        reward_funcs.append(
            DimJudgeReward(
                sanity_judge=sanity_judge,
                action_generator_model=args.action_generator_model,
                dim_judge_model=args.dim_judge_model,
                prompt_config=prompt_config,
                dim_judge_max_retries=args.dim_judge_max_retries,
                judge_concurrency=args.judge_concurrency,
                action_temperature=args.action_temperature,
                dim_judge_temperature=args.dim_judge_temperature,
                action_generator_api_base=args.action_generator_api_base or None,
                dim_judge_api_base=args.dim_judge_api_base or None,
            )
        )
        reward_weights.append(args.reward_weight_dim_judge)
    if not reward_funcs:
        raise ValueError("At least one reward weight must be non-zero")

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(val_examples) if val_examples else None
    has_eval = eval_dataset is not None

    eval_steps = None
    if has_eval and args.num_evals_per_epoch > 1:
        steps_per_epoch = math.ceil(
            len(train_examples) * args.num_generations
            / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        )
        eval_steps = max(1, round(steps_per_epoch / (args.num_evals_per_epoch - 1)))
    save_strategy = "steps" if args.save_steps is not None else "epoch"

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size or args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        temperature=args.temperature,
        beta=args.beta,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_max_model_length=16384,
        eval_strategy="steps" if has_eval and args.num_evals_per_epoch > 1 else ("epoch" if has_eval else "no"),
        eval_on_start=has_eval,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if save_strategy == "steps" else None,
        save_only_model=True,
        logging_steps=args.logging_steps,
        report_to="wandb" if args.wandb_entity and args.wandb_project else "none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        log_completions=args.log_completions,
        reward_weights=reward_weights,
        multi_objective_aggregation=args.multi_objective_aggregation,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )
    trainer.ref_model = trainer.accelerator.prepare_model(ref_model, evaluation_mode=True)

    trainer.train(resume_from_checkpoint=_resolve_resume(args.resume_from_checkpoint))
    _save_final_model(args.output_dir, trainer, tokenizer)


def main() -> None:
    load_dotenv()
    import logging
    import litellm
    litellm.suppress_debug_info = True
    logging.getLogger().setLevel(logging.WARNING)
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    for h in litellm_logger.handlers:
        h.setLevel(logging.WARNING)
    run_training(parse_args())


if __name__ == "__main__":
    main()
