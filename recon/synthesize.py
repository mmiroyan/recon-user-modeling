"""Augment retrieved (context, action) pairs with synthesized reasoning traces.

Methods
-------
backward      Backward Synthesis: one rationalization per pair, r = M_r(c, a*).
recon_select  Recon-Select: sample N rationalizations from M_r(c, a*), reconstruct the action
              from each with a_i = M_a(c, r_i), score alignment(a_i, a*) with an LM judge, and
              keep the rationalization whose reconstruction scores highest.

To synthesize with a Recon-GRPO-trained reasoning model, serve the LoRA adapter with vLLM and
run ``--method backward`` with ``--model`` pointing at it (see recon/grpo/README.md).

    python -m recon.synthesize --method recon_select \\
        --retrievals runs/example/retrievals.json --data data/example/sessions.json \\
        --domain_config configs/example.json --output runs/example/recon_select.json \\
        --model hosted_vllm/Qwen/Qwen3-8B --api_base http://localhost:8000/v1
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from recon.common import (
    LLM,
    chunk_action,
    format_chunk_context,
    load_domain_config,
    load_json,
    preamble_block,
    run_parallel,
    save_json,
    target_author_of,
    with_retries,
)
from recon.prompts import ACTION_RECONSTRUCTION_PROMPT, ALIGNMENT_JUDGE_PROMPT, REASONING_SYNTHESIS

JUDGE_DIMS = ("style", "intent", "values")


def unique_chunks(retrievals: dict) -> dict[str, list[dict]]:
    """Group retrieved chunks by (session, target turn); a chunk may be retrieved many times."""
    groups: dict[str, list[dict]] = {}
    for turn_map in retrievals.values():
        for chunks in turn_map.values():
            for chunk in chunks:
                groups.setdefault(f"{chunk['session_id']}:{chunk['end_turn_idx']}", []).append(chunk)
    return groups


def reasoning_prompt(context: str, action: str, cfg: dict) -> str:
    return REASONING_SYNTHESIS.format(
        preamble_block=preamble_block(cfg["reasoning_preamble"]),
        context=context,
        ground_truth_response=action,
    )


# ---------------------------------------------------------------------------
# Backward Synthesis
# ---------------------------------------------------------------------------

def backward_synthesis(examples: dict[str, dict], cfg: dict, reasoner: LLM, max_workers: int) -> dict[str, dict]:
    tasks = [
        (key, lambda ex=ex: reasoner.complete(reasoning_prompt(ex["context"], ex["action"], cfg), seed=42))
        for key, ex in examples.items()
    ]
    results = run_parallel(tasks, max_workers, "Backward Synthesis")
    return {key: {"reasoning": text} for key, text in results.items()}


# ---------------------------------------------------------------------------
# Recon-Select
# ---------------------------------------------------------------------------

def _judge_schema(n: int) -> dict:
    score = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "alignment_rationale": {"type": "string"},
            "alignment_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["alignment_rationale", "alignment_score"],
    }
    response = {
        "type": "object",
        "additionalProperties": False,
        "properties": {d: score for d in JUDGE_DIMS},
        "required": list(JUDGE_DIMS),
    }
    keys = [str(i) for i in range(1, n + 1)]
    return {"type": "object", "additionalProperties": False,
            "properties": {k: response for k in keys}, "required": keys}


def alignment_judge_prompt(context: str, action: str, reconstructions: list[str]) -> str:
    return ALIGNMENT_JUDGE_PROMPT.format(
        context=context,
        ground_truth_response=action,
        generated_responses="\n\n".join(f"Response {i}: {r}" for i, r in enumerate(reconstructions, 1)),
        k=len(reconstructions),
    )


def alignment_scores(judgement: dict, n: int) -> list[float]:
    """Mean alignment score over style / intent / values for each of the n reconstructions."""
    scores = []
    for i in range(1, n + 1):
        dims = judgement.get(str(i), {}) if judgement else {}
        vals = [float(dims[d]["alignment_score"]) for d in JUDGE_DIMS if d in dims]
        scores.append(sum(vals) / len(vals) if vals else 0.0)
    return scores


def recon_select(
    examples: dict[str, dict],
    cfg: dict,
    reasoner: LLM,
    reconstructor: LLM,
    judge: LLM,
    num_candidates: int,
    max_workers: int,
) -> dict[str, dict]:
    # 1. Sample N candidate rationalizations r_i ~ M_r(c, a*).
    tasks = [
        ((key, i), lambda ex=ex: with_retries(
            lambda: reasoner.complete(reasoning_prompt(ex["context"], ex["action"], cfg)), "reasoning"))
        for key, ex in examples.items()
        for i in range(num_candidates)
    ]
    candidates = run_parallel(tasks, max_workers, "Recon-Select 1/3: candidates")

    # 2. Reconstruct the action from the context and each candidate: a_i = M_a(c, r_i).
    tasks = []
    for key, ex in examples.items():
        for i in range(num_candidates):
            prompt = ACTION_RECONSTRUCTION_PROMPT.format(
                preamble_block=preamble_block(cfg["action_preamble"]),
                reasoning=candidates.get((key, i), ""),
                active_context=ex["context"],
                response_format=cfg["response_format"],
            )
            tasks.append(((key, i), lambda p=prompt: with_retries(lambda: reconstructor.complete(p), "reconstruction")))
    reconstructions = run_parallel(tasks, max_workers, "Recon-Select 2/3: reconstruction")

    # 3. Score every reconstruction against a* in a single judge call per pair.
    tasks = []
    for key, ex in examples.items():
        recs = [reconstructions.get((key, i), "") for i in range(num_candidates)]
        if not any(recs):
            continue
        prompt = alignment_judge_prompt(ex["context"], ex["action"], recs)
        tasks.append((key, lambda p=prompt: judge.complete_json(p, _judge_schema(num_candidates), "JudgeResponse")))
    judgements = run_parallel(tasks, max_workers, "Recon-Select 3/3: scoring")

    out: dict[str, dict] = {}
    for key in examples:
        scores = alignment_scores(judgements.get(key, {}), num_candidates)
        best = max(range(num_candidates), key=lambda i: scores[i])
        out[key] = {
            "reasoning": candidates.get((key, best), ""),
            "candidates": [
                {"reasoning": candidates.get((key, i), ""),
                 "reconstruction": reconstructions.get((key, i), ""),
                 "score": scores[i]}
                for i in range(num_candidates)
            ],
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _bool(x: str) -> bool:
    return {"true": True, "false": False}[x.lower()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=["backward", "recon_select"])
    p.add_argument("--retrievals", required=True, help="Output of recon.retrieve")
    p.add_argument("--data", required=True, help="Sessions JSON (used for the target author's name)")
    p.add_argument("--domain_config", required=True, help="Prompt config, e.g. configs/example.json")
    p.add_argument("--output", required=True)
    p.add_argument("--target_alias", default="I",
                   help="Label for the target author's own turns in the context")
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--save_candidates", action="store_true",
                   help="recon_select: keep all candidates, reconstructions and scores in the output")
    # Reasoning model M_r
    p.add_argument("--model", required=True, help="litellm model string for the reasoning model M_r")
    p.add_argument("--api_base", default=None)
    p.add_argument("--temperature", type=float, default=None,
                   help="Default: 0.7 for backward, 1.0 for recon_select candidates")
    p.add_argument("--enable_thinking", type=_bool, default=None,
                   help="Pass enable_thinking to a vLLM chat template (true/false)")
    p.add_argument("--reasoning_effort", default=None, help="For OpenAI reasoning models")
    # Recon-Select: action reconstructor M_a and alignment judge
    p.add_argument("--num_candidates", type=int, default=4, help="N candidates per pair")
    p.add_argument("--action_model", default=None, help="Reconstruction model M_a (default: --model)")
    p.add_argument("--action_api_base", default=None, help="Default: --api_base")
    p.add_argument("--action_temperature", type=float, default=0.7)
    p.add_argument("--judge_model", default="gemini/gemini-3.1-flash-lite-preview")
    p.add_argument("--judge_api_base", default=None)
    p.add_argument("--judge_temperature", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    retrievals = load_json(args.retrievals)
    target_author = target_author_of(load_json(args.data))
    cfg = load_domain_config(args.domain_config)

    groups = unique_chunks(retrievals)
    examples = {
        key: {
            "context": format_chunk_context(chunks[0], target_author, args.target_alias),
            "action": chunk_action(chunks[0]),
        }
        for key, chunks in groups.items()
    }
    print(f"Target author: {target_author} | unique retrieved pairs: {len(examples)}")

    temperature = args.temperature
    if temperature is None:
        temperature = 1.0 if args.method == "recon_select" else 0.7
    reasoner = LLM(args.model, args.api_base, temperature, args.timeout,
                   args.reasoning_effort, args.enable_thinking)

    if args.method == "backward":
        results = backward_synthesis(examples, cfg, reasoner, args.max_workers)
    else:
        reconstructor = LLM(args.action_model or args.model, args.action_api_base or args.api_base,
                            args.action_temperature, args.timeout, args.reasoning_effort, args.enable_thinking)
        judge = LLM(args.judge_model, args.judge_api_base, args.judge_temperature, args.timeout)
        results = recon_select(examples, cfg, reasoner, reconstructor, judge,
                               args.num_candidates, args.max_workers)

    missing = 0
    for key, chunks in groups.items():
        result = results.get(key)
        if result is None:
            missing += 1
            continue
        for chunk in chunks:
            chunk["reasoning"] = result["reasoning"]
            if args.save_candidates and "candidates" in result:
                chunk["candidates"] = result["candidates"]
    if missing:
        print(f"[warn] no reasoning for {missing}/{len(groups)} pairs")
    save_json(args.output, retrievals)
    print(f"Wrote augmented retrievals to {args.output}")


if __name__ == "__main__":
    main()
