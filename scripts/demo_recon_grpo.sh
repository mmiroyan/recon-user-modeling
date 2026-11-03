#!/usr/bin/env bash
# Recon-GRPO: train a reasoning model with action-reconstruction rewards, then use it to
# augment the retrieval corpus and compare against Backward Synthesis from the same base model.
#
#   bash scripts/demo_recon_grpo.sh train      # GRPO on the train_1 split (needs a GPU)
#   bash scripts/demo_recon_grpo.sh evaluate   # synthesize with the trained LoRA, generate, judge
#
# Prerequisites (see recon/grpo/README.md):
#   - a frozen action model M_a behind an OpenAI-compatible server, e.g.
#       vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --port 8000
#   - judge credentials in .env (default judge: Gemini 3.1 Flash-Lite -> GEMINI_API_KEY)
#   - for `evaluate`: the base reasoning model served with the trained adapter, e.g.
#       vllm serve Qwen/Qwen3-4B --reasoning-parser qwen3 --port 8001 \
#         --enable-lora --max-lora-rank 128 --lora-modules recon-grpo=runs/example/recon_grpo/final
#   - retrievals from step 1 of scripts/demo_recon_select.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:?usage: $0 train|evaluate}"

DATA="${DATA:-data/example/sessions.json}"
SPLIT="${SPLIT:-data/example/splits.json}"
DOMAIN="${DOMAIN:-configs/example.json}"
OUT="${OUT:-runs/example}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"                 # reasoning model M_r being trained
ACTION_MODEL="${ACTION_MODEL:-hosted_vllm/Qwen/Qwen3-8B}" # frozen reconstructor / action model M_a
ACTION_API_BASE="${ACTION_API_BASE:-http://localhost:8000/v1}"
REASONER_API_BASE="${REASONER_API_BASE:-http://localhost:8001/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini/gemini-3.1-flash-lite-preview}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
RUN_DIR="$OUT/recon_grpo"

set -x
case "$STAGE" in
train)
  # Settings used for the PMQ experiments in the paper (Appendix C.3).
  python -m recon.grpo.train \
    --run_name recon_grpo --output_dir "$RUN_DIR" \
    --data_path "$DATA" --split_path "$SPLIT" --split_names train_1 \
    --proportion 1.0 --mode random --num_val_sessions 0 \
    --domain_config "$DOMAIN" --target_alias I --max_context_length 2048 \
    --base_model "$BASE_MODEL" --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --num_generations 4 --temperature 1.0 --max_completion_length 1024 --beta 0.03 \
    --num_train_epochs 1 --learning_rate 1e-5 --lr_scheduler_type cosine \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing \
    --logging_steps 4 --save_steps 128 \
    --action_generator_model "$ACTION_MODEL" --action_generator_api_base "$ACTION_API_BASE" \
    --action_temperature 1.0 \
    --dim_judge_model "$JUDGE_MODEL" --dim_judge_temperature 1.0 \
    --sanity_judge_model "$JUDGE_MODEL" --sanity_judge_temperature 1.0 \
    --reward_weight_dim_judge 1.0 --reward_weight_sanity 2.0 --sanity_failure_reward -1.0 \
    --judge_concurrency 32
  ;;
evaluate)
  # Baseline: Backward Synthesis with the untrained base model.
  python -m recon.synthesize --method backward \
    --retrievals "$OUT/retrievals.json" --data "$DATA" --domain_config "$DOMAIN" \
    --model "hosted_vllm/$BASE_MODEL" --api_base "$REASONER_API_BASE" \
    --output "$OUT/synthesis/backward_base.json"

  # Recon-GRPO: the trained adapter writes one trace per pair, prompted exactly as in training.
  python -m recon.synthesize --method backward \
    --retrievals "$OUT/retrievals.json" --data "$DATA" --domain_config "$DOMAIN" \
    --model hosted_vllm/recon-grpo --api_base "$REASONER_API_BASE" --enable_thinking false \
    --output "$OUT/synthesis/recon_grpo.json"

  for method in backward_base recon_grpo; do
    python -m recon.generate \
      --data "$DATA" --split "$SPLIT" --domain_config "$DOMAIN" --tokenizer "$TOKENIZER" \
      --retrievals "$OUT/synthesis/$method.json" \
      --model "$ACTION_MODEL" --api_base "$ACTION_API_BASE" \
      --output "$OUT/generations/$method.json"
  done

  python -m recon.evaluate \
    --data "$DATA" --split "$SPLIT" --tokenizer "$TOKENIZER" --judge_model "$JUDGE_MODEL" \
    --generations_a "$OUT/generations/recon_grpo.json" \
    --generations_b "$OUT/generations/backward_base.json" \
    --output "$OUT/eval/recon_grpo_vs_backward.json"
  python -m recon.summarize --path "$OUT/eval/recon_grpo_vs_backward.json"
  ;;
*)
  echo "unknown stage: $STAGE (expected train or evaluate)" >&2
  exit 1
  ;;
esac
