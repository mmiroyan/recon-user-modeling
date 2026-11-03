#!/usr/bin/env bash
# Recon-Select end to end on one target user, compared against Backward Synthesis.
#
#   retrieve -> synthesize (backward, recon_select) -> generate (both) -> evaluate -> summarize
#
# Prerequisites (see README.md):
#   - an OpenAI-compatible server for the reasoning/action model, e.g.
#       vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --port 8000
#   - credentials for the judge model in .env (default judge: Gemini 3.1 Flash-Lite -> GEMINI_API_KEY)
#
# Every setting below can be overridden from the environment, e.g.
#   DATA=data/my_user/sessions.json SPLIT=data/my_user/splits.json DOMAIN=configs/my_domain.json \
#     bash scripts/demo_recon_select.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/example/sessions.json}"
SPLIT="${SPLIT:-data/example/splits.json}"
DOMAIN="${DOMAIN:-configs/example.json}"
OUT="${OUT:-runs/example}"

MODEL="${MODEL:-hosted_vllm/Qwen/Qwen3-8B}"          # reasoning model M_r and action model M_a
API_BASE="${API_BASE:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini/gemini-3.1-flash-lite-preview}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"              # only used to count tokens for context budgets
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}"
DEVICE="${DEVICE:-cuda}"                             # embedding device: cuda or cpu
NUM_CANDIDATES="${NUM_CANDIDATES:-4}"
TOP_K="${TOP_K:-8}"

set -x

# 1. Retrieve similar past (context, action) pairs for each test turn.
python -m recon.retrieve \
  --data "$DATA" --split "$SPLIT" --train_splits train_0 train_1 --test_splits test_0 \
  --tokenizer "$TOKENIZER" --embedding_model "$EMBEDDING_MODEL" --device "$DEVICE" --top_k "$TOP_K" \
  --output "$OUT/retrievals.json"

# 2a. Baseline: Backward Synthesis (one rationalization per pair).
python -m recon.synthesize --method backward \
  --retrievals "$OUT/retrievals.json" --data "$DATA" --domain_config "$DOMAIN" \
  --model "$MODEL" --api_base "$API_BASE" \
  --output "$OUT/synthesis/backward.json"

# 2b. Recon-Select: N candidates, reconstruct the action from each, keep the best-aligned one.
python -m recon.synthesize --method recon_select \
  --retrievals "$OUT/retrievals.json" --data "$DATA" --domain_config "$DOMAIN" \
  --model "$MODEL" --api_base "$API_BASE" --num_candidates "$NUM_CANDIDATES" \
  --judge_model "$JUDGE_MODEL" --save_candidates \
  --output "$OUT/synthesis/recon_select.json"

# 3. Predict each test action from the augmented retrievals.
for method in backward recon_select; do
  python -m recon.generate \
    --data "$DATA" --split "$SPLIT" --domain_config "$DOMAIN" --tokenizer "$TOKENIZER" \
    --retrievals "$OUT/synthesis/$method.json" \
    --model "$MODEL" --api_base "$API_BASE" \
    --output "$OUT/generations/$method.json"
done

# 4. Pairwise LM judge against the ground truth, then win rates (ties excluded).
python -m recon.evaluate \
  --data "$DATA" --split "$SPLIT" --tokenizer "$TOKENIZER" --judge_model "$JUDGE_MODEL" \
  --generations_a "$OUT/generations/recon_select.json" \
  --generations_b "$OUT/generations/backward.json" \
  --output "$OUT/eval/recon_select_vs_backward.json"

python -m recon.summarize --path "$OUT/eval/recon_select_vs_backward.json"
