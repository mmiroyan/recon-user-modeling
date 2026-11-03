# Recon-GRPO

Recon-GRPO trains the reasoning model `M_r` with GRPO, using action reconstruction as the reward
(paper Sec. 3.3, App. C.3). The action model `M_a` stays frozen. Because the reward depends only on how
well `M_a` recovers the action from the trace, the policy has to write reasoning that guides `M_a`,
much like an advisor model steering a black-box model.

## Reward

For a target turn `(c, a*)` from the training split:

1. The policy samples a group of rationalizations `r_1..r_G ~ M_r(c, a*)` (G = `--num_generations`,
   4 in the paper). The prompt is the same reasoning-synthesis prompt used by Backward Synthesis and
   Recon-Select ([`recon/prompts.py`](../prompts.py)).
2. **Copy check** ([`judges/sanity_check.py`](judges/sanity_check.py)): an LM judge flags traces that
   contain `a*` verbatim. A flagged trace gets `--sanity_failure_reward` (−1) and its alignment reward
   is 0.
3. **Reconstruction** ([`action_generator.py`](action_generator.py)): `M_a` writes `a_i = M_a(c, r_i)`
   for each remaining trace.
4. **Alignment** ([`judges/dim_judge.py`](judges/dim_judge.py)): an LM judge scores every `a_i` in the
   group against `a*` on style, intent and values. The alignment reward is the mean of the three
   scores.

The total reward is `--reward_weight_dim_judge` × alignment + `--reward_weight_sanity` × copy penalty,
i.e. `1 · alignment − 2 · copied` in the paper.

## Training

```bash
bash scripts/demo_recon_grpo.sh train
```

This calls `python -m recon.grpo.train` with the paper's settings: LoRA rank 128 (alpha 256), group
size 4, sampling temperature 1.0, KL coefficient 0.03, learning rate 1e-5 with cosine decay, one epoch,
per-device batch size 1 with 8 gradient-accumulation steps, and up to 2048 context and 1024 completion
tokens. The policy starts
from the base model (no SFT warm-up). Training data are the target turns in `--split_names` (`train_1`
by default).

`M_a` and the judges are called through litellm, so they need to be served separately: the demo
expects `M_a` (Qwen3-8B) behind a vLLM server on port 8000 and a Gemini key for the judges. Training a
Qwen3-4B policy on about 3k turns took roughly 12 hours on a single Hopper-class GPU. To use more
GPUs, launch the same arguments with `torchrun --nproc_per_node=N -m recon.grpo.train ...`.

The adapter is saved to `<output_dir>/final`. Add `--wandb_entity` / `--wandb_project` to log to
Weights & Biases.

## Using the trained model

At inference time, the trained model writes one trace per retrieved pair, so Recon-GRPO augmentation
is Backward Synthesis with the trained adapter as `M_r`:

```bash
vllm serve Qwen/Qwen3-4B --reasoning-parser qwen3 --port 8001 \
  --enable-lora --max-lora-rank 128 --lora-modules recon-grpo=runs/example/recon_grpo/final

python -m recon.synthesize --method backward \
  --model hosted_vllm/recon-grpo --api_base http://localhost:8001/v1 --enable_thinking false \
  --retrievals runs/example/retrievals.json --data data/example/sessions.json \
  --domain_config configs/example.json --output runs/example/synthesis/recon_grpo.json
```

`--enable_thinking false` matches the chat template the policy was trained with. `bash
scripts/demo_recon_grpo.sh evaluate` runs this step, then generation and pairwise evaluation against
Backward Synthesis from the untrained base model (the comparison in paper Fig. 6).

## Files

| File | Contents |
|---|---|
| `train.py` | CLI, model/LoRA setup, TRL `GRPOTrainer` |
| `data.py`, `examples.py` | Selecting sessions and target turns, building policy prompts |
| `prompt_format.py` | Context packing and the reconstruction, judge and copy-check prompts |
| `rewards.py` | Copy-check and alignment rewards (computed per GRPO group) |
| `action_generator.py` | Calls the frozen action model |
| `judges/` | LM-judge calls for alignment and copy checking |
| `rollout_parse.py` | Cleans policy completions before scoring |
