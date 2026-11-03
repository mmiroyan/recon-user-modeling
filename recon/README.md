# `recon`

The pipeline has four stages. Each one is a module you run with `python -m` from the repository
root; each reads JSON files and writes JSON files.

```text
sessions.json + splits.json
        │
        ▼
 recon.retrieve      top-k similar past (context, action) pairs for each test turn
        │                                                    → retrievals.json
        ▼
 recon.synthesize    add a reasoning trace to every retrieved pair
        │              --method backward       Backward Synthesis (baseline)
        │              --method recon_select   Recon-Select
        │              (Recon-GRPO: --method backward with the trained model, see grpo/)
        │                                                    → augmented retrievals
        ▼
 recon.generate      predict each test action from the augmented examples (Eq. 1)
        │                                                    → generations
        ▼
 recon.evaluate      pairwise LM judge of two generation files vs. the ground truth
 recon.summarize     win rates, ties excluded
```

[`scripts/demo_recon_select.sh`](../scripts/demo_recon_select.sh) runs every stage with the paper's
settings. [`grpo/`](grpo/) holds the Recon-GRPO trainer.

| Module | Paper |
|---|---|
| `retrieve.py` | Retrieval for in-context learning (Sec. 3.1, App. D.2): Qwen3-Embedding-4B, k = 8 |
| `synthesize.py` | Backward Synthesis (Sec. 4.2), Recon-Select (Sec. 3.3, App. C.2) |
| `generate.py` | Action generation `M_a({(c_i, r_i, a*_i)}, c_T)` (Eq. 1, App. D.2) |
| `evaluate.py`, `summarize.py` | Pairwise evaluation (Sec. 4.3, App. D) |
| `grpo/` | Recon-GRPO (Sec. 3.3, App. C.3) |
| `prompts.py` | All prompt templates |
| `common.py` | I/O, domain configs, turn formatting, LLM calls |

## Models

Models are [litellm](https://docs.litellm.ai/docs/providers) model strings, so the same code runs
against hosted APIs (`gpt-5-mini`, `gemini/gemini-3.1-flash-lite-preview`) and local
OpenAI-compatible servers (`hosted_vllm/Qwen/Qwen3-8B` with `--api_base http://localhost:8000/v1`).
API keys are read from `.env` (see [`.env.example`](../.env.example)).

In the paper, the reasoning model `M_r` and action model `M_a` are the same model unless stated
otherwise. The judge for both Recon-Select scoring and evaluation is Gemini 3.1 Flash-Lite. Qwen models
were served with vLLM's `--reasoning-parser qwen3`, so the returned content excludes the model's own
thinking.

## Domain configs

Prompts are shared across domains. A domain config (`--domain_config`) only supplies short role
descriptions that are inserted into them. [`configs/README.md`](../configs/README.md) lists the
required fields and where each is used, and [`configs/example.json`](../configs/example.json) matches
the example dataset.

## Stages

Run any module with `--help` for the full list of options.

### `recon.retrieve`

```bash
python -m recon.retrieve --data <sessions.json> --split <splits.json> --output <retrievals.json> \
  [--train_splits train_0 train_1] [--test_splits test_0] [--top_k 8] [--device cuda]
```

The retrieval corpus contains one entry per target turn in the training splits. The embedding key is
the preceding context (up to `--chunk_tokens`, default 1024); the stored value is that context plus the
target turn. Each test turn is queried with its own preceding context (up to `--query_tokens`, default
2048). Output:

```json
{
  "<test session id>": {
    "<test turn index>": [
      {
        "rank": 0, "score": 0.91,
        "session_id": "<train session id>", "start_turn_idx": 0, "end_turn_idx": 3,
        "turns": [{"author": "...", "action": "..."}, "...", {"author": "<target>", "action": "<a*>"}]
      }
    ]
  }
}
```

The last entry of `turns` is always the target user's action `a*`; everything before it is the
context `c`.

### `recon.synthesize`

```bash
python -m recon.synthesize --method {backward,recon_select} \
  --retrievals <retrievals.json> --data <sessions.json> --domain_config configs/<domain>.json \
  --model <M_r> [--api_base URL] --output <augmented.json>
```

Each distinct retrieved pair is synthesized once and the trace is attached to every copy of it as
`"reasoning"`. The output has the same shape as the retrievals file.

- `backward`: one trace `r = M_r(c, a*)` per pair (temperature 0.7).
- `recon_select`: `--num_candidates` traces per pair (default 4, temperature 1.0). `M_a`
  (`--action_model`, default `--model`) reconstructs the action from `(c, r_i)` alone. The judge
  (`--judge_model`) scores every reconstruction against `a*` on style, intent and values in a single
  call, and the trace with the highest mean score is kept. `--save_candidates` also stores every
  candidate, its reconstruction and its score under `"candidates"`. This takes `2N + 1` LM calls per
  pair.

### `recon.generate`

```bash
python -m recon.generate --data <sessions.json> --split <splits.json> \
  --retrievals <augmented.json> --domain_config configs/<domain>.json \
  --model <M_a> [--api_base URL] --output <generations.json> [--no_reasoning]
```

For each target turn in the test split, the prompt contains the retrieved examples as (context,
thought process, response) triplets, followed by the current context (up to `--max_context_tokens`,
default 2048). The model writes a thought process and a response. The output is the test sessions with
`generated_action` (and `generated_reasoning`) added to each target turn. With `--no_reasoning` the
examples are (context, response) pairs: the retrieval-only baseline.

### `recon.evaluate` and `recon.summarize`

```bash
python -m recon.evaluate --data <sessions.json> --split <splits.json> \
  --generations_a <a.json> --generations_b <b.json> --output <judged.json>
python -m recon.summarize --path <judged.json>
```

For every test turn, the judge sees the context, the ground-truth action and the two generated actions.
The A/B order is randomized per turn and undone afterwards. The judge picks a winner (`A`, `B`, `tie`,
`tie (bad)`, or `NA` per dimension) for style, intent, values and overall. Winners are recorded under
the file stems of `--generations_a` / `--generations_b`, so the two files need different names.
`summarize` reports the win rate of A excluding ties, with a 95% normal-approximation interval, for
each dimension and overall.
