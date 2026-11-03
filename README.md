# Recon: Reconstruction-Guided Reasoning Synthesis for User Modeling

Alan Zhu\*, Mihran Miroyan\*, Carolyn Wang, Andrew Zhou, Lisa Dunlap, Narges Norouzi, Joseph E. Gonzalez
<br>NeurIPS 2026 · [Paper](https://arxiv.org/abs/2605.26969)

*Equal contribution.

User modeling uses a language model to predict what a specific person would say or do next, given their
past context–action pairs. A common way to help is to augment those past examples with synthesized
reasoning traces. These traces are usually generated from both the context and the observed action, so
they are post-hoc rationalizations: consistent with the action, but not necessarily the reasoning that
leads to it.

**Recon** scores a candidate trace by **action reconstruction**: given only the context and the trace,
can a model reconstruct the user's actual action? The more closely the reconstruction matches the real
action, the better the trace. We use this signal in two ways:

- **Recon-Select** (training-free): sample N rationalizations per example, reconstruct the action from
  each, and keep the trace whose reconstruction best matches the ground truth.
- **Recon-GRPO**: train the reasoning model with GRPO, using reconstruction fidelity as the reward.

The traces augment a retrieval corpus. Downstream, an action model predicts the user's next action from
retrieved (context, reasoning, action) examples.

## Repository layout

```text
recon/            the method: retrieval, reasoning synthesis, action generation, evaluation
recon/grpo/       Recon-GRPO training
configs/          domain prompt config: an example plus a README on writing your own
data/             data format documentation and a small made-up example dataset
scripts/          example scripts for end-to-end Recon-Select and Recon-GRPO
```

- [`recon/README.md`](recon/README.md): pipeline stages, command-line options and file formats
- [`recon/grpo/README.md`](recon/grpo/README.md): Recon-GRPO reward and training
- [`data/README.md`](data/README.md): input data and splits format
- [`configs/README.md`](configs/README.md): writing a domain config for your data

## Setup

```bash
uv sync                      # method + evaluation
uv sync --extra training     # + Recon-GRPO training (TRL, PEFT)
uv sync --extra serve        # + vLLM for serving local models (Linux)
source .venv/bin/activate
cp .env.example .env         # then fill in API keys
```

All model calls go through [litellm](https://docs.litellm.ai/docs/providers). Any hosted model or
OpenAI-compatible server works for the reasoning, action and judge models. The paper uses Qwen3 and
Llama 3.1 models served with vLLM, GPT-5-mini, and Gemini 3.1 Flash-Lite as the judge.

## Quickstart: Recon-Select

Serve the reasoning/action model, then run the full pipeline on the example data:

```bash
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --port 8000
bash scripts/demo_recon_select.sh
```

The demo runs five stages:

1. retrieves similar past (context, action) pairs for every test turn;
2. augments them with Backward Synthesis (the baseline) and with Recon-Select;
3. predicts every test action from each augmented corpus;
4. has the LM judge compare the two sets of predictions against the ground truth;
5. prints win rates.

Outputs go to `runs/example/`. To use your own data, set `DATA`, `SPLIT` and `DOMAIN` (see the top of
the script). The example dataset in this repo is tiny and serves only to demo the code.

## Quickstart: Recon-GRPO

```bash
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --port 8000       # frozen action model
bash scripts/demo_recon_grpo.sh train                               # GRPO on train_1
vllm serve Qwen/Qwen3-4B --reasoning-parser qwen3 --port 8001 \
  --enable-lora --max-lora-rank 128 --lora-modules recon-grpo=runs/example/recon_grpo/final
bash scripts/demo_recon_grpo.sh evaluate                            # vs. Backward Synthesis
```

`evaluate` reuses `runs/example/retrievals.json` from the Recon-Select demo. See
[`recon/grpo/README.md`](recon/grpo/README.md) for details.

## Data

For privacy and ethical reasons, we do not release the datasets from the paper. The paper datasets were
built from U.S. Supreme Court oral arguments, U.K. Prime Minister's Questions, podcast interviews, and
Reddit (r/AmItheAsshole), with two individuals per domain. [`data/README.md`](data/README.md) describes
the expected format, which is one JSON file of conversations per target user plus a split file.

## Citation

```bibtex
@inproceedings{zhu2026recon,
  title     = {{Recon}: Reconstruction-Guided Reasoning Synthesis for User Modeling},
  author    = {Zhu, Alan and Miroyan, Mihran and Wang, Carolyn and Zhou, Andrew and
               Dunlap, Lisa and Norouzi, Narges and Gonzalez, Joseph E.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
