# Domain configs

The prompt templates in [`recon/prompts.py`](../recon/prompts.py) are the same for every domain. A
domain config only supplies a few sentences describing who the target user is and what kind of
conversation they are in. These are inserted at the top of the prompts. Pass a config with
`--domain_config` to `recon.synthesize`, `recon.generate` and `recon.grpo.train`.

[`example.json`](example.json) is the config for the example dataset in [`data/example/`](../data/example/):
a podcast host talking with guests. To run Recon on a new domain, copy it and rewrite each field.

## Fields

All five keys are required; each value is a string.

| Key | Used in | What to write |
|---|---|---|
| `reasoning_preamble` | Reasoning synthesis: Backward Synthesis, Recon-Select candidates, the Recon-GRPO policy | Address the model as the target user. Say that its task is to reconstruct its internal thought process, what the context is, and that it is also given its next-turn response. |
| `action_preamble` | Action generation (`recon.generate`) and action reconstruction in Recon-Select | Address the model as the target user taking part in the conversation, and say what the current context is. |
| `reconstruction_preamble` | Action reconstruction inside the Recon-GRPO reward | Like `action_preamble`, but also say that it is given its thought process and should write a next-turn response consistent with it. |
| `judge_preamble` | Alignment judge inside the Recon-GRPO reward | Address the judge in the third person: say what the context is, that it is given the target user's ground-truth next turn and a list of generated responses, and that it should compare them. |
| `response_format` | Action generation and reconstruction prompts | One sentence describing the medium of the action, e.g. "It is a transcribed spoken turn in an oral conversation." or "It is a comment on a post on r/AmItheAsshole." |

The preambles are written in terms of the user's **role** (a podcast host, a commenter on a forum,
a judge in an oral argument). We don't use the user's name to avoid pre-existing biases. The user's
own earlier turns are relabeled `I` in the context rather than a name.

## Example

```json
{
  "reasoning_preamble": "You are a podcast host. Your current task is to reconstruct your internal thought process for while you were hosting a guest on your podcast. The provided current context is a podcast conversation. You are also given your next-turn spoken response to the current conversation context.",
  "action_preamble": "You are a podcast host participating in a conversation with a guest on your podcast. The provided current context is a podcast conversation.",
  "reconstruction_preamble": "You are a podcast host participating in a conversation with a guest on your podcast. The provided current context is a podcast conversation. You are provided with your thought process that connects the provided conversation with your next-turn response. Generate your next-turn spoken response that would be consistent with the provided conversation and thought process.",
  "judge_preamble": "You are given a podcast conversation between a podcast host and their guest. You are also given the podcast host's next-turn ground truth response to the provided context, along with a list of generated responses. Your task is to analyze the ground truth and compare the generated responses to the ground truth response.",
  "response_format": "It is a transcribed spoken turn in an oral conversation."
}
```
