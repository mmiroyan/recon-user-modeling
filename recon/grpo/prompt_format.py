"""Prompt templates and context packing for Recon-GRPO."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from recon.prompts import REASONING_SYNTHESIS as REASONING_GEN_PROMPT


@dataclass(frozen=True)
class PromptConfig:
    preamble_reasoning_gen: str = ""
    preamble_action_gen: str = ""
    preamble_judge: str = ""
    response_format: str = ""
    target_alias: str = "I"


def load_prompt_config(path: str | Path) -> PromptConfig:
    """Read a domain config (configs/*.json)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PromptConfig(
        preamble_reasoning_gen=str(data.get("reasoning_preamble", "")),
        preamble_action_gen=str(data.get("reconstruction_preamble", "")),
        preamble_judge=str(data.get("judge_preamble", "")),
        response_format=str(data.get("response_format", "")),
    )


def prompt_config_to_dict(cfg: PromptConfig) -> dict[str, str]:
    return asdict(cfg)


def prompt_config_from_dict(data: dict[str, str]) -> PromptConfig:
    return PromptConfig(**data)


SANITY_CHECK_PROMPT = """\
You are given a response and reasoning trace. You need to determine whether the response **explicitly** appears in the reasoning trace.

## Input

<|Start Reasoning Trace|>
{reasoning_trace}
<|End of Reasoning Trace|>

<|Start Response|>
{ground_truth_response}
<|End of Response|>

## Instructions

- Check if the response text **explicitly** appears in the reasoning trace.
- True means the response text is present verbatim in the reasoning trace. False means it is not present verbatim.
- Consider **minor** formatting differences and paraphrases as matches.

## Output format (JSON):

{{
    "rationale": <brief explanation of your analysis>,
    "found": <true or false>
}}"""


ACTION_GEN_PROMPT = """\
{preamble_block}

## Input

<|Start Context|>
{context}
<|End of Context|>

<|Start Thought Process|>
{reasoning_trace}
<|End of Thought Process|>

## Instructions

- You are a participant in the conversation in the Current Context. It is now your turn to speak in the conversation. Generate your response to the provided context.
- The generated response should be consistent with the provided thought process.
- The generated response does not have to cover all the points mentioned in the thought process.
- The response should be from the first point of view.
- Your previous turns in the provided context (if any) have been labeled with <turn author=\"I\">. Do not get confused by other participants referring to themselves in the first person; only turns labeled with <turn author=\"I\"> are you. 
- {response_format}

## Output format

- Only output the response without any additional text or commentary.\
"""


DIM_JUDGE_PROMPT = """\
{preamble_block}

Score how well each generated response aligns with the ground truth response along three interpretive dimensions: **style**, **intent**, and **values**.

Each **alignment_score** must reflect both (a) how well the generated response matches the ground truth on that dimension and (b) any need to downweight the score because of **extra or conflicting** content relative to the ground truth—unsupported additions, contradictions, or stylistic or pragmatic moves that diverge in ways that are not harmless paraphrases.

Consider extra or conflicting content and features in each generated response relative to the ground truth when assigning each score. Verbose generated responses that include content and features not supported by the ground truth should receive **lower** scores on the relevant dimensions than more concise responses that better capture the core meaning and style of the ground truth.

## Input

<|Start Context|>
{context}
<|End of Context|>

<|Start Ground Truth Response|>
{ground_truth_response}
<|End of Ground Truth Response|>

<|Start Generated Responses|>
{generated_responses}
<|End of Generated Responses|>

## Evaluation

### Dimension definitions

- Style: The surface-level linguistic properties of the utterance, entirely independent of meaning. Consider: sentence length and syntactic complexity, formality and register, vocabulary and lexical choices, use of hedges, discourse markers, and fillers, and overall tone (assertive, tentative, emotive, dry).
- Intent: The specific propositional and pragmatic content of the utterance as a response to the given context. Consider: the information conveyed, the stance taken on the immediate topic, the speech act performed (assertion, concession, challenge, deflection, etc.), and how it addresses what the prior turn raised.
- Values: The high-level, stable outlook of the speaker as revealed by the utterance. Consider: the beliefs and assumptions about the world the speaker treats as given, the values and moral or social principles that shape their framing, and the ideological or philosophical commitments implicit in how they engage with the topic — characteristics that would remain consistent across different conversations and topics.

## Instructions

- Analyze the provided context and the ground truth response.
- Analyze all {k} generated responses and compare each one independently to the ground truth response.
- For each generated response and each of the three dimensions (**style**, **intent**, **values**):
  * Identify a few **key points** in the **ground truth response** (and context when needed) that are salient for that dimension.
  * Assign an **alignment_score** in [0, 1] for how well the generated response matches the ground truth on that dimension, already incorporating penalties for unsupported extras, contradictions, or harmful verbosity as they affect this dimension.
  * In **alignment_rationale**, briefly list the key ground-truth points you used, then explain the score with concrete comparisons to the generated response, including how extras, conflicts, or verbosity affected the score if applicable.
- Score each generated response independently. Multiple responses may receive similar or identical scores.

### Score anchors

| Score | Meaning |
|------|--------|
| 1.0 | Matches the ground truth on this dimension very closely, without meaningful unsupported extras |
| 0.7-0.9 | Strong match with minor gaps or very mild extra content |
| 0.4-0.6 | Partial match or noticeable extras or tensions |
| 0.1-0.3 | Weak match or clear conflicting or unsupported material on this dimension |
| 0.0 | Missing, contradicted, or heavily misaligned on this dimension |

## Output format (JSON)

Return a JSON object whose keys are the response indices `"1"` ... `"{k}"`. Each key-value pair must have this shape:

{{
  "1": {{
    "style":  {{"alignment_rationale": string, "alignment_score": float}},
    "intent": {{"alignment_rationale": string, "alignment_score": float}},
    "values": {{"alignment_rationale": string, "alignment_score": float}}
  }}
}}\
"""


def _render_context_xml(turns: list[dict], target_lower: str, target_alias: str | None) -> str:
    lines: list[str] = []
    for turn in turns:
        author = turn["author"]
        author_lower = author.lower()
        if target_alias is not None and (target_lower in author_lower or author_lower in target_lower):
            author = target_alias
        lines.append(f'<turn author="{author}">\n{turn["action"].strip()}\n</turn>')
    return "\n".join(lines)


def _apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("Tokenizer must have a non-empty chat_template")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _preamble_block(text: str) -> str:
    text = (text or "").strip()
    return f"{text}\n\n" if text else ""


def build_reasoning_gen_prompt(
    *,
    context: str,
    ground_truth_response: str,
    prompt_config: Optional[PromptConfig] = None,
) -> str:
    cfg = prompt_config or PromptConfig()
    return REASONING_GEN_PROMPT.format(
        preamble_block=_preamble_block(cfg.preamble_reasoning_gen),
        context=context,
        ground_truth_response=ground_truth_response,
    )


def build_action_gen_prompt(
    *,
    context: str,
    reasoning_trace: str,
    prompt_config: Optional[PromptConfig] = None,
) -> str:
    cfg = prompt_config or PromptConfig()
    return ACTION_GEN_PROMPT.format(
        preamble_block=_preamble_block(cfg.preamble_action_gen),
        context=context,
        reasoning_trace=reasoning_trace,
        response_format=cfg.response_format,
    )


def build_dim_judge_prompt(
    *,
    context: str,
    ground_truth_response: str,
    generated_responses: list[str],
    prompt_config: Optional[PromptConfig] = None,
) -> str:
    cfg = prompt_config or PromptConfig()
    responses_block = "\n\n".join(
        f"Response {i}: {r}" for i, r in enumerate(generated_responses, start=1)
    )
    return DIM_JUDGE_PROMPT.format(
        preamble_block=_preamble_block(cfg.preamble_judge),
        k=len(generated_responses),
        context=context,
        ground_truth_response=ground_truth_response,
        generated_responses=responses_block,
    )


def pack_context(
    session: dict,
    turn_idx: int,
    tokenizer,
    max_policy_tokens: int,
    prompt_config: Optional[PromptConfig] = None,
) -> str:
    if max_policy_tokens == 0:
        return ""
    cfg = prompt_config or PromptConfig()
    target_lower = session["metadata"]["target_author_name"].strip().lower()
    target_alias = cfg.target_alias or None
    turns = session["data"]
    kept_rev: list[dict] = []
    for idx in range(turn_idx - 1, -1, -1):
        kept_rev.append(turns[idx])
        trial_ctx = _render_context_xml(list(reversed(kept_rev)), target_lower, target_alias)
        ntok = len(tokenizer.encode(trial_ctx, add_special_tokens=False))
        if ntok > max_policy_tokens:
            kept_rev.pop()
            break
    if not kept_rev:
        return ""
    return _render_context_xml(list(reversed(kept_rev)), target_lower, target_alias)


def build_policy_prompt(
    context: str,
    gt_action: str,
    tokenizer,
    prompt_config: Optional[PromptConfig] = None,
) -> str:
    messages = [{
        "role": "user",
        "content": build_reasoning_gen_prompt(
            context=context,
            ground_truth_response=gt_action,
            prompt_config=prompt_config,
        ),
    }]
    return _apply_chat_template(tokenizer, messages)
