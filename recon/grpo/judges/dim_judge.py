"""Dimension judge: score all generated responses against the ground truth at once."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import litellm

from recon.grpo.prompt_format import PromptConfig, build_dim_judge_prompt

logger = logging.getLogger(__name__)

_DIMS = ("style", "intent", "values")

_DIM_SCHEMA = {
    "type": "object",
    "properties": {
        "alignment_rationale": {"type": "string"},
        "alignment_score": {"type": "number"},
    },
    "required": ["alignment_rationale", "alignment_score"],
}

_PER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {dim: _DIM_SCHEMA for dim in _DIMS},
    "required": list(_DIMS),
}


def _response_schema(n: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {str(i): _PER_RESPONSE_SCHEMA for i in range(1, n + 1)},
        "required": [str(i) for i in range(1, n + 1)],
    }


def _parse(content: str, n: int) -> Optional[dict[str, dict[str, dict[str, Any]]]]:
    try:
        data = json.loads(content)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    for idx in range(1, n + 1):
        key = str(idx)
        item = data.get(key)
        if not isinstance(item, dict):
            return None
        parsed[key] = {}
        for dim in _DIMS:
            dim_item = item.get(dim)
            if not isinstance(dim_item, dict):
                return None
            rationale = dim_item.get("alignment_rationale")
            score = dim_item.get("alignment_score")
            if not isinstance(rationale, str):
                return None
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                return None
            if not (0.0 <= score_f <= 1.0):
                return None
            parsed[key][dim] = {"alignment_rationale": rationale, "alignment_score": score_f}
    return parsed


def judge_responses(
    *,
    context: str,
    ground_truth_response: str,
    responses: list[str],
    model: str,
    prompt_config: Optional[PromptConfig] = None,
    max_retries: int = 3,
    temperature: float = 0.0,
    api_base: Optional[str] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[tuple[dict[str, dict[str, dict[str, Any]]], list[float]]]:
    n = len(responses)
    if n == 0:
        return None
    user = build_dim_judge_prompt(
        context=context,
        ground_truth_response=ground_truth_response,
        generated_responses=responses,
        prompt_config=prompt_config,
    )
    kw: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "dim_judge_response", "schema": _response_schema(n)},
        },
    }
    if api_base:
        kw["api_base"] = api_base
    if extra_kwargs:
        kw.update(extra_kwargs)
    attempts = max(1, 1 + int(max_retries))
    for attempt in range(attempts):
        try:
            resp = litellm.completion(**kw)
        except Exception as e:  # noqa: BLE001
            logger.warning("dim_judge litellm call failed (attempt %d/%d): %s", attempt + 1, attempts, e)
            continue
        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            continue
        parsed = _parse(content, n)
        if parsed is None:
            logger.warning("dim_judge schema mismatch (attempt %d/%d, n=%d):\n%s", attempt + 1, attempts, n, content)
            continue
        avg_scores = [
            sum(parsed[str(i)][dim]["alignment_score"] for dim in _DIMS) / len(_DIMS)
            for i in range(1, n + 1)
        ]
        return parsed, avg_scores
    return None
