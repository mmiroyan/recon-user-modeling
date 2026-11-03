"""Sanity-check judge (PLAN.md Step 2): is the ground-truth action verbatim in the trace?"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import litellm
from pydantic import BaseModel, Field

from recon.grpo.prompt_format import SANITY_CHECK_PROMPT


logger = logging.getLogger(__name__)


_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "found": {"type": "boolean"},
    },
    "required": ["rationale", "found"],
}


class SanityCheckResult(BaseModel):
    rationale: str = Field(description="Brief explanation of the analysis")
    found: bool = Field(description="True if the action appears in the reasoning trace")


def _parse(content: str) -> Optional[SanityCheckResult]:
    try:
        data = json.loads(content)
        return SanityCheckResult(**data)
    except Exception:  # noqa: BLE001
        return None


def check_verbatim(
    *,
    gt_action: str,
    reasoning_trace: str,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 3,
    api_base: Optional[str] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[SanityCheckResult]:
    """Return the parsed judge result, or ``None`` if all attempts fail."""
    user = SANITY_CHECK_PROMPT.format(
        ground_truth_response=gt_action, reasoning_trace=reasoning_trace
    )
    kw: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "sanity_check_response", "schema": _SCHEMA},
        },
    }
    if api_base:
        kw["api_base"] = api_base
    if extra_kwargs:
        kw.update(extra_kwargs)
    attempts = 1 + max_retries
    for attempt in range(attempts):
        try:
            resp = litellm.completion(**kw)
        except Exception as e:  # noqa: BLE001
            logger.warning("sanity_check failed (attempt %d/%d): %s", attempt + 1, attempts, e)
            continue
        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            continue
        result = _parse(content)
        if result is None:
            logger.warning("sanity_check parse failed (attempt %d/%d): %s", attempt + 1, attempts, content[:200])
            continue
        return result
    return None
