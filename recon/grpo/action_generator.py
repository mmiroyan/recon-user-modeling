"""Litellm-backed action generator (Step 3 of PLAN.md).

Fixed API model (e.g. ``gpt-4o-mini``). Given (context, reasoning_trace) -> action.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import litellm

from recon.grpo.prompt_format import (
    PromptConfig,
    build_action_gen_prompt,
)


logger = logging.getLogger(__name__)


def generate_action(
    *,
    context: str,
    reasoning_trace: str,
    model: str,
    prompt_config: Optional[PromptConfig] = None,
    temperature: float = 0.7,
    max_retries: int = 3,
    api_base: Optional[str] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Return the generated action text, or ``None`` on failure."""
    user = build_action_gen_prompt(
        context=context,
        reasoning_trace=reasoning_trace,
        prompt_config=prompt_config,
    )
    kw: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "temperature": temperature,
    }
    if api_base:
        kw["api_base"] = api_base
    if extra_kwargs:
        kw.update(extra_kwargs)
    attempts = 1 + max_retries
    for attempt in range(attempts):
        try:
            resp = litellm.completion(**kw)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("action_generator failed (attempt %d/%d): %s", attempt + 1, attempts, e)
    return None
