"""Normalize reasoning-trace completions from vLLM / GRPO rollouts."""

from __future__ import annotations


def normalize_reasoning_trace(completion: str) -> str:
    """Strip whitespace. The reasoning generator emits free-form prose (no tags);
    we intentionally do not strip formatting beyond surrounding whitespace.
    """
    return (completion or "").strip()
