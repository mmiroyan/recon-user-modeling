"""Data loading, splitting, and filtering for Recon-GRPO."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer


TURN_FILTERS: dict[str, str] = {
    "inaudible": r"(?i)inaudible",
    "interrupted": r"--",
}


_AUG_SUFFIX = re.compile(r"_(\d+)$")


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(base_tokenizer: str | None, base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(
        base_tokenizer or base_model, token=os.getenv("HF_TOKEN")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def is_target_author(author: str, target_author: str) -> bool:
    author_norm = author.strip().lower()
    target_norm = target_author.strip().lower()
    return (target_norm in author_norm) or (author_norm in target_norm)


def original_session_root(session_id: str) -> str:
    """Strip trailing ``_123`` augmentation suffix for grouping train/val by conversation root."""
    m = _AUG_SUFFIX.search(session_id)
    return session_id[: m.start()] if m else session_id


def select_sessions(
    session_ids: list[str],
    *,
    proportion: float,
    mode: str,
    max_sessions: Optional[int],
    seed: int,
) -> list[str]:
    sorted_ids = sorted(session_ids)
    n = round(proportion * len(sorted_ids))
    if max_sessions is not None:
        n = min(n, max_sessions)
    n = max(n, 0)
    if mode == "start":
        return sorted_ids[:n]
    if mode == "end":
        return sorted_ids[-n:] if n > 0 else []
    return random.Random(seed).sample(sorted_ids, min(n, len(sorted_ids)))


def _split_original_roots_train_val(
    original_keys: list[str],
    *,
    seed: int,
    num_val_sessions: int,
    max_train_sessions: Optional[int],
) -> tuple[list[str], list[str]]:
    if num_val_sessions < 0:
        raise ValueError("--num_val_sessions must be >= 0")
    keys = list(original_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    if num_val_sessions > len(keys):
        raise ValueError(
            f"--num_val_sessions ({num_val_sessions}) exceeds number of conversation roots ({len(keys)})."
        )
    val_roots = keys[:num_val_sessions]
    rest = keys[num_val_sessions:]
    if not rest:
        raise ValueError(
            "No training roots left after val split; use a smaller --num_val_sessions."
        )
    if max_train_sessions is not None:
        if len(rest) < max_train_sessions:
            raise ValueError(
                f"--max_train_sessions ({max_train_sessions}) too large: only {len(rest)} roots "
                f"remain after reserving {num_val_sessions} for val (total roots {len(keys)})."
            )
        train_roots = rest[:max_train_sessions]
    else:
        train_roots = rest
    return train_roots, val_roots


def split_session_keys_by_original_roots(
    selected_ids: list[str],
    *,
    seed: int,
    num_val_sessions: int,
    max_train_sessions: Optional[int],
) -> tuple[list[str], list[str]]:
    original_members: dict[str, list[str]] = {}
    seen: set[str] = set()
    for sid in selected_ids:
        if sid in seen:
            continue
        seen.add(sid)
        original_members.setdefault(original_session_root(sid), []).append(sid)
    for oid in original_members:
        original_members[oid].sort()
    original_keys = sorted(original_members.keys())
    train_roots, val_roots = _split_original_roots_train_val(
        original_keys,
        seed=seed,
        num_val_sessions=num_val_sessions,
        max_train_sessions=max_train_sessions,
    )
    train_keys = [sid for o in train_roots for sid in original_members[o]]
    val_keys = [sid for o in val_roots for sid in original_members[o]]
    return train_keys, val_keys


def collect_split_session_ids(split: dict, split_names: list[str]) -> set[str]:
    ids: set[str] = set()
    for name in split_names:
        if name not in split:
            raise ValueError(f"Split name '{name}' not found. Available: {sorted(split)}")
        ids.update(split[name])
    return ids


def enumerate_turn_jobs(
    sessions: dict,
    allowed_session_ids: Optional[set[str]],
    turn_filter_names: Optional[list[str]],
) -> list[tuple[str, int]]:
    """``(session_id, turn_idx)`` for each target-author turn surviving the regex filters."""
    compiled = {name: re.compile(TURN_FILTERS[name]) for name in tuple(turn_filter_names or ())}
    jobs: list[tuple[str, int]] = []
    ids_iter = sorted(allowed_session_ids) if allowed_session_ids is not None else sorted(sessions.keys())
    for session_id in ids_iter:
        if session_id not in sessions:
            continue
        session = sessions[session_id]
        target_author = session["metadata"]["target_author_name"]
        for t, turn in enumerate(session["data"]):
            if not is_target_author(turn["author"], target_author):
                continue
            if compiled and any(p.search(turn["action"]) for p in compiled.values()):
                continue
            jobs.append((session_id, t))
    return jobs


def subsample_jobs(
    jobs: list[tuple[str, int]],
    cap: Optional[int],
    rng: random.Random,
) -> list[tuple[str, int]]:
    if cap is None:
        return list(jobs)
    if len(jobs) > cap:
        return rng.sample(jobs, cap)
    out = list(jobs)
    rng.shuffle(out)
    return out
