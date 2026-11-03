"""Build GRPO training rows for reasoning synthesis.

Each row = one target-author turn. Row schema:
    {prompt, context, gt_action, target_author, session_id, turn_idx}
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from tqdm import tqdm

from recon.grpo.data import (
    TURN_FILTERS,
    is_target_author,
    load_tokenizer,
)
from recon.grpo.prompt_format import (
    PromptConfig,
    build_policy_prompt,
    pack_context,
    prompt_config_from_dict,
    prompt_config_to_dict,
)


_PARALLEL_TOKENIZER = None


def _parallel_init(base_tokenizer: Optional[str], base_model: str) -> None:
    global _PARALLEL_TOKENIZER
    _PARALLEL_TOKENIZER = load_tokenizer(base_tokenizer, base_model)


def _build_one_session(
    session_id: str,
    session: dict,
    tokenizer,
    compiled_filters: dict[str, re.Pattern],
    max_context_length: int,
    turn_indices: Optional[list[int]],
    prompt_config: Optional[PromptConfig],
) -> list[dict]:
    target_author = session["metadata"]["target_author_name"]
    turns = session["data"]
    if turn_indices is None:
        iter_indices = list(range(len(turns)))
    else:
        iter_indices = sorted(set(t for t in turn_indices if 0 <= t < len(turns)))
    out: list[dict] = []
    for t in iter_indices:
        turn = turns[t]
        if not is_target_author(turn["author"], target_author):
            continue
        if compiled_filters and any(p.search(turn["action"]) for p in compiled_filters.values()):
            continue
        gt_action = turn["action"].strip()
        context = pack_context(
            session, t, tokenizer, max_context_length, prompt_config=prompt_config
        )
        prompt = build_policy_prompt(
            context, gt_action, tokenizer, prompt_config=prompt_config
        )
        out.append(
            {
                "prompt": prompt,
                "context": context,
                "gt_action": gt_action,
                "target_author": target_author,
                "session_id": session_id,
                "turn_idx": t,
            }
        )
    return out


def _worker(payload: tuple) -> list[dict]:
    global _PARALLEL_TOKENIZER
    if _PARALLEL_TOKENIZER is None:
        raise RuntimeError("parallel worker: tokenizer not initialized")
    session_id, session, max_context_length, turn_filter_tuple, turn_indices, prompt_config_dict = payload
    compiled = {name: re.compile(TURN_FILTERS[name]) for name in turn_filter_tuple}
    prompt_config = prompt_config_from_dict(prompt_config_dict)
    return _build_one_session(
        session_id, session, _PARALLEL_TOKENIZER, compiled, max_context_length, turn_indices, prompt_config
    )


def build_examples(
    sessions: dict,
    tokenizer,
    max_context_length: int,
    *,
    allowed_session_ids: Optional[set[str]] = None,
    turn_filter_names: Optional[list[str]] = None,
    turn_jobs: Optional[list[tuple[str, int]]] = None,
    num_workers: int = 0,
    parallel_base_tokenizer: Optional[str] = None,
    parallel_base_model: Optional[str] = None,
    prompt_config: Optional[PromptConfig] = None,
) -> list[dict]:
    turn_tuple = tuple(turn_filter_names or ())

    if turn_jobs is not None:
        allowed = allowed_session_ids or set(sessions.keys())
        by_session: dict[str, list[int]] = {}
        for sid, t in turn_jobs:
            if sid not in allowed or sid not in sessions:
                continue
            by_session.setdefault(sid, []).append(t)
        for sid in by_session:
            by_session[sid].sort()
        session_pairs = [(sid, sessions[sid]) for sid in sorted(by_session)]
    else:
        session_pairs = [
            (sid, sessions[sid])
            for sid in sorted(sessions)
            if allowed_session_ids is None or sid in allowed_session_ids
        ]
        by_session = None

    n_sessions = len(session_pairs)

    if num_workers > 1:
        reload_model = parallel_base_model or getattr(tokenizer, "name_or_path", "") or ""
        if not reload_model:
            raise ValueError(
                "build_examples(..., num_workers>1) needs parallel_base_model or tokenizer.name_or_path"
            )
        payloads = []
        for sid, sess in session_pairs:
            indices = by_session[sid] if by_session is not None else None
            payloads.append(
                (sid, sess, max_context_length, turn_tuple, indices, prompt_config_to_dict(prompt_config or PromptConfig()))
            )
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_parallel_init,
            initargs=(parallel_base_tokenizer, reload_model),
        ) as ex:
            nested = list(
                tqdm(
                    ex.map(_worker, payloads),
                    total=n_sessions,
                    desc="build_examples (parallel)",
                    unit="session",
                )
            )
        return [row for block in nested for row in block]

    compiled = {name: re.compile(TURN_FILTERS[name]) for name in turn_tuple}
    out: list[dict] = []
    for sid, sess in tqdm(session_pairs, total=n_sessions, desc="build_examples", unit="session"):
        indices = by_session[sid] if by_session is not None else None
        out.extend(_build_one_session(sid, sess, tokenizer, compiled, max_context_length, indices, prompt_config))
    return out
