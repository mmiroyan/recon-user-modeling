"""Retrieve similar past (context, action) pairs for every target turn in the test split.

Each training target turn becomes one chunk. The chunk's key is the context preceding the
turn (greedily packed, newest first, up to ``--chunk_tokens``); its value is that context
plus the target turn. Test queries are built the same way and matched by cosine similarity.

    python -m recon.retrieve \\
        --data data/example/sessions.json --split data/example/splits.json \\
        --train_splits train_0 train_1 --test_splits test_0 \\
        --output runs/example/retrievals.json
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from recon.common import (
    compile_turn_filters,
    is_target_author,
    load_json,
    load_split_ids,
    passes_filters,
    save_json,
    target_author_of,
)


@dataclass
class RetrievedChunk:
    rank: int
    score: float
    session_id: str
    start_turn_idx: int
    end_turn_idx: int   # index of the target-author turn (inclusive)
    turns: list[dict]   # [{author, action}]: context + target turn


def _format_turns(turns: list[dict]) -> str:
    return "\n".join(f"{t['author']}: {t['action']}" for t in turns)


def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _pack_preceding(turns: list[dict], turn_idx: int, tokenizer, max_tokens: int) -> int:
    """Index of the earliest turn that fits when packing turns before ``turn_idx`` newest-first."""
    start = turn_idx
    budget = max_tokens
    for t in reversed(range(turn_idx)):
        n = _token_len(f"{turns[t]['author']}: {turns[t]['action']}", tokenizer)
        if n > budget:
            break
        start = t
        budget -= n
    return start


def build_index(
    train_sessions: dict,
    target_author: str,
    tokenizer,
    embed_model,
    chunk_tokens: int,
    batch_size: int,
    device: str,
    filters: dict[str, re.Pattern],
) -> tuple[np.ndarray, list[dict]]:
    chunks: list[dict] = []
    keys: list[str] = []
    for session_id, session in train_sessions.items():
        turns = session["data"]
        for i, turn in enumerate(turns):
            if not is_target_author(turn["author"], target_author):
                continue
            if not passes_filters(turn["action"], filters):
                continue
            start = _pack_preceding(turns, i, tokenizer, chunk_tokens)
            chunks.append({"session_id": session_id, "ctx_start": start, "target_turn_idx": i})
            keys.append(_format_turns(turns[start:i]))
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32), []
    embs = embed_model.encode(
        keys, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size, device=device,
    )
    return np.array(embs, dtype=np.float32), chunks


def retrieve(
    queries: list[str],
    index_embs: np.ndarray,
    chunk_meta: list[dict],
    embed_model,
    train_sessions: dict,
    top_k: int,
    batch_size: int,
    device: str,
) -> list[list[RetrievedChunk]]:
    if not queries or index_embs.shape[0] == 0:
        return [[] for _ in queries]
    query_embs = embed_model.encode(
        queries, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size, device=device,
    )
    scores = query_embs @ index_embs.T
    results: list[list[RetrievedChunk]] = []
    for q_scores in tqdm(scores, total=len(queries), unit="query", desc="Retrieving"):
        retrieved: list[RetrievedChunk] = []
        for idx in np.argsort(q_scores)[::-1][:top_k]:
            meta = chunk_meta[idx]
            start, end = meta["ctx_start"], meta["target_turn_idx"]
            turns = train_sessions[meta["session_id"]]["data"]
            retrieved.append(RetrievedChunk(
                rank=len(retrieved),
                score=float(q_scores[idx]),
                session_id=meta["session_id"],
                start_turn_idx=start,
                end_turn_idx=end,
                turns=[{"author": t["author"], "action": t["action"]} for t in turns[start: end + 1]],
            ))
        results.append(retrieved)
    return results


def retrieve_for_sessions(
    test_sessions: dict,
    train_sessions: dict,
    target_author: str,
    index_embs: np.ndarray,
    chunk_meta: list[dict],
    tokenizer,
    embed_model,
    query_tokens: int,
    top_k: int,
    batch_size: int,
    device: str,
    filters: dict[str, re.Pattern],
) -> dict:
    """``{session_id: {turn_idx: [RetrievedChunk, ...]}}`` for every target turn in ``test_sessions``."""
    query_keys: list[tuple[str, int]] = []
    query_texts: list[str] = []
    for session_id, session in test_sessions.items():
        turns = session["data"]
        for i, turn in enumerate(turns):
            if not is_target_author(turn["author"], target_author):
                continue
            if not passes_filters(turn["action"], filters):
                continue
            start = _pack_preceding(turns, i, tokenizer, query_tokens)
            if start == i:
                continue  # no preceding context fits
            query_keys.append((session_id, i))
            query_texts.append(_format_turns(turns[start:i]))

    results = retrieve(query_texts, index_embs, chunk_meta, embed_model, train_sessions, top_k, batch_size, device)
    out: dict = {}
    for (session_id, turn_idx), chunks in zip(query_keys, results):
        out.setdefault(session_id, {})[str(turn_idx)] = [asdict(c) for c in chunks]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Sessions JSON for one target user")
    p.add_argument("--split", required=True, help="Split JSON mapping split names to session IDs")
    p.add_argument("--train_splits", nargs="+", default=["train_0", "train_1"],
                   help="Splits that form the retrieval pool")
    p.add_argument("--test_splits", nargs="+", default=["test_0"], help="Splits to retrieve for")
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="HF tokenizer used for token budgets")
    p.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--chunk_tokens", type=int, default=1024, help="Context budget for index keys")
    p.add_argument("--query_tokens", type=int, default=2048, help="Context budget for queries")
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--index_cache", default=None, help="Directory to cache the embedded index")
    p.add_argument("--turn_filters", nargs="*", default=None,
                   help="Drop target turns matching these filters (inaudible, interrupted)")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    sessions = load_json(args.data)
    target_author = target_author_of(sessions)
    filters = compile_turn_filters(args.turn_filters)
    train_ids = load_split_ids(args.split, args.train_splits)
    test_ids = load_split_ids(args.split, args.test_splits)
    train_sessions = {sid: sessions[sid] for sid in train_ids if sid in sessions}
    test_sessions = {sid: sessions[sid] for sid in test_ids if sid in sessions}
    print(f"Target author: {target_author} | train sessions: {len(train_sessions)} | "
          f"test sessions: {len(test_sessions)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=os.getenv("HF_TOKEN"))
    embed_model = SentenceTransformer(args.embedding_model, model_kwargs={"dtype": torch.float16})

    cache = Path(args.index_cache) if args.index_cache else None
    if cache and (cache / "embeddings.npz").exists() and (cache / "chunks.pkl").exists():
        index_embs = np.load(cache / "embeddings.npz")["embeddings"]
        with open(cache / "chunks.pkl", "rb") as f:
            chunk_meta = pickle.load(f)
    else:
        index_embs, chunk_meta = build_index(
            train_sessions, target_author, tokenizer, embed_model,
            args.chunk_tokens, args.batch_size, args.device, filters,
        )
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache / "embeddings.npz", embeddings=index_embs)
            with open(cache / "chunks.pkl", "wb") as f:
                pickle.dump(chunk_meta, f)
    print(f"Index size: {len(chunk_meta)} chunks")

    out = retrieve_for_sessions(
        test_sessions, train_sessions, target_author, index_embs, chunk_meta, tokenizer, embed_model,
        args.query_tokens, args.top_k, args.batch_size, args.device, filters,
    )
    save_json(args.output, out)
    print(f"Wrote retrievals for {sum(len(v) for v in out.values())} test turns to {args.output}")


if __name__ == "__main__":
    main()
