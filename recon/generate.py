"""Predict the target user's action for every test turn from retrieved examples (Eq. 1).

The action model M_a sees the retrieved (context, action) pairs, optionally augmented with
synthesized reasoning, followed by the current context. It writes a thought process and a
response; the response is stored as ``generated_action`` on the test turn.

    python -m recon.generate \\
        --data data/example/sessions.json --split data/example/splits.json \\
        --retrievals runs/example/recon_select.json --domain_config configs/example.json \\
        --output runs/example/generations/recon_select.json \\
        --model hosted_vllm/Qwen/Qwen3-8B --api_base http://localhost:8000/v1

Use ``--no_reasoning`` to drop the reasoning traces (retrieval-only RAG baseline).
"""

from __future__ import annotations

import argparse
import os
import re
import time

from dotenv import load_dotenv
from tqdm import tqdm

from recon.common import (
    LLM,
    RETRY_BACKOFF,
    compile_turn_filters,
    format_turn,
    is_target_author,
    load_domain_config,
    load_json,
    load_split_ids,
    passes_filters,
    preamble_block,
    resolve_author,
    run_parallel,
    save_json,
    target_author_of,
)
from recon.prompts import RAG_PROMPT, RAG_PROMPT_WITH_REASONING

# Closing tags may be written as <|... End|> or </|... End|>.
THOUGHT_RE = re.compile(r"<\|Thought Process Start\|>(.*?)</?\|Thought Process End\|>", re.DOTALL)
RESPONSE_RE = re.compile(r"<\|Response Start\|>(.*?)(?:</?\|Response End\|>|$)", re.DOTALL)
MAX_ATTEMPTS = 3


def format_active_context(
    turns: list[dict], turn_idx: int, target_author: str, target_alias: str | None, tokenizer, max_tokens: int,
) -> str:
    """Pack the turns before ``turn_idx`` newest-first up to ``max_tokens``."""
    selected: list[str] = []
    budget = max_tokens
    for t in reversed(range(turn_idx)):
        text = format_turn(resolve_author(turns[t]["author"], target_author, target_alias), turns[t]["action"])
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n > budget:
            break
        selected.append(text)
        budget -= n
    return "\n".join(reversed(selected))


def format_retrieved_examples(
    chunks: list[dict], target_author: str, target_alias: str | None, include_reasoning: bool,
) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        context = "\n".join(
            format_turn(resolve_author(t["author"], target_author, target_alias), t["action"])
            for t in chunk["turns"][:-1]
        )
        lines = [f"<|Example {i} Start|>", "<|Context Start|>", context, "<|Context End|>"]
        if include_reasoning:
            lines += ["<|Thought Process Start|>", chunk.get("reasoning", ""), "<|Thought Process End|>"]
        lines += ["<|Response Start|>", chunk["turns"][-1]["action"], "<|Response End|>", f"<|Example {i} End|>"]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def build_prompt(
    session: dict, turn_idx: int, chunks: list[dict], cfg: dict, target_author: str,
    target_alias: str | None, tokenizer, max_context_tokens: int, include_reasoning: bool,
) -> str:
    template = RAG_PROMPT_WITH_REASONING if include_reasoning else RAG_PROMPT
    return template.format(
        preamble_block=preamble_block(cfg["action_preamble"]),
        retrieved_examples=format_retrieved_examples(chunks, target_author, target_alias, include_reasoning),
        active_context=format_active_context(
            session["data"], turn_idx, target_author, target_alias, tokenizer, max_context_tokens,
        ),
        response_format=cfg["response_format"],
    )


def generate_one(prompt: str, llm: LLM) -> tuple[str, str]:
    """Return (thought process, response); retry until exactly one of each is produced."""
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            time.sleep(RETRY_BACKOFF[attempt - 1])
        try:
            content = llm.complete(prompt, seed=42)
        except Exception as exc:
            tqdm.write(f"  [retry {attempt + 1}/{MAX_ATTEMPTS}] call error: {exc}")
            continue
        thoughts, responses = THOUGHT_RE.findall(content), RESPONSE_RE.findall(content)
        if len(thoughts) == 1 and len(responses) == 1:
            return thoughts[0].strip(), responses[0].strip()
        tqdm.write(f"  [retry {attempt + 1}/{MAX_ATTEMPTS}] parse failed: "
                   f"{len(thoughts)} thought(s), {len(responses)} response(s)")
    return "", ""


def generate_for_sessions(
    test_sessions: dict, retrievals: dict, cfg: dict, target_author: str, target_alias: str | None,
    tokenizer, max_context_tokens: int, include_reasoning: bool, filters: dict, llm: LLM, max_workers: int,
) -> None:
    """Write ``generated_action`` (and ``generated_reasoning``) onto each target turn in place."""
    tasks = []
    for sid, session in test_sessions.items():
        for t, turn in enumerate(session["data"]):
            if not is_target_author(turn["author"], target_author) or not passes_filters(turn["action"], filters):
                continue
            chunks = retrievals.get(sid, {}).get(str(t), [])
            prompt = build_prompt(session, t, chunks, cfg, target_author, target_alias,
                                  tokenizer, max_context_tokens, include_reasoning)
            tasks.append(((sid, t), lambda p=prompt: generate_one(p, llm)))

    results = run_parallel(tasks, max_workers, "Generating")
    for (sid, t), (thought, response) in results.items():
        turn = test_sessions[sid]["data"][t]
        turn["generated_action"] = response
        if thought:
            turn["generated_reasoning"] = thought


def _bool(x: str) -> bool:
    return {"true": True, "false": False}[x.lower()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--test_splits", nargs="+", default=["test_0"])
    p.add_argument("--retrievals", required=True,
                   help="Output of recon.retrieve or recon.synthesize")
    p.add_argument("--domain_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no_reasoning", action="store_true",
                   help="Ignore reasoning traces: plain retrieval-augmented generation")
    p.add_argument("--target_alias", default="I")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="HF tokenizer used for the context budget")
    p.add_argument("--max_context_tokens", type=int, default=2048)
    p.add_argument("--turn_filters", nargs="*", default=None)
    p.add_argument("--model", required=True, help="litellm model string for the action model M_a")
    p.add_argument("--api_base", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--enable_thinking", type=_bool, default=None)
    p.add_argument("--reasoning_effort", default=None)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_workers", type=int, default=32)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    from transformers import AutoTokenizer

    sessions = load_json(args.data)
    target_author = target_author_of(sessions)
    test_sessions = {sid: sessions[sid] for sid in load_split_ids(args.split, args.test_splits) if sid in sessions}
    retrievals = load_json(args.retrievals)
    cfg = load_domain_config(args.domain_config)
    filters = compile_turn_filters(args.turn_filters)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=os.getenv("HF_TOKEN"))
    llm = LLM(args.model, args.api_base, args.temperature, args.timeout,
              args.reasoning_effort, args.enable_thinking)
    print(f"Target author: {target_author} | test sessions: {len(test_sessions)}")
    generate_for_sessions(
        test_sessions, retrievals, cfg, target_author, args.target_alias, tokenizer,
        args.max_context_tokens, not args.no_reasoning, filters, llm, args.max_workers,
    )
    save_json(args.output, test_sessions)
    print(f"Wrote generations to {args.output}")


if __name__ == "__main__":
    main()
