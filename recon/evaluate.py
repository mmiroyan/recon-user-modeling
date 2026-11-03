"""Pairwise LM-judge evaluation of two sets of generated actions against the ground truth.

For every test turn, the judge sees the context, the ground-truth action and the two generated
actions (A/B order randomized per turn) and picks a winner on style, intent, values and overall.
Winners are stored under the file stems of ``--generations_a`` / ``--generations_b``.

    python -m recon.evaluate \\
        --data data/example/sessions.json --split data/example/splits.json \\
        --generations_a runs/example/generations/recon_select.json \\
        --generations_b runs/example/generations/backward.json \\
        --output runs/example/eval/recon_select_vs_backward.json

Then summarize win rates with ``python -m recon.summarize --path <output>``.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from pathlib import Path
from typing import Literal

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

from recon.common import (
    compile_turn_filters,
    load_json,
    load_split_ids,
    passes_filters,
    run_parallel,
    save_json,
)
from recon.prompts import PAIRWISE_DIMENSION_DEFINITIONS, PAIRWISE_JUDGE_PROMPT

DIMENSIONS = ("style", "intent", "values")
MAX_ATTEMPTS = 3


class DimensionWinner(BaseModel):
    rationale: str
    winner: Literal["A", "B", "tie", "tie (bad)", "NA"]


class OverallWinner(BaseModel):
    rationale: str
    winner: Literal["A", "B", "tie", "tie (bad)"]


class PairwiseJudgement(BaseModel):
    style: DimensionWinner
    intent: DimensionWinner
    values: DimensionWinner
    overall: OverallWinner


def is_target(author: str, target_author: str) -> bool:
    a, t = author.strip().lower(), target_author.strip().lower()
    return t in a or a in t


def build_context(session: dict, turn_idx: int, tokenizer, max_tokens: int) -> str:
    """Pack turns before ``turn_idx`` newest-first; the target author is shown as PERSONA."""
    target_author = session["metadata"]["target_author_name"]
    kept: list[str] = []
    used = 0
    for t in reversed(session["data"][:turn_idx]):
        author = "PERSONA" if is_target(t["author"], target_author) else t["author"]
        text = f'<turn author="{author}">\n{t["action"].strip()}\n</turn>\n'
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if used + n > max_tokens:
            break
        kept.append(text)
        used += n
    return "".join(reversed(kept))


def call_judge(prompt: str, model: str, timeout: float) -> PairwiseJudgement:
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=PairwiseJudgement,
                timeout=timeout,
            )
            return PairwiseJudgement.model_validate_json(resp.choices[0].message.content)
        except Exception as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            print(f"  [retry {attempt + 1}/{MAX_ATTEMPTS}] judge call failed: {exc}")
            time.sleep(3)


def judge_turn(
    session: dict, turn_idx: int, action_a: str, action_b: str, label_a: str, label_b: str,
    tokenizer, context_tokens: int, model: str, timeout: float,
) -> dict:
    swap = random.random() < 0.5
    shown = [(label_b, action_b), (label_a, action_a)] if swap else [(label_a, action_a), (label_b, action_b)]
    context = build_context(session, turn_idx, tokenizer, context_tokens)
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        style_intent_values_definitions=PAIRWISE_DIMENSION_DEFINITIONS.strip(),
        context=context or "(no prior context)\n",
        action=session["data"][turn_idx]["action"].strip(),
        generated_action_a=shown[0][1].strip(),
        generated_action_b=shown[1][1].strip(),
    )
    raw = call_judge(prompt, model, timeout).model_dump()

    def resolve(winner: str) -> str:  # map the judge's A/B back to the model labels
        return {"A": shown[0][0], "B": shown[1][0]}.get(winner, winner)

    return {
        "compared": [label_a, label_b],  # order of --generations_a / --generations_b
        # model_a / model_b are in the order the judge saw them, which is randomized per turn
        "model_a": shown[0][0],
        "model_b": shown[1][0],
        "generated_action_a": shown[0][1],
        "generated_action_b": shown[1][1],
        "pairwise_judgement": {
            k: {"rationale": raw[k]["rationale"], "winner": resolve(raw[k]["winner"])}
            for k in (*DIMENSIONS, "overall")
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--test_splits", nargs="+", default=["test_0"])
    p.add_argument("--generations_a", required=True, help="Output of recon.generate")
    p.add_argument("--generations_b", required=True, help="Output of recon.generate")
    p.add_argument("--output", required=True)
    p.add_argument("--judge_model", default="gemini/gemini-3.1-flash-lite-preview")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="HF tokenizer used for the context budget")
    p.add_argument("--context_tokens", type=int, default=2048)
    p.add_argument("--turn_filters", nargs="*", default=None)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max_workers", type=int, default=32)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    from transformers import AutoTokenizer

    label_a, label_b = Path(args.generations_a).stem, Path(args.generations_b).stem
    if label_a == label_b:
        raise ValueError("--generations_a and --generations_b must have different file names")
    data = load_json(args.data)
    session_ids = sorted(set(load_split_ids(args.split, args.test_splits)) & set(data))
    gen_a, gen_b = load_json(args.generations_a), load_json(args.generations_b)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=os.getenv("HF_TOKEN"))
    filters = compile_turn_filters(args.turn_filters)

    tasks = []
    for sid in session_ids:
        if sid not in gen_a or sid not in gen_b:
            print(f"  [warn] session {sid} missing from a generations file, skipping")
            continue
        session = data[sid]
        target_author = session["metadata"]["target_author_name"]
        for t, turn in enumerate(session["data"]):
            if not is_target(turn["author"], target_author) or not passes_filters(turn["action"], filters):
                continue
            ta, tb = gen_a[sid]["data"][t], gen_b[sid]["data"][t]
            if "generated_action" not in ta or "generated_action" not in tb:
                continue
            tasks.append(((sid, t), lambda s=session, i=t, a=ta["generated_action"], b=tb["generated_action"]:
                          judge_turn(s, i, a, b, label_a, label_b, tokenizer, args.context_tokens,
                                     args.judge_model, args.timeout)))
    if not tasks:
        raise ValueError("No test turns with generations in both files")
    print(f"Judging {len(tasks)} turns: {label_a} vs {label_b}")

    results = run_parallel(tasks, args.max_workers, "Judging")
    out = {sid: copy.deepcopy(data[sid]) for sid in session_ids}
    for (sid, t), result in results.items():
        out[sid]["data"][t].update(result)
    save_json(args.output, out)
    print(f"Wrote judgements to {args.output}")


if __name__ == "__main__":
    main()
