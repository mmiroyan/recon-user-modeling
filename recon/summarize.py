"""Summarize a recon.evaluate output into win rates.

The headline number (as reported in the paper) is the win rate of A over B after excluding
ties, i.e. wins_A / (wins_A + wins_B), with a normal-approximation 95% interval.

    python -m recon.summarize --path runs/example/eval/recon_select_vs_backward.json [--json]
"""

from __future__ import annotations

import argparse
import json
import math

from recon.common import load_json

DIMENSIONS = ("style", "intent", "values", "overall")


def labels_of(data: dict) -> tuple[str, str]:
    """(A, B) in the order they were passed to recon.evaluate."""
    pairs = {tuple(t["compared"]) for s in data.values() for t in s["data"] if "compared" in t}
    if len(pairs) != 1:
        raise ValueError(f"Expected one compared pair, found {sorted(pairs)}")
    a, b = pairs.pop()
    return a, b


def summarize(data: dict, label_a: str, label_b: str) -> dict:
    counts = {d: {"a": 0, "b": 0, "tie": 0, "tie (bad)": 0, "NA": 0} for d in DIMENSIONS}
    for session in data.values():
        for turn in session["data"]:
            judgement = turn.get("pairwise_judgement")
            if not judgement:
                continue
            for d in DIMENSIONS:
                w = judgement[d]["winner"]
                counts[d]["a" if w == label_a else "b" if w == label_b else w] += 1

    out = {}
    for d, c in counts.items():
        decided = c["a"] + c["b"]
        p = c["a"] / decided if decided else float("nan")
        out[d] = {
            **c,
            "n_non_tied": decided,
            "win_rate_a": p,
            "ci95": 1.96 * math.sqrt(p * (1 - p) / decided) if decided else float("nan"),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True, help="Output of recon.evaluate")
    p.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    args = p.parse_args()

    data = load_json(args.path)
    label_a, label_b = labels_of(data)
    summary = summarize(data, label_a, label_b)

    if args.json:
        print(json.dumps({"label_a": label_a, "label_b": label_b, **summary}, indent=2))
        return
    print(f"A = {label_a}\nB = {label_b}\n")
    print(f"{'dimension':<10} {'win rate A':>11} {'±95%':>7} {'n':>5} {'A':>5} {'B':>5} {'tie':>5} {'tie(bad)':>9} {'NA':>5}")
    for d, s in summary.items():
        print(f"{d:<10} {s['win_rate_a']:>11.3f} {s['ci95']:>7.3f} {s['n_non_tied']:>5} {s['a']:>5} "
              f"{s['b']:>5} {s['tie']:>5} {s['tie (bad)']:>9} {s['NA']:>5}")


if __name__ == "__main__":
    main()
