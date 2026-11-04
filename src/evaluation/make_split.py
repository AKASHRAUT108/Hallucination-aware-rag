"""Split an evaluation file into a tuning half and a test half, stratified by question type.

Run once:  python -m src.evaluation.make_split --in data/eval/eval.jsonl
Then choose the threshold, window and verifier model on tune.jsonl ONLY, and run test.jsonl once at the end.
Do not re-split with a different seed after you have seen test results: that would leak the test set.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split_rows(rows: list[dict], test_frac: float = 0.5, seed: int = 42) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(bool(r.get("answerable", True)), r.get("type", ""))].append(r)
    rng = random.Random(seed)
    tune: list[dict] = []
    test: list[dict] = []
    for key in sorted(groups, key=str):
        g = list(groups[key])
        rng.shuffle(g)
        n_test = round(len(g) * test_frac)
        if len(g) >= 2:
            n_test = min(max(n_test, 1), len(g) - 1)     # both halves get at least one of each kind
        test += g[:n_test]
        tune += g[n_test:]
    order = {id(r): i for i, r in enumerate(rows)}
    tune.sort(key=lambda r: order[id(r)])
    test.sort(key=lambda r: order[id(r)])
    return tune, test


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    from src.utils.helpers import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/eval/eval.jsonl")
    ap.add_argument("--tune-out", default="data/eval/tune.jsonl")
    ap.add_argument("--test-out", default="data/eval/test.jsonl")
    ap.add_argument("--test-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    with open(ROOT / args.src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tune, test = split_rows(rows, args.test_frac, args.seed)
    write(ROOT / args.tune_out, tune)
    write(ROOT / args.test_out, test)
    for name, part in (("tune", tune), ("test", test)):
        un = sum(1 for r in part if r.get("answerable") is False)
        print(f"{name}: {len(part)} questions ({len(part) - un} answerable, {un} unanswerable)")


if __name__ == "__main__":
    main()
