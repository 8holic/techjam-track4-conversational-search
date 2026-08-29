from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the public set into train/eval slices")
    parser.add_argument("--input", default="data/public_set.jsonl")
    parser.add_argument("--eval", type=int, default=40, help="number of sessions for the eval slice")
    parser.add_argument("--seed", type=int, default=0, help="random seed for a reproducible split")
    parser.add_argument("--output-eval", default="testing/eval_split.jsonl")
    parser.add_argument("--output-train", default="testing/train_split.jsonl")
    args = parser.parse_args()

    samples = load(args.input)
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["scenario_type"]].append(sample)

    eval_rows: list[dict] = []
    train_rows: list[dict] = []
    for scenario, items in groups.items():
        rng.shuffle(items)
        keep = round(len(items) * args.eval / len(samples))
        eval_rows.extend(items[:keep])
        train_rows.extend(items[keep:])

    write(args.output_eval, eval_rows)
    write(args.output_train, train_rows)
    print(f"eval  -> {args.output_eval}: {len(eval_rows)} {dict(Counter(s['scenario_type'] for s in eval_rows))}")
    print(f"train -> {args.output_train}: {len(train_rows)} {dict(Counter(s['scenario_type'] for s in train_rows))}")


if __name__ == "__main__":
    main()
