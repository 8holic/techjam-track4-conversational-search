from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print result files in baseline_results.json shape")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--name", nargs="+", default=None)
    parser.add_argument("--dataset", nargs="+", default=None)
    args = parser.parse_args()

    rows = []
    for i, path in enumerate(args.input):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        name = args.name[i] if args.name and i < len(args.name) else Path(path).stem
        dataset = args.dataset[i] if args.dataset and i < len(args.dataset) else data.get("dataset", path)
        score = data.get("recommended_technical_score", data.get("technical_score", 0.0))
        rows.append({
            "baseline": name,
            "dataset": dataset,
            "sample_count": data["sample_count"],
            "hit_rate_at_10": data["hit_rate_at_10"],
            "mrr": data["mrr"],
            "mttc": data["mttc"],
            "efficiency": data["efficiency"],
            "technical_score": score,
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
