from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from evaluator.local_evaluator import intent_card, classify_constraint, searchable_text

MATERIALS = ["cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon"]
COLORS = ["black", "white", "blue", "red", "pink", "green", "brown", "gray", "purple", "yellow"]
FEATURES = ["Button closure", "Snap closure", "Zipper closure", "Machine Washable",
            "Hand Wash Only", "Water Resistant", "Drawstring closure", "Pull On closure"]
TAGS_POOL = ["material", "color", "size", "style", "fit", "comfort", "durability",
             "budget", "performance", "functionality", "aesthetic"]
RATING_STYLES = ["usually positive", "critical", "mixed", "positive"]
FREQUENCIES = ["1-2 prior purchases", "3-4 prior purchases", "5+ prior purchases"]
DIFFICULTIES = ["easy", "medium", "hard"]


def make_profile(rng: random.Random) -> dict:
    tags = rng.sample(TAGS_POOL, rng.randint(2, 4))
    style = rng.choice(RATING_STYLES)
    freq = rng.choice(FREQUENCIES)
    avg = rng.choice([1.0, 2.0, 3.0, 4.0, 4.5, 5.0, None])
    return {
        "average_prior_rating": avg,
        "preference_tags": tags,
        "purchase_frequency": freq,
        "rating_style": style,
        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {style}.",
    }


def harsh_override(sample_id: str, product: dict) -> tuple[dict, dict]:
    card = intent_card(product)
    strings = [*card["hard_constraints"], *card["soft_preferences"]]
    rng = random.Random("harsh_" + sample_id)
    axis = None
    for s in strings:
        if classify_constraint(s) == "color":
            new_value, axis = s, "color"
            break
    if axis is None:
        for s in strings:
            if classify_constraint(s) == "material":
                new_value, axis = s, "material"
                break
    if axis is None:
        new_value, axis = strings[0], "feature"

    text = searchable_text(product).lower()
    if axis == "color":
        pool = [c for c in COLORS if c not in text and c != new_value.lower()]
        old_value = "color: " + rng.choice(pool) if pool else "color: black"
    elif axis == "material":
        pool = [m for m in MATERIALS if m not in text and m != new_value.lower()]
        old_value = rng.choice(pool) if pool else "polyester"
    else:
        pool = [f for f in FEATURES if f.lower() not in text and f != new_value]
        old_value = rng.choice(pool) if pool else "Button closure"

    turn = rng.choice([3, 4])
    behavior = {
        "scenario_type": "intent_override",
        "override": {
            "turn": turn,
            "old_value": old_value,
            "new_value": new_value,
            "message": f"Actually, ignore my earlier preference. What I need is: {new_value}.",
        },
    }
    return card, behavior


def make_set(products: list[dict], n: int, seed: int, tag: str, used: set[str]) -> list[dict]:
    rng = random.Random(seed)
    available = [p for p in products if p["parent_asin"] not in used]
    rng.shuffle(available)
    chosen = available[:n]
    for p in chosen:
        used.add(p["parent_asin"])

    browsing = int(0.40 * n)
    buying = int(0.40 * n)
    override = int(0.15 * n)
    boundary = n - browsing - buying - override
    scenarios = ["browsing"] * browsing + ["buying"] * buying + \
                ["intent_override"] * override + ["boundary"] * boundary
    rng.shuffle(scenarios)

    rows = []
    for i, (product, scenario) in enumerate(zip(chosen, scenarios)):
        sample_id = f"{tag}_{i + 1:04d}"
        sample = {
            "category_bucket": "clothing",
            "difficulty_bucket": rng.choice(DIFFICULTIES),
            "ground_truth": {"parent_asin": product["parent_asin"]},
            "sample_id": sample_id,
            "scenario_type": scenario,
            "user_profile": make_profile(rng),
        }
        if scenario == "intent_override":
            card, behavior = harsh_override(sample_id, product)
            sample["intent_card"] = card
            sample["behavior"] = behavior
        rows.append(sample)
    return rows


def write(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test sets (40/40/15/5)")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--outdir", default="testing")
    args = parser.parse_args()

    products = [json.loads(l) for l in Path(args.catalog).open(encoding="utf-8") if l.strip()]
    public_rows = [json.loads(l) for l in Path(args.public).open(encoding="utf-8") if l.strip()]
    public_asins = {r["ground_truth"]["parent_asin"] for r in public_rows}
    used = set(public_asins)

    specs = [("set_a_200", 200, 101), ("set_b_200", 200, 202),
             ("set_c_800", 800, 303), ("set_d_800", 800, 404)]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, n, seed in specs:
        rows = make_set(products, n, seed, tag, used)
        path = outdir / f"{tag}.jsonl"
        write(path, rows)
        print(f"{tag}: {len(rows)} rows -> {path}")
        print(f"   scenario mix: {dict(Counter(r['scenario_type'] for r in rows))}")
        override_count = sum(1 for r in rows if r["scenario_type"] == "intent_override")
        harsh = 0
        for r in rows:
            if r.get("behavior") and r["behavior"]["override"]["old_value"]:
                harsh += 1
        print(f"   baked harsh overrides: {harsh}/{override_count}")


if __name__ == "__main__":
    main()
