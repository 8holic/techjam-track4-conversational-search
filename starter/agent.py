from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

COLUMN_WEIGHTS = (0.0, 6.0, 4.0, 3.0, 3.0, 1.0, 1.0)
DEFAULT_BUCKETS = ["feature", "material", "color", "style", "size", "use_case"]
TAG_TO_BUCKET = {
    "material": "material",
    "fit": "feature",
    "comfort": "feature",
    "durability": "feature",
    "style": "style",
    "color": "color",
    "size": "size",
}
CONVERGE_COUNT = 25


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Constraint-driven agent: polls attribute buckets, accumulates revealed
    constraints, prunes the catalog via AND-intersection and ranks by BM25."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._build_index()
        self._build_idf()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def _build_idf(self) -> None:
        df: Counter[str] = Counter()
        total = 0
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                total += 1
                for token in set(_terms(
                    _text(product.get("title")) + " " + _text(product.get("categories")) + " "
                    + _text(product.get("features")) + " " + _text(product.get("details")) + " "
                    + _text(product.get("store")) + " " + _text(product.get("description"))
                )):
                    df[token] += 1
        self._df = df
        self._total = total
        self._idf = {token: math.log(total / max(count, 1)) for token, count in df.items()}

    def reset(self, session_id: str, user_profile: dict) -> None:
        tags = user_profile.get("preference_tags", []) if isinstance(user_profile, dict) else []
        boosted = [TAG_TO_BUCKET[tag] for tag in tags if TAG_TO_BUCKET.get(tag)]
        priority = [b for b in boosted if b in DEFAULT_BUCKETS] + \
                   [b for b in DEFAULT_BUCKETS if b not in boosted]
        self._sessions[session_id] = {
            "confirmed": [],
            "category_terms": [],
            "asked": set(),
            "exhausted": set(),
            "boundary_used": False,
            "overridden": False,
            "priority": priority,
        }

    def _add_constraint(self, state: dict, text: str, provisional: bool) -> None:
        text = text.strip().strip(".")
        if text and not any(text == existing for existing, _ in state["confirmed"]):
            state["confirmed"].append((text, provisional))

    def _absorb(self, state: dict, message: str) -> None:
        message = message.strip()
        if not state["category_terms"]:
            match = re.search(r"I'm looking for (.+?)[.,]", message)
            if match:
                state["category_terms"] = _terms(match.group(1))
            match = re.search(r"A key requirement is:\s*(.+)$", message)
            if match:
                self._add_constraint(state, match.group(1), provisional=False)
        match = re.search(r"what matters is:\s*(.+)$", message)
        if match:
            for part in match.group(1).rstrip(".").split("; "):
                self._add_constraint(state, part, provisional=False)
        match = re.search(r"What I need is:\s*(.+)$", message)
        if match:
            state["overridden"] = True
            state["confirmed"] = [(text, provisional) for text, provisional in state["confirmed"] if not provisional]
            self._add_constraint(state, match.group(1), provisional=False)
        match = re.search(r"I don't have an additional preference for (\w+)", message)
        if match:
            state["exhausted"].add(match.group(1))
        match = re.search(r"I don't have a preference for (\w+)", message)
        if match:
            state["boundary_used"] = True

    def _execute(self, groups: list[str], top_k: int) -> list[str]:
        if not groups:
            return []
        query = " AND ".join(groups)
        weights = ", ".join(str(weight) for weight in COLUMN_WEIGHTS)
        sql = (
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {weights}) LIMIT ?"
        )
        try:
            rows = self.connection.execute(sql, (query, top_k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    def _candidate_count(self, groups: list[str]) -> int:
        if not groups:
            return self._total
        query = " AND ".join(groups)
        try:
            row = self.connection.execute(
                "SELECT count(*) FROM products WHERE products MATCH ?", (query,)
            ).fetchone()
        except sqlite3.OperationalError:
            return self._total
        return int(row[0]) if row else 0

    def _rank(self, state: dict, top_k: int) -> tuple[list[dict], int]:
        entries: list[list] = []
        if state["category_terms"]:
            terms = state["category_terms"]
            entries.append([0.0, "(" + " OR ".join(f'"{t}"' for t in terms) + ")", True])
        for text, provisional in state["confirmed"]:
            terms = _terms(text)
            if not terms:
                continue
            distinctiveness = max(self._idf.get(t, 0.0) for t in terms)
            entries.append([distinctiveness, "(" + " OR ".join(f'"{t}"' for t in terms) + ")", True])

        best: list[str] = []
        while True:
            active = [entry[1] for entry in entries if entry[2]]
            candidates = self._execute(active, top_k) if active else []
            if candidates or len(active) <= 1:
                best = candidates
                break
            drop = min((entry for entry in entries if entry[2]), key=lambda entry: entry[0])
            drop[2] = False

        if not best:
            fallback: list[str] = []
            for text, _provisional in state["confirmed"]:
                fallback.extend(_terms(text))
            fallback.extend(state["category_terms"])
            if fallback:
                best = self._execute(["(" + " OR ".join(f'"{t}"' for t in fallback) + ")"], top_k)

        active = [entry[1] for entry in entries if entry[2]]
        count = self._candidate_count(active) if active else self._total
        return [{"parent_asin": asin} for asin in best], count

    def _next_ask(self, state: dict) -> str | None:
        for bucket in state["priority"]:
            if bucket not in state["asked"] and bucket not in state["exhausted"]:
                return bucket
        if "other" not in state["asked"] and "other" not in state["exhausted"]:
            return "other"
        return None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        self._absorb(state, user_message)

        recommendations, candidate_count = self._rank(state, top_k)
        converged = len(state["confirmed"]) >= 2 and candidate_count <= CONVERGE_COUNT
        ask_attribute = None if (turn >= 9 or converged) else self._next_ask(state)
        if ask_attribute is not None:
            state["asked"].add(ask_attribute)

        return {
            "message": "Here are my current best matches.",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
