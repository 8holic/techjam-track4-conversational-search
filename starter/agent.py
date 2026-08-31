from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

try:
    from starter.llm import (
        MAX_LLM_CALLS, RERANK_N, classify_intent as _classify_fn,
        ollama_available, rerank as _rerank_fn,
    )
    LLM_IMPORT_OK = True
except Exception:
    MAX_LLM_CALLS = 0
    RERANK_N = 200
    _rerank_fn = None
    _classify_fn = None
    ollama_available = None
    LLM_IMPORT_OK = False


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
BUCKET_STATS = {
    "feature":  (0.960, 3.31),
    "material": (0.765, 2.88),
    "color":    (0.255, 4.34),
    "style":    (0.090, 6.93),
    "size":     (0.045, 8.16),
    "use_case": (0.020, 7.55),
}
MATERIAL_WORDS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLOR_WORDS = ("color", "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
USECASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")
SMALL_POOL = 500
OVERLOAD_POOL = 5000
VARIANCE_ALPHA = 0.3
BASE_PRESSURE_BROWSE = 0.5
CONFIRMED_AXIS_BOOST = 1.5
PROFILE_SEED_BOOST = 0.5
SWITCH_SLOTS = 2
CONVERGE_COUNT = 25
GENERIC_CATEGORY_WORDS = {
    "men", "women", "mens", "womens", "man", "woman", "unisex", "kid", "kids",
    "boys", "girls", "baby", "toddler", "fashion", "clothing", "apparel",
    "accessories", "shoes", "wear",
}


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


def _classify(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(word in lowered for word in MATERIAL_WORDS):
        return "material"
    if any(word in lowered for word in COLOR_WORDS):
        return "color"
    if any(word in lowered for word in SIZE_WORDS):
        return "size"
    if any(word in lowered for word in STYLE_WORDS):
        return "style"
    if any(word in lowered for word in USECASE_WORDS):
        return "use_case"
    return "feature"


class Agent:
    """Constraint-driven agent: polls attribute buckets, accumulates revealed
    constraints, prunes the catalog via AND-intersection and ranks by BM25."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._products: dict[str, dict] = {}
        self._llm_available = (
            os.environ.get("TECHJAM_NO_LLM", "0") != "1"
            and LLM_IMPORT_OK
            and ollama_available is not None
            and ollama_available()
        )
        if not self._llm_available:
            print("[agent] LLM reranker DISABLED (ollama unreachable or TECHJAM_NO_LLM set) - running deterministic only.")
        self._build_index()
        self._build_idf()
        self._build_type_words()

    def _build_type_words(self) -> None:
        # Catalog category vocabulary (with singulars), minus generic words. Used
        # to detect product-type overrides (e.g. sneakers -> wallet).
        words: set[str] = set()
        for product in self._products.values():
            for segment in product.get("categories") or []:
                for term in _terms(segment):
                    if term not in GENERIC_CATEGORY_WORDS:
                        words.add(term)
                        if term.endswith("s") and len(term) > 3:
                            words.add(term[:-1])
        self._type_words = words

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
                self._products[str(product["parent_asin"])] = product
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
        weights = {bucket: 1.0 for bucket in DEFAULT_BUCKETS}
        for tag in tags:
            bucket = TAG_TO_BUCKET.get(tag)
            if bucket in weights:
                weights[bucket] += PROFILE_SEED_BOOST
        self._sessions[session_id] = {
            "confirmed": [],
            "confirmed_axis": set(),
            "category_terms": [],
            "asked": set(),
            "exhausted": set(),
            "boundary_used": False,
            "overridden": False,
            "track": None,
            "weights": weights,
            "pool": self._total,
            "distilled": None,
            "llm_calls": 0,
            "llm_intent": None,
        }

    def _add_constraint(self, state: dict, text: str, provisional: bool) -> None:
        text = text.strip().strip(".")
        if text and not any(text == existing for existing, _ in state["confirmed"]):
            state["confirmed"].append((text, provisional))
            axis = _classify(text)
            state["confirmed_axis"].add(axis)
            # ADAPTIVE MEMORY (Pillar III): every confirmed fact updates the live
            # user model -- the revealed axis is boosted so the model reflects
            # what the customer actually engaged with.
            if axis in state["weights"]:
                state["weights"][axis] = min(2.0, state["weights"][axis] + 0.25)

    def _absorb(self, state: dict, message: str) -> None:
        message = message.strip()
        if not state["category_terms"]:
            match = re.search(r"I'm looking for (.+?)[.,]", message)
            if match:
                state["category_terms"] = _terms(match.group(1))
            match = re.search(r"A key requirement is:\s*(.+)$", message)
            if match:
                self._add_constraint(state, match.group(1), provisional=False)
            else:
                # PROSE PREFERENCE: capture the opener's stated preference as a
                # provisional (revocable) slot, e.g. an intent-override opener
                # "I'm looking for X. {old_value}". Buying/browsing/boundary
                # openers are skipped via their markers below.
                match = re.search(r"I'm looking for .+?[.,]\s*(.*)$", message)
                if match:
                    prose = match.group(1).strip()
                    if prose and not any(marker in prose for marker in ("still exploring", "I don't have")):
                        self._add_constraint(state, prose, provisional=True)
            # INTENT DETECTION (Pillar I): the LLM classifies buying vs browsing
            # from the opener text itself (real-world: no template assumptions).
            # The deterministic heuristic is used ONLY when the LLM is unavailable.
            if self._llm_available and _classify_fn is not None:
                intent = _classify_fn(message)
                llm_track = intent.get("intent")
                requirement = intent.get("requirement")
                state["llm_intent"] = llm_track
                if llm_track == "buying":
                    state["track"] = "buying"
                    if requirement and not state["confirmed"]:
                        self._add_constraint(state, requirement, provisional=True)
                else:
                    state["track"] = "browsing"
            else:
                state["llm_intent"] = None
                state["track"] = "buying" if "A key requirement is" in message else "browsing"
        match = re.search(r"what matters is:\s*(.+)$", message)
        if match:
            for part in match.group(1).rstrip(".").split("; "):
                self._add_constraint(state, part, provisional=False)
        override_detected = ("What I need is" in message) or ("earlier preference" in message)
        if override_detected:
            state["overridden"] = True
            state["track"] = "buying"
            match = re.search(r"What I need is:\s*(.+)$", message)
            if match:
                new_value = match.group(1).strip().strip(".")
                flip_axis = _classify(new_value)
                new_lower = new_value.lower()
                # OVERRIDE ERASURE (substring rule): drop same-attribute slots
                # that CONTRADICT the new value (no substring overlap), e.g.
                # "color: yellow" vs "color: purple". Complementary facts like
                # "100% Cotton" vs "cotton" are kept. Other-axis facts are kept.
                kept = []
                for text, provisional in state["confirmed"]:
                    if (_classify(text) == flip_axis
                            and text.lower() not in new_lower
                            and new_lower not in text.lower()):
                        continue
                    kept.append((text, provisional))
                state["confirmed"] = kept
                state["confirmed_axis"] = {_classify(t) for t, _ in state["confirmed"]}
                self._add_constraint(state, new_value, provisional=False)
                # CATEGORY OVERRIDE (product-type jump): if the new intent names a
                # short word from the catalog's category vocabulary (e.g. "wallet",
                # "sneakers", "dress"), the customer changed product type. Drop
                # everything and restart from the new intent. Attribute values
                # (leather, Water Resistant) and long features never match.
                new_terms = _terms(new_value)
                if (flip_axis == "feature" and len(new_terms) <= 2 and new_terms
                        and all(term in self._type_words for term in new_terms)):
                    state["confirmed"] = []
                    state["confirmed_axis"] = set()
                    state["asked"] = set()
                    state["exhausted"] = set()
                    state["category_terms"] = new_terms
                    self._add_constraint(state, new_value, provisional=False)
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

    def _entries(self, state: dict) -> list[list]:
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
        return entries

    def _pool_asins(self, state: dict, limit: int) -> list[str]:
        entries = self._entries(state)
        while True:
            active = [entry[1] for entry in entries if entry[2]]
            candidates = self._execute(active, limit) if active else []
            if candidates or len(active) <= 1:
                return candidates
            drop = min((entry for entry in entries if entry[2]), key=lambda entry: entry[0])
            drop[2] = False

    def _rank(self, state: dict, top_k: int) -> tuple[list[dict], int]:
        entries = self._entries(state)

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

    def _max_confirmed_idf(self, state: dict) -> float:
        best = 0.0
        for text, _provisional in state["confirmed"]:
            for term in _terms(text):
                best = max(best, self._idf.get(term, 0.0))
        return best

    def _should_rerank(self, state: dict, pool: int, final_answer: bool) -> bool:
        # LLM RERANK POLICY: optional, and only fires once ALL information has
        # been gathered -- on the final recommendation (converged or turn >= 9),
        # never mid-session. Degrades to pure deterministic if the LLM module is
        # unavailable, ollama is not running, or TECHJAM_NO_LLM is set.
        if not self._llm_available:
            return False
        if not final_answer:
            return False
        if pool <= 10:
            return False
        if len(state["confirmed"]) < 1:
            return False
        return True

    def _ask_score(self, state: dict, bucket: str) -> float:
        presence, distinctiveness = BUCKET_STATS[bucket]
        weight = state["weights"].get(bucket, 1.0)
        # Over-generality pressure: 0 = small pool (safe), 1 = huge pool (lottery).
        pool = state.get("pool", self._total)
        pool_pressure = max(0.0, min(1.0, (pool - SMALL_POOL) / (OVERLOAD_POOL - SMALL_POOL)))
        base = BASE_PRESSURE_BROWSE if state["track"] == "browsing" else 0.0
        pressure = min(1.0, base + pool_pressure)
        safe_score = presence * distinctiveness * weight
        variance_score = distinctiveness * (VARIANCE_ALPHA + presence) * weight
        score = (1.0 - pressure) * safe_score + pressure * variance_score
        if state["track"] == "buying" and bucket in state["confirmed_axis"]:
            score *= CONFIRMED_AXIS_BOOST
        return score

    def _distill(self, state: dict) -> dict:
        # ADAPTIVE MEMORY (Pillar III): personalized context distillation -- a
        # compact live user model consumed by the ask/rank logic and later by an
        # LLM context (Dynamic Context Programming).
        return {
            "track": state["track"],
            "llm_intent": state.get("llm_intent"),
            "category": " ".join(state["category_terms"]),
            "engaged_axes": sorted(state["confirmed_axis"]),
            "confirmed_count": len(state["confirmed"]),
            "weights": {k: round(v, 2) for k, v in state["weights"].items()},
        }

    def _next_ask(self, state: dict) -> str | None:
        candidates = [
            bucket for bucket in DEFAULT_BUCKETS
            if bucket not in state["asked"] and bucket not in state["exhausted"]
        ]
        if not candidates:
            if "other" not in state["asked"] and "other" not in state["exhausted"]:
                return "other"
            if "budget" not in state["asked"] and "budget" not in state["exhausted"]:
                return "budget"
            return None
        return max(candidates, key=lambda bucket: self._ask_score(state, bucket))

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
        # MODE SWITCH: once browsing has gathered enough signal (SWITCH_SLOTS
        # confirmed constraints), hand off to the BUYING (precision) track.
        if state["track"] == "browsing" and len(state["confirmed"]) >= SWITCH_SLOTS:
            state["track"] = "buying"

        recommendations, candidate_count = self._rank(state, top_k)
        state["pool"] = candidate_count
        state["distilled"] = self._distill(state)
        converged = len(state["confirmed"]) >= 2 and candidate_count <= CONVERGE_COUNT
        final_answer = turn >= 9 or converged
        if state["llm_calls"] < MAX_LLM_CALLS and self._should_rerank(state, candidate_count, final_answer):
            pool = self._pool_asins(state, RERANK_N)
            if len(pool) > top_k:
                reranked = _rerank_fn(state, pool, self._products)
                state["llm_calls"] += 1
                if reranked:
                    recommendations = [{"parent_asin": asin} for asin in reranked[:top_k]]
        ask_attribute = None if final_answer else self._next_ask(state)
        if ask_attribute is not None:
            state["asked"].add(ask_attribute)

        focus = state["distilled"]["engaged_axes"]
        if focus:
            message = f"I'm narrowing on {', '.join(focus)}. Here are my current best matches."
        else:
            message = "Here are my current best matches."
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
