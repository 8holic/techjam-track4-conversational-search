from __future__ import annotations

import json
import re
import urllib.request

LLM_MODEL = "qwen3:1.7b"
LLM_URL = "http://localhost:11434/api/chat"
LLM_TIMEOUT = 45
LLM_TEMP = 0.2
RERANK_N = 30
MAX_LLM_CALLS = 2

SYSTEM_INTENT = (
    "You are a shopping-assistant intent classifier. Decide if a shopper is "
    "BUYING or BROWSING.\n"
    "- BUYING: they express a FIRM, decisive must-have requirement with explicit "
    "framing, e.g. \"A key requirement is...\", \"I need...\", \"I must have...\", "
    "\"I want exactly...\".\n"
    "- BROWSING: they are exploring, mention only a category, or state a bare "
    "attribute WITHOUT any requirement framing. IMPORTANT: a single word or short "
    "phrase after the category is a TENTATIVE attribute mention, NOT a firm "
    "requirement. Example: \"I'm looking for shoes. black.\" is BROWSING; "
    "\"I'm looking for shoes. A key requirement is black.\" is BUYING.\n"
    'Reply with ONLY JSON: {"intent": "buying" or "browsing", '
    '"requirement": "the specific must-have requirement if clearly stated, else null"}'
)

SYSTEM_RERANK = (
    "You are a shopping-assistant reranker. Given the customer's situation and the "
    "candidate products, pick the product the customer most likely wants.\n"
    "Output ONLY the TOP 10 candidate numbers, most likely first, comma-separated, "
    "e.g. 7, 2, 14, 5, 11, 3, 9, 1, 6, 12"
)


def ollama_available(timeout: float = 2.0) -> bool:
    try:
        request = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except Exception:
        return False


def _candidate_text(products: dict, asin: str) -> str:
    product = products.get(asin)
    if not product:
        return asin
    title = str(product.get("title") or "")[:90]
    price = product.get("price")
    price_txt = f" ${price}" if price else ""
    details = product.get("details") or {}
    det = " ".join(f"{k}: {v}" for k, v in list(details.items())[:5])[:150]
    features = product.get("features") or []
    feat = " ".join(str(f) for f in features[:2])[:120]
    return f"{title}{price_txt} | {det} | {feat}"


def _prompt(state: dict, pool: list[str], products: dict) -> str:
    category = " ".join(state["category_terms"]) or "unknown"
    slots = "; ".join(text for text, _prov in state["confirmed"]) or "none yet"
    axes = ", ".join(sorted(state["confirmed_axis"])) or "none"
    lines = "\n".join(
        f"{i}. {_candidate_text(products, asin)}" for i, asin in enumerate(pool[:RERANK_N], 1)
    )
    return (
        f"Customer is looking for: {category}\n"
        f"Confirmed requirements: {slots}\n"
        f"Profile emphasis: {axes}\n\n"
        f"Candidates:\n{lines}"
    )


def _chat(system: str, user: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": LLM_TEMP, "num_predict": 64},
    }
    request = urllib.request.Request(
        LLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("message", {}).get("content", ""))


def classify_intent(message: str) -> dict:
    """Classify a session opener as buying or browsing (+ optional requirement).
    Returns {"intent": "buying"|"browsing", "requirement": str|None}; returns a
    safe browsing default on any failure."""
    try:
        raw = _chat(SYSTEM_INTENT, f"Customer message: {message}")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            intent = str(data.get("intent", "")).lower().strip()
            requirement = data.get("requirement")
            if intent in ("buying", "browsing"):
                return {"intent": intent, "requirement": str(requirement) if requirement else None}
        return {"intent": "browsing", "requirement": None}
    except Exception:
        return {"intent": "browsing", "requirement": None}


def rerank(state: dict, pool: list[str], products: dict) -> list[str]:
    """Semantically re-rank the BM25 candidate pool. Returns the reordered
    pool, or the ORIGINAL pool unchanged on any failure (model down, timeout,
    unparseable output)."""
    try:
        raw = _chat(SYSTEM_RERANK, _prompt(state, pool, products))
        seen: set[str] = set()
        ordered: list[str] = []
        for number in re.findall(r"\d+", raw):
            index = int(number)
            if 1 <= index <= len(pool) and pool[index - 1] not in seen:
                seen.add(pool[index - 1])
                ordered.append(pool[index - 1])
        for asin in pool:
            if asin not in seen:
                seen.add(asin)
                ordered.append(asin)
        return ordered
    except Exception:
        return pool
