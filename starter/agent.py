from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

PRICE_RE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
ATTRIBUTE_ORDER = ["use_case", "feature", "material", "color", "style", "size", "budget"]

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "don", "additional", "preference", "matters", "closest", "found",
    "actually", "ignore", "earlier", "what", "need", "not", "quite",
    "right", "yet", "ask", "one", "specific", "attribute", "options",
    "those", "here", "matches", "have",
}


def _safe_price(value: object) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


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
    """Editable weak baseline: stateful BM25 retrieval with clarifying questions, no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._prices: dict[str, float | None] = {}
        self._build_index()

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
                parent_asin = str(product["parent_asin"])

                price = product.get("price")
                self._prices[parent_asin] = _safe_price(price)

                batch.append(
                    (
                        parent_asin,
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

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "terms": [],
            "budget_max": None,
            "asked": set(),
            "turn": 0,
            "raw_text": "",
        }

    def _extract_budget(self, text: str) -> float | None:
        lowered = text.lower()
        if "budget" in lowered or "under" in lowered or "$" in lowered or "less than" in lowered:
            match = PRICE_RE.search(text)
            if match:
                return float(match.group(1))
        return None

    def _apply_budget_filter(self, candidates: list[str], budget_max: float | None) -> list[str]:
        if budget_max is None:
            return candidates
        filtered = []
        for pid in candidates:
            price = self._prices.get(pid)
            if price is None or price <= budget_max:
                filtered.append(pid)
        return filtered

    def _decide_ask_attribute(self, state: dict) -> str | None:
        for attr in ATTRIBUTE_ORDER:
            if attr == "budget" and state["budget_max"] is not None:
                continue
            if attr not in state["asked"]:
                state["asked"].add(attr)
                return attr
        return None

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        state = self._sessions[session_id]
        state["turn"] = turn

        new_terms = _terms(user_message)
        state["terms"].extend(new_terms)
        state["raw_text"] += " " + user_message

        budget = self._extract_budget(user_message)
        if budget is not None:
            state["budget_max"] = budget

        unique_terms = list(dict.fromkeys(state["terms"]))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)

        candidates = []
        if expression:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k * 3),
            ).fetchall()
            candidates = [str(row[0]) for row in rows]

        candidates = self._apply_budget_filter(candidates, state["budget_max"])
        recommendations = [{"parent_asin": pid} for pid in candidates[:top_k]]

        ask_attribute = self._decide_ask_attribute(state)
        message = (
            f"Could you tell me more about your {ask_attribute} preference?"
            if ask_attribute
            else "Here are the closest matches I found."
        )

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
