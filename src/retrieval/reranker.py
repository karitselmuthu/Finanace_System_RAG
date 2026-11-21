from __future__ import annotations

import re

from src.utils.logger import logger

# Keywords that boost relevance for transaction-domain queries
_TRANSACTION_TERMS = re.compile(
    r"\b(credit|debit|balance|amount|payment|deposit|withdrawal|transfer|"
    r"failed|pending|reversed|completed|status|date|reference)\b",
    re.IGNORECASE,
)


def _keyword_boost(text: str, query: str) -> float:
    """
    Simple keyword overlap boost. Returns a small float in [0, 0.2].
    Rewards chunks that share query terms, especially transaction keywords.
    """
    query_words = set(query.lower().split())
    text_lower = text.lower()
    overlap = sum(1 for w in query_words if w in text_lower)
    domain_hits = len(_TRANSACTION_TERMS.findall(text))
    return min(0.1 * overlap + 0.01 * domain_hits, 0.2)


def rerank(hits: list[dict], query: str, top_n: int | None = None) -> list[dict]:
    """
    Re-score and re-sort retrieved chunks using a lightweight keyword heuristic.
    For production, replace with a cross-encoder model.

    Args:
        hits:   Output of retriever.retrieve() — each item has 'text' and 'score'.
        query:  The original user query.
        top_n:  Trim to top-N after reranking (None = keep all).

    Returns:
        Re-sorted list with a new 'rerank_score' field.
    """
    for hit in hits:
        boost = _keyword_boost(hit["text"], query)
        hit["rerank_score"] = round(hit["score"] + boost, 4)

    hits.sort(key=lambda h: h["rerank_score"], reverse=True)

    if top_n:
        hits = hits[:top_n]

    logger.debug(f"Reranked {len(hits)} chunks (top score: {hits[0]['rerank_score'] if hits else 'n/a'})")
    return hits
