from __future__ import annotations

from src.embedding.vector_store import query_similar
from src.utils.config import settings
from src.utils.logger import logger


def retrieve(
    query: str,
    top_k: int | None = None,
    doc_id: str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a natural-language query.

    Args:
        query:   The user's question.
        top_k:   Number of chunks to return (defaults to settings.top_k_results).
        doc_id:  Optional — restrict search to a single document.

    Returns:
        List of dicts: { id, text, metadata, distance, score }
        where score = 1 - distance (cosine similarity, higher is better).
    """
    k = top_k or settings.top_k_results
    where = {"doc_id": doc_id} if doc_id else None

    hits = query_similar(query, n_results=k, where=where)

    # Convert cosine distance → similarity score (0-1)
    for hit in hits:
        hit["score"] = round(1.0 - hit["distance"], 4)

    if hits:
        logger.info(
            f"Retrieval scores — top: {hits[0]['score']}, "
            f"bottom: {hits[-1]['score']}, threshold: {settings.similarity_threshold}"
        )

    # Filter by minimum similarity threshold
    hits = [h for h in hits if h["score"] >= settings.similarity_threshold]

    logger.debug(
        f"Retrieval: query='{query[:60]}...' → {len(hits)} hits above threshold"
    )
    return hits
