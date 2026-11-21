from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI

from src.utils.config import settings
from src.utils.logger import logger

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts via OpenAI.
    Batches are handled automatically by the SDK.
    Returns a list of float vectors in the same order as input.
    """
    if not texts:
        return []

    logger.debug(f"Embedding {len(texts)} texts with {settings.embedding_model}")
    response = _get_client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    logger.debug(f"  → {len(embeddings)} embeddings (dim={len(embeddings[0])})")
    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
