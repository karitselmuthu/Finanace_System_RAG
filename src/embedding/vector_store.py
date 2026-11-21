from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.chunker import Chunk
from src.embedding.embedder import embed_texts, embed_query
from src.utils.config import settings
from src.utils.logger import logger

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{settings.chroma_collection_name}' "
            f"loaded ({_collection.count()} existing vectors)"
        )
    return _collection


def upsert_chunks(chunks: list[Chunk]) -> None:
    """Embed and store chunks in ChromaDB. Existing IDs are updated."""
    if not chunks:
        return

    collection = _get_collection()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c.metadata for c in chunks],
    )
    logger.info(f"Upserted {len(chunks)} chunks into vector store")


def query_similar(
    query: str,
    n_results: int | None = None,
    where: dict | None = None,
) -> list[dict]:
    """
    Return the top-N most similar chunks to the query.

    Each result dict contains:
        id, text, metadata, distance
    """
    k = n_results or settings.top_k_results
    collection = _get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append(
            {
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to a document."""
    collection = _get_collection()
    collection.delete(where={"doc_id": doc_id})
    logger.info(f"Deleted all chunks for doc_id={doc_id}")
