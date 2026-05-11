"""
RAG knowledge base backed by ChromaDB + its built-in ONNX embedding function.
Uses all-MiniLM-L6-v2 via onnxruntime — no PyTorch required.

Call ingest_review() to add documents, rag_search() to query.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "music_reviews"

_ef = None
_collection = None


def _get_ef():
    global _ef
    if _ef is None:
        _ef = DefaultEmbeddingFunction()
    return _ef


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def ingest_review(doc_id: str, title: str, content: str, metadata: dict | None = None) -> None:
    """
    Add or update a document in the vector store.

    doc_id   – stable unique key (e.g. the article URL hash)
    title    – human-readable label stored in metadata
    content  – full text to embed and retrieve
    metadata – any extra fields (artist, genre, source, …)
    """
    col = _get_collection()
    meta = {"title": title, **(metadata or {})}
    # upsert so re-runs don't duplicate
    col.upsert(ids=[doc_id], documents=[content], metadatas=[meta])


def rag_search(query: str, n_results: int = 4) -> list[dict]:
    """
    Return the top-n most relevant documents for a natural-language query.
    Each result dict has: text, title, score, and any extra metadata keys.
    """
    col = _get_collection()
    count = col.count()
    if count == 0:
        return []

    results = col.query(
        query_texts=[query],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"text": doc, "score": round(1 - dist, 4), **meta})
    return out


def collection_count() -> int:
    return _get_collection().count()
