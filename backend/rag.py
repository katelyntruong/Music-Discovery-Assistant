"""
RAG knowledge base backed by ChromaDB + its built-in ONNX embedding function.
Uses all-MiniLM-L6-v2 via onnxruntime — no PyTorch required.

Reviews are split into overlapping word-window chunks before embedding:
MiniLM truncates input at 256 tokens (~190 words), so embedding a whole
800-word review would silently ignore most of it. Each chunk is stored with
its parent review's id/title/url, and rag_search returns at most one chunk
(the best-matching) per review.

Call ingest_review() to add documents, rag_search() to query.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "music_reviews"

# ~150 words ≈ 200 tokens, leaving room for the title prefix under MiniLM's 256 cap
CHUNK_WORDS = 150
CHUNK_OVERLAP = 30

_ef = None
_client = None
_collection = None


def _get_ef():
    global _ef
    if _ef is None:
        _ef = DefaultEmbeddingFunction()
    return _ef


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
    return _client


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_collection() -> None:
    """Delete every document (drop and recreate the collection)."""
    global _collection
    try:
        _get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet
    _collection = None
    _get_collection()


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into windows of `size` words, each overlapping the previous by `overlap`."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def ingest_review(doc_id: str, title: str, content: str, metadata: dict | None = None) -> None:
    """
    Add or update a document in the vector store, split into chunks.

    doc_id   – stable unique key (e.g. the article URL hash)
    title    – human-readable label stored in metadata (and prefixed to each
               chunk so every chunk knows which artist/album it's about)
    content  – full text to chunk, embed and retrieve
    metadata – any extra fields (artist, genre, source, …)
    """
    col = _get_collection()
    # drop this review's old chunks first, in case it now splits into fewer
    col.delete(where={"parent_id": doc_id})

    chunks = chunk_text(content)
    if not chunks:
        return
    base_meta = {"title": title, "parent_id": doc_id, **(metadata or {})}
    col.upsert(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=[f"{title}\n{chunk}" for chunk in chunks],
        metadatas=[{**base_meta, "chunk_index": i} for i in range(len(chunks))],
    )


def rag_search(query: str, n_results: int = 4) -> list[dict]:
    """
    Return the top-n most relevant reviews for a natural-language query.
    Each result dict has: text (the best-matching chunk of that review),
    title, score, and any extra metadata keys.
    """
    col = _get_collection()
    count = col.count()
    if count == 0:
        return []

    # over-fetch chunks, then keep only the best chunk per review
    results = col.query(
        query_texts=[query],
        n_results=min(n_results * 5, count),
        include=["documents", "metadatas", "distances"],
    )

    out = []
    seen_parents: set[str] = set()
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        parent = meta.get("parent_id")
        if parent in seen_parents:
            continue
        seen_parents.add(parent)
        out.append({"text": doc, "score": round(1 - dist, 4), **meta})
        if len(out) == n_results:
            break
    return out


def collection_count() -> int:
    return _get_collection().count()
