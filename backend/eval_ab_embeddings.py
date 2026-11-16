"""
A/B test of two embedding models for RAG retrieval.

    A (control):    all-MiniLM-L6-v2   -- Chroma's default, what rag.py uses
    B (challenger): BAAI/bge-small-en-v1.5 via fastembed (ONNX, no PyTorch)

Everything except the embedding model is held fixed: B's index is built from
the exact chunks (ids, text, metadata) already in backend/chroma_db, and both
models go through the same search logic as rag.rag_search (over-fetch chunks,
keep the best chunk per review). B's index lives in backend/chroma_ab/, so the
production collection is never touched.

BGE is asymmetric: queries get its retrieval instruction prefix, passages
don't (per the model card's recommendation for short query -> passage search).

Metrics (per model, on the same queries):
  - Recall@K: at least one relevant review in the top K (K = 1, 3, 5, 10)
  - MRR@10:   mean of 1/rank of the first relevant review (0 if not in top 10)
  - Recall@3 by question type (the dataset's optional "type" field)
  - query latency (embed + search) and index build time
Significance, on Recall@3 (the agent's default n_results):
  - McNemar's exact test on the per-query hit/miss disagreements
  - bootstrap 95% CI for the B - A difference

Use the dev split for model selection; run the test split once at the end.

Usage:
    cd backend
    python eval_ab_embeddings.py                              # dev split
    python eval_ab_embeddings.py --data eval_data/rag_test.json
    python eval_ab_embeddings.py --rebuild                    # re-embed B's index
    python eval_ab_embeddings.py --csv results.csv

Requires a populated backend/chroma_db (run scraper.py first) and fastembed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import chromadb

import rag

AB_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_ab")
B_COLLECTION = "music_reviews_bge_small"
B_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_DATA = Path(__file__).parent / "eval_data" / "rag_dev.json"
KS = [1, 3, 5, 10]
PRIMARY_K = 3


# ---------------------------------------------------------------------------
# The two retrievers -- identical apart from how vectors are made
# ---------------------------------------------------------------------------

def _best_chunk_per_review(col: chromadb.Collection, query_embedding: list[float], n: int) -> list[str]:
    """Same over-fetch + dedupe as rag.rag_search; returns review urls, best first."""
    res = col.query(query_embeddings=[query_embedding], n_results=min(n * 5, col.count()),
                    include=["metadatas"])
    urls: list[str] = []
    for meta in res["metadatas"][0]:
        if meta["url"] not in urls:
            urls.append(meta["url"])
        if len(urls) == n:
            break
    return urls


class MiniLM:
    name = "A: all-MiniLM-L6-v2"

    def __init__(self):
        self.col = rag._get_collection()
        self.ef = rag._get_ef()
        self.build_seconds = None  # already built by scraper.py

    def search(self, query: str, n: int) -> list[str]:
        return _best_chunk_per_review(self.col, [float(x) for x in self.ef([query])[0]], n)


class BgeSmall:
    name = "B: bge-small-en-v1.5"

    def __init__(self, rebuild: bool):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(B_MODEL)
        client = chromadb.PersistentClient(path=AB_DB_PATH)
        source = rag._get_collection().get(include=["documents", "metadatas"])
        self.build_seconds = None

        existing = {c.name for c in client.list_collections()}
        if B_COLLECTION in existing and not rebuild:
            self.col = client.get_collection(B_COLLECTION)
            if sorted(self.col.get()["ids"]) == sorted(source["ids"]):
                return
            print("B index is out of date with chroma_db -- rebuilding.")
        if B_COLLECTION in existing:
            client.delete_collection(B_COLLECTION)

        # no embedding_function: vectors are always passed in explicitly
        self.col = client.create_collection(B_COLLECTION, metadata={"hnsw:space": "cosine"},
                                            embedding_function=None)
        start = time.perf_counter()
        vectors = [v.tolist() for v in self.model.embed(source["documents"])]
        self.build_seconds = time.perf_counter() - start
        self.col.add(ids=source["ids"], documents=source["documents"],
                     metadatas=source["metadatas"], embeddings=vectors)

    def search(self, query: str, n: int) -> list[str]:
        vector = next(iter(self.model.embed([BGE_QUERY_PREFIX + query]))).tolist()
        return _best_chunk_per_review(self.col, vector, n)


# ---------------------------------------------------------------------------
# Metrics and significance
# ---------------------------------------------------------------------------

def first_relevant_rank(retrieved: list[str], relevant: set[str]) -> int | None:
    for i, url in enumerate(retrieved, 1):
        if url in relevant:
            return i
    return None


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two disagreement counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_diff_ci(a_hits: list[int], b_hits: list[int], iters: int = 10_000,
                      seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a_hits)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b_hits[i] - a_hits[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]


def evaluate(model, rows: list[dict]) -> dict:
    ranks, latencies = [], []
    for r in rows:
        start = time.perf_counter()
        retrieved = model.search(r["query"], max(KS))
        latencies.append((time.perf_counter() - start) * 1000)
        ranks.append(first_relevant_rank(retrieved, set(r["relevant_docs"])))
    return {"ranks": ranks, "latency_ms": latencies}


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test: MiniLM vs bge-small for retrieval")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                        help="Dataset (JSON list of {query, relevant_docs, type?}); default is the dev split")
    parser.add_argument("--rebuild", action="store_true", help="Re-embed B's index from chroma_db")
    parser.add_argument("--csv", type=str, default=None,
                        help="Also write the per-query results table to this CSV file")
    args = parser.parse_args()

    if rag.collection_count() == 0:
        print("ChromaDB collection is empty -- run scraper.py first.")
        sys.exit(1)

    with open(args.data) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} queries from {args.data}\n")

    models = [MiniLM(), BgeSmall(rebuild=args.rebuild)]
    results = {m.name: evaluate(m, rows) for m in models}
    names = [m.name for m in models]

    print("=" * 72)
    print(f"{'metric':22s}" + "".join(f"{n:>25s}" for n in names))
    print("=" * 72)
    for k in KS:
        vals = [sum(rk is not None and rk <= k for rk in results[n]["ranks"]) / len(rows) for n in names]
        print(f"{f'Recall@{k}':22s}" + "".join(f"{v:>25.1%}" for v in vals))
    mrr = [sum(1 / rk for rk in results[n]["ranks"] if rk) / len(rows) for n in names]
    print(f"{'MRR@10':22s}" + "".join(f"{v:>25.3f}" for v in mrr))
    lat = [statistics.median(results[n]["latency_ms"]) for n in names]
    print(f"{'median query ms':22s}" + "".join(f"{v:>25.1f}" for v in lat))
    build = ["(prebuilt)" if m.build_seconds is None else f"{m.build_seconds:.1f}s" for m in models]
    print(f"{'index build time':22s}" + "".join(f"{v:>25s}" for v in build))

    hits = {n: [int(rk is not None and rk <= PRIMARY_K) for rk in results[n]["ranks"]] for n in names}
    a, b = hits[names[0]], hits[names[1]]

    if any("type" in r for r in rows):
        print(f"\nRecall@{PRIMARY_K} by question type:")
        by_type = defaultdict(list)
        for i, r in enumerate(rows):
            by_type[r.get("type", "untyped")].append(i)
        for t, idx in sorted(by_type.items()):
            va, vb = sum(a[i] for i in idx) / len(idx), sum(b[i] for i in idx) / len(idx)
            print(f"  {t:12s} n={len(idx):<3d} A={va:6.1%}  B={vb:6.1%}  diff={vb - va:+6.1%}")

    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    lo, hi = bootstrap_diff_ci(a, b)
    print(f"\nRecall@{PRIMARY_K} significance (paired, same {len(rows)} queries):")
    print(f"  only A hit: {only_a}   only B hit: {only_b}   both/neither: {len(rows) - only_a - only_b}")
    print(f"  B - A = {(sum(b) - sum(a)) / len(rows):+.1%}   bootstrap 95% CI [{lo:+.1%}, {hi:+.1%}]")
    print(f"  McNemar exact p = {mcnemar_exact_p(only_a, only_b):.3f}"
          "   (p < 0.05 suggests a real difference)")

    flips = [(r["query"], x, y) for r, x, y in zip(rows, a, b) if x != y]
    if flips:
        print(f"\nQueries where the models disagree at Recall@{PRIMARY_K}:")
        for q, x, y in flips:
            print(f"  {'B wins' if y else 'A wins'}: {q!r}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query", "type", "a_first_relevant_rank", "b_first_relevant_rank"])
            for r, ra, rb in zip(rows, results[names[0]]["ranks"], results[names[1]]["ranks"]):
                w.writerow([r["query"], r.get("type", ""), ra or "", rb or ""])
        print(f"\nPer-query results written to {args.csv}")


if __name__ == "__main__":
    main()
