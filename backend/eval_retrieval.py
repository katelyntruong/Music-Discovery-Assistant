"""
RAG retrieval eval — Recall@K / Precision@K for the Rolling Stone review
knowledge base (rag.py + ChromaDB).

Question this answers: when a query needs the knowledge base, does
retrieval actually surface the right article(s) — independent of whatever
the LLM does with them afterwards?

    query -> embed -> ChromaDB -> top-K documents   <- evaluated here
                                        |
                                        v
                                (LLM generation; not this script's concern)

You need ground-truth labels: for each eval query, the article URL(s) that
actually answer it. There's no way to generate these automatically — you
have to scrape (`python scraper.py --pages N`), read a handful of the
resulting articles, and record which one(s) are relevant to each query.
Start with 30-50 queries.

Dataset format (JSON list, default: eval_data/rag_retrieval.json):
    [
      {"query": "What did Rolling Stone say about Radiohead?",
       "relevant_docs": ["https://www.rollingstone.com/.../radiohead-review/"]},
      ...
    ]
A retrieved result counts as relevant if its `url` metadata matches any
string in that query's relevant_docs.

Usage:
    cd backend
    python eval_retrieval.py                  # default dataset, K = 1,3,5,10
    python eval_retrieval.py --k 1 5 10
    python eval_retrieval.py --data path/to.json
    python eval_retrieval.py --csv results.csv

Requires backend/chroma_db to already be populated (run scraper.py first).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from rag import rag_search, collection_count

DEFAULT_DATA = Path(__file__).parent / "eval_data" / "rag_retrieval.json"
DEFAULT_KS = [1, 3, 5, 10]


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    for r in rows:
        r["relevant_docs"] = set(r.get("relevant_docs", []))
    return rows


def retrieve(query: str, max_k: int) -> list[str]:
    """Retrieved doc identifiers, best match first (url, falling back to
    title for any hand-ingested doc that has no url metadata)."""
    results = rag_search(query, n_results=max_k)
    return [r.get("url") or r.get("title") or "" for r in results]


def recall_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    """Did at least one relevant doc appear in the top K?"""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    """Of the top K retrieved, what fraction were actually relevant?"""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval eval (Recall@K / Precision@K)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                         help="Path to the eval dataset (JSON list of {query, relevant_docs})")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS,
                         help="Cutoffs to evaluate (default: 1 3 5 10)")
    parser.add_argument("--csv", type=str, default=None,
                         help="Also write the per-query results table to this CSV file")
    args = parser.parse_args()

    count = collection_count()
    if count == 0:
        print(
            "ChromaDB collection is empty. Run `python scraper.py --pages N` first "
            "to ingest Rolling Stone reviews, then fill in eval_data/rag_retrieval.json "
            "with real article URLs before running this eval."
        )
        sys.exit(1)

    ks = sorted(set(args.k))
    max_k = max(ks)
    if count < max_k:
        print(f"Note: collection only has {count} document(s), fewer than the largest "
              f"K={max_k} requested -- Precision@{max_k} will be pulled down by that, "
              f"not by retrieval quality.\n")

    dataset = load_dataset(Path(args.data))
    print(f"Loaded {len(dataset)} queries from {args.data}\n")

    rows = []
    for item in dataset:
        query = item["query"]
        relevant = item["relevant_docs"]
        retrieved = retrieve(query, max_k)

        per_query = {"query": query, "retrieved": retrieved}
        for k in ks:
            per_query[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            per_query[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        rows.append(per_query)

        hit = "HIT " if any(per_query[f"recall@{k}"] for k in ks) else "MISS"
        print(f"[{hit}] {query!r}")
        print(f"       top-{max_k}: {retrieved}")
        print(f"       relevant:  {sorted(relevant)}")

    n = len(rows)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Queries evaluated: {n}\n")
    for k in ks:
        r = sum(row[f"recall@{k}"] for row in rows) / n
        p = sum(row[f"precision@{k}"] for row in rows) / n
        print(f"Recall@{k:<2} = {r:.1%}    Precision@{k:<2} = {p:.1%}")

    misses = [row["query"] for row in rows if not any(row[f"recall@{k}"] for k in ks)]
    if misses:
        print(f"\n{len(misses)} quer{'y' if len(misses) == 1 else 'ies'} never surfaced "
              f"a relevant doc within top-{max_k}:")
        for q in misses:
            print(f"  {q!r}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            fieldnames = ["query"] + [f"recall@{k}" for k in ks] + [f"precision@{k}" for k in ks]
            w = csv.writer(f)
            w.writerow(fieldnames)
            for row in rows:
                w.writerow(
                    [row["query"]]
                    + [f"{row[f'recall@{k}']:.3f}" for k in ks]
                    + [f"{row[f'precision@{k}']:.3f}" for k in ks]
                )
        print(f"\nPer-query results written to {args.csv}")


if __name__ == "__main__":
    main()
