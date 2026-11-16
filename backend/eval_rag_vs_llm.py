"""
LLM-only vs LLM+RAG comparison for the music agent.

Question this answers: does giving the model retrieved review text actually
improve its answers, compared with the model answering from memory?

Both conditions use the same model and the same neutral system prompt. The
only difference is that LLM+RAG also receives the top-K rag_search results.
agent.SYSTEM is deliberately not used: it tells the model to rely on tool
results, which would make the LLM-only condition refuse nearly everything.

Each query's two answers go to one judge call, shuffled and labeled A/B so the
judge can't tell which is which. The judge scores both against the full text
of the review labeled relevant in the dataset (not the retrieved text), so
LLM-only isn't penalized just for not having seen the retrieved chunks.

Metrics, per condition:
  - avg correctness (1-5) and % correct (score >= 4)
  - % abstained: said it didn't know
  - % confidently wrong: gave an answer scoring <= 2
  - avg unsupported claims: claims the review doesn't back up. This does NOT
    mean false -- a true fact from memory that isn't in the review counts too.
LLM+RAG is also split by whether retrieval found the right review, since a
retrieval miss limits what generation can do.

Dataset: eval_data/rag_retrieval.json. Only queries with exactly one relevant
doc are used, since the judge needs a single review to grade against.

Usage:
    cd backend
    python eval_rag_vs_llm.py                    # random sample of 30 queries
    python eval_rag_vs_llm.py --n 50 --seed 7
    python eval_rag_vs_llm.py --csv results.csv

Requires GROQ_API_KEY and a populated backend/chroma_db. Makes three real LLM
calls per query (two to generate, one to judge).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import eval_generation
from eval_generation import _extract_json, create_completion
from rag import CHUNK_OVERLAP, _get_collection, collection_count, rag_search

DEFAULT_DATA = Path(__file__).parent / "eval_data" / "rag_retrieval.json"
DEFAULT_JUDGE = "openai/gpt-oss-120b"
CONDITIONS = ["llm_only", "llm_rag"]
TOP_K = 3  # same default as agent._run_tool

ANSWER_SYSTEM = """You are a music assistant answering questions about recent album reviews
from Rolling Stone. Answer in 2-4 sentences. If you don't know, say so rather than guessing."""

JUDGE_SYSTEM = """You are a strict evaluator of a music assistant. You will be given a question,
the full text of the Rolling Stone review that answers it, and two candidate answers, A and B.
Treat the review as the source of truth.

For each answer return:
- correctness: 1 (wrong album or wrong facts) to 5 (fully correct and answers the question).
  An answer that declines to answer scores 1.
- abstained: true if it says it doesn't know or can't answer.
- unsupported_claims: specific claims about this album or artist that the review doesn't back up.

Return ONLY a JSON object of this exact shape, no other text:
{
  "A": {"correctness": <1-5>, "abstained": <true|false>, "unsupported_claims": ["<claim>", ...]},
  "B": {"correctness": <1-5>, "abstained": <true|false>, "unsupported_claims": ["<claim>", ...]}
}"""

JUDGE_USER_TEMPLATE = """Question: {query}

Review:
{review}

Answer A:
{answer_a}

Answer B:
{answer_b}"""


def load_reviews() -> dict[str, str]:
    """url -> full review text, rebuilt from its overlapping chunks."""
    data = _get_collection().get(include=["metadatas", "documents"])
    chunks: dict[str, dict[int, str]] = {}
    titles: dict[str, str] = {}
    for meta, doc in zip(data["metadatas"], data["documents"]):
        chunks.setdefault(meta["url"], {})[meta["chunk_index"]] = doc.split("\n", 1)[1]
        titles[meta["url"]] = meta["title"]

    reviews = {}
    for url, parts in chunks.items():
        ordered = [parts[i] for i in sorted(parts)]
        words = ordered[0].split()
        for chunk in ordered[1:]:
            words += chunk.split()[CHUNK_OVERLAP:]
        reviews[url] = f"{titles[url]}\n{' '.join(words)}"
    return reviews


def sample_dataset(path: Path, n: int, seed: int) -> list[dict]:
    with open(path) as f:
        rows = [r for r in json.load(f) if len(r.get("relevant_docs", [])) == 1]
    return random.Random(seed).sample(rows, min(n, len(rows)))


def generate(query: str, excerpts: list[dict] | None) -> str:
    user_msg = query
    if excerpts is not None:
        block = "\n\n".join(f"[{r['title']}]\n{r['text']}" for r in excerpts)
        user_msg = f"Retrieved review excerpts:\n{block}\n\nQuestion: {query}"
    messages = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user_msg}]
    # gpt-oss occasionally returns empty content; one retry usually clears it
    for _ in range(2):
        response = create_completion(model=eval_generation.agent.MODEL, messages=messages)
        answer = response.choices[0].message.content or ""
        if answer.strip():
            return answer
    return ""


def judge(query: str, review: str, answers: dict[str, str], judge_model: str,
          rng: random.Random) -> dict | None:
    """Returns {condition: verdict}, with the blind A/B labels mapped back."""
    order = list(CONDITIONS)
    rng.shuffle(order)
    user_msg = JUDGE_USER_TEMPLATE.format(query=query, review=review,
                                          answer_a=answers[order[0]], answer_b=answers[order[1]])
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user_msg}]
    response = create_completion(model=judge_model, messages=messages,
                                 response_format={"type": "json_object"})
    verdict = _extract_json(response.choices[0].message.content or "")
    if not verdict or "A" not in verdict or "B" not in verdict:
        return None
    return {order[0]: verdict["A"], order[1]: verdict["B"]}


def summarize(label: str, verdicts: list[dict]) -> None:
    n = len(verdicts)
    if not n:
        print(f"{label:30s} (no queries)")
        return
    scores = [int(v.get("correctness", 1)) for v in verdicts]
    abstained = [bool(v.get("abstained")) for v in verdicts]
    wrong = sum(s <= 2 and not a for s, a in zip(scores, abstained))
    claims = sum(len(v.get("unsupported_claims") or []) for v in verdicts)
    print(f"{label:30s} n={n:<3d} avg={sum(scores) / n:.2f}/5  "
          f"correct={sum(s >= 4 for s in scores) / n:6.1%}  "
          f"abstained={sum(abstained) / n:6.1%}  "
          f"confidently_wrong={wrong / n:6.1%}  "
          f"unsupported/answer={claims / n:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-only vs LLM+RAG comparison")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                        help="Path to the dataset (JSON list of {query, relevant_docs})")
    parser.add_argument("--n", type=int, default=30, help="Number of queries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the sample")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE, help="Model to use as judge")
    parser.add_argument("--csv", type=str, default=None,
                        help="Also write the per-query results table to this CSV file")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set (checked environment and .env).")
        sys.exit(1)
    if collection_count() == 0:
        print("ChromaDB collection is empty -- run scraper.py first.")
        sys.exit(1)

    import agent
    eval_generation.agent = agent

    reviews = load_reviews()
    dataset = sample_dataset(Path(args.data), args.n, args.seed)
    rng = random.Random(args.seed)
    print(f"Sampled {len(dataset)} queries. Generator: {agent.MODEL}  Judge: {args.judge_model}\n")

    rows = []
    for item in dataset:
        query, url = item["query"], item["relevant_docs"][0]
        try:
            excerpts = rag_search(query, n_results=TOP_K)
            hit = url in [r.get("url") for r in excerpts]
            answers = {"llm_only": generate(query, None), "llm_rag": generate(query, excerpts)}
            verdicts = judge(query, reviews[url], answers, args.judge_model, rng)
            error = None if verdicts else "judge returned unparseable output"
        except Exception as e:
            answers, verdicts, hit, error = {}, None, False, f"{type(e).__name__}: {e}"

        if error:
            print(f"[ERROR] {query!r}: {error}", flush=True)
            continue

        rows.append({"query": query, "hit": hit, "answers": answers, "verdicts": verdicts})
        print(f"[{'hit ' if hit else 'miss'}] {query!r}  "
              f"llm_only={verdicts['llm_only'].get('correctness')}  "
              f"llm_rag={verdicts['llm_rag'].get('correctness')}", flush=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Queries scored: {len(rows)}/{len(dataset)}\n")
    summarize("LLM only", [r["verdicts"]["llm_only"] for r in rows])
    summarize("LLM + RAG", [r["verdicts"]["llm_rag"] for r in rows])
    summarize("LLM + RAG, retrieval hit", [r["verdicts"]["llm_rag"] for r in rows if r["hit"]])
    summarize("LLM + RAG, retrieval miss", [r["verdicts"]["llm_rag"] for r in rows if not r["hit"]])

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query", "retrieval_hit", "llm_only_score", "llm_rag_score",
                        "llm_only_answer", "llm_rag_answer"])
            for r in rows:
                w.writerow([r["query"], r["hit"], r["verdicts"]["llm_only"].get("correctness"),
                            r["verdicts"]["llm_rag"].get("correctness"),
                            r["answers"]["llm_only"], r["answers"]["llm_rag"]])
        print(f"\nPer-query results written to {args.csv}")


if __name__ == "__main__":
    main()
