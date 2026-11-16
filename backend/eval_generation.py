"""
Generation eval for the music agent.

Question this answers: given the query and the information that was
actually retrieved (RAG docs and/or tool results), did the LLM's answer
correctly use that information?

    query
      |
      v
    retrieved context (fixed/given here -- retrieval itself is
      |                 eval_retrieval.py's job, tool choice is
      |                 eval_tools.py's job)
      v
    LLM answer   <- EVALUATED HERE, by an LLM judge, on 4 axes:

  - correctness:   are the answer's claims actually true / do they match
                    the reference_answer when one is given?
  - faithfulness:   is every claim in the answer supported by the given
                    context? (this is the hallucination check -- a claim
                    can be faithful without being correct, and vice versa)
  - relevance:      does the answer actually address what was asked?
  - completeness:   does it make use of the useful information available
                    in the context, rather than leaving things out?

Each query's context is fed to the model in the same shape production uses
-- as synthetic tool-call/tool-result turns ahead of the real system prompt
-- then the model is asked to produce its final answer with no further
tools available, isolating the generation step from tool selection and
retrieval (both covered by the other two eval scripts).

Dataset format (JSON list, default: eval_data/generation.json):
    [
      {"query": "What did critics say about Radiohead's In Rainbows?",
       "context": [{"source": "rag_search", "content": "..."}],
       "reference_answer": "..."}   // optional; omit or null if you don't have one
    ]

Scoring is done by an LLM judge (1-5 per axis) since these are inherently
qualitative judgments -- there's no exact-match check for "did it use the
context well". By default the judge is the SAME model being evaluated,
which is a known weakness (self-preference bias); pass --judge-model to
point at a different/stronger model if you have one available on Groq
(GET /openai/v1/models lists what your account can use).

Usage:
    cd backend
    python eval_generation.py
    python eval_generation.py --data path/to.json
    python eval_generation.py --judge-model openai/gpt-oss-120b
    python eval_generation.py --csv results.csv

Requires GROQ_API_KEY (via .env or env var) -- this makes two real LLM
calls per query (one to generate, one to judge). Rate-limit (429) errors are
retried after the wait Groq asks for (up to MAX_RATE_LIMIT_WAIT seconds per
attempt), so hitting the per-minute or rolling daily token cap pauses the run
instead of dropping queries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

# imported lazily in main(), after argparse has handled --help and after we
# know GROQ_API_KEY is set
agent = None

DEFAULT_DATA = Path(__file__).parent / "eval_data" / "generation.json"
DIMENSIONS = ["correctness", "faithfulness", "relevance", "completeness"]

# 429 handling: wait as long as Groq says (capped), up to this many attempts per call
MAX_RATE_LIMIT_RETRIES = 6
MAX_RATE_LIMIT_WAIT = 15 * 60  # seconds

JUDGE_SYSTEM = """You are a strict, careful evaluator of an AI music assistant's generated answers.
You will be given the user's question, the retrieved context the assistant had access to, the
assistant's generated answer, and optionally a reference answer.

Score the generated answer on each of these 4 criteria, from 1 (very poor) to 5 (excellent):

- correctness: Do the answer's claims match reality? If a reference answer is given, judge
  against it directly. If no reference answer is given, judge whether the claims are accurate
  and consistent with the provided context.
- faithfulness: Is every factual claim in the answer directly supported by the retrieved context?
  Penalize invented songs, artists, quotes, or facts not present in the context -- this is about
  hallucination, independent of whether a claim happens to be true in the real world. If the
  context contains no useful information, the only faithful answer says so.
- relevance: Does the answer directly address what the user asked, without padding or refusing
  to engage where it reasonably could?
- completeness: Does the answer make use of the useful information available in the context,
  rather than leaving out relevant facts that were right there to use?

Return ONLY a JSON object of this exact shape, no other text:
{
  "correctness":  {"score": <1-5>, "reasoning": "<one sentence>"},
  "faithfulness": {"score": <1-5>, "reasoning": "<one sentence>", "unsupported_claims": ["<claim>", ...]},
  "relevance":    {"score": <1-5>, "reasoning": "<one sentence>"},
  "completeness": {"score": <1-5>, "reasoning": "<one sentence>"}
}"""

JUDGE_USER_TEMPLATE = """Question: {query}

Retrieved context:
{context_block}

Reference answer (may be absent): {reference_answer}

Assistant's generated answer:
{answer}"""


# ---------------------------------------------------------------------------
# Generation step -- isolated from the tool-calling loop: the context is
# injected as synthetic tool-call/tool-result turns (matching agent.py's
# real message shape), then the model is asked to answer with no tools
# available, so this call can only produce a final text answer.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rate-limit-aware completion call
# ---------------------------------------------------------------------------

def _retry_after_seconds(err: openai.RateLimitError) -> float:
    """How long Groq asked us to wait: retry-after header, else the
    'Please try again in 1m58.3s' text in the message, else 30s."""
    try:
        header = err.response.headers.get("retry-after")
        if header:
            return float(header)
    except Exception:
        pass
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(err))
    if m and any(m.groups()):
        h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mins * 60 + secs
    return 30.0


def create_completion(**kwargs):
    client = agent._get_client()
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            wait = min(_retry_after_seconds(e) + 1, MAX_RATE_LIMIT_WAIT)
            print(f"       rate limited on {kwargs.get('model')} -- waiting {wait:.0f}s "
                  f"(attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})", flush=True)
            time.sleep(wait)


def _context_to_tool_content(item: dict):
    content = item["content"]
    source = item.get("source", "rag_search")
    if isinstance(content, str) and source == "rag_search":
        # mirror rag.rag_search's real return shape
        return [{"text": content, "title": item.get("title", "Provided context"), "score": 1.0}]
    return content


def build_messages(query: str, context_items: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": agent.SYSTEM}, {"role": "user", "content": query}]
    for i, item in enumerate(context_items):
        source = item.get("source", "rag_search")
        call_id = f"eval_call_{i}"
        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": source, "arguments": "{}"}}],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id,
            "content": json.dumps(_context_to_tool_content(item)),
        })
    return messages


def generate_answer(query: str, context_items: list[dict]) -> str:
    # same nudge + tool_use_failed retry as agent.final_answer (production's
    # no-tools path), but routed through create_completion for 429 handling
    messages = build_messages(query, context_items)
    messages.append({"role": "system", "content": agent.FINAL_ANSWER_NUDGE})
    for attempt in range(2):
        try:
            response = create_completion(model=agent.MODEL, messages=messages)
            return response.choices[0].message.content or ""
        except openai.BadRequestError as e:
            if getattr(e, "code", None) != "tool_use_failed" or attempt == 1:
                raise


# ---------------------------------------------------------------------------
# Judge step
# ---------------------------------------------------------------------------

def format_context_for_judge(context_items: list[dict]) -> str:
    if not context_items:
        return "(no context was retrieved for this query)"
    lines = []
    for i, item in enumerate(context_items, 1):
        content = item["content"]
        if not isinstance(content, str):
            content = json.dumps(content)
        lines.append(f"[{i}] (from {item.get('source', 'rag_search')})\n{content}")
    return "\n\n".join(lines)


def _extract_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def judge_answer(query: str, context_items: list[dict], answer: str,
                  reference_answer: str | None, judge_model: str) -> dict | None:
    user_msg = JUDGE_USER_TEMPLATE.format(
        query=query,
        context_block=format_context_for_judge(context_items),
        reference_answer=reference_answer or "(none provided)",
        answer=answer,
    )
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user_msg}]
    try:
        resp = create_completion(model=judge_model, messages=messages,
                                 response_format={"type": "json_object"})
    except openai.BadRequestError:
        # model doesn't support JSON mode -- fall back to plain output + _extract_json
        resp = create_completion(model=judge_model, messages=messages)
    return _extract_json(resp.choices[0].message.content or "")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation eval for the music agent")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                         help="Path to the eval dataset (JSON list of {query, context, reference_answer})")
    parser.add_argument("--judge-model", type=str, default=None,
                         help="Model to use as judge (default: same model being evaluated)")
    parser.add_argument("--csv", type=str, default=None,
                         help="Also write the per-query results table to this CSV file")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print(
            "GROQ_API_KEY not set (checked environment and .env).\n"
            "This eval makes real calls to the LLM (one to generate an answer, one to\n"
            "judge it) -- set the key and retry."
        )
        sys.exit(1)

    global agent
    import agent as _agent_module
    agent = _agent_module
    judge_model = args.judge_model or agent.MODEL

    dataset = load_dataset(Path(args.data))
    print(f"Loaded {len(dataset)} queries from {args.data}")
    print(f"Judge model: {judge_model}\n")

    rows = []
    for item in dataset:
        query = item["query"]
        context_items = item.get("context", [])
        reference_answer = item.get("reference_answer")

        try:
            answer = generate_answer(query, context_items)
            verdict = judge_answer(query, context_items, answer, reference_answer, judge_model)
            error = None if verdict else "judge returned unparseable output"
        except Exception as e:
            answer, verdict, error = "", None, f"{type(e).__name__}: {e}"

        scores = {}
        unsupported_claims = []
        if verdict:
            for dim in DIMENSIONS:
                try:
                    scores[dim] = int(verdict.get(dim, {}).get("score", 0))
                except (TypeError, ValueError):
                    scores[dim] = 0
            unsupported_claims = verdict.get("faithfulness", {}).get("unsupported_claims", []) or []

        rows.append({
            "query": query, "answer": answer, "scores": scores,
            "unsupported_claims": unsupported_claims, "verdict": verdict, "error": error,
        })

        if error:
            print(f"[ERROR] {query!r}: {error}")
            continue

        avg = sum(scores.values()) / len(DIMENSIONS)
        flag = "PASS" if avg >= 4 and scores.get("faithfulness", 0) >= 4 else \
               ("HALLUCINATION" if scores.get("faithfulness", 0) <= 2 else "LOW")
        print(f"[{flag}] {query!r}  (avg {avg:.1f}/5)")
        print(f"       {scores}")
        if unsupported_claims:
            print(f"       unsupported claims: {unsupported_claims}")

    scored_rows = [r for r in rows if not r["error"]]
    n = len(scored_rows)
    if n == 0:
        print("\nNo queries scored successfully -- nothing to summarize.")
        return

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Queries scored: {n}/{len(rows)}\n")
    for dim in DIMENSIONS:
        avg = sum(r["scores"][dim] for r in scored_rows) / n
        print(f"{dim:14s} avg = {avg:.2f}/5")

    hallucinations = [r for r in scored_rows if r["scores"].get("faithfulness", 0) <= 2]
    if hallucinations:
        print(f"\n{len(hallucinations)} possible hallucination(s) (faithfulness <= 2):")
        for r in hallucinations:
            print(f"  {r['query']!r}")
            if r["unsupported_claims"]:
                print(f"    unsupported claims: {r['unsupported_claims']}")
    else:
        print("\nNo low-faithfulness answers flagged.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query", "answer"] + DIMENSIONS + ["unsupported_claims"])
            for r in rows:
                w.writerow([
                    r["query"], r["answer"],
                    *[r["scores"].get(d, "") for d in DIMENSIONS],
                    "; ".join(r["unsupported_claims"]),
                ])
        print(f"\nPer-query results written to {args.csv}")


if __name__ == "__main__":
    main()
