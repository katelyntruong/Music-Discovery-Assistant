"""
Tool-selection eval for the music agent.

Question this answers: given a user query, does the LLM (agent.chat) request
the tool(s) it should?

Tool execution is mocked (canned fixtures — no real Spotify/ChromaDB calls),
so only the model's SELECTION decision is under test. The agent's normal
turn loop still runs (mocked results feed back to the model) since a
multi-tool query needs that to produce its full call sequence — Groq is
called with parallel_tool_calls=False, so multi-tool decisions only show up
across turns, never in one completion.

Dataset format (JSON list, default: eval_data/tool_selection.json):
    [
      {"query": "What have I listened to recently?", "expected_tools": ["get_recently_played"]},
      {"query": "Find songs by Radiohead",            "expected_tools": ["spotify_search"]},
      {"query": "Based on my recent listening, recommend Radiohead songs",
       "expected_tools": ["get_recently_played", "spotify_search"]},
      {"query": "Thanks, that's helpful!",             "expected_tools": []}
    ]

Metrics:
  - exact-match accuracy: predicted tool set == expected tool set
  - precision: of the tools it called, how many were actually needed
  - recall:    of the tools it needed, how many did it call
  (precision/recall reduce to plain right/wrong for single-tool rows, and
  become genuinely informative once a query needs more than one tool)

Usage:
    cd backend
    python eval_tools.py                       # run the default dataset
    python eval_tools.py --data path/to.json
    python eval_tools.py --csv results.csv     # also write the per-query table

Requires GROQ_API_KEY (via .env or env var) — this makes real calls to the
LLM; only tool EXECUTION is mocked. A rate-limited (429) query is re-run from
scratch after the wait Groq asks for, instead of being scored as a failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import openai

from dotenv import load_dotenv

load_dotenv()

# imported lazily in main(), after argparse has handled --help and after we
# know GROQ_API_KEY is set (agent.py talks to Groq as soon as chat() runs)
agent = None

DEFAULT_DATA = Path(__file__).parent / "eval_data" / "tool_selection.json"

MAX_RATE_LIMIT_RETRIES = 6
MAX_RATE_LIMIT_WAIT = 15 * 60  # seconds


# ---------------------------------------------------------------------------
# Fixtures — stub every tool the agent can call so no real Spotify/ChromaDB
# request ever happens; only the model's tool-selection decision is real.
# ---------------------------------------------------------------------------

def fake_rag_search(query, n_results=3):
    return [{"text": "Fixture review snippet.", "title": "Fixture Review",
             "score": 0.9, "source": "rolling_stone"}]


def fake_search_tracks(query, limit=10):
    return [{"id": "t1", "uri": "spotify:track:t1", "name": "Fixture Track",
             "artist": "Fixture Artist", "album": "Fixture Album", "release_year": "2022",
             "popularity": 70, "preview_url": None,
             "spotify_url": "https://open.spotify.com/track/t1"}]


def fake_get_artist_info(artist_name):
    return {"id": "a1", "name": artist_name, "genres": ["fixture-genre"], "popularity": 80,
            "followers": 1_000_000, "spotify_url": "https://open.spotify.com/artist/a1"}


def fake_create_playlist(name, track_uris, description=""):
    return {"playlist_id": "p1", "playlist_url": "https://open.spotify.com/playlist/p1",
            "name": name, "track_count": len(track_uris)}


def fake_get_recently_played(limit=20):
    return [{"id": "t2", "uri": "spotify:track:t2", "name": "Fixture Recent Track",
             "artist": "Fixture Artist", "played_at": "2026-09-14T20:00:00Z"}]


def fake_get_top_tracks(time_range="medium_term", limit=20):
    return [{"id": "t3", "uri": "spotify:track:t3", "name": "Fixture Top Track",
             "artist": "Fixture Artist", "popularity": 75}]


def _install_fixtures() -> None:
    agent.rag_search = fake_rag_search
    agent.search_tracks = fake_search_tracks
    agent.get_artist_info = fake_get_artist_info
    agent.create_playlist = fake_create_playlist
    agent.get_recently_played = fake_get_recently_played
    agent.get_top_tracks = fake_get_top_tracks


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    for r in rows:
        r["expected_tools"] = list(r.get("expected_tools", []))
    return rows


def run_query(query: str) -> list[str]:
    """Run the agent loop (tools mocked) and return the tools it called, in order.
    On a 429 the whole query is re-run, since a partial tool sequence can't be resumed."""
    from eval_generation import _retry_after_seconds

    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return agent.chat([], query)["tools_called"]
        except openai.RateLimitError as e:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            wait = min(_retry_after_seconds(e) + 1, MAX_RATE_LIMIT_WAIT)
            print(f"       rate limited -- waiting {wait:.0f}s and re-running "
                  f"(attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})", flush=True)
            time.sleep(wait)


def score(expected: list[str], predicted: list[str]) -> dict:
    exp, pred = set(expected), set(predicted)
    correct = exp & pred
    precision = (len(correct) / len(pred)) if pred else 1.0   # nothing called -> no wrong calls
    recall = (len(correct) / len(exp)) if exp else 1.0        # nothing needed -> trivially satisfied
    return {"precision": precision, "recall": recall, "exact_match": exp == pred}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-selection eval for the music agent")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                         help="Path to the eval dataset (JSON list of {query, expected_tools})")
    parser.add_argument("--csv", type=str, default=None,
                         help="Also write the per-query results table to this CSV file")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print(
            "GROQ_API_KEY not set (checked environment and .env).\n"
            "This eval makes real calls to the LLM to test its tool-selection\n"
            "decisions -- only tool EXECUTION is mocked. Set the key and retry."
        )
        sys.exit(1)

    global agent
    import agent as _agent_module
    agent = _agent_module
    _install_fixtures()

    dataset = load_dataset(Path(args.data))
    print(f"Loaded {len(dataset)} queries from {args.data}\n")

    rows = []
    for item in dataset:
        query = item["query"]
        expected = item["expected_tools"]
        try:
            predicted = run_query(query)
            error = None
        except Exception as e:
            predicted, error = [], f"{type(e).__name__}: {e}"

        s = score(expected, predicted)
        rows.append({"query": query, "expected": expected, "predicted": predicted, **s, "error": error})

        mark = "PASS" if s["exact_match"] else "FAIL"
        print(f"[{mark}] {query!r}")
        print(f"       expected={expected}  predicted={predicted}")
        if error:
            print(f"       ERROR: {error}")

    n = len(rows)
    n_correct = sum(r["exact_match"] for r in rows)
    accuracy = n_correct / n
    avg_precision = sum(r["precision"] for r in rows) / n
    avg_recall = sum(r["recall"] for r in rows) / n
    avg_f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) else 0.0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Queries evaluated:     {n}")
    print(f"Exact-match accuracy:  {n_correct}/{n} = {accuracy:.1%}")
    print(f"Avg tool precision:    {avg_precision:.1%}  (avoided unnecessary tool calls?)")
    print(f"Avg tool recall:       {avg_recall:.1%}  (called every tool it needed?)")
    print(f"Avg tool F1:           {avg_f1:.1%}")

    failures = [r for r in rows if not r["exact_match"]]
    if failures:
        print(f"\n{len(failures)} failing quer{'y' if len(failures) == 1 else 'ies'}:")
        for r in failures:
            print(f"  {r['query']!r}: expected={r['expected']} got={r['predicted']}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query", "expected_tools", "predicted_tools", "correct", "precision", "recall"])
            for r in rows:
                w.writerow([r["query"], ";".join(r["expected"]), ";".join(r["predicted"]),
                            int(r["exact_match"]), f"{r['precision']:.3f}", f"{r['recall']:.3f}"])
        print(f"\nPer-query results written to {args.csv}")


if __name__ == "__main__":
    main()
