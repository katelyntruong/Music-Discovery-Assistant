# Partify — AI Music Discovery Assistant

An agentic AI music assistant driven by an LLM (`openai/gpt-oss-20b` served on Groq) that answers natural language queries to build Spotify playlists, deep-dive artists, and surface personalized listening trends.

<img width="1439" height="745" alt="Screen Shot 2026-05-10 at 10 15 24 PM" src="https://github.com/user-attachments/assets/e8914577-b995-42b7-92e7-51267e9255af" />

<img width="1414" height="805" alt="Screen Shot 2026-05-10 at 10 15 35 PM" src="https://github.com/user-attachments/assets/0b43ae02-dc79-45ed-8aee-c0cb097636cf" />

## Features

- **Natural language queries** — ask for music by mood, genre, artist, or vibe
- **Agentic tool-use loop** — LLM autonomously decides which tools to call across multiple steps
- **RAG pipeline** — Rolling Stone album reviews split into overlapping chunks (150 words, 30-word overlap), embedded into ChromaDB, and retrieved at query time (best chunk per review) to ground recommendations in real editorial context
- **Spotify integration** — real-time track search, artist profiling, and playlist creation via the Spotify API
- **Chat UI** — full-stack web app with session memory, live tool-call visibility, and markdown rendering
- **Automated ingestion** — weekly cron job keeps the knowledge base current
- **Evaluation suite** — separate scripts measure tool selection, retrieval, generation quality, and an embedding-model A/B test (see [Evaluation](#evaluation))

## Tech Stack

| Layer | Tech |
|-------|------|
| LLM | `openai/gpt-oss-20b` via Groq |
| LLM SDK | OpenAI Python SDK (pointed at Groq) |
| Backend | FastAPI (Python) |
| Vector store | ChromaDB + ONNX embeddings (all-MiniLM-L6-v2) |
| Evaluation | LLM-as-judge, Recall/Precision@K, MRR, McNemar test, bootstrap CIs |
| Music data | Spotify Web API |
| Knowledge base | Rolling Stone album reviews (scraped) |
| Frontend | Vanilla JS + CSS served by FastAPI |
| Automation | System cron job |

## Project Structure

```
App Partify/
├── backend/
│   ├── main.py          # FastAPI server + session management
│   ├── agent.py         # Agentic LLM loop + tool dispatcher
│   ├── rag.py           # ChromaDB RAG pipeline
│   ├── scraper.py       # Rolling Stone review scraper
│   ├── spotify_tools.py # Spotify API wrappers
│   ├── cron_scrape.sh   # Weekly automation script
│   ├── eval_tools.py          # Eval: tool selection
│   ├── eval_retrieval.py      # Eval: RAG retrieval (Recall/Precision@K)
│   ├── eval_generation.py     # Eval: answer quality (LLM judge)
│   ├── eval_ab_embeddings.py  # Eval: A/B test of embedding models
│   └── eval_data/             # Hand-labeled eval datasets
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

## Setup

### 1. Prerequisites

- Python 3.12
- `uv` package manager (`pip install uv`)
- Groq API key — [console.groq.com](https://console.groq.com)
- Spotify app credentials — [developer.spotify.com](https://developer.spotify.com/dashboard)

### 2. Install dependencies

```bash
cd backend
uv venv --python 3.12
uv pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

### 4. Seed the knowledge base

```bash
cd backend
python scraper.py --pages 3
```

### 5. Run

```bash
cd backend
.venv/bin/uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Automation

The scraper runs automatically via cron:

```bash
# Install
(crontab -l 2>/dev/null; echo "0 2 * * 0 \"/path/to/backend/cron_scrape.sh\"") | crontab -

# View logs
cat backend/logs/scraper.log
```

## How It Works

1. User sends a message → FastAPI stores it in session history
2. Agent sends history + 6 tool definitions to the LLM
3. LLM decides which tools to call (Spotify search, RAG lookup, playlist creation, etc.)
4. Agent executes the tools and feeds results back to the LLM
5. LLM produces a final response — loop ends, reply is returned

## Evaluation

The agent is evaluated one stage at a time, so a bad answer can be traced to the step that caused it:

```
user query
   │
   ▼
tool selection ──────── eval_tools.py          did the LLM pick the right tool(s)?
   │
   ▼
RAG retrieval ───────── eval_retrieval.py      did ChromaDB return the right review?
   │                    eval_ab_embeddings.py  which embedding model retrieves better?
   ▼
answer generation ───── eval_generation.py     did the LLM use the context correctly?
```

Each script holds the other stages fixed (mocked tools, given context, the same chunks), so its score reflects only the stage it tests.

All scripts run from `backend/`, print one line per query and a summary, and accept `--data <path>` for another dataset and `--csv <path>` to save per-query results.

| Script | What it tests | Metrics | Needs |
|--------|---------------|---------|-------|
| `eval_tools.py` | Tool choice | Exact-match accuracy, tool precision / recall / F1 | `GROQ_API_KEY` |
| `eval_retrieval.py` | Retrieval quality | Recall@K, Precision@K (K = 1, 3, 5, 10) | Populated `chroma_db` |
| `eval_generation.py` | Answer quality | Correctness, faithfulness, relevance, completeness (1–5) | `GROQ_API_KEY` |
| `eval_ab_embeddings.py` | Embedding model A vs B | Recall@K, MRR@10, latency, McNemar p, bootstrap CI | Populated `chroma_db`, `fastembed` |

### Datasets (`backend/eval_data/`)

All labels were written by hand: relevant review URLs come from reading the scraped articles.

| File | Size | Used by | Format |
|------|------|---------|--------|
| `tool_selection.json` | 44 queries (30 single-tool, 9 multi-tool, 5 no-tool) | `eval_tools.py` | `{query, expected_tools}` |
| `rag_retrieval.json` | 111 queries | `eval_retrieval.py` | `{query, type, relevant_docs}` |
| `rag_dev.json` | 78 queries | `eval_ab_embeddings.py` (default) | same as above |
| `rag_test.json` | 33 queries | `eval_ab_embeddings.py --data` | same as above |
| `rag_test_held_out_reviews.json` | 15 review URLs | — | list of the review URLs the test split is built around |
| `generation.json` | 12 queries | `eval_generation.py` | `{query, context, reference_answer?}` |

Retrieval queries are tagged by `type` so results can be broken down by difficulty: `direct` (names the artist/album), `paraphrase` (describes it in other words), `deep` (asks about a detail inside the review), `multi` (more than one relevant review), and `genre_mood` (asks by vibe rather than by name).

`rag_dev.json` and `rag_test.json` split `rag_retrieval.json` into a dev set for choosing models and a test set for the final check. The test set's queries are built around the 15 reviews listed in `rag_test_held_out_reviews.json`, so the test score also covers reviews that weren't the focus while tuning. (3 of those reviews are also relevant to some dev queries.)

### 1. Tool selection — `eval_tools.py`

```bash
python eval_tools.py
```

Runs the real agent loop (`agent.chat`) against the real LLM, but replaces every tool with a canned fixture: no Spotify or ChromaDB calls are made. The full loop still runs because Groq is called with `parallel_tool_calls=False`, so a query that needs several tools only shows them across turns.

- **Exact match** — the set of tools called equals the expected set
- **Precision** — of the tools called, the share that were needed (penalizes extra calls)
- **Recall** — of the tools needed, the share that were called (penalizes missed calls)
- **F1** — combines precision and recall

The no-tool queries (e.g. "Thanks, that's helpful!") check that the agent doesn't call tools when it shouldn't.

### 2. Retrieval — `eval_retrieval.py`

```bash
python eval_retrieval.py              # K = 1 3 5 10
python eval_retrieval.py --k 1 5
```

Sends each query through `rag.rag_search` and compares the returned review URLs with the labeled `relevant_docs`. The LLM is not involved.

- **Recall@K** — at least one relevant review is in the top K
- **Precision@K** — the share of the top K that is relevant

The script lists the queries that never retrieved a relevant review, and warns if the collection holds fewer documents than the largest K (which would pull down precision for reasons unrelated to retrieval quality).

### 3. Generation — `eval_generation.py`

```bash
python eval_generation.py
python eval_generation.py --judge-model openai/gpt-oss-120b
```

Each query comes with a fixed context. The context is inserted as fake tool-call/tool-result messages, matching what production sends, and the model then answers with no tools available. That isolates generation from tool choice and retrieval. A second LLM call acts as judge and scores the answer from 1 to 5 on:

- **Correctness** — claims are true / match the reference answer
- **Faithfulness** — every claim is backed by the context (hallucination check); the judge also lists unsupported claims
- **Relevance** — the answer addresses the question
- **Completeness** — it uses the useful information the context provides

An answer is marked `PASS` when its average is ≥ 4 and faithfulness is ≥ 4, and `HALLUCINATION` when faithfulness is ≤ 2.

By default the judge is the same model being evaluated, which can bias it toward its own answers. Pass `--judge-model` to use a different, stronger model.

### 4. Embedding A/B test — `eval_ab_embeddings.py`

```bash
uv pip install fastembed                                 # one-time
python eval_ab_embeddings.py                             # dev split
python eval_ab_embeddings.py --data eval_data/rag_test.json   # final test, run once
python eval_ab_embeddings.py --rebuild                   # re-embed model B's index
```

| | Model | Notes |
|--|-------|-------|
| A (control) | `all-MiniLM-L6-v2` | Chroma's default; what production uses |
| B (challenger) | `BAAI/bge-small-en-v1.5` | Via fastembed (ONNX, no PyTorch); queries get BGE's retrieval prefix |

Only the embedding model differs. B's index is built from the exact chunks already in `chroma_db`, both models use the same search logic as `rag_search` (over-fetch, keep the best chunk per review), and B is stored separately in `backend/chroma_ab/` so the production index is never touched. B's index is rebuilt automatically if `chroma_db` changes.

Reported for each model:

- Recall@1/3/5/10 and MRR@10
- Recall@3 for each question `type`
- Median query latency and index build time

Significance is tested on Recall@3 (the agent's default `n_results`), because both models answer the same queries:

- **McNemar's exact test** on the queries where exactly one model succeeds (p < 0.05 suggests a real difference)
- **Bootstrap 95% CI** for the B − A difference
- A list of the queries where the models disagree

Choose the model using the dev split, then run the test split once at the end so the final number isn't tuned to it.

### Results

Run on 2026-11-15 against the current `chroma_db`, with `openai/gpt-oss-20b` as the agent model. The LLM-based evals are not deterministic, so expect a few points of variation between runs.

#### At a glance

| Eval | Headline result |
|------|-----------------|
| Tool selection (44 queries) | **84.1%** exact match · 96.6% precision · 89.4% recall · 92.9% F1 |
| Retrieval (111 queries) | **78.4%** Recall@3 · 64.0% Recall@1 · 91.0% Recall@10 |
| Generation (12 queries) | Correctness 4.67 · **Faithfulness 3.00** · Relevance 5.00 · Completeness 4.92 (out of 5) |
| Embedding A/B (test split) | bge-small +6.1 pts Recall@3, not significant (p = 0.754) |

#### Tool selection

| Metric | Score |
|--------|-------|
| Exact-match accuracy | 37 / 44 = 84.1% |
| Avg tool precision | 96.6% |
| Avg tool recall | 89.4% |
| Avg tool F1 | 92.9% |

The agent rarely calls a tool it doesn't need (high precision). Most errors are missed tools (lower recall), in three patterns:

- **`rag_search` skipped on multi-step requests.** For "something like Frank Ocean" and "best indie albums … find me tracks", the agent went straight to `spotify_search` without consulting the reviews.
- **Top vs. recent listening confused.** "What's been on repeat for me lately?" called `get_recently_played` instead of `get_top_tracks`. Two personal-history queries called no tool at all.
- **Playlist flows left incomplete.** "Build me an energetic gym playlist" searched four times but never called `create_playlist`. "Make me a playlist based on what I've been playing" skipped `spotify_search`.

#### Retrieval

| K | Recall@K | Precision@K |
|---|----------|-------------|
| 1 | 64.0% | 64.0% |
| 3 | 78.4% | 27.9% |
| 5 | 82.0% | 17.7% |
| 10 | 91.0% | 10.3% |

Precision@K is capped by design: most queries have a single relevant review, so Precision@3 can't exceed ~33%. Recall is the main metric.

Recall by question type:

| Type | n | R@1 | R@3 | R@5 | R@10 |
|------|---|-----|-----|-----|------|
| direct | 20 | 100.0% | 100.0% | 100.0% | 100.0% |
| paraphrase | 36 | 75.0% | 83.3% | 86.1% | 100.0% |
| genre_mood | 9 | 44.4% | 77.8% | 77.8% | 77.8% |
| deep | 32 | 43.8% | 65.6% | 71.9% | 78.1% |
| multi | 14 | 42.9% | 64.3% | 71.4% | 92.9% |

Queries that name the artist or album always succeed. The weak spot is **`deep` questions about a detail inside a review** ("track that samples Glen Campbell", "recorded without click tracks at Bear Creek Studios"). 10 queries never surfaced the right review in the top 10, and most are of this kind: the detail usually sits in a chunk that doesn't mention the artist or album.

#### Generation

Judge: `openai/gpt-oss-120b` (a different, larger model than the one being evaluated).

| Dimension | Avg (1–5) |
|-----------|-----------|
| Correctness | 4.67 |
| Faithfulness | **3.00** |
| Relevance | 5.00 |
| Completeness | 4.92 |

3 PASS · 3 LOW · 6 flagged as possible hallucination (faithfulness ≤ 2).

The answers are on-topic and broadly accurate, but the model **adds descriptive detail that isn't in its context**. Examples: describing tracks as "glitchy synths with eerie vocal samples" when only the track names were retrieved, or adding critics' praise that the review doesn't contain. The judge rated most of these claims plausible (correctness stays high); they just aren't supported by the context. The failures happen most on queries where the context is thin Spotify data (track lists, playlists) rather than review text. Queries backed by a full review ("What did Rolling Stone say about Chromakopia?") and the out-of-scope weather question scored 5/5.

#### Embedding A/B: MiniLM (A) vs bge-small (B)

| Metric | Dev A | Dev B | Test A | Test B |
|--------|-------|-------|--------|--------|
| Recall@1 | 67.9% | 66.7% | 54.5% | 63.6% |
| Recall@3 | 83.3% | 83.3% | 66.7% | 72.7% |
| Recall@5 | 87.2% | 91.0% | 69.7% | 78.8% |
| Recall@10 | 93.6% | 96.2% | 84.8% | 87.9% |
| MRR@10 | 0.764 | 0.769 | 0.624 | 0.704 |
| Median query latency | 517 ms | 71 ms | 494 ms | 96 ms |

Recall@3 significance:

| Split | Only A hit | Only B hit | B − A | 95% CI | McNemar p |
|-------|-----------|-----------|-------|--------|-----------|
| Dev (78) | 7 | 7 | +0.0% | [−9.0%, +9.0%] | 1.000 |
| Test (33) | 4 | 6 | +6.1% | [−12.1%, +24.2%] | 0.754 |

**Conclusion:** bge-small is ahead or tied on almost every metric and was about 5–7× faster per query in this setup. However, the Recall@3 difference is **not statistically significant** on either split; both confidence intervals include zero. By type on dev, B did better on `multi` and `genre_mood` but worse on `deep`; on test, B did better on `deep`. With these sample sizes, the per-type differences are noise. Switching models is defensible on latency, but more labeled queries are needed to show a quality gain.

### Rate limits

The LLM-based evals (`eval_tools.py`, `eval_generation.py`) handle Groq `429` errors by waiting as long as Groq asks (up to 15 minutes, 6 attempts), so hitting a per-minute or daily token cap pauses the run instead of dropping queries. In `eval_tools.py` a rate-limited query is re-run from the start, because a partial tool sequence can't be resumed.
