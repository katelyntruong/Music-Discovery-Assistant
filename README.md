# Partify — AI Music Discovery Assistant

An agentic AI music assistant driven by an LLM (Groq / Llama 4) that answers natural language queries to build Spotify playlists, deep-dive artists, and surface personalized listening trends.

<img width="1439" height="745" alt="Screen Shot 2026-05-10 at 10 15 24 PM" src="https://github.com/user-attachments/assets/e8914577-b995-42b7-92e7-51267e9255af" />

<img width="1414" height="805" alt="Screen Shot 2026-05-10 at 10 15 35 PM" src="https://github.com/user-attachments/assets/0b43ae02-dc79-45ed-8aee-c0cb097636cf" />

## Features

- **Natural language queries** — ask for music by mood, genre, artist, or vibe
- **Agentic tool-use loop** — LLM autonomously decides which tools to call across multiple steps
- **RAG pipeline** — Rolling Stone album reviews ingested into ChromaDB, retrieved at query time to ground recommendations in real editorial context
- **Spotify integration** — real-time track search, artist profiling, and playlist creation via the Spotify API
- **Chat UI** — full-stack web app with session memory, live tool-call visibility, and markdown rendering
- **Automated ingestion** — weekly cron job keeps the knowledge base current

## Tech Stack

| Layer | Tech |
|-------|------|
| LLM | Llama 4 Scout via Groq |
| LLM SDK | OpenAI Python SDK (pointed at Groq) |
| Backend | FastAPI (Python) |
| Vector store | ChromaDB + ONNX embeddings |
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
│   └── cron_scrape.sh   # Weekly automation script
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
