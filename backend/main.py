"""
FastAPI backend for the AI Music Discovery Assistant.

Endpoints:
  POST /chat           – send a message, get a reply
  DELETE /session/{id} – clear conversation history
  GET  /session/{id}   – inspect current history (debug)
  GET  /health         – liveness check
"""

import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from agent import chat

load_dotenv()

FRONTEND = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="AI Music Discovery Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id → message history
# Swap for Redis when you want persistence across restarts
_sessions: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tools_called: list[str]


class NewSessionResponse(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_ui():
    return FileResponse(FRONTEND / "index.html")


@app.post("/session", response_model=NewSessionResponse)
def new_session():
    """Create a fresh conversation session and return its ID."""
    sid = str(uuid.uuid4())
    _sessions[sid] = []
    return {"session_id": sid}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    history = _sessions.get(req.session_id, [])

    # pass user_message separately — agent manages Gemini-format history internally
    result = chat(history, req.message)

    _sessions[req.session_id] = result["messages"]
    return {
        "session_id": req.session_id,
        "reply": result["reply"],
        "tools_called": result["tools_called"],
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "message_count": len(_sessions[session_id])}


# Serve frontend static files — must be mounted after all API routes
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
