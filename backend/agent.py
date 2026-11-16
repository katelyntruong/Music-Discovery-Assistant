"""
Music agent powered by Groq (openai/gpt-oss-20b) with function calling.

Note: Groq's hosted model catalog changes over time -- Meta's Llama chat
models (previously llama-4-scout-17b-16e-instruct) were removed from Groq
entirely at some point, which is why this points at an OpenAI open-weight
model instead. If MODEL below starts 404ing again, check
https://console.groq.com/docs/models or `GET /openai/v1/models` for what's
currently hosted, and pick another tool-calling-capable chat model.

Loop:
  1. User message arrives
  2. LLM decides which tools to call
  3. We execute them and return results
  4. Repeat until the model produces a final text response
"""

import json
import os
from openai import BadRequestError, OpenAI
from dotenv import load_dotenv
from rag import rag_search
from spotify_tools import (
    search_tracks,
    get_artist_info,
    create_playlist,
    get_recently_played,
    get_top_tracks,
)

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


MODEL = "openai/gpt-oss-20b"

FINAL_ANSWER_NUDGE = (
    "Stop calling tools now. Write your final answer to the user using only the "
    "tool results already in this conversation. Do not call any tool."
)


def final_answer(messages: list) -> str:
    """
    Get a text-only answer with no tools available.

    gpt-oss sometimes emits a tool call even when tools are disabled, which
    Groq rejects with a 400 (code "tool_use_failed"). Nudge it explicitly,
    retry once, and fall back to a canned reply rather than crashing.
    """
    nudged = list(messages) + [{"role": "system", "content": FINAL_ANSWER_NUDGE}]
    for _ in range(2):
        try:
            response = _get_client().chat.completions.create(model=MODEL, messages=nudged)
            return response.choices[0].message.content or ""
        except BadRequestError as e:
            if getattr(e, "code", None) != "tool_use_failed":
                raise
    return ("Sorry — I gathered some results but couldn't put together an answer. "
            "Could you try rephrasing your request?")


# Guardrails against runaway tool loops (repeated rag_search, double playlists, …)
MAX_TOOL_ROUNDS = 6
TOOL_CALL_LIMITS = {
    "rag_search": 2,
    "spotify_search": 4,
    "create_playlist": 1,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM = """You are an expert music discovery assistant. You help users find music,
understand artists, and create personalised playlists through conversation.

## Your tools
- rag_search — search Rolling Stone reviews for critical/editorial context (what critics said, reception, style analysis). NOT for finding tracks.
- spotify_search — search Spotify for real tracks by artist, album, song title, mood, or genre text
- get_artist_info — get an artist's Spotify genres, popularity, and followers
- get_recently_played — user's recently played tracks (requires login)
- get_top_tracks — user's top tracks over a time period (requires login)
- create_playlist — build a real Spotify playlist and return the URL

## When to use rag_search
Only call rag_search when the user wants critical or editorial context: reviews,
reception, what critics/the press think, or an analysis of an artist's style.
Do NOT call rag_search for plain lookups like "find songs by X", "play something
by Y", "tracks from album Z", or "search for workout songs" — use spotify_search alone.

## How to handle requests

**Track lookups** ("find songs by Radiohead", "play something by Beyoncé", "tracks from Random Access Memories"):
1. Call spotify_search (once or twice is enough). No rag_search.

**Discovery requests** ("give me chill jazz", "something like Frank Ocean"):
1. Call get_artist_info if a specific artist is mentioned
2. Call spotify_search with a few different angle queries to gather tracks
3. Curate the best results and explain WHY each fits the request
(Only add rag_search if the user asks for critical context.)

**Playlist requests** — any time the user says "build", "make", or "create" a playlist
("build me a gym playlist", "make a playlist of jazz", "using my top tracks, build me a playlist"):
1. Gather tracks: call spotify_search (and get_top_tracks / get_recently_played if it's based on their listening)
2. You MUST then call create_playlist exactly once with the final URIs, and share the Spotify URL.
   Never finish a build/make playlist request without calling create_playlist.

**Critical / artist deep-dives** ("what did critics say about Blonde", "summarise Frank Ocean's style"):
1. Call rag_search for critical writing
2. Call get_artist_info for their Spotify genres (if an artist is involved)
3. Call spotify_search for key tracks if the user wants recommendations
4. Synthesise into a rich explanation

**Personal recommendations** ("based on my recent listening"):
1. Call get_recently_played (recent plays) or get_top_tracks ("on repeat", "my favourites", "this month")
2. Identify patterns in artists and genres
3. Call spotify_search to find similar music and explain the connection

## Tool-call budget
- Be efficient: most requests need 1–4 tool calls in total.
- Never repeat a tool call with the same or near-identical arguments — reuse the result you already have.
- Call rag_search at most once or twice per request; one good query is usually enough.
- Call create_playlist at most once per request.
- Once you have enough information, stop calling tools and write your answer.

## Grounding
- Only name tracks, albums, quotes, dates and numbers that appear in tool results.
- If the results don't cover something, say so rather than filling in from memory.

## Response style
- Be specific — name actual songs, albums, and production details
- Explain the WHY: mood, instrumentation, lyrical themes, cultural context
- Keep lists scannable: **bold the track name**, then one sentence of context
- When you create a playlist, always share the Spotify URL"""


# ---------------------------------------------------------------------------
# Tool definitions — OpenAI function calling format
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the music review knowledge base (Rolling Stone articles) for critical "
                "and editorial context: reviews, critical reception, and style analysis. "
                "Only use when the user asks what critics/the press think or wants an analysis "
                "of an artist's style. Do NOT use for finding or looking up tracks — use spotify_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query, e.g. 'Frank Ocean neo-soul emotional'",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-6)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_search",
            "description": (
                "Search Spotify for tracks by song title, album, artist name, mood, genre, or "
                "descriptive text. This is the tool for any track lookup. Use a few distinct "
                "queries for variety; never repeat the same query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'late night jazz instrumental'",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of tracks to return (1-50)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_artist_info",
            "description": (
                "Get an artist's Spotify profile: name, genres, popularity, followers. "
                "Use when the user mentions a specific artist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {
                        "type": "string",
                        "description": "Artist name, e.g. 'Frank Ocean'",
                    },
                },
                "required": ["artist_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_playlist",
            "description": (
                "Create a Spotify playlist for the logged-in user and populate it with tracks. "
                "Returns the playlist URL. Call this exactly once whenever the user asks to build, "
                "make, create, or save a playlist — after gathering tracks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Playlist name"},
                    "track_uris": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Spotify track URIs (format: spotify:track:<id>)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short playlist description shown in Spotify",
                    },
                },
                "required": ["name", "track_uris"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recently_played",
            "description": "Get the user's recently played Spotify tracks (requires user login).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent tracks (1-50)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_tracks",
            "description": (
                "Get the user's top Spotify tracks over a time range (requires user login). "
                "Use to personalise recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_range": {
                        "type": "string",
                        "description": "short_term=4 weeks, medium_term=6 months, long_term=all time",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of tracks (1-50)",
                    },
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _run_tool(name: str, args: dict) -> str:
    try:
        if name == "rag_search":
            result = rag_search(args["query"], n_results=args.get("n_results", 3))
        elif name == "spotify_search":
            result = search_tracks(args["query"], limit=args.get("limit", 10))
        elif name == "get_artist_info":
            result = get_artist_info(args["artist_name"])
        elif name == "create_playlist":
            result = create_playlist(
                name=args["name"],
                track_uris=args["track_uris"],
                description=args.get("description", ""),
            )
        elif name == "get_recently_played":
            result = get_recently_played(limit=args.get("limit", 20))
        elif name == "get_top_tracks":
            result = get_top_tracks(
                time_range=args.get("time_range", "medium_term"),
                limit=args.get("limit", 20),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def chat(history: list, user_message: str) -> dict:
    """
    Run the agentic loop for one user turn.

    history      – previous turns (list of role/content dicts, no system message)
    user_message – the new user message string

    Returns:
      {
        "reply":        str,
        "messages":     list,   # updated history (without system message) to store in session
        "tools_called": list[str],
      }
    """
    messages = (
        [{"role": "system", "content": SYSTEM}]
        + list(history)
        + [{"role": "user", "content": user_message}]
    )
    tools_called: list[str] = []
    # (name, canonical args) -> result, so repeated identical calls aren't re-executed
    call_cache: dict[tuple[str, str], str] = {}
    tool_counts: dict[str, int] = {}
    rounds = 0

    while True:
        rounds += 1
        if rounds > MAX_TOOL_ROUNDS:
            # Out of budget: force a final text answer from what we already have
            reply = final_answer(messages)
            messages.append({"role": "assistant", "content": reply})
            return {"reply": reply, "messages": messages[1:], "tools_called": tools_called}

        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            # Append assistant turn with tool call requests
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool and append results
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                key = (name, json.dumps(args, sort_keys=True))
                limit = TOOL_CALL_LIMITS.get(name)

                if key in call_cache:
                    result_str = json.dumps({
                        "note": "Duplicate call — you already have this result. Do not repeat it.",
                        "result": json.loads(call_cache[key]),
                    })
                elif limit is not None and tool_counts.get(name, 0) >= limit:
                    result_str = json.dumps({
                        "error": f"{name} call limit ({limit}) reached for this request. "
                                 "Use the results you already have."
                    })
                else:
                    tools_called.append(name)
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    result_str = _run_tool(name, args)
                    call_cache[key] = result_str
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            continue

        # No tool calls — extract final text
        reply = msg.content or ""
        messages.append({"role": "assistant", "content": reply})

        # Return history without the leading system message
        return {
            "reply": reply,
            "messages": messages[1:],
            "tools_called": tools_called,
        }
