"""
Music agent powered by Groq (llama-3.3-70b) with function calling.

Loop:
  1. User message arrives
  2. LLM decides which tools to call
  3. We execute them and return results
  4. Repeat until the model produces a final text response
"""

import json
import os
from openai import OpenAI
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


MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM = """You are an expert music discovery assistant. You help users find music,
understand artists, and create personalised playlists through conversation.

## Your tools
- rag_search — search Rolling Stone review database for critical context and style descriptions
- spotify_search — search Spotify for real tracks by artist, mood, or genre text
- get_artist_info — get an artist's Spotify genres, popularity, and followers
- get_recently_played — user's recently played tracks (requires login)
- get_top_tracks — user's top tracks over a time period (requires login)
- create_playlist — build a real Spotify playlist and return the URL

## How to handle requests

**Discovery requests** ("give me chill jazz", "something like Frank Ocean"):
1. Call rag_search to get critical context about the style or artist
2. Call get_artist_info if a specific artist is mentioned
3. Call spotify_search multiple times with different angle queries to gather tracks
4. Curate the best results and explain WHY each fits the request

**Playlist requests** ("build me a gym playlist", "studying playlist"):
1. Call spotify_search with several genre/mood queries to gather enough tracks
2. Call create_playlist with the final URIs and return the Spotify URL

**Artist deep-dives** ("summarise Frank Ocean's style"):
1. Call rag_search for critical writing about the artist
2. Call get_artist_info for their Spotify genres
3. Call spotify_search for their key tracks
4. Synthesise into a rich explanation

**Personal recommendations** ("based on my recent listening"):
1. Call get_recently_played or get_top_tracks to see what they like
2. Identify patterns in artists and genres
3. Call spotify_search to find similar music and explain the connection

## Response style
- Be specific — name actual songs, albums, and production details
- Explain the WHY: mood, instrumentation, lyrical themes, cultural context
- Keep lists scannable: **bold the track name**, then one sentence of context
- When you create a playlist, always share the Spotify URL
- Run multiple spotify_search calls with different queries to get variety"""


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
                "context, style descriptions, genre analysis, and artist write-ups. "
                "Use this first when the user mentions a specific artist, genre, or vibe."
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
                "Search Spotify for tracks by mood, genre, artist name, or descriptive text. "
                "Call multiple times with different queries to get variety."
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
                "Returns the playlist URL. Only call when the user explicitly asks to build or save a playlist."
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

    while True:
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
                tools_called.append(tc.function.name)
                args = json.loads(tc.function.arguments)
                result_str = _run_tool(tc.function.name, args)
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
