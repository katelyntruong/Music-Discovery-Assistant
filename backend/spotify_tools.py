"""
Spotify API wrappers for the music agent.

Auth modes:
  - SpotifyClientCredentials → search, artist lookup (no user login needed)
  - SpotifyOAuth             → create playlists, listening history (one-time browser login,
                               token cached in .spotify_cache)

Endpoints removed (restricted to Spotify Extended Quota apps since late 2024):
  audio-features, recommendations, related-artists, artist/top-tracks, genre-seeds
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

_OAUTH_SCOPE = (
    "playlist-modify-public "
    "playlist-modify-private "
    "user-read-recently-played "
    "user-top-read"
)

_client: spotipy.Spotify | None = None
_oauth_client: spotipy.Spotify | None = None


def _get_client() -> spotipy.Spotify:
    global _client
    if _client is None:
        _client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            )
        )
    return _client


def _get_oauth_client() -> spotipy.Spotify:
    global _oauth_client
    if _oauth_client is None:
        _oauth_client = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
                redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback"),
                scope=_OAUTH_SCOPE,
                cache_path=os.path.join(os.path.dirname(__file__), ".spotify_cache"),
                open_browser=True,
            )
        )
    return _oauth_client


# ---------------------------------------------------------------------------
# Search — client credentials
# ---------------------------------------------------------------------------

def search_tracks(query: str, limit: int = 10) -> list[dict]:
    """Search Spotify for tracks by mood, genre, artist, or descriptive text."""
    sp = _get_client()
    results = sp.search(q=query, type="track", limit=min(limit, 50))
    return [
        {
            "id": t["id"],
            "uri": t["uri"],
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "artist_id": t["artists"][0]["id"],
            "album": t["album"]["name"],
            "release_year": t["album"].get("release_date", "")[:4],
            "popularity": t.get("popularity", 0),
            "preview_url": t.get("preview_url"),
            "spotify_url": t["external_urls"]["spotify"],
        }
        for t in results["tracks"]["items"]
    ]


# ---------------------------------------------------------------------------
# Artist — client credentials
# ---------------------------------------------------------------------------

def get_artist_info(artist_name: str) -> dict:
    """
    Look up an artist's Spotify profile: name, genres, popularity, followers.
    top_tracks and related_artists are not available (Extended Quota restricted).
    """
    sp = _get_client()
    results = sp.search(q=artist_name, type="artist", limit=1)
    items = results["artists"]["items"]
    if not items:
        return {"error": f"Artist '{artist_name}' not found"}

    a = items[0]
    return {
        "id": a["id"],
        "name": a["name"],
        "genres": a.get("genres", []),
        "popularity": a.get("popularity", 0),
        "followers": a.get("followers", {}).get("total", 0),
        "spotify_url": a["external_urls"]["spotify"],
    }


# ---------------------------------------------------------------------------
# Playlist creation — OAuth
# ---------------------------------------------------------------------------

def get_current_user_id() -> str:
    """Return the Spotify user ID of the logged-in user."""
    return _get_oauth_client().current_user()["id"]


def create_playlist(name: str, track_uris: list[str], description: str = "") -> dict:
    """Create a public playlist for the logged-in user and return its URL."""
    sp = _get_oauth_client()
    # Use POST /me/playlists — user_playlist_create() hits the deprecated /users/{id}/playlists
    pl = sp._post(
        "me/playlists",
        payload={"name": name, "public": True, "description": description},
    )
    for i in range(0, len(track_uris), 100):
        sp.playlist_add_items(pl["id"], track_uris[i : i + 100])
    return {
        "playlist_id": pl["id"],
        "playlist_url": pl["external_urls"]["spotify"],
        "name": pl["name"],
        "track_count": len(track_uris),
    }


# ---------------------------------------------------------------------------
# Listening history — OAuth
# ---------------------------------------------------------------------------

def get_recently_played(limit: int = 20) -> list[dict]:
    """Return the user's recently played tracks."""
    sp = _get_oauth_client()
    raw = sp.current_user_recently_played(limit=min(limit, 50))
    return [
        {
            "id": item["track"]["id"],
            "uri": item["track"]["uri"],
            "name": item["track"]["name"],
            "artist": item["track"]["artists"][0]["name"],
            "played_at": item["played_at"],
        }
        for item in raw["items"]
    ]


def get_top_tracks(time_range: str = "medium_term", limit: int = 20) -> list[dict]:
    """
    Return the user's top tracks.
    time_range: 'short_term' (4 weeks), 'medium_term' (6 months), 'long_term' (all time)
    """
    sp = _get_oauth_client()
    raw = sp.current_user_top_tracks(limit=min(limit, 50), time_range=time_range)
    return [
        {
            "id": t["id"],
            "uri": t["uri"],
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "popularity": t.get("popularity", 0),
        }
        for t in raw["items"]
    ]
