import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


DEFAULT_TIMEOUT = 10.0
DEFAULT_BASE_URL = "https://song-of-the-week.onrender.com"


def _get_base_url() -> str:
    env_url = os.getenv("SOTW_API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")

    return DEFAULT_BASE_URL


BASE_URL = _get_base_url()


class APIClientError(RuntimeError):
    """Simple wrapper for API-related errors."""


def _build_url(path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{BASE_URL.rstrip('/')}{path}"


def _handle_response(response: requests.Response) -> Dict[str, Any]:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        # Try to surface backend error message, if any
        try:
            payload = response.json()
            detail = payload.get("detail") or payload
        except Exception:
            detail = response.text
        raise APIClientError(f"API error {response.status_code}: {detail}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("API returned non-JSON response") from exc


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = _build_url(path)
    resp = requests.get(url, params=params or {}, timeout=DEFAULT_TIMEOUT)
    return _handle_response(resp)


def _post(path: str, json: Dict[str, Any]) -> Dict[str, Any]:
    url = _build_url(path)
    resp = requests.post(url, json=json, timeout=DEFAULT_TIMEOUT)
    return _handle_response(resp)


# -------- Public helpers -------- #


def healthcheck() -> Dict[str, Any]:
    return _get("/health")


def predict_candidates(
    candidates: List[Dict[str, Any]],
    mode: str = "auto",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "mode": mode,
        "candidates": candidates,
    }
    return _post("/predict", json=payload)


def get_random_examples(count: int = 1) -> Dict[str, Any]:
    return _get("/examples/random", params={"count": count})


def get_favorite_examples(count: int = 1) -> Dict[str, Any]:
    return _get("/examples/favorites", params={"count": count})


def get_spotify_candidate_from_url(url: str) -> Dict[str, Any]:
    return _get("/spotify/candidate_from_url", params={"url": url})


def get_spotify_candidate_from_id(track_id: str) -> Dict[str, Any]:
    return _get("/spotify/candidate_from_id", params={"track_id": track_id})


def spotify_search_track(
    artist: str,
    track: str,
    limit: int = 5,
) -> Dict[str, Any]:
    params = {
        "artist": artist,
        "track": track,
        "limit": limit,
    }
    return _get("/spotify/search_track", params=params)
