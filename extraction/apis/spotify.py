
import logging
import time
import base64
from collections import OrderedDict

import requests

logger = logging.getLogger(__name__)


class SpotifyAPI:
    """
    Thin synchronous Spotify client (pure, no file I/O).

    Auth: Client Credentials (pass client_id/client_secret via constructor).
    Headers: 'User-Agent' passed by caller (project.yaml).
    Timeouts/sleep: taken from constructor (yaml via entrypoint).
    """

    token_url = "https://accounts.spotify.com/api/token"
    base_url = "https://api.spotify.com/v1"

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        user_agent: str,
        timeout_secs: int,
        sleep_secs: float = 0.0,
        max_retry_after_secs: int = 120,
        artist_cache_size: int = 5000,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout_secs = timeout_secs
        self.sleep_secs = sleep_secs
        self.max_retry_after_secs = max_retry_after_secs
        self.artist_cache_size = artist_cache_size
        self._artist_cache = OrderedDict()  # artist_id -> artist json

        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

        self.access_token = None
        self.token_expires_at = 0.0  # epoch seconds

    # -------- auth --------

    def _encode_basic(self) -> str:
        raw = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        return base64.b64encode(raw).decode("utf-8")

    def _ensure_token(self) -> None:
        now = time.time()
        if self.access_token and now < self.token_expires_at - 30:
            return

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                resp = self.session.post(
                    self.token_url,
                    headers={"Authorization": f"Basic {self._encode_basic()}"},
                    data={"grant_type": "client_credentials"},
                    timeout=self.timeout_secs,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self.access_token = data.get("access_token")
                    expires_in = int(data.get("expires_in", 3600))
                    self.token_expires_at = time.time() + expires_in
                    return
                elif resp.status_code == 502:
                    wait_time = 2 ** attempt
                    logger.info(f"Spotify auth 502. Retrying in {wait_time}s (attempt {attempt+1}).")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Spotify auth failed: {resp.status_code} {resp.text}")
                    break
            except requests.ConnectionError as e:
                wait_time = 2 ** attempt
                logger.info(f"Spotify auth connection error: {e}. Retrying in {wait_time}s (attempt {attempt+1}).")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Spotify auth unexpected error: {e}")
                break

        raise RuntimeError("Failed to authenticate with Spotify after retries.")

    # -------- request helper --------

    def _get(self, path: str, params: dict | None = None) -> dict:
        self._ensure_token()
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        # simple retry loop (handles 429 + transient 5xx)
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                resp = self.session.get(url, headers=headers, params=params or {}, timeout=self.timeout_secs)

                # polite sleep for burst control
                if self.sleep_secs:
                    time.sleep(self.sleep_secs)

                # rate limit
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "1"))
                    if retry_after > self.max_retry_after_secs:
                        logger.info(
                            f"Spotify 429 with Retry-After={retry_after}s exceeds cap "
                            f"({self.max_retry_after_secs}s). Stopping so you can resume later."
                        )
                        raise RuntimeError("Rate limit window too long; aborting current run.")
                    logger.info(f"Spotify 429. Sleeping {retry_after}s and retrying.")
                    time.sleep(retry_after)
                    continue

                # transient server errors
                if resp.status_code in (502, 503, 504):
                    wait_time = 2 ** attempt
                    logger.info(f"Spotify {resp.status_code}. Retrying in {wait_time}s (attempt {attempt+1}).")
                    time.sleep(wait_time)
                    continue

                if 200 <= resp.status_code < 300:
                    return resp.json()

                # audio-features can return 403; preserve old behavior with empty dict
                if resp.status_code == 403 and "audio-features" in path:
                    return {}

                # other errors -> raise
                raise RuntimeError(f"Spotify GET {path} failed: {resp.status_code} {resp.text}")

            except requests.ConnectionError as e:
                wait_time = 2 ** attempt
                logger.info(f"Spotify connection error: {e}. Retrying in {wait_time}s (attempt {attempt+1}).")
                time.sleep(wait_time)

        raise RuntimeError(f"Max retries exceeded for Spotify GET {path}.")

    # -------- small utilities (ported from async impl) --------

    def _clean_artist_name(self, artist_name: str) -> str:
        """
        Truncate at first unwanted pattern:
        ' feat.', ' Ft.', ' ft.', ' [', ' X ' (radio-style credits).
        """
        import re

        patterns = [
            r"\sfeat\.",
            r"\sFeat\.",
            r"\sft\.",
            r"\sFt\.",
            r"\s\[",
            r"\sX\s",
        ]
        for pat in patterns:
            m = re.search(pat, artist_name)
            if m:
                return artist_name[: m.start()].strip()
        return artist_name.strip()

    # -------- public API --------

    def search_track_first(self, *, track_name: str, artist_name: str) -> dict | None:
        """
        Return the first matching track item (or None).
        Tries with original artist name, then a cleaned variant.
        """
        q = f"track:{track_name} artist:{artist_name}"
        data = self._get("search", {"q": q, "type": "track", "limit": 1})
        items = (data.get("tracks") or {}).get("items") or []
        if items:
            return items[0]

        cleaned = self._clean_artist_name(artist_name)
        if cleaned != artist_name:
            q = f"track:{track_name} artist:{cleaned}"
            data = self._get("search", {"q": q, "type": "track", "limit": 1})
            items = (data.get("tracks") or {}).get("items") or []
            if items:
                return items[0]
        return None

    def get_artist(self, artist_id: str) -> dict:
        return self._get(f"artists/{artist_id}")

    def get_artist_cached(self, artist_id: str) -> dict:
        """
        LRU cache for artist lookups. Avoids repeated calls for the same artist_id.
        """
        cached = self._artist_cache.get(artist_id)
        if cached is not None:
            # refresh LRU order
            self._artist_cache.move_to_end(artist_id)
            return cached

        data = self._get(f"artists/{artist_id}")
        # insert with simple LRU eviction
        self._artist_cache[artist_id] = data
        if len(self._artist_cache) > self.artist_cache_size:
            self._artist_cache.popitem(last=False)
        return data

    def get_audio_features_or_empty(self, track_id: str) -> dict:
        """
        Audio features for a track.
        If unavailable (e.g., 403), return {} to keep pipeline schema handling simple.
        """
        return self._get(f"audio-features/{track_id}")

    def iter_playlist_items(self, playlist_id: str, page_limit: int | None = None):
        """
        Iterate playlist items (tracks). If page_limit is set, stop after that many pages.
        """
        limit = 100
        offset = 0
        page = 0
        while True:
            page += 1
            data = self._get(f"playlists/{playlist_id}/tracks", {"limit": limit, "offset": offset})
            items = data.get("items", []) or []
            for it in items:
                yield it
            if not data.get("next"):
                break
            offset += limit
            if page_limit and page >= page_limit:
                break

    # -------- convenience: assemble track features row --------

    def fetch_track_features(self, *, track_name: str, artist_name: str) -> dict | None:
        """
        Search the track, fetch artist genres + audio features, and return a normalized row.
        Returns None if no track match is found.
        """
        track = self.search_track_first(track_name=track_name, artist_name=artist_name)
        if not track:
            return None

        track_id = track.get("id")
        album = track.get("album", {}) or {}
        artists = track.get("artists") or []
        artist_id = artists[0].get("id") if artists else None

        genre_primary = None
        if artist_id:
            artist_info = self.get_artist_cached(artist_id)
            genres_list = artist_info.get("genres") or []
            genre_primary = genres_list[0] if genres_list else None

        # features = self.get_audio_features_or_empty(track_id) if track_id else {}
        features = {}

        # keep field names aligned with the schema
        row = {
            "artist_name": artist_name,
            "track_name": track_name,
            "spotify_track_id": track_id,
            "spotify_album": album.get("name"),
            "spotify_release_date": album.get("release_date"),
            "spotify_duration_ms": track.get("duration_ms"),
            "spotify_popularity": track.get("popularity"),
            "spotify_genres": genre_primary,
            "spotify_danceability": features.get("danceability"),
            "spotify_energy": features.get("energy"),
            "spotify_valence": features.get("valence"),
            "spotify_acousticness": features.get("acousticness"),
            "spotify_instrumentalness": features.get("instrumentalness"),
            "spotify_liveness": features.get("liveness"),
            "spotify_speechiness": features.get("speechiness"),
            "spotify_tempo": features.get("tempo"),
            "spotify_mode": features.get("mode"),
            "spotify_loudness": features.get("loudness"),
            "spotify_time_signature": features.get("time_signature"),
        }
        return row
