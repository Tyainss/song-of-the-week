
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

from common.utils.helper import (
    get_image_text,
    unix_from_lastfm_datetime,
)
from common.utils.io import read_csv

logger = logging.getLogger(__name__)


class LastFMAPI:
    """
    Last.fm API client with helpers for recent tracks,
    and basic metadata for user, track, album, and artist.
    """

    def __init__(
        self,
        api_key: str,
        username: str,
        *,
        user_agent: str,
        base_url: str = "https://ws.audioscrobbler.com/2.0/",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.username = username
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def _get(self, params: dict[str, Any]) -> dict:
        """
        Execute a GET request to Last.fm and return the JSON payload.
        """
        # response = self.session.get(self.base_url, params=params, timeout=30)
        # response.raise_for_status()
        # data = response.json()
        response = requests.get(self.base_url, params=params)
        try:
            data = response.json()
        except Exception as e:
            logger.error(f'Error {e} fetching data')
            data = {}

        return data

    def fetch_user_profile(self) -> list[dict[str, Any]]:
        """
        Get basic profile information for the configured user.
        """
        params = {
            "method": "user.getinfo",
            "user": self.username,
            "api_key": self.api_key,
            "format": "json",
        }
        data = self._get(params)
        user_info = data["user"]
        user_data = [
            {
                "username": user_info["name"],
                "user_playcount": user_info["playcount"],
                "user_artist_count": user_info["artist_count"],
                "user_album_count": user_info["album_count"],
                "user_track_count": user_info["track_count"],
                "user_image": get_image_text(user_info.get("image", []), "extralarge"),
                "user_country": user_info.get("country"),
            }
        ]
        return user_data
    
    def fetch_track_info(self, *, artist: str, track: str) -> dict[str, Any]:
        """
        Track metadata (e.g., duration, listeners, playcount).
        """
        params = {
            "method": "track.getInfo",
            "artist": artist,
            "track": track,
            "api_key": self.api_key,
            "format": "json",
        }
        data = self._get(params)
        track_data = data.get("track", {}) if isinstance(data, dict) else {}
        track_duration = int(track_data.get('duration')) // 1000    # in seconds

        track_details = {
            "artist_name": artist,
            "track_name": track,
            "track_duration": track_duration
        }
        return track_details

    def fetch_album_info(self, *, artist: str, album: str) -> list[dict[str, Any]]:
        """
        Album metadata + per-track durations. Handles single-track albums (singles).
        Returns a list of per-track dicts with album stats included.
        """
        params = {
            "method": "album.getInfo",
            "artist": artist,
            "album": album,
            "api_key": self.api_key,
            "format": "json",
        }
        data = self._get(params)
        
        # Extract album data
        album_info = data.get('album', {})
        album_listeners = album_info.get('listeners', 0)
        album_playcount = album_info.get('playcount', 0)
        
        album_details = {
            'artist_name': artist,
            'album_name': album,
            'album_listeners': album_listeners,
            'album_playcount': album_playcount,
        }
        return album_details
    
    def fetch_artist_info(self, *, artist: str) -> dict[str, Any]:
        """
        Artist metadata (stats, tags, bio summary).
        """
        params = {
            "method": "artist.getInfo",
            'artist': artist,
            "api_key": self.api_key,
            "format": "json",
        }
        data = self._get(params)
        artist_info = data.get('artist', {})
        artist_data = {
            'artist_name': artist
            # , 'artist_mbid2': artist_info.get('mbid', '')
            , 'artist_listeners': artist_info.get('stats', {}).get('listeners', 0)
            , 'artist_playcount': artist_info.get('stats', {}).get('playcount', 0)
            # , 'artist_image': get_image_text(artist_info.get('image', {}), 'extralarge')
        }

        return artist_data

    def total_pages(self, from_unix: str | None, to_unix: str | None, limit: int = 1000) -> int:
        """
        Inspect the paginator to determine how many pages are available for the window.
        """
        params = {
            "method": "user.getrecenttracks",
            "user": self.username,
            "api_key": self.api_key,
            "format": "json",
            "limit": limit,
        }
        if from_unix:
            params["from"] = from_unix
        if to_unix:
            params["to"] = to_unix

        data = self._get(params)
        total_pages = int(data["recenttracks"]["@attr"]["totalPages"])
        return total_pages

    def extract_tracks(
        self,
        from_unix: str | None,
        to_unix: str | None,
        *,
        limit: int = 1000,
        scrobble_start: int = 0,
        number_pages: int | None = None,
        courtesy_sleep_secs: float = 0.2,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent tracks newest → oldest, respecting from/to and page window.
        Continues scrobble_number from an existing CSV if provided.
        """
        if scrobble_start == 0:
            logger.info("Starting fresh scrobble numbering from 1.")

        total = self.total_pages(from_unix, to_unix, limit=limit)
        if number_pages is None:
            page_goal = 1
        else:
            page_goal = max(total - number_pages + 1, 1)

        all_tracks: list[dict[str, Any]] = []
        scrobble_number = scrobble_start

        logger.info(f"Total pages: {total} | Fetching from page {total} down to {page_goal}")

        page = total
        while page >= page_goal:
            logger.info(f'Page {page} up until {page_goal}')
            params = {
                "method": "user.getrecenttracks",
                "user": self.username,
                "api_key": self.api_key,
                "format": "json",
                "page": page,
                "limit": limit,
                "extended": 0,
            }
            if from_unix:
                params["from"] = from_unix
            if to_unix:
                params["to"] = to_unix

            data = self._get(params)
            print(params)
            tracks = data.get("recenttracks", {}).get("track", [])
            if not tracks:
                page -= 1
                continue

            # drop now-playing at the head if present
            if isinstance(tracks[0], dict) and tracks[0].get("@attr", {}).get("nowplaying") == "true":
                tracks = tracks[1:]

            # early skip if page’s newest is older/equal than from_unix
            most_recent_txt = (tracks[0].get("date") or {}).get("#text") if tracks else None
            if most_recent_txt and from_unix:
                most_recent_dt = pd.to_datetime(most_recent_txt, format="%d %b %Y, %H:%M", utc=True)
                from_dt = pd.to_datetime(from_unix, unit="s", utc=True)
                if from_dt >= most_recent_dt:
                    logger.info(f"Page {page}: newest item <= from_date ({most_recent_dt}); skipping page.")
                    page -= 1
                    if courtesy_sleep_secs:
                        time.sleep(courtesy_sleep_secs)
                    continue

            for track in tracks:
                if track.get("@attr", {}).get("nowplaying") == "true":
                    continue

                date_txt = (track.get("date") or {}).get("#text")
                if not date_txt:
                    continue

                # per-row bounds
                row_dt = pd.to_datetime(date_txt, format="%d %b %Y, %H:%M", utc=True)
                if from_unix:
                    from_dt = pd.to_datetime(from_unix, unit="s", utc=True)
                    if row_dt <= from_dt:
                        continue
                if to_unix:
                    to_dt = pd.to_datetime(to_unix, unit="s", utc=True)
                    if row_dt > to_dt:
                        continue

                track_name = track.get("name", "")
                track_mbid = track.get("mbid", "")
                artist_name = track["artist"].get("#text", "")
                album_name = track["album"].get("#text", "")

                scrobble_number += 1
                entry = {
                    "scrobble_number": scrobble_number,
                    "username": self.username,
                    "track_name": track_name,
                    "track_mbid": track_mbid,
                    "date": date_txt,
                    "artist_name": artist_name,
                    "artist_mbid": track["artist"].get("mbid", ""),
                    "album_name": album_name,
                    "album_mbid": track["album"].get("mbid", ""),
                }
                all_tracks.append(entry)

            page -= 1
            if courtesy_sleep_secs:
                time.sleep(courtesy_sleep_secs)

        logger.info("Extraction finished: %d tracks collected", len(all_tracks))
        return all_tracks

    def extract_tracks_paged(
        self,
        *,
        from_unix: str | None,
        to_unix: str | None,
        limit: int,
        number_pages: int | None,
        courtesy_sleep_secs: float,
        scrobble_start: int = 0,
    ) -> list[list[dict[str, Any]]]:
        """
        Same logic as extract_tracks, but yields page-sized lists so callers can
        write progress to CSV periodically. Ordering: newest → oldest.
        """
        total = self.total_pages(limit=limit, from_unix=from_unix, to_unix=to_unix)
        first_page = 1 if number_pages is None else max(total - number_pages + 1, 1)

        from_dt = pd.to_datetime(from_unix, unit="s", utc=True) if from_unix else None
        to_dt = pd.to_datetime(to_unix, unit="s", utc=True) if to_unix else None

        logger.info("Total pages: %d | Fetching from page %d down to %d", total, total, first_page)

        scrobble_number = scrobble_start
        page = total
        page_batches: list[list[dict[str, Any]]] = []

        while page >= first_page:
            params = {
                "method": "user.getrecenttracks",
                "user": self.username,
                "api_key": self.api_key,
                "format": "json",
                "page": page,
                "limit": limit,
                "extended": 0,
            }
            if from_unix:
                params["from"] = from_unix
            if to_unix:
                params["to"] = to_unix

            data = self._get(params)
            tracks = (data.get("recenttracks") or {}).get("track", [])
            if not tracks:
                page -= 1
                continue

            if isinstance(tracks[0], dict) and tracks[0].get("@attr", {}).get("nowplaying") == "true":
                tracks = tracks[1:]

            most_recent_txt = (tracks[0].get("date") or {}).get("#text") if tracks else None
            if most_recent_txt and from_dt is not None:
                most_recent_dt = pd.to_datetime(most_recent_txt, format="%d %b %Y, %H:%M", utc=True)
                if from_dt >= most_recent_dt:
                    logger.info("Page %d: newest item <= from_date; skipping page.", page)
                    page -= 1
                    if courtesy_sleep_secs:
                        time.sleep(courtesy_sleep_secs)
                    continue

            batch: list[dict[str, Any]] = []
            kept_in_page = 0
            for t in tracks:
                if (t.get("@attr", {}) or {}).get("nowplaying") == "true":
                    continue
                date_txt = (t.get("date") or {}).get("#text")
                if not date_txt:
                    continue

                row_dt = pd.to_datetime(date_txt, format="%d %b %Y, %H:%M", utc=True)
                if from_dt is not None and row_dt <= from_dt:
                    continue
                if to_dt is not None and row_dt > to_dt:
                    continue

                scrobble_number += 1
                record = {
                    "scrobble_number": scrobble_number,
                    "username": self.username,
                    "track_name": t.get("name", "") or "",
                    "track_mbid": t.get("mbid", "") or "",
                    "date": date_txt,
                    "artist_name": (t.get("artist") or {}).get("#text", "") or "",
                    "artist_mbid": (t.get("artist") or {}).get("mbid", "") or "",
                    "album_name": (t.get("album") or {}).get("#text", "") or "",
                    "album_mbid": (t.get("album") or {}).get("mbid", "") or "",
                }
                batch.append(record)
                kept_in_page += 1

            logger.info("Page %d processed: kept %d tracks", page, kept_in_page)
            if batch:
                page_batches.append(batch)

            page -= 1
            if courtesy_sleep_secs:
                time.sleep(courtesy_sleep_secs)

        logger.info("Paged extraction finished with %d non-empty batches", len(page_batches))
        return page_batches