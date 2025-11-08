import time
import logging
from typing import Any
import requests

from common.utils import helper

logger = logging.getLogger(__name__)


class MusicBrainzAPI:
    """
    Thin MusicBrainz client for artist metadata.

    Notes
    - Respects a per-call sleep to avoid rate limits.
    - Uses User-Agent and timeout from the caller (yaml via entrypoint).
    """

    def __init__(
        self,
        *,
        user_agent: str,
        sleep_secs: float = 1.0,
        base_url: str = "https://musicbrainz.org/ws/2",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.sleep_secs = sleep_secs

        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def _get(self, path: str, params: dict[str, Any]) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_artist_info_by_mbid(self, artist_mbid: str) -> dict[str, Any]:
        """
        Fetch enriched artist info by MBID.
        """
        # rate-limit
        if self.sleep_secs:
            time.sleep(self.sleep_secs)

        params = {
            "inc": "aliases+tags+ratings+works+url-rels",
            "fmt": "json",
        }
        data = self._get(f"artist/{artist_mbid}", params=params)

        tags = data.get("tags", [])
        if tags:
            tags.sort(reverse=True, key=lambda x: x.get("count", 0))
            main_genre = tags[0].get("name")
        else:
            main_genre = None

        country_1 = data.get("country", "")
        area = data.get("area")
        country_2 = area.get("iso-3166-1-codes", [""])[0] if area else None
        country_name = helper.country_name_from_iso2(country_2 if country_2 else country_1)

        life_span = data.get("life-span", {})
        career_begin = helper.format_date(life_span.get("begin"))
        career_end = helper.format_date(life_span.get("end"))
        career_ended = life_span.get("ended")
        artist_type = data.get("type", "")

        out = {
            "artist_mbid": artist_mbid,
            "mb_artist_country": country_name,
            "mb_artist_main_genre": main_genre,
            "mb_artist_type": artist_type,
            "mb_artist_career_begin": career_begin,
            "mb_artist_career_end": career_end,
            "mb_artist_career_ended": career_ended,
        }
        return out

    def search_artist_mbid_by_name(self, artist_name: str) -> str | None:
        """
        Resolve an artist MBID by name (first match).
        """
        params = {
            "query": f"artist:{artist_name}", 
            "fmt": 
            "json", "limit": 1
        }
        data = self._get("artist", params=params)
        artists = data.get("artists") or []
        if not artists:
            return None
        return artists[0].get("id")

    def fetch_artist_info_by_name(self, artist_name: str) -> dict[str, Any] | None:
        """
        Convenience: resolve MBID then fetch info.
        """
        mbid = self.search_artist_mbid_by_name(artist_name)
        if not mbid:
            return None
        return self.fetch_artist_info_by_mbid(mbid)
