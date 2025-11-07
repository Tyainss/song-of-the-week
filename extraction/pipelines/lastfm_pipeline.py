from pathlib import Path
from typing import Any
import logging
from datetime import datetime, timezone

import pandas as pd

from common.utils.io import read_csv, write_csv
from extraction.apis.lastfm import LastFMAPI

logger = logging.getLogger(__name__)


def _unique_not_in(existing_csv: Path, columns: list[str], new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique rows from new_df[columns] that are not present in existing_csv[columns].
    """
    if existing_csv.exists():
        try:
            existing = pd.read_csv(existing_csv, usecols=columns)
        except Exception:
            existing = pd.read_csv(existing_csv)

        merged = (
            new_df.merge(
                existing.drop_duplicates(),
                on=columns,
                how="left",
                indicator=True,
            )
            .query('_merge == "left_only"')
            .drop(columns=["_merge"])
        )
        return merged[columns].drop_duplicates()

    return new_df[columns].drop_duplicates()


def _resolve_window(cfg_lastfm: dict[str, Any], scrobbles_csv: Path) -> tuple[str | None, str | None]:
    """
    Determine (from_unix, to_unix) for the extraction window.

    Rules:
      - If cfg has explicit values, use them.
      - Otherwise:
          from_unix = latest 'date' in existing scrobbles CSV (non-inclusive)
          to_unix   = now (UTC)
    """
    from_cfg = cfg_lastfm.get("from_unix")
    to_cfg = cfg_lastfm.get("to_unix")

    # Start with config-provided values (may be None)
    resolved_from_unix = from_cfg
    resolved_to_unix = to_cfg

    if resolved_from_unix is None and scrobbles_csv.exists():
        try:
            existing_scrobbles = pd.read_csv(scrobbles_csv, usecols=["date"])
        except Exception:
            existing_scrobbles = pd.read_csv(scrobbles_csv)

        if not existing_scrobbles.empty:
            s = existing_scrobbles["date"].dropna()
            # Try Last.fm display format first; if it fails, fall back to generic parser (ISO, etc.)
            try:
                dt = pd.to_datetime(s, format="%d %b %Y, %H:%M", utc=True, errors="raise")
            except Exception:
                dt = pd.to_datetime(s, utc=True, errors="coerce")
            latest_dt = dt.max()
            if pd.notna(latest_dt):
                # non-inclusive: start 1s after the last saved scrobble
                resolved_from_unix = str(int(latest_dt.timestamp()) + 1)

    if resolved_to_unix is None:
        resolved_to_unix = str(int(datetime.now(timezone.utc).timestamp()))

    logger.info("Resolved window: from_unix=%s | to_unix=%s", resolved_from_unix, resolved_to_unix)
    return resolved_from_unix, resolved_to_unix


def run_incremental(
    *,
    api: LastFMAPI,
    cfg_lastfm: dict[str, Any],
    curated_dir: Path,
    scrobble_start: int,
) -> None:
    """
    Incremental Last.fm pipeline:
      1) Extract scrobbles and append in batches.
      2) Extract missing artist info and append.
      3) Extract missing track info and append.
      4) Extract missing album info and append.
    """
    curated_dir.mkdir(parents=True, exist_ok=True)

    scrobbles_csv = curated_dir / cfg_lastfm["outputs"]["scrobbles_csv"]
    artists_csv   = curated_dir / cfg_lastfm["outputs"]["artists_csv"]
    tracks_csv    = curated_dir / cfg_lastfm["outputs"]["tracks_csv"]
    albums_csv    = curated_dir / cfg_lastfm["outputs"]["albums_csv"]

    batch_size   = cfg_lastfm["batch_size"]
    page_size    = cfg_lastfm["page_size"]
    page_limit   = cfg_lastfm["page_limit"]
    sleep_secs   = cfg_lastfm["courtesy_sleep_secs"]

    # Resolve from/to window if not provided
    from_unix, to_unix = _resolve_window(cfg_lastfm, scrobbles_csv)
    logger.info(f"Window: from_unix={from_unix} to_unix={to_unix}")

    # 1) Scrobbles (paged write via on_batch)
    logger.info("Step 1/4: extracting scrobbles in pages (batch writes)")
    scrobble_columns = [
        "scrobble_number",
        "username",
        "track_name",
        "track_mbid",
        "date",
        "artist_name",
        "artist_mbid",
        "album_name",
        "album_mbid",
    ]

    written = {"n": 0}

    def _flush_scrobble_batch(batch: list[dict[str, Any]]) -> None:
        df_batch = pd.DataFrame(batch, columns=scrobble_columns)
        if not df_batch.empty:
            write_csv(scrobbles_csv, df_batch, append=scrobbles_csv.exists())
            written["n"] += len(df_batch)
            logger.info(f"Scrobbles appended: {len(df_batch)} rows (total {written['n']})")

    api.extract_tracks(
        from_unix=from_unix,
        to_unix=to_unix,
        limit=page_size,
        number_pages=page_limit,
        courtesy_sleep_secs=sleep_secs,
        scrobble_start=scrobble_start,
        on_batch=_flush_scrobble_batch,
    )

    if not scrobbles_csv.exists():
        logger.info("No scrobbles extracted; stopping.")
        return

    # Prepare deduplicated bases for enrichment steps
    scrobbles_df = pd.read_csv(
        scrobbles_csv,
        usecols=["artist_name", "track_name", "album_name"],
    ).drop_duplicates()

    # 2) Artists (missing only)
    logger.info("Step 2/4: extracting missing artist info")

    artists_to_fetch = _unique_not_in(
        artists_csv,
        ["artist_name"],
        scrobbles_df[["artist_name"]].rename(columns={"artist_name": "artist_name"}),
    )

    if not artists_to_fetch.empty:
        pending_artists = artists_to_fetch["artist_name"].tolist()
        artist_rows: list[dict[str, Any]] = []
        total_artists = len(pending_artists)
        logger.info(f"Artists to fetch: {total_artists}")

        for idx, artist_name in enumerate(pending_artists, start=1):
            artist_info = api.fetch_artist_info(artist=artist_name)
            artist_rows.append(artist_info)

            if idx % batch_size == 0:
                df_artist = pd.DataFrame(artist_rows)
                write_csv(artists_csv, df_artist, append=artists_csv.exists())
                logger.info(f"Artists appended ({len(df_artist)} rows). Progress: {idx}/{total_artists}")
                artist_rows.clear()

        if artist_rows:
            df_artist = pd.DataFrame(artist_rows)
            write_csv(artists_csv, df_artist, append=artists_csv.exists())
            logger.info(f"Artists appended ({len(df_artist)} rows). Progress: {total_artists}/{total_artists}")

    # 3) Tracks (missing only)
    logger.info("Step 3/4: extracting missing track info")

    if tracks_csv.exists():
        try:
            existing_tracks = pd.read_csv(tracks_csv, usecols=["artist", "name"])
        except Exception:
            existing_tracks = pd.read_csv(tracks_csv)
        existing_tracks = existing_tracks.rename(columns={"artist": "artist_name", "name": "track_name"})
    else:
        existing_tracks = pd.DataFrame(columns=["artist_name", "track_name"])

    track_pairs = (
        scrobbles_df[["artist_name", "track_name"]]
        .dropna()
        .drop_duplicates()
        .merge(
            existing_tracks.drop_duplicates(),
            on=["artist_name", "track_name"],
            how="left",
            indicator=True,
        )
        .query('_merge == "left_only"')
        .drop(columns=["_merge", "track_duration"])
    )

    if not track_pairs.empty:
        track_rows: list[dict[str, Any]] = []
        total_tracks = len(track_pairs)
        logger.info(f"Track pairs to fetch: {total_tracks}")
        # print(track_pairs)

        for idx, (artist_name, track_name) in enumerate(track_pairs.itertuples(index=False), start=1):
            track_info = api.fetch_track_info(artist=artist_name, track=track_name)
            track_rows.append(track_info)


            if idx % batch_size == 0:
                df_track = pd.DataFrame(track_rows)
                write_csv(tracks_csv, df_track, append=tracks_csv.exists())
                logger.info(f"Tracks appended ({len(df_track)} rows). Progress: {idx}/{total_tracks}")
                track_rows.clear()

        if track_rows:
            df_track = pd.DataFrame(track_rows)
            write_csv(tracks_csv, df_track, append=tracks_csv.exists())
            logger.info(f"Tracks appended ({len(df_track)} rows). Progress: {total_tracks}/{total_tracks}")

    # 4) Albums (missing only)
    logger.info("Step 4/4: extracting missing album info")

    if albums_csv.exists():
        try:
            existing_albums = pd.read_csv(albums_csv, usecols=["artist_name", "album_name", "track_name"])
        except Exception:
            existing_albums = pd.read_csv(albums_csv)
    else:
        existing_albums = pd.DataFrame(columns=["artist_name", "album_name", "track_name"])

    album_pairs = (
        scrobbles_df[["artist_name", "album_name"]]
        .dropna()
        .drop_duplicates()
        .merge(
            existing_albums[["artist_name", "album_name"]].drop_duplicates(),
            on=["artist_name", "album_name"],
            how="left",
            indicator=True,
        )
        .query('_merge == "left_only"')
        .drop(columns=["_merge"])
    )

    if not album_pairs.empty:
        album_rows: list[dict[str, Any]] = []
        total_albums = len(album_pairs)
        logger.info(f"Album pairs to fetch: {total_albums}")

        for idx, (artist_name, album_name) in enumerate(album_pairs.itertuples(index=False), start=1):
            # fetch_album_info returns a list of per-track dicts for that album
            rows_for_album = api.fetch_album_info(artist=artist_name, album=album_name)
            album_rows.extend(rows_for_album)

            if idx % batch_size == 0:
                df_album = pd.DataFrame(album_rows)
                write_csv(albums_csv, df_album, append=albums_csv.exists())
                logger.info(f"Albums appended ({len(df_album)} rows). Progress: {idx}/{total_albums}")
                album_rows.clear()

        if album_rows:
            df_album = pd.DataFrame(album_rows)
            write_csv(albums_csv, df_album, append=albums_csv.exists())
            logger.info(f"Albums appended ({len(df_album)} rows). Progress: {total_albums}/{total_albums}")
