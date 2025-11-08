
from pathlib import Path
import logging

import pandas as pd

from common.utils.io import read_csv, write_csv
from extraction.apis.spotify import SpotifyAPI

logger = logging.getLogger(__name__)


def _dedupe_missing_tracks(existing_csv: Path, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Unique (artist_name, track_name) from pairs not yet present in existing_csv.
    """
    cols = ["artist_name", "track_name"]
    if existing_csv.exists():
        existing = read_csv(existing_csv, usecols=cols, safe=True)
        missing = (
            pairs[cols]
            .dropna()
            .drop_duplicates()
            .merge(existing.drop_duplicates(), on=cols, how="left", indicator=True)
            .query('_merge == "left_only"')
            .drop(columns=["_merge"])
        )
        return missing[cols].drop_duplicates()
    return pairs[cols].dropna().drop_duplicates()


def _generate_track_feature_rows(api: SpotifyAPI, pending_pairs: pd.DataFrame):
    """
    Yield feature rows for each (artist_name, track_name) pair in pending_pairs.
    """
    for pair in pending_pairs.itertuples(index=False):
        artist_name = getattr(pair, "artist_name")
        track_name = getattr(pair, "track_name")
        row = api.fetch_track_features(track_name=track_name, artist_name=artist_name)
        if row:
            yield row


def _pull_playlist(api: SpotifyAPI, cfg_spotify, curated_dir: Path) -> None:
    """
    Read a playlist and write curated favorites with a chosen Saturday anchor.
    """
    outputs = cfg_spotify.get("outputs", {})
    playlist_id = cfg_spotify.get("favorites_playlist_id")
    favorites_csv = outputs.get("favorites_csv")

    if not playlist_id or not favorites_csv:
        return

    out_csv = curated_dir / favorites_csv
    rows = []

    logger.info(f"Fetching playlist items for {playlist_id}")
    for item in api.iter_playlist_items(playlist_id, page_limit=cfg_spotify.get("page_limit")):
        track = (item or {}).get("track") or {}
        if not track:
            continue
        artists = track.get("artists") or []
        rows.append(
            {
                "artist_name": artists[0].get("name") if artists else None,
                "track_name": track.get("name"),
                "spotify_track_id": track.get("id"),
                "added_at_raw": item.get("added_at"),
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Normalize timestamp to UTC tz-naive string
    # Spotify added_at is RFC3339 with Z (UTC). Keep tz-naive strings in curated layer.
    dt = pd.to_datetime(df["added_at_raw"], utc=True, errors="coerce")
    df["added_at_utc"] = dt.dt.tz_convert("UTC").dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Compute nearest Saturday within ±3 days; otherwise bias to previous Saturday
    # weekday(): Monday=0 ... Saturday=5, Sunday=6
    weekday = dt.dt.weekday
    days_since_prev_sat = (weekday - 5) % 7
    prev_sat = (dt - pd.to_timedelta(days_since_prev_sat, unit="D")).dt.tz_convert("UTC")
    next_sat = prev_sat + pd.Timedelta(days=7)
    diff_prev = (dt - prev_sat).dt.total_seconds().abs() / 86400.0
    diff_next = (next_sat - dt).dt.total_seconds().abs() / 86400.0
    choose_next = (diff_next < diff_prev) & (diff_next <= 3)
    choose_prev = (~choose_next) | (diff_prev <= 3)
    chosen_sat = prev_sat.where(choose_prev, next_sat)

    # When both diffs > 3, we bias to previous Saturday (keeps cadence stable)
    both_far = (diff_prev > 3) & (diff_next > 3)
    chosen_sat = chosen_sat.where(~both_far, prev_sat)

    df["week_saturday_utc"] = chosen_sat.dt.tz_localize(None).dt.strftime("%Y-%m-%d 00:00:00")

    # Final curated columns and label flag
    df_curated = df[[
        "artist_name",
        "track_name",
        "spotify_track_id",
        "added_at_utc",
        "week_saturday_utc",
    ]].copy()
    df_curated["is_week_favorite"] = 1

    write_csv(out_csv, df_curated, append=out_csv.exists())
    logger.info(f"Favorites rows appended: {len(df_curated)}")


def run_incremental(
    *,
    api: SpotifyAPI,
    cfg_spotify,
    curated_dir: Path,
) -> None:
    """
    Incremental Spotify enrichment:
      1) Read scrobbles, build unique (artist_name, track_name).
      2) Dedupe against existing Spotify CSV.
      3) Fetch features/metadata in batches; append.
      4) Optional: pull favorites playlist when configured.
    """
    curated_dir.mkdir(parents=True, exist_ok=True)

    inputs = cfg_spotify["inputs"]
    outputs = cfg_spotify["outputs"]

    scrobbles_csv = curated_dir / inputs["scrobbles_csv"]
    tracks_output_csv = curated_dir / outputs["tracks_csv"]

    if not scrobbles_csv.exists():
        logger.info(f"Scrobbles file not found: {scrobbles_csv}")
        return

    batch_size = cfg_spotify["batch_size"]

    # Candidate pairs from scrobbles
    scrobbles_pairs = read_csv(
        scrobbles_csv, usecols=["artist_name", "track_name"], safe=True
    ).dropna().drop_duplicates()
    pending_pairs = _dedupe_missing_tracks(tracks_output_csv, scrobbles_pairs)
    total_pending = len(pending_pairs.index)
    logger.info(f"Spotify tracks to fetch: {total_pending}")

    if total_pending > 0:
        batch_rows = []
        for processed_count, row in enumerate(_generate_track_feature_rows(api, pending_pairs), start=1):
            batch_rows.append(row)
            if processed_count % batch_size == 0:
                df = pd.DataFrame(batch_rows)
                if not df.empty:
                    write_csv(tracks_output_csv, df, append=tracks_output_csv.exists())
                    logger.info(f"Spotify rows appended ({len(df)}). Progress: {processed_count}/{total_pending}")
                batch_rows.clear()

        if batch_rows:
            df = pd.DataFrame(batch_rows)
            write_csv(tracks_output_csv, df, append=tracks_output_csv.exists())
            logger.info(f"Spotify rows appended ({len(df)}). Progress: {total_pending}/{total_pending}")

    # Curated favorites (labels)
    _pull_playlist(api, cfg_spotify, curated_dir)
