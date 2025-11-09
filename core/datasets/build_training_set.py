
from pathlib import Path
import logging
import pandas as pd

from common.config_manager import ConfigManager
from common.logging import setup_logging
from common.utils.io import read_csv, write_csv

logger = logging.getLogger(__name__)


def _safe_read(path, usecols=None):
    path_obj = Path(path)
    if not path_obj.exists():
        return pd.DataFrame(columns=usecols or [])
    # Delegate to io.read_csv with safe=True to tolerate missing columns while keeping a single call site.
    return read_csv(path_obj, usecols=usecols, dtype=None, safe=True)


def build_unified_dataset(repo_root="."):
    config_manager = ConfigManager(Path(repo_root).resolve())
    project_config = config_manager.project()
    setup_logging(project_config)

    curated_dir = Path(project_config["paths"]["extraction_curated"])

    lastfm_config = config_manager.lastfm()
    musicbrainz_config = config_manager.musicbrainz()
    spotify_config = config_manager.spotify()

    scrobbles_path = curated_dir / lastfm_config["outputs"]["scrobbles_csv"]
    musicbrainz_artists_path = curated_dir / musicbrainz_config["outputs"]["artists_csv"]
    spotify_tracks_path = curated_dir / spotify_config["outputs"]["tracks_csv"]
    spotify_favorites_path = curated_dir / spotify_config["outputs"].get("favorites_csv", "")

    if not scrobbles_path.exists():
        logger.info(f"No scrobbles found at {scrobbles_path}. Nothing to build.")
        return

    # Load curated sources via io.read_csv
    scrobbles_df = _safe_read(scrobbles_path)
    scrobbles_columns = list(scrobbles_df.columns)

    musicbrainz_artists_df = _safe_read(musicbrainz_artists_path)
    spotify_tracks_df = _safe_read(spotify_tracks_path)
    spotify_favorites_df = _safe_read(spotify_favorites_path)

    # Normalize join keys as strings (assumes curated casing already applied upstream)
    for frame in (scrobbles_df, musicbrainz_artists_df, spotify_tracks_df, spotify_favorites_df):
        for column in ("artist_name", "track_name"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)

    # Anchor scrobbles to the same Saturday used by favorites for a correct join
    if "date" in scrobbles_df.columns:
        ts = pd.to_datetime(scrobbles_df["date"], utc=True, errors="coerce")
        weekday = ts.dt.weekday  # Mon=0 ... Sat=5, Sun=6
        days_since_prev_sat = (weekday - 5) % 7
        prev_sat = ts - pd.to_timedelta(days_since_prev_sat, unit="D")
        scrobbles_df["week_saturday_utc"] = (
            prev_sat.dt.tz_convert("UTC").dt.tz_localize(None).dt.strftime("%Y-%m-%d 00:00:00")
        )

    # scrobbles + MB by artist_name
    unified = scrobbles_df
    if not musicbrainz_artists_df.empty and "artist_name" in musicbrainz_artists_df.columns:
        unified = unified.merge(
            musicbrainz_artists_df,
            how="left",
            on=["artist_name"],
            suffixes=("", "_mb"),
            sort=False,
        )

    # + Spotify by (artist_name, track_name)
    if not spotify_tracks_df.empty and {"artist_name", "track_name"}.issubset(spotify_tracks_df.columns):
        unified = unified.merge(
            spotify_tracks_df,
            how="left",
            on=["artist_name", "track_name"],
            suffixes=("", "_sp"),
            sort=False,
        )

    # + Favorites labels by (artist_name, track_name, week_saturday_utc)
    if (
        not spotify_favorites_df.empty
        and {"artist_name", "track_name", "week_saturday_utc"}.issubset(spotify_favorites_df.columns)
        and "week_saturday_utc" in unified.columns
    ):
        label_cols = ["artist_name", "track_name", "week_saturday_utc", "is_week_favorite", "added_at_utc"]
        label_cols = [c for c in label_cols if c in spotify_favorites_df.columns]
        unified = unified.merge(
            spotify_favorites_df[label_cols],
            how="left",
            on=["artist_name", "track_name", "week_saturday_utc"],
            sort=False,
        )
        if "is_week_favorite" in unified.columns:
            unified["is_week_favorite"] = unified["is_week_favorite"].fillna(0).astype(int)

    # Column order: scrobbles first, then MB-only, then Spotify-only, then any extras
    musicbrainz_only_columns = (
        [c for c in musicbrainz_artists_df.columns if c not in ("artist_name",) and c not in scrobbles_columns]
        if not musicbrainz_artists_df.empty else []
    )
    spotify_only_columns = (
        [c for c in spotify_tracks_df.columns if c not in ("artist_name", "track_name") and c not in scrobbles_columns and c not in musicbrainz_only_columns]
        if not spotify_tracks_df.empty else []
    )

    # Put labels at the end for readability
    label_tail = [c for c in ["week_saturday_utc", "added_at_utc", "is_week_favorite"] if c in unified.columns and c not in scrobbles_columns]

    ordered_columns = (
        scrobbles_columns
        + [c for c in musicbrainz_only_columns if c in unified.columns]
        + [c for c in spotify_only_columns if c in unified.columns]
        + [c for c in label_tail if c in unified.columns]
    )
    ordered_columns += [c for c in unified.columns if c not in ordered_columns]
    unified = unified[ordered_columns]

    # Write to core processed area
    paths_cfg = project_config.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed", "core/data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "dataset_full.csv"
    write_csv(output_path, unified, append=False)
    logger.info(f"Unified dataset built at {output_path} (rows={len(unified)})")
