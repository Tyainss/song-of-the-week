
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
    return read_csv(path_obj, usecols=usecols, dtype=None, safe=True)


def _load_inputs(config_manager, project_config):
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
        return None

    scrobbles_df = _safe_read(scrobbles_path)
    musicbrainz_artists_df = _safe_read(musicbrainz_artists_path)
    spotify_tracks_df = _safe_read(spotify_tracks_path)
    spotify_favorites_df = _safe_read(spotify_favorites_path)

    return {
        "scrobbles": scrobbles_df,
        "mb_artists": musicbrainz_artists_df,
        "sp_tracks": spotify_tracks_df,
        "sp_favorites": spotify_favorites_df,
    }


def _normalize_join_keys(frames):
    for frame in frames:
        for column in ("artist_name", "track_name"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)
    return frames


def _anchor_scrobbles_week(scrobbles_df):
    if "date" not in scrobbles_df.columns:
        return scrobbles_df
    ts = pd.to_datetime(scrobbles_df["date"], utc=True, errors="coerce")
    weekday = ts.dt.weekday  # Mon=0 ... Sat=5, Sun=6
    days_since_prev_sat = (weekday - 5) % 7
    prev_sat = ts - pd.to_timedelta(days_since_prev_sat, unit="D")
    scrobbles_df["week_saturday_utc"] = (
        prev_sat.dt.tz_convert("UTC").dt.tz_localize(None).dt.strftime("%Y-%m-%d 00:00:00")
    )
    return scrobbles_df


def _merge_musicbrainz(unified_df, musicbrainz_artists_df):
    if musicbrainz_artists_df.empty or "artist_name" not in musicbrainz_artists_df.columns:
        return unified_df
    return unified_df.merge(
        musicbrainz_artists_df,
        how="left",
        on=["artist_name"],
        suffixes=("", "_mb"),
        sort=False,
    )


def _merge_spotify_tracks(unified_df, spotify_tracks_df):
    if spotify_tracks_df.empty or not {"artist_name", "track_name"}.issubset(spotify_tracks_df.columns):
        return unified_df
    return unified_df.merge(
        spotify_tracks_df,
        how="left",
        on=["artist_name", "track_name"],
        suffixes=("", "_sp"),
        sort=False,
    )


def _merge_favorites(unified_df, spotify_favorites_df):
    need_cols = {"artist_name", "track_name", "week_saturday_utc"}
    if spotify_favorites_df.empty or not need_cols.issubset(spotify_favorites_df.columns):
        return unified_df
    label_cols = [c for c in ["artist_name", "track_name", "week_saturday_utc", "is_week_favorite", "added_at_utc"] if c in spotify_favorites_df.columns]
    out = unified_df.merge(
        spotify_favorites_df[label_cols],
        how="left",
        on=["artist_name", "track_name", "week_saturday_utc"],
        sort=False,
    )
    if "is_week_favorite" in out.columns:
        out["is_week_favorite"] = out["is_week_favorite"].fillna(0).astype(int)
    return out


def _order_columns(unified_df, scrobbles_columns, musicbrainz_artists_df, spotify_tracks_df):
    mb_only = [c for c in musicbrainz_artists_df.columns if c not in ("artist_name",) and c not in scrobbles_columns] if not musicbrainz_artists_df.empty else []
    sp_only = [c for c in spotify_tracks_df.columns if c not in ("artist_name", "track_name") and c not in scrobbles_columns and c not in mb_only] if not spotify_tracks_df.empty else []
    label_tail = [c for c in ["week_saturday_utc", "added_at_utc", "is_week_favorite"] if c in unified_df.columns and c not in scrobbles_columns]
    ordered = scrobbles_columns + [c for c in mb_only if c in unified_df.columns] + [c for c in sp_only if c in unified_df.columns] + [c for c in label_tail if c in unified_df.columns]
    ordered += [c for c in unified_df.columns if c not in ordered]
    return unified_df[ordered]


def _write_output(unified_df, project_config):
    processed_dir = Path(project_config["paths"]["core_processed"])  # no fallback; fail if missing
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "dataset_full.csv"
    write_csv(output_path, unified_df, append=False)
    logger.info(f"Unified dataset built at {output_path} (rows={len(unified_df)})")

def build_unified_dataset(repo_root="."):
    config_manager = ConfigManager(Path(repo_root).resolve())
    project_config = config_manager.project()
    setup_logging(project_config)

    inputs = _load_inputs(config_manager, project_config)
    if inputs is None:
        return

    scrobbles_df = inputs["scrobbles"]
    musicbrainz_artists_df = inputs["mb_artists"]
    spotify_tracks_df = inputs["sp_tracks"]
    spotify_favorites_df = inputs["sp_favorites"]

    scrobbles_columns = list(scrobbles_df.columns)
    _normalize_join_keys([scrobbles_df, musicbrainz_artists_df, spotify_tracks_df, spotify_favorites_df])
    scrobbles_df = _anchor_scrobbles_week(scrobbles_df)

    unified = scrobbles_df
    unified = _merge_musicbrainz(unified, musicbrainz_artists_df)
    unified = _merge_spotify_tracks(unified, spotify_tracks_df)
    unified = _merge_favorites(unified, spotify_favorites_df)
    unified = _order_columns(unified, scrobbles_columns, musicbrainz_artists_df, spotify_tracks_df)

    _write_output(unified, project_config)