
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

    lastfm_scrobbles_path = curated_dir / lastfm_config["outputs"]["scrobbles_csv"]
    lastfm_tracks_path = curated_dir / lastfm_config["outputs"]["tracks_csv"]
    lastfm_artists_path = curated_dir / lastfm_config["outputs"]["artists_csv"]
    lastfm_albums_path = curated_dir / lastfm_config["outputs"]["albums_csv"]
    musicbrainz_artists_path = curated_dir / musicbrainz_config["outputs"]["artists_csv"]
    spotify_tracks_path = curated_dir / spotify_config["outputs"]["tracks_csv"]
    spotify_favorites_path = curated_dir / spotify_config["outputs"].get("favorites_csv", "")

    if not lastfm_scrobbles_path.exists():
        logger.info(f"No scrobbles found at {lastfm_scrobbles_path}. Nothing to build.")
        return None

    lastfm_scrobbles_df = _safe_read(lastfm_scrobbles_path)
    lastfm_tracks_df = _safe_read(lastfm_tracks_path)
    lastfm_artists_df = _safe_read(lastfm_artists_path)
    lastfm_albums_df = _safe_read(lastfm_albums_path)
    musicbrainz_artists_df = _safe_read(musicbrainz_artists_path)
    spotify_tracks_df = _safe_read(spotify_tracks_path)
    spotify_favorites_df = _safe_read(spotify_favorites_path)

    return {
        "lf_scrobbles": lastfm_scrobbles_df,
        "lf_tracks": lastfm_tracks_df,
        "lf_artists": lastfm_artists_df,
        "lf_albums": lastfm_albums_df,
        "mb_artists": musicbrainz_artists_df,
        "sp_tracks": spotify_tracks_df,
        "sp_favorites": spotify_favorites_df,
    }


def _normalize_join_keys(frames):
    for frame in frames:
        for column in ("artist_name", "track_name", "album_name"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)
    return frames


def _anchor_scrobbles_week(scrobbles_df):
    if "date" not in scrobbles_df.columns:
        return scrobbles_df
    ts = pd.to_datetime(scrobbles_df["date"], utc=True, errors="coerce")
    weekday = ts.dt.weekday  # Mon=0 ... Sat=5, Sun=6
    # Map any date to the Saturday on or after it (same "Sat-ended week").
    days_until_sat = (5 - weekday) % 7
    next_sat = ts + pd.to_timedelta(days_until_sat, unit="D")
    scrobbles_df["week_saturday_utc"] = (
        next_sat.dt.tz_convert("UTC").dt.tz_localize(None).dt.strftime("%Y-%m-%d 00:00:00")
    )
    return scrobbles_df

def _merge_lastfm_tracks(unified_df: pd.DataFrame, lastfm_tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Last.fm track-level info (e.g., track_duration) on ['artist_name', 'track_name'].
    """
    need = {"artist_name", "track_name"}
    if lastfm_tracks_df.empty or not need.issubset(lastfm_tracks_df.columns):
        return unified_df
    right = lastfm_tracks_df.drop_duplicates(subset=["artist_name", "track_name"])
    return unified_df.merge(
        right,
        how="left",
        on=["artist_name", "track_name"],
        sort=False,
    )


def _merge_lastfm_artists(unified_df: pd.DataFrame, lastfm_artists_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Last.fm artist-level stats (artist_listeners, artist_playcount) on ['artist_name'].
    """
    if lastfm_artists_df.empty or "artist_name" not in lastfm_artists_df.columns:
        return unified_df
    right = lastfm_artists_df.drop_duplicates(subset=["artist_name"])
    return unified_df.merge(
        right,
        how="left",
        on=["artist_name"],
        sort=False,
    )


def _merge_lastfm_albums(unified_df: pd.DataFrame, lastfm_albums_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Last.fm album-level stats (album_listeners, album_playcount) on ['artist_name', 'album_name'].
    """
    need = {"artist_name", "album_name"}
    if lastfm_albums_df.empty or not need.issubset(lastfm_albums_df.columns):
        return unified_df
    right = lastfm_albums_df.drop_duplicates(subset=["artist_name", "album_name"])
    return unified_df.merge(
        right,
        how="left",
        on=["artist_name", "album_name"],
        sort=False,
    )


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


def _order_columns(
    unified_df: pd.DataFrame,
    scrobbles_columns: list[str],
    musicbrainz_artists_df: pd.DataFrame,
    spotify_tracks_df: pd.DataFrame,
    lastfm_tracks_df: pd.DataFrame,
    lastfm_artists_df: pd.DataFrame,
    lastfm_albums_df: pd.DataFrame,
):
    # Ignore any merge-suffix variants to avoid reintroducing duplicates after coalescing
    cols_filtered = [c for c in unified_df.columns if not (c.endswith("_x") or c.endswith("_y"))]
    
    lf_tracks_only = [c for c in lastfm_tracks_df.columns if c not in ("artist_name", "track_name") and c not in scrobbles_columns] if not lastfm_tracks_df.empty else []
    lf_artists_only = [c for c in lastfm_artists_df.columns if c not in ("artist_name",) and c not in scrobbles_columns] if not lastfm_artists_df.empty else []
    lf_albums_only = [c for c in lastfm_albums_df.columns if c not in ("artist_name", "album_name") and c not in scrobbles_columns] if not lastfm_albums_df.empty else []

    mb_only = [c for c in musicbrainz_artists_df.columns if c not in ("artist_name",) and c not in scrobbles_columns] if not musicbrainz_artists_df.empty else []
    sp_only = [c for c in spotify_tracks_df.columns if c not in ("artist_name", "track_name") and c not in scrobbles_columns and c not in mb_only] if not spotify_tracks_df.empty else []
    label_tail = [c for c in ["week_saturday_utc", "added_at_utc", "is_week_favorite"] if c in cols_filtered and c not in scrobbles_columns]


    preferred = (
        [c for c in scrobbles_columns if c in cols_filtered]
        + [c for c in lf_artists_only if c in cols_filtered]
        + [c for c in lf_tracks_only if c in cols_filtered]
        + [c for c in lf_albums_only if c in cols_filtered]
        + [c for c in mb_only if c in cols_filtered]
        + [c for c in sp_only if c in cols_filtered]
        + [c for c in label_tail if c in cols_filtered]
    )
    # De-duplicate while preserving order
    seen = set()
    preferred_unique = [c for c in preferred if not (c in seen or seen.add(c))]
    the_rest = [c for c in cols_filtered if c not in preferred_unique]
    return unified_df[preferred_unique + the_rest]

def _coalesce_col(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # If merges created column_x / column_y, prefer column's (x) and
    # fallback to the (y). Produce a single column.
    x, y = f"{column}_x", f"{column}_y"
    if x in df.columns and y in df.columns:
        df[column] = df[x].where(df[x].notna(), df[y])
        df = df.drop(columns=[x, y])
    elif x in df.columns:
        df = df.rename(columns={x: column})
    elif y in df.columns:
        df = df.rename(columns={y: column})
    # if column already existed cleanly, do nothing
    return df

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

    scrobbles_df = inputs["lf_scrobbles"]
    lastfm_tracks_df = inputs["lf_tracks"]
    lastfm_artists_df = inputs["lf_artists"]
    lastfm_albums_df = inputs["lf_albums"]
    musicbrainz_artists_df = inputs["mb_artists"]
    spotify_tracks_df = inputs["sp_tracks"]
    spotify_favorites_df = inputs["sp_favorites"]

    scrobbles_columns = list(scrobbles_df.columns)
    _normalize_join_keys([
        scrobbles_df,
        lastfm_tracks_df,
        lastfm_artists_df,
        lastfm_albums_df,
        musicbrainz_artists_df,
        spotify_tracks_df,
        spotify_favorites_df,
    ])
    scrobbles_df = _anchor_scrobbles_week(scrobbles_df)

    unified = scrobbles_df
    unified = _merge_lastfm_artists(unified, lastfm_artists_df)
    unified = _merge_lastfm_tracks(unified, lastfm_tracks_df)
    unified = _merge_lastfm_albums(unified, lastfm_albums_df)
    unified = _merge_musicbrainz(unified, musicbrainz_artists_df)
    unified = _merge_spotify_tracks(unified, spotify_tracks_df)
    unified = _merge_favorites(unified, spotify_favorites_df)

    unified = _coalesce_col(unified, column='album_mbid')
    unified = _coalesce_col(unified, column='artist_mbid')

    unified = _order_columns(
        unified,
        scrobbles_columns,
        musicbrainz_artists_df,
        spotify_tracks_df,
        lastfm_tracks_df,
        lastfm_artists_df,
        lastfm_albums_df,
    )

    _write_output(unified, project_config)