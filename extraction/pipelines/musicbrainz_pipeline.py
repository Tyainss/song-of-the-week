from pathlib import Path
from typing import Any
import logging

import pandas as pd

from common.utils.io import write_csv
from extraction.apis.musicbrainz import MusicBrainzAPI

logger = logging.getLogger(__name__)


def _dedupe_missing_by_mbid(artists_csv: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique artist_mbid values in candidates that are not yet present in artists_csv.
    """
    if artists_csv.exists():
        try:
            existing = pd.read_csv(artists_csv, usecols=["artist_mbid"])
        except Exception:
            existing = pd.read_csv(artists_csv)
        missing = (
            candidates[["artist_mbid"]]
            .dropna()
            .drop_duplicates()
            .merge(existing.drop_duplicates(), on=["artist_mbid"], how="left", indicator=True)
            .query('_merge == "left_only"')
            .drop(columns=["_merge"])
        )
        return missing
    return candidates[["artist_mbid"]].dropna().drop_duplicates()


def _dedupe_missing_by_name(artists_csv: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique artist_name values (where MBID is not available) that are not yet present in artists_csv.
    """
    if artists_csv.exists():
        try:
            existing = pd.read_csv(artists_csv, usecols=["artist_name", "artist_mbid"])
        except Exception:
            existing = pd.read_csv(artists_csv)
    else:
        existing = pd.DataFrame(columns=["artist_name", "artist_mbid"])

    # Only candidates with no MBID
    name_only = candidates[candidates["artist_mbid"].isna()][["artist_name"]].dropna().drop_duplicates()

    missing = (
        name_only.merge(existing[["artist_name"]].dropna().drop_duplicates(), on="artist_name", how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns=["_merge"])
    )
    return missing.drop_duplicates()


def run_incremental(
    *,
    api: MusicBrainzAPI,
    cfg_mb: dict[str, Any],
    curated_dir: Path,
) -> None:
    """
    Incremental MB enrichment:
      1) Read scrobbles (artists seen so far).
      2) Enrich artists with MBID first; then artist names without MBID.
      3) Append to CSV in batches.
    """
    curated_dir.mkdir(parents=True, exist_ok=True)

    scrobbles_csv = curated_dir / cfg_mb["inputs"]["scrobbles_csv"]
    artists_csv = curated_dir / cfg_mb["outputs"]["artists_csv"]

    if not scrobbles_csv.exists():
        logger.info(f"Scrobbles file not found: {scrobbles_csv}")
        return

    batch_size = cfg_mb["batch_size"]

    # Base candidates from scrobbles
    sc = pd.read_csv(scrobbles_csv, usecols=["artist_name", "artist_mbid"]).drop_duplicates()

    # 1) Missing by MBID
    logger.info("MB Step 1/2: fetching artists by MBID")
    mbid_missing = _dedupe_missing_by_mbid(artists_csv, sc)
    total_mbid = len(mbid_missing.index)
    logger.info(f"Artists with MBID to fetch: {total_mbid}")

    if total_mbid > 0:
        rows: list[dict[str, Any]] = []
        for idx, mbid in enumerate(mbid_missing["artist_mbid"].tolist(), start=1):
            info = api.fetch_artist_info_by_mbid(mbid)
            # preserve artist_name if available from scrobbles for traceability
            name = sc.loc[sc["artist_mbid"] == mbid, "artist_name"].iloc[0] if not sc.loc[sc["artist_mbid"] == mbid].empty else None
            info["artist_name"] = name
            rows.append(info)

            if idx % batch_size == 0:
                df = pd.DataFrame(rows)
                write_csv(artists_csv, df, append=artists_csv.exists())
                logger.info(f"Artists (MBID) appended ({len(df)} rows). Progress: {idx}/{total_mbid}")
                rows.clear()

        if rows:
            df = pd.DataFrame(rows)
            write_csv(artists_csv, df, append=artists_csv.exists())
            logger.info(f"Artists (MBID) appended ({len(df)} rows). Progress: {total_mbid}/{total_mbid}")

    # 2) Missing by Name (no MBID available)
    logger.info("MB Step 2/2: fetching artists by name (no MBID present)")
    name_missing = _dedupe_missing_by_name(artists_csv, sc)
    total_names = len(name_missing.index)
    logger.info(f"Artists by name to fetch: {total_names}")

    if total_names > 0:
        rows: list[dict[str, Any]] = []
        for idx, artist_name in enumerate(name_missing["artist_name"].tolist(), start=1):
            info = api.fetch_artist_info_by_name(artist_name)
            if info:
                info["artist_name"] = artist_name
                rows.append(info)
            # if None, still skip silently; you asked to avoid extra guardrails

            if idx % batch_size == 0 and rows:
                df = pd.DataFrame(rows)
                write_csv(artists_csv, df, append=artists_csv.exists())
                logger.info(f"Artists (Name) appended ({len(df)} rows). Progress: {idx}/{total_names}")
                rows.clear()

        if rows:
            df = pd.DataFrame(rows)
            write_csv(artists_csv, df, append=artists_csv.exists())
            logger.info(f"Artists (Name) appended ({len(df)} rows). Progress: {total_names}/{total_names}")
