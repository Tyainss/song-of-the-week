from pathlib import Path
from typing import Any
import logging

import pandas as pd

from common.utils.io import read_csv, write_csv
from extraction.apis.musicbrainz import MusicBrainzAPI

logger = logging.getLogger(__name__)

COLUMNS_ORDER = [
    "artist_mbid",
    "mb_artist_country",
    "mb_artist_type",
    "mb_artist_main_genre",
    "mb_artist_career_begin",
    "mb_artist_career_end",
    "mb_artist_career_ended",
]

def _dedupe_missing_by_mbid(artists_csv: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique artist_mbid values in candidates that are not yet present in artists_csv.
    """
    if artists_csv.exists():
        existing = read_csv(artists_csv, usecols=["artist_mbid"], safe=True)
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




def run_incremental(
    *,
    api: MusicBrainzAPI,
    cfg_mb: dict[str, Any],
    curated_dir: Path,
) -> None:
    """
    Incremental MB enrichment:
      1) Read scrobbles (artists seen so far).
      2) Enrich artists with MBID
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
    sc = read_csv(scrobbles_csv, usecols=["artist_name", "artist_mbid"], safe=True).drop_duplicates()

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
                # enforce exact schema/order (drop anything extra)
                df = df.reindex(columns=COLUMNS_ORDER)
                write_csv(artists_csv, df, append=artists_csv.exists())
                logger.info(f"Artists (MBID) appended ({len(df)} rows). Progress: {idx}/{total_mbid}")
                rows.clear()

        if rows:
            df = pd.DataFrame(rows)
            df = df.reindex(columns=COLUMNS_ORDER)
            write_csv(artists_csv, df, append=artists_csv.exists())
            logger.info(f"Artists (MBID) appended ({len(df)} rows). Progress: {total_mbid}/{total_mbid}")
