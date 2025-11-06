from pathlib import Path
from typing import Any, Iterable
import logging
import pandas as pd

from common.utils.io import read_csv, write_csv
from extraction.apis.lastfm import LastFMAPI

logger = logging.getLogger(__name__)


def _unique_not_in(existing_csv: Path, cols: list[str], new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_csv.exists():
        try:
            existing = pd.read_csv(existing_csv, usecols=cols)
        except Exception:
            existing = pd.read_csv(existing_csv)
        merged = (
            new_df.merge(existing.drop_duplicates(), on=cols, how="left", indicator=True)
                 .query('_merge == "left_only"')
                 .drop(columns=["_merge"])
        )
        return merged[cols].drop_duplicates()
    return new_df[cols].drop_duplicates()


def run_pipeline_incremental(
    *,
    api: LastFMAPI,
    cfg_lastfm: dict[str, Any],
    curated_dir: Path,
    scrobble_start: int,
) -> None:
    """
    1) Extract scrobbles and append in batches.
    2) Extract missing artist info and append.
    3) Extract missing track info and append.
    4) Extract missing album info and append.
    """
    curated_dir.mkdir(parents=True, exist_ok=True)

    out_scrobbles = curated_dir / cfg_lastfm["outputs"]["scrobbles_csv"]
    out_artists   = curated_dir / cfg_lastfm["outputs"]["artists_csv"]
    out_tracks    = curated_dir / cfg_lastfm["outputs"]["tracks_csv"]
    out_albums    = curated_dir / cfg_lastfm["outputs"]["albums_csv"]

    batch_size = cfg_lastfm["batch_size"]  # raises if missing (intended)

    # 1) SCROBBLES (paged write)
    logger.info("Step 1/4: extracting scrobbles in pages (batch writes)")
    batches = api.extract_tracks_paged(
        from_unix=cfg_lastfm["from_unix"],
        to_unix=cfg_lastfm["to_unix"],
        limit=cfg_lastfm["page_size"],
        number_pages=cfg_lastfm["page_limit"],
        courtesy_sleep_secs=cfg_lastfm["courtesy_sleep_secs"],
        scrobble_start=scrobble_start,
    )
    cols_sc = ["scrobble_number","username","track_name","track_mbid","date","artist_name","artist_mbid","album_name","album_mbid"]
    written = 0
    for i, batch in enumerate(batches, start=1):
        df = pd.DataFrame(batch, columns=cols_sc)
        if not df.empty:
            write_csv(out_scrobbles, df, append=out_scrobbles.exists())
            written += len(df)
            logger.info("Scrobbles page %d appended (%d rows, total written %d)", i, len(df), written)

    if not out_scrobbles.exists():
        logger.info("No scrobbles extracted; stopping.")
        return

    sc = pd.read_csv(out_scrobbles, usecols=["artist_name","track_name","album_name"]).drop_duplicates()

    # 2) ARTISTS (missing only)
    logger.info("Step 2/4: extracting missing artist info")
    missing_artists = _unique_not_in(out_artists, ["artist_name"], sc.rename(columns={"artist_name":"artist_name"}))
    if not missing_artists.empty:
        to_fetch = missing_artists["artist_name"].tolist()
        artist_rows: list[dict[str, Any]] = []
        for idx, name in enumerate(to_fetch, start=1):
            data = api.fetch_artist_info(artist=name)
            artist_rows.append(data)
            if idx % batch_size == 0:
                df_a = pd.DataFrame(artist_rows)
                write_csv(out_artists, df_a, append=out_artists.exists())
                logger.info("Artists appended (%d rows)", len(df_a))
                artist_rows.clear()
        if artist_rows:
            df_a = pd.DataFrame(artist_rows)
            write_csv(out_artists, df_a, append=out_artists.exists())
            logger.info("Artists appended (%d rows)", len(df_a))

    # 3) TRACKS (missing only)
    logger.info("Step 3/4: extracting missing track info")
    # Expecting track table to store at least artist_name, track_name
    if out_tracks.exists():
        try:
            existing_tracks = pd.read_csv(out_tracks, usecols=["artist","name"])
        except Exception:
            existing_tracks = pd.read_csv(out_tracks)
        existing_tracks = existing_tracks.rename(columns={"artist":"artist_name","name":"track_name"})
    else:
        existing_tracks = pd.DataFrame(columns=["artist_name","track_name"])
    pairs = sc[["artist_name","track_name"]].dropna().drop_duplicates()
    pairs = (
        pairs.merge(existing_tracks.drop_duplicates(), on=["artist_name","track_name"], how="left", indicator=True)
             .query('_merge == "left_only"')
             .drop(columns=["_merge"])
    )
    if not pairs.empty:
        track_rows: list[dict[str, Any]] = []
        for idx, (artist, track) in enumerate(pairs.itertuples(index=False), start=1):
            info = api.fetch_track_info(artist=artist, track=track)
            track_rows.append(info)
            if idx % batch_size == 0:
                df_t = pd.DataFrame(track_rows)
                write_csv(out_tracks, df_t, append=out_tracks.exists())
                logger.info("Tracks appended (%d rows)", len(df_t))
                track_rows.clear()
        if track_rows:
            df_t = pd.DataFrame(track_rows)
            write_csv(out_tracks, df_t, append=out_tracks.exists())
            logger.info("Tracks appended (%d rows)", len(df_t))

    # 4) ALBUMS (missing only)
    logger.info("Step 4/4: extracting missing album info")
    if out_albums.exists():
        try:
            existing_albums = pd.read_csv(out_albums, usecols=["artist_name","album_name","track_name"])
        except Exception:
            existing_albums = pd.read_csv(out_albums)
    else:
        existing_albums = pd.DataFrame(columns=["artist_name","album_name","track_name"])

    album_pairs = sc[["artist_name","album_name"]].dropna().drop_duplicates()
    album_pairs = (
        album_pairs.merge(existing_albums[["artist_name","album_name"]].drop_duplicates(),
                          on=["artist_name","album_name"], how="left", indicator=True)
                   .query('_merge == "left_only"')
                   .drop(columns=["_merge"])
    )
    if not album_pairs.empty:
        album_rows: list[dict[str, Any]] = []
        for idx, (artist, album) in enumerate(album_pairs.itertuples(index=False), start=1):
            rows = api.fetch_album_info(artist=artist, album=album)
            album_rows.extend(rows)
            if idx % batch_size == 0:
                df_al = pd.DataFrame(album_rows)
                write_csv(out_albums, df_al, append=out_albums.exists())
                logger.info("Albums appended (%d rows)", len(df_al))
                album_rows.clear()
        if album_rows:
            df_al = pd.DataFrame(album_rows)
            write_csv(out_albums, df_al, append=out_albums.exists())
            logger.info("Albums appended (%d rows)", len(df_al))


def run_pipeline(
    *,
    api: LastFMAPI,
    cfg_lastfm: dict[str, Any],
    scrobble_start: int,
) -> pd.DataFrame:
    """
    Run the Last.fm extraction and return a DataFrame with your expected columns.
    No file I/O here.
    """
    rows = api.extract_tracks(
        from_unix=cfg_lastfm["from_unix"],
        to_unix=cfg_lastfm["to_unix"],
        limit=cfg_lastfm["page_size"],
        number_pages=cfg_lastfm["page_limit"],
        courtesy_sleep_secs=cfg_lastfm["courtesy_sleep_secs"],
        scrobble_start=scrobble_start,
    )

    cols = [
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
    df = pd.DataFrame(rows, columns=cols)
    return df
