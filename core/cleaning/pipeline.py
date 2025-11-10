
from pathlib import Path
from common.config_manager import ConfigManager
from common.utils.io import read_csv, write_csv

from .cleaning_steps import (
    trim_text_columns,
    standardize_spotify_release_date,
    add_genre_bucket,
    to_numeric,
    fill_with_median_and_flag,
    drop_columns,
    add_date_missing_flags,
    round_counts_to_int,
)

def run_pipeline():
    cm = ConfigManager(Path.cwd())
    project_cfg = cm.project()

    processed_dir = Path(project_cfg["paths"]["core_processed"])
    input_csv = processed_dir / "dataset_full.csv"
    output_csv = processed_dir / "dataset_clean.csv"

    df = read_csv(input_csv)

    # Text cleanup
    text_cols = ["artist_name", "track_name", "album_name", "username", "spotify_album"]
    df = trim_text_columns(df, text_cols)

    # Dates
    df = standardize_spotify_release_date(df, col="spotify_release_date")
    date_cols = ["date", "added_at_utc", "week_saturday_utc", "spotify_release_date"]
    df = add_date_missing_flags(df, date_cols)

    # Genres
    df = add_genre_bucket(df, source_col="spotify_genres")

    # Numeric columns
    # We only use track_duration for duration. Others are kept but cleaned.
    numeric_cols = [
        "track_duration",
        "artist_listeners", "artist_playcount",
        "album_listeners", "album_playcount",
        "spotify_popularity",
    ]
    df = to_numeric(df, numeric_cols)

    # Fill selected with median + add flags
    fill_cols = [
        "artist_listeners", "artist_playcount",
        "album_listeners", "album_playcount",
        "spotify_popularity",
    ]
    df = fill_with_median_and_flag(df, fill_cols)

    count_cols = ["artist_listeners", "artist_playcount", "album_listeners", "album_playcount"]
    df = round_counts_to_int(df, count_cols)

    # Keep only track_duration (seconds); drop spotify_duration_ms if present
    df = drop_columns(df, ["spotify_duration_ms"])

    # Save
    write_csv(output_csv, df)
