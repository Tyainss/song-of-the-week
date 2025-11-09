
from pathlib import Path
import argparse

from common.config_manager import ConfigManager
from common.logging import setup_logging
from extraction.apis.spotify import SpotifyAPI
from extraction.pipelines.spotify_pipeline import run_incremental


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    cfg = ConfigManager(root)

    project = cfg.project()
    setup_logging(project)

    cfg_spotify = cfg.spotify()

    curated_dir = Path(project["paths"]["extraction_curated"])
    curated_dir.mkdir(parents=True, exist_ok=True)

    api = SpotifyAPI(
        client_id=cfg.env("SPOTIFY_CLIENT_ID", required=True),
        client_secret=cfg.env("SPOTIFY_CLIENT_SECRET", required=True),
        user_agent=project["user_agent"],
        timeout_secs=cfg_spotify["timeout_secs"],
        sleep_secs=cfg_spotify.get("sleep_secs", 0.0),
        artist_cache_size=cfg_spotify.get("artist_cache_size", 5000),
    )

    run_incremental(
        api=api,
        cfg_spotify=cfg_spotify,
        curated_dir=curated_dir,
    )


if __name__ == "__main__":
    main()
