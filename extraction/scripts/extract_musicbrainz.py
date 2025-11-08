from pathlib import Path
import argparse

from common.config_manager import ConfigManager
from common.logging import setup_logging
from extraction.apis.musicbrainz import MusicBrainzAPI
from extraction.pipelines.musicbrainz_pipeline import run_incremental


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    cfg = ConfigManager(root)

    project = cfg.project()
    setup_logging(project)

    mb = cfg.musicbrainz()

    curated_dir = Path(project["paths"]["extraction_curated"])
    curated_dir.mkdir(parents=True, exist_ok=True)

    api = MusicBrainzAPI(
        user_agent=project["user_agent"],
        sleep_secs=mb["sleep_secs"],
    )

    run_incremental(
        api=api,
        cfg_mb=mb,
        curated_dir=curated_dir,
    )


if __name__ == "__main__":
    main()
