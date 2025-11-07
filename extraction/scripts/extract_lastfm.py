from pathlib import Path
import argparse
import pandas as pd

from common.config_manager import ConfigManager
from common.logging import setup_logging
from common.utils.io import write_csv 
from extraction.apis.lastfm import LastFMAPI
from extraction.pipelines.lastfm_pipeline import run_pipeline_incremental # run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    cfg = ConfigManager(root)

    project = cfg.project()
    lastfm = cfg.lastfm()

    # logging
    setup_logging(project)

    # paths
    curated_dir = Path(project["paths"]["extraction_curated"])
    curated_dir.mkdir(parents=True, exist_ok=True)
    out_csv = curated_dir / lastfm["outputs"]["scrobbles_csv"]

    # compute scrobble_start from existing CSV (outside the API)
    scrobble_start = 0
    if out_csv.exists():
        # only the 'scrobble_number' column is needed
        existing = pd.read_csv(out_csv, usecols=["scrobble_number"])
        scrobble_start = int(existing["scrobble_number"].max()) if not existing.empty else 0

    # API client (timeout_secs + user_agent from yaml)
    api = LastFMAPI(
        api_key=cfg.env("LASTFM_API_KEY", required=True),
        username=lastfm["username"],
        user_agent=project["user_agent"],
    )

    # run pipeline and append to CSV
    # df = run_pipeline(api=api, cfg_lastfm=lastfm, scrobble_start=scrobble_start)
    # run full incremental pipeline (writes batches internally)
    # if not df.empty:
    #     write_csv(out_csv, df, append=out_csv.exists())
    
    run_pipeline_incremental(
        api=api,
        cfg_lastfm=lastfm,
        curated_dir=curated_dir,
        scrobble_start=scrobble_start,
    )


if __name__ == "__main__":
    main()
