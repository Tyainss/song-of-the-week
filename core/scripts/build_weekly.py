from pathlib import Path
import argparse
import logging

from common.config_manager import ConfigManager
from common.logging import setup_logging
from common.utils.io import read_csv, write_csv

from core.features.aggregations import (
    build_weekly_base,
    compute_core_v1_features,
)

logger = logging.getLogger(__name__)


def run(repo_root: Path) -> Path:
    cm = ConfigManager(repo_root)
    project = cm.project()
    setup_logging(project)

    processed_dir = Path(project["paths"]["core_processed"])
    src = processed_dir / "dataset_full.csv"
    dst = processed_dir / "weekly_table.csv"

    if not src.exists():
        raise FileNotFoundError(f"dataset_full.csv not found at: {src}")

    df = read_csv(src, safe=True)
    logger.info(f"Loaded dataset_full.csv: {len(df)} rows")

    weekly_base = build_weekly_base(df)
    logger.info(f"Weekly base built: {len(weekly_base)} rows")

    weekly = compute_core_v1_features(weekly_base)
    logger.info("Core V1 features computed")

    write_csv(dst, weekly, append=False)
    logger.info(f"Wrote weekly_table.csv ({len(weekly)} rows) -> {dst}")
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    run(repo_root)


if __name__ == "__main__":
    main()
