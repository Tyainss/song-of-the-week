from pathlib import Path
import argparse
import logging

import pandas as pd

from common.config_manager import ConfigManager
from common.logging import setup_logging
from common.utils.io import read_csv, write_csv

from core.features.featurize import (
    get_label_start_dt,
    make_weekly_for_model,
    make_X_y,
)

logger = logging.getLogger(__name__)


def run(repo_root: Path) -> dict[str, Path]:
    cm = ConfigManager(repo_root)
    project = cm.project()
    setup_logging(project)

    processed_dir = Path(project["paths"]["core_processed"])
    features_dir = Path(project["paths"].get("core_features", "core/data/features"))
    features_dir.mkdir(parents=True, exist_ok=True)

    weekly_table = processed_dir / "weekly_table.csv"
    if not weekly_table.exists():
        raise FileNotFoundError(
            f"weekly_table.csv not found at {weekly_table}. "
            f"Build it first via core/scripts/build_weekly.py."
        )

    df_weekly = read_csv(weekly_table, safe=True)
    if "week_saturday_dt" not in df_weekly.columns:
        df_weekly["week_saturday_dt"] = pd.to_datetime(
            df_weekly["week_saturday_utc"], utc=True, errors="coerce"
        )

    label_start_dt = get_label_start_dt(project)
    logger.info(f"Label start: {label_start_dt.date()}")

    weekly_for_model = make_weekly_for_model(df_weekly, label_start_dt)

    # Persist the modeling view (keeps keys & potentially leaky cols for audits)
    weekly_for_model_path = features_dir / "weekly_for_model.csv"
    write_csv(weekly_for_model_path, weekly_for_model, append=False)
    logger.info(f"Wrote weekly_for_model.csv ({len(weekly_for_model)}) -> {weekly_for_model_path}")

    # # Build X, y
    # X, y = make_X_y(weekly_for_model)
    # X_path = features_dir / "X.csv"
    # y_path = features_dir / "y.csv"
    # write_csv(X_path, X, append=False)
    # write_csv(y_path, y, append=False)
    # logger.info(f"Wrote X.csv {tuple(X.shape)} and y.csv {tuple(y.shape)} in {features_dir}")

    return {
        "weekly_for_model": weekly_for_model_path, 
        # "X": X_path, 
        # "y": y_path
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    run(repo_root)


if __name__ == "__main__":
    main()
