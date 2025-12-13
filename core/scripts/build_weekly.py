from pathlib import Path
import argparse
import logging
import pandas as pd

from common.config_manager import ConfigManager
from common.logging import setup_logging
from common.utils.io import read_csv, write_csv

from core.features.aggregations import (
    build_weekly_base,
    compute_core_v1_features,
)

logger = logging.getLogger(__name__)


def _log_weekly_sanity_checks(df: pd.DataFrame) -> None:
    """
    Lightweight invariants to catch data/logic issues early.
    Logs warnings only (does not fail the pipeline).
    """
    checks = {}

    nonneg_cols = [
        "scrobbles_week",
        "unique_days_week",
        "scrobbles_last_fri_sat",
        "scrobbles_saturday",
        "last_scrobble_gap_days",
        "days_since_release",
    ]
    for c in nonneg_cols:
        if c in df.columns:
            checks[f"{c}_neg_rows"] = int(
                (pd.to_numeric(df[c], errors="coerce") < 0).sum()
            )

    if "unique_days_week" in df.columns:
        checks["unique_days_week_gt_7_rows"] = int(
            (pd.to_numeric(df["unique_days_week"], errors="coerce") > 7).sum()
        )

    if {"scrobbles_last_fri_sat", "scrobbles_week"}.issubset(df.columns):
        a = pd.to_numeric(df["scrobbles_last_fri_sat"], errors="coerce")
        b = pd.to_numeric(df["scrobbles_week"], errors="coerce")
        checks["fri_sat_gt_week_rows"] = int((a > b).sum())

    if {"scrobbles_saturday", "scrobbles_week"}.issubset(df.columns):
        a = pd.to_numeric(df["scrobbles_saturday"], errors="coerce")
        b = pd.to_numeric(df["scrobbles_week"], errors="coerce")
        checks["sat_gt_week_rows"] = int((a > b).sum())

    bad = {k: v for k, v in checks.items() if v > 0}
    if bad:
        logger.warning(f"Weekly sanity checks flagged potential issues: {bad}")
    else:
        logger.info("Weekly sanity checks passed (no obvious invariant violations).")


def run(repo_root: Path) -> Path:
    cm = ConfigManager(repo_root)
    project = cm.project()
    setup_logging(project)

    processed_dir = Path(project["paths"]["core_processed"])
    src = processed_dir / "dataset_clean.csv"
    dst = processed_dir / "weekly_table.csv"

    if not src.exists():
        raise FileNotFoundError(f"dataset_clean.csv not found at: {src}")

    df = read_csv(src, safe=True)
    logger.info(f"Loaded dataset_clean.csv: {len(df)} rows")

    weekly_base = build_weekly_base(df)
    logger.info(f"Weekly base built: {len(weekly_base)} rows")

    weekly = compute_core_v1_features(weekly_base)
    logger.info("Core V1 features computed")

    _log_weekly_sanity_checks(weekly)

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
