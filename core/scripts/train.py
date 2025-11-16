
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

from common.config_manager import ConfigManager
from common.logging import setup_logging
from common.utils.io import read_csv, write_json

from core.features.featurize import (
    _ensure_week_saturday_dt,
    fit_dv_ohe,
    transform_dv_ohe,
    select_feature_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------

def _time_based_split(
    df: pd.DataFrame,
    date_col: str = "week_saturday_dt",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split by unique weeks:
    - earliest 60% weeks  -> train
    - next   20% weeks    -> val
    - last   20% weeks    -> test
    """
    out = df.copy()
    out = _ensure_week_saturday_dt(out)

    unique_weeks = np.sort(out[date_col].dropna().unique())
    n_weeks = len(unique_weeks)
    if n_weeks < 5:
        raise ValueError(f"Not enough weeks to split: {n_weeks}")

    train_end = int(n_weeks * train_frac)
    val_end = train_end + int(n_weeks * val_frac)

    train_weeks = unique_weeks[:train_end]
    val_weeks = unique_weeks[train_end:val_end]
    test_weeks = unique_weeks[val_end:]

    df_train = out[out[date_col].isin(train_weeks)].copy()
    df_val = out[out[date_col].isin(val_weeks)].copy()
    df_test = out[out[date_col].isin(test_weeks)].copy()

    logger.info(
        "Split weeks into train/val/test: "
        f"{len(train_weeks)}/{len(val_weeks)}/{len(test_weeks)} "
        f"weeks; rows = {len(df_train)}/{len(df_val)}/{len(df_test)}"
    )

    return df_train, df_val, df_test


def _build_X_y(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Apply OHE on genre_bucket (train only), then build X/y for train/val/test,
    reusing the same DictVectorizer and feature list everywhere.
    """
    # Fit DV on train only
    df_train_ohe, dv, _ = fit_dv_ohe(
        df_train,
        column="genre_bucket",
        min_freq=20,
        prefix="genre",
        keep_original=True,
    )

    # Apply DV to val/test
    df_val_ohe = transform_dv_ohe(
        df_val,
        dv=dv,
        column="genre_bucket",
        prefix="genre",
        keep_original=True,
    )
    df_test_ohe = transform_dv_ohe(
        df_test,
        dv=dv,
        column="genre_bucket",
        prefix="genre",
        keep_original=True,
    )

    # Feature columns (Core V1 + released_within_*d + genre__*)
    feat_cols = select_feature_columns(pd.concat([df_train_ohe, df_val_ohe, df_test_ohe], axis=0))

    def _XY(df_ohe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_df = df_ohe[feat_cols].copy()
        X_df = X_df.fillna(0.0)
        X = X_df.to_numpy(dtype=float)
        y = df_ohe["is_week_favorite"].astype(int).to_numpy()
        return X, y

    X_train, y_train = _XY(df_train_ohe)
    X_val, y_val = _XY(df_val_ohe)
    X_test, y_test = _XY(df_test_ohe)

    meta = {
        "dv": dv,
        "feature_columns": feat_cols,
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, meta


def _tune_threshold_on_val(
    y_val: np.ndarray,
    p_val: np.ndarray,
) -> float:
    """
    Choose the probability threshold that maximizes F1 on the validation set.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)

    # precision_recall_curve returns len(thresholds) + 1 points
    # Clip to common length
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    logger.info(
        "Best threshold on validation (by F1): "
        f"{best_threshold:.4f} (F1={f1_scores[best_idx]:.3f}, "
        f"precision={precision[best_idx]:.3f}, recall={recall[best_idx]:.3f})"
    )
    return best_threshold


def _compute_holdout_metrics(
    y_true: np.ndarray,
    p_scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Basic ROC/PR + thresholded metrics (precision/recall/F1).
    """
    roc_auc = float(roc_auc_score(y_true, p_scores))
    pr_auc = float(average_precision_score(y_true, p_scores))

    y_pred = (p_scores >= threshold).astype(int)
    precision, recall, f1 = 0.0, 0.0, 0.0
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------
# Main training routine
# ---------------------------

def run(repo_root: Path) -> Path:
    """
    Train the final Logistic Regression model on train+val and save artifacts.

    Artifacts saved:
      - core/data/models/model.bin  (pickle with model, DV, feature_columns, threshold)
      - core/data/metrics/logreg_final_metrics.json  (test metrics)
    """
    cm = ConfigManager(repo_root)
    project_cfg = cm.project()
    setup_logging(project_cfg)

    paths_cfg = project_cfg["paths"]
    modeling_cfg = project_cfg.get("modeling", {})

    features_dir = Path(paths_cfg.get("core_features", "core/data/features"))
    models_dir = Path(paths_cfg.get("core_models", "core/data/models"))
    metrics_dir = Path(paths_cfg.get("core_metrics", "core/data/metrics"))

    weekly_for_model_filename = modeling_cfg.get(
        "weekly_for_model_filename",
        "weekly_for_model.csv",
    )
    model_filename = modeling_cfg.get(
        "model_filename",
        "model.bin",
    )
    metrics_filename = modeling_cfg.get(
        "logreg_metrics_filename",
        "logreg_final_metrics.json",
    )

    weekly_for_model_path = features_dir / weekly_for_model_filename
    metrics_path = metrics_dir / metrics_filename
    model_path = models_dir / model_filename

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not weekly_for_model_path.exists():
        raise FileNotFoundError(
            f"weekly_for_model.csv not found at {weekly_for_model_path}. "
            "Build it first via core/scripts/featurize.py."
        )

    logger.info(f"Loading weekly_for_model from {weekly_for_model_path}")
    df_weekly = read_csv(weekly_for_model_path, safe=True)

    # Time-based split by week (60/20/20)
    df_train, df_val, df_test = _time_based_split(df_weekly)

    # Build X/y with OHE applied post-split
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        meta,
    ) = _build_X_y(df_train, df_val, df_test)

    # Class imbalance logging only
    pos_rate_train = float(y_train.mean())
    pos_rate_val = float(y_val.mean())
    pos_rate_test = float(y_test.mean())
    logger.info(
        f"Positive rate (train/val/test): "
        f"{pos_rate_train:.4f} / {pos_rate_val:.4f} / {pos_rate_test:.4f}"
    )

    # ---------------------------
    # Train on train set, tune threshold on val
    # ---------------------------
    random_state = 42
    n_jobs = -1
    best_C = 0.01  # Found in the notebook via validation PR-AUC

    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=n_jobs,
        random_state=random_state,
        C=best_C,
    )
    logger.info("Fitting Logistic Regression on train split...")
    logreg.fit(X_train, y_train)

    p_val = logreg.predict_proba(X_val)[:, 1]
    tuned_threshold = _tune_threshold_on_val(y_val, p_val)

    # ---------------------------
    # Retrain on train + val with the chosen C
    # ---------------------------
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    logreg_full = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=n_jobs,
        random_state=random_state,
        C=best_C,
    )
    logger.info("Fitting Logistic Regression on train+val...")
    logreg_full.fit(X_trainval, y_trainval)

    # Evaluate on test (for information / documentation only)
    p_test = logreg_full.predict_proba(X_test)[:, 1]
    test_metrics = _compute_holdout_metrics(y_test, p_test, threshold=tuned_threshold)
    logger.info(
        "Final Logistic Regression performance on test: "
        f"ROC-AUC={test_metrics['roc_auc']:.3f} | "
        f"PR-AUC={test_metrics['pr_auc']:.3f} | "
        f"F1={test_metrics['f1']:.3f} "
        f"(precision={test_metrics['precision']:.3f}, "
        f"recall={test_metrics['recall']:.3f})"
    )

    # Save metrics
    write_json(metrics_path, test_metrics)
    logger.info(f"Wrote test metrics -> {metrics_path}")

    # Save model + DV + feature list + threshold
    model_artifacts = {
        "model_type": "logreg",
        "model": logreg_full,
        "dv": meta["dv"],
        "feature_columns": meta["feature_columns"],
        "threshold": float(tuned_threshold),
    }

    with model_path.open("wb") as f:
        pickle.dump(model_artifacts, f)

    logger.info(f"Saved model artifacts -> {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run(repo_root)


if __name__ == "__main__":
    main()
