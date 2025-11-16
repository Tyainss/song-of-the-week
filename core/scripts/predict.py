
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common.config_manager import ConfigManager
from common.logging import setup_logging

from core.features.featurize import transform_dv_ohe

logger = logging.getLogger(__name__)

app = FastAPI(
    title="song-of-the-week",
    description="Logistic Regression model for predicting weekly favourite songs.",
    version="0.1.0",
)


class TrackPrediction(BaseModel):
    """
    Per-track prediction details.

    Fields
    ------
    index:
        Position of the track in the input list.
    probability:
        Predicted probability that this track is the weekly favourite.
    prediction:
        Final 0/1 flag for this track:
        - In single-track mode: 1 if probability >= threshold, else 0.
        - In ranking mode: 1 only for the selected winner, 0 for all others.
    above_threshold:
        1 if probability >= global threshold (F1-tuned), else 0.
        This is always computed, regardless of mode.
    """

    index: int
    probability: float
    prediction: int
    above_threshold: int


class PredictionResponse(BaseModel):
    """
    Response for /predict.

    Fields
    ------
    n_tracks:
        Number of tracks in the request.
    threshold:
        Global probability threshold tuned on validation to maximize F1.
    mode:
        "single_threshold" when exactly one track is provided;
        "ranking" when multiple tracks are provided.
    winner_index:
        Index (in the input list) of the "winner" track:
        - In single_threshold mode, always 0.
        - In ranking mode, the track with the highest probability.
    results:
        Per-track prediction details.
    """

    n_tracks: int
    threshold: float
    mode: str
    winner_index: int
    results: List[TrackPrediction]


class PredictionRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "tracks": [
                    {
                        "spotify_popularity": 45,
                        "track_duration": 210,
                        "scrobbles_week": 12,
                        "unique_days_week": 3,
                        "scrobbles_last_fri_sat": 5,
                        "scrobbles_saturday": 3,
                        "last_scrobble_gap_days": 0.5,
                        "within_week_rank_by_scrobbles": 2,
                        "scrobbles_prev_1w": 8,
                        "scrobbles_prev_4w": 20,
                        "week_over_week_change": 4,
                        "momentum_4w_ratio": 1.2,
                        "prior_scrobbles_all_time": 30,
                        "first_seen_week": 0,
                        "days_since_release": 10,
                        "released_within_28d": 1,
                        "genre_bucket": "hip_hop_rap"
                    }
                ]
            }
        }
    }


def _load_artifacts(repo_root: Path) -> Dict[str, Any]:
    cm = ConfigManager(repo_root)
    project_cfg = cm.project()
    setup_logging(project_cfg)

    paths_cfg = project_cfg["paths"]
    modeling_cfg = project_cfg.get("modeling", {})

    models_dir = Path(paths_cfg.get("core_models", "core/data/models"))
    model_filename = modeling_cfg.get("model_filename", "model.bin")
    model_path = models_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found at {model_path}. "
            "Train the model first with train.py."
        )

    logger.info(f"Loading model artifacts from {model_path}")
    with model_path.open("rb") as f:
        artifacts = pickle.load(f)

    required_keys = {"model", "feature_columns", "threshold"}
    missing = required_keys - set(artifacts.keys())
    if missing:
        raise KeyError(f"Missing keys in model artifacts: {missing}")

    return artifacts


# Load artifacts once at startup
REPO_ROOT = Path(".").resolve()
ARTIFACTS = _load_artifacts(REPO_ROOT)
MODEL = ARTIFACTS["model"]
FEATURE_COLUMNS: List[str] = list(ARTIFACTS["feature_columns"])
THRESHOLD: float = float(ARTIFACTS["threshold"])
DV = ARTIFACTS.get("dv")


@app.get("/", summary="Healthcheck")
def healthcheck() -> Dict[str, Any]:
    """
    Simple healthcheck endpoint.
    """
    return {
        "status": "ok",
        "message": "song-of-the-week model is ready",
        "threshold": THRESHOLD,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict weekly favourite probability for tracks",
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Expect JSON payload like:

    {
      "tracks": [
        {
          "spotify_popularity": 45,
          "track_duration": 210,
          "scrobbles_week": 12,
          "unique_days_week": 3,
          "scrobbles_last_fri_sat": 5,
          "scrobbles_saturday": 3,
          "last_scrobble_gap_days": 0.5,
          "within_week_rank_by_scrobbles": 2,
          "scrobbles_prev_1w": 8,
          "scrobbles_prev_4w": 20,
          "week_over_week_change": 4,
          "momentum_4w_ratio": 1.2,
          "prior_scrobbles_all_time": 30,
          "first_seen_week": 0,
          "days_since_release": 10,
          "released_within_28d": 1,
          "genre_bucket": "hip_hop_rap"
        }
      ]
    }

    Behaviour
    ---------
    - If a single track is provided:
        * "single_threshold" mode.
        * prediction = 1 iff probability >= global threshold.
        * above_threshold mirrors that same comparison.

    - If multiple tracks are provided:
        * "ranking" mode.
        * Exactly one "winner" is chosen: the track with the highest probability.
        * prediction = 1 only for the winner (0 for all others).
        * above_threshold still indicates which tracks are above the global
          threshold, but it does not affect the choice of winner.

    In both modes, FEATURE_COLUMNS and the DictVectorizer (for genre_bucket)
    are applied exactly as in training.
    """
    rows = request.tracks
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="'tracks' must be a non-empty list",
        )

    df = pd.DataFrame(rows)

    # If genre_bucket is missing, fall back to "unknown"
    if "genre_bucket" not in df.columns:
        logger.info("genre_bucket not provided; defaulting to 'unknown'")
        df["genre_bucket"] = "unknown"

    # Apply OHE for genre_bucket if DV is available
    if DV is not None:
        df = transform_dv_ohe(
            df,
            dv=DV,
            column="genre_bucket",
            prefix="genre",
            keep_original=True,
        )

    # Ensure all expected feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    X_df = df[FEATURE_COLUMNS].copy().fillna(0.0)
    X = X_df.to_numpy(dtype=float)

    probs = MODEL.predict_proba(X)[:, 1]
    above_threshold = (probs >= THRESHOLD).astype(int)
    n_tracks = len(df)

    # Decide behaviour based on number of tracks
    if n_tracks == 1:
        mode = "single_threshold"
        winner_index = 0
        preds = above_threshold.copy()
    else:
        mode = "ranking"
        winner_index = int(np.argmax(probs))
        preds = np.zeros_like(above_threshold, dtype=int)
        preds[winner_index] = 1

    results: List[TrackPrediction] = []
    for i in range(n_tracks):
        results.append(
            TrackPrediction(
                index=int(i),
                probability=float(probs[i]),
                prediction=int(preds[i]),
                above_threshold=int(above_threshold[i]),
            )
        )

    return PredictionResponse(
        n_tracks=n_tracks,
        threshold=THRESHOLD,
        mode=mode,
        winner_index=winner_index,
        results=results,
    )