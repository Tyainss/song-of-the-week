
import os
import logging
import pickle
from pathlib import Path
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

import time
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse

from common.config_manager import ConfigManager
from common.logging import setup_logging

from core.features.featurize import transform_dv_ohe

logger = logging.getLogger(__name__)

app = FastAPI(
    title="song-of-the-week",
    description="Logistic Regression model for predicting weekly favourite songs.",
    version="0.1.0",
)


MODEL_INPUT_FIELDS: List[str] = [
    "spotify_popularity",
    "track_duration",
    "scrobbles_week",
    "unique_days_week",
    "scrobbles_last_fri_sat",
    "scrobbles_saturday",
    "last_scrobble_gap_days",
    "within_week_rank_by_scrobbles",
    "scrobbles_prev_1w",
    "scrobbles_prev_4w",
    "week_over_week_change",
    "momentum_4w_ratio",
    "prior_scrobbles_all_time",
    "first_seen_week",
    "days_since_release",
    "released_within_28d",
    "genre_bucket",
]


class CandidateSource(str, Enum):
    SPOTIFY = "spotify"
    RANDOM_EXAMPLE = "random_example"
    FAVOURITE_EXAMPLE = "favourite_example"
    MANUAL = "manual"


class PredictMode(str, Enum):
    AUTO = "auto"
    SINGLE = "single"
    RANKING = "ranking"

class Candidate(BaseModel):
    """
    One candidate song for a given week.

    Contains:
    - Minimal metadata for the UI (IDs, names, source).
    - All raw features expected by the model (before DictVectorizer/OHE).
    """

    # --- UI / bookkeeping ---
    candidate_id: Optional[str] = None
    source: Optional[CandidateSource] = None

    track_name: Optional[str] = None
    artist_name: Optional[str] = None
    week_start: Optional[date] = None
    spotify_track_id: Optional[str] = None

    # --- Model input features (pre-DV/OHE) ---
    spotify_popularity: float
    track_duration: float
    scrobbles_week: float
    unique_days_week: float
    scrobbles_last_fri_sat: float
    scrobbles_saturday: float
    last_scrobble_gap_days: float
    within_week_rank_by_scrobbles: float
    scrobbles_prev_1w: float
    scrobbles_prev_4w: float
    week_over_week_change: float
    momentum_4w_ratio: float
    prior_scrobbles_all_time: float
    first_seen_week: float
    days_since_release: float
    released_within_28d: float
    genre_bucket: str = "unknown"

    model_config = {
        "json_schema_extra": {
            "example": {
                "candidate_id": "cand_001",
                "source": "spotify",
                "track_name": "Example Song",
                "artist_name": "Example Artist",
                "spotify_track_id": "1234abcd",
                "week_start": "2023-05-06",
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
                "genre_bucket": "hip_hop_rap",
            }
        }
    }

    def to_model_row(self) -> Dict[str, Any]:
        """
        Canonical mapping from Candidate -> raw feature row for the model.

        Only the fields in MODEL_INPUT_FIELDS are included; metadata is ignored.
        """
        row: Dict[str, Any] = {}
        for field in MODEL_INPUT_FIELDS:
            row[field] = getattr(self, field)
        return row

class ExampleMetadata(BaseModel):
    """
    Lightweight metadata for an example row from the weekly dataset.
    """

    track_name: Optional[str] = None
    artist_name: Optional[str] = None
    week_start: Optional[date] = None
    is_week_favorite: Optional[bool] = None
    spotify_track_id: Optional[str] = None


class ExampleCandidate(BaseModel):
    """
    Wrapper combining a model-ready Candidate with human-friendly metadata.
    """

    candidate: Candidate
    metadata: ExampleMetadata


class ExamplesResponse(BaseModel):
    """
    Response for /examples/* endpoints.
    """

    candidates: List[ExampleCandidate]


class TrackPrediction(BaseModel):
    """
    Per-candidate prediction details.

    Fields
    ------
    index:
        Position of the candidate in the input list.
    candidate_id:
        Optional client-side ID, echoed back for convenience.
    probability:
        Predicted probability that this candidate is the weekly favourite.
    prediction:
        Final 0/1 flag:
        - In single mode: 1 if probability >= threshold, else 0.
        - In ranking mode: 1 only for the selected winner, 0 for all others.
    above_threshold:
        1 if probability >= global threshold (F1-tuned), else 0.
    rank:
        1-based rank by probability (1 = highest score), only meaningful
        in ranking mode.
    """

    index: int
    candidate_id: Optional[str] = None
    probability: float
    prediction: int
    above_threshold: int
    rank: Optional[int] = None


class PredictionResponse(BaseModel):
    """
    Response for /predict.

    Fields
    ------
    n_candidates:
        Number of candidates in the request.
    threshold:
        Global probability threshold tuned on validation to maximize F1.
    mode:
        "single" when single-song mode is applied;
        "ranking" when multiple candidates are ranked.
    winner_index:
        Index (in the input list) of the "winner" candidate.
    winner_candidate_id:
        Echo of the candidate_id for the winner, if provided.
    results:
        Per-candidate prediction details.
    """

    n_candidates: int
    threshold: float
    mode: PredictMode
    winner_index: int
    winner_candidate_id: Optional[str] = None
    results: List[TrackPrediction]

class SpotifyTrack(BaseModel):
    """
    Normalized Spotify track information returned by helper endpoints.
    """

    track_id: str
    track_name: str
    artist_name: str

    album_name: Optional[str] = None
    popularity: Optional[int] = None
    duration_ms: Optional[int] = None
    duration_sec: Optional[float] = None
    external_url: Optional[str] = None
    preview_url: Optional[str] = None


class SpotifySearchResponse(BaseModel):
    """
    Response for /spotify/search_track.

    items:
        List of candidate tracks that match the search query.
    """

    items: List[SpotifyTrack]


class PredictionRequest(BaseModel):
    """
    Request body for /predict.

    - candidates: one or more Candidate objects.
    - mode:
        * "auto" (default): backend infers mode from number of candidates.
        * "single": force single-song / threshold mode.
        * "ranking": force ranking mode.
    """

    candidates: List[Candidate]
    mode: PredictMode = PredictMode.AUTO

    model_config = {
        "json_schema_extra": {
            "example": {
                "mode": "auto",
                "candidates": [
                    {
                        "candidate_id": "cand_001",
                        "source": "manual",
                        "track_name": "Example Song",
                        "artist_name": "Example Artist",
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
                        "genre_bucket": "hip_hop_rap",
                    }
                ],
            }
        }
    }


def build_model_dataframe(candidates: List[Candidate]) -> pd.DataFrame:
    """
    Convert a list of Candidate objects into a DataFrame of raw features.

    - Uses Candidate.to_model_row() so metadata is ignored.
    - Only fields in MODEL_INPUT_FIELDS are included.
    """
    if not candidates:
        raise ValueError("candidates must be a non-empty list")

    rows: List[Dict[str, Any]] = [c.to_model_row() for c in candidates]
    df = pd.DataFrame(rows)

    if "genre_bucket" not in df.columns:
        df["genre_bucket"] = "unknown"

    return df

def spotify_track_to_candidate_template(
    track: SpotifyTrack,
) -> Candidate:
    """
    Map a SpotifyTrack into a Candidate template.

    - Uses Spotify popularity and duration for the corresponding features.
    - Initializes all other numeric features to 0.0 so the UI can tweak them.
    """
    popularity = float(track.popularity) if track.popularity is not None else 0.0
    duration_sec = float(track.duration_sec) if track.duration_sec is not None else 0.0

    return Candidate(
        source=CandidateSource.SPOTIFY,
        track_name=track.track_name,
        artist_name=track.artist_name,
        spotify_track_id=track.track_id,
        spotify_popularity=popularity,
        track_duration=duration_sec,
        scrobbles_week=0.0,
        unique_days_week=0.0,
        scrobbles_last_fri_sat=0.0,
        scrobbles_saturday=0.0,
        last_scrobble_gap_days=0.0,
        within_week_rank_by_scrobbles=1.0,
        scrobbles_prev_1w=0.0,
        scrobbles_prev_4w=0.0,
        week_over_week_change=0.0,
        momentum_4w_ratio=0.0,
        prior_scrobbles_all_time=0.0,
        first_seen_week=0.0,
        days_since_release=0.0,
        released_within_28d=0.0,
        genre_bucket="unknown",
    )


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

def _get_spotify_credentials() -> Dict[str, str]:
    """
    Load Spotify client credentials.

    For security, they are expected in environment variables:
      - SPOTIFY_CLIENT_ID
      - SPOTIFY_CLIENT_SECRET
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Spotify credentials missing. Please set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET environment variables."
        )

    return {"client_id": client_id, "client_secret": client_secret}


SPOTIFY_TOKEN: Optional[str] = None
SPOTIFY_TOKEN_EXPIRES_AT: float = 0.0


def _get_spotify_token() -> str:
    """
    Retrieve (and cache) a Spotify access token using Client Credentials flow.
    """
    global SPOTIFY_TOKEN, SPOTIFY_TOKEN_EXPIRES_AT

    now = time.time()
    if SPOTIFY_TOKEN and now < SPOTIFY_TOKEN_EXPIRES_AT - 30:
        return SPOTIFY_TOKEN

    creds = _get_spotify_credentials()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(creds["client_id"], creds["client_secret"]),
        timeout=10,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to obtain Spotify token: {resp.status_code} {resp.text}"
        )

    payload = resp.json()
    SPOTIFY_TOKEN = payload["access_token"]
    expires_in = int(payload.get("expires_in", 3600))
    SPOTIFY_TOKEN_EXPIRES_AT = now + expires_in
    return SPOTIFY_TOKEN


def _spotify_api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Helper to call Spotify Web API (GET).
    """
    token = _get_spotify_token()
    url = f"https://api.spotify.com/v1{path}"
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(url, headers=headers, params=params or {}, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Spotify API error ({resp.status_code}): {resp.text}"
        )
    return resp.json()

def _load_examples_df(repo_root: Path) -> pd.DataFrame:
    """
    Load the weekly model-ready dataset used to serve /examples endpoints.

    Expects a 'paths.weekly_dataset' entry in project.yaml pointing to a CSV
    or Parquet file relative to the repo root, for example:

        paths:
          weekly_dataset: "core/data/weekly_tracks.parquet"

    You can adjust the config key/path as needed; this function centralizes
    the loading logic so the rest of the code stays clean.
    """
    cm = ConfigManager(repo_root)
    project_cfg = cm.project()
    paths_cfg = project_cfg["paths"]

    weekly_dataset_rel = paths_cfg.get("weekly_dataset")
    if not weekly_dataset_rel:
        raise KeyError(
            "Config 'paths.weekly_dataset' is required for /examples endpoints."
        )

    weekly_dataset_path = (repo_root / weekly_dataset_rel).resolve()
    if not weekly_dataset_path.exists():
        raise FileNotFoundError(
            f"Weekly dataset not found at {weekly_dataset_path}. "
            "Please ensure 'paths.weekly_dataset' points to a valid file."
        )

    if weekly_dataset_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(weekly_dataset_path)
    if weekly_dataset_path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(weekly_dataset_path)

    raise ValueError(
        f"Unsupported weekly_dataset file format: {weekly_dataset_path.suffix}"
    )


# Load artifacts once at startup
REPO_ROOT = Path(".").resolve()
ARTIFACTS = _load_artifacts(REPO_ROOT)
MODEL = ARTIFACTS["model"]
FEATURE_COLUMNS: List[str] = list(ARTIFACTS["feature_columns"])
THRESHOLD: float = float(ARTIFACTS["threshold"])
DV = ARTIFACTS.get("dv")
EXAMPLES_DF = _load_examples_df(REPO_ROOT)


@app.get("/", summary="Healthcheck")
@app.get("/health", summary="Healthcheck")
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
    summary="Predict weekly favourite probability for candidates",
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Expect JSON payload like:

    {
      "mode": "auto",
      "candidates": [
        {
          "candidate_id": "cand_001",
          "source": "manual",
          "track_name": "Example Song",
          "artist_name": "Example Artist",
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
    - If a single candidate is provided (or mode='single'):
        * "single" mode.
        * prediction = 1 iff probability >= global threshold.
        * above_threshold mirrors that same comparison.

    - If multiple candidates are provided (or mode='ranking'):
        * "ranking" mode.
        * Exactly one "winner" is chosen: the candidate with the highest probability.
        * prediction = 1 only for the winner (0 for all others).
        * above_threshold still indicates which candidates are above the global
          threshold, but it does not affect the choice of winner.

    In both modes, FEATURE_COLUMNS and the DictVectorizer (for genre_bucket)
    are applied exactly as in training.
    """
    candidates = request.candidates
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="'candidates' must be a non-empty list",
        )

    # Build base DataFrame from candidates
    try:
        df = build_model_dataframe(candidates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    n_candidates = len(df)

    # Decide behaviour based on mode + number of candidates
    if request.mode == PredictMode.AUTO:
        if n_candidates == 1:
            effective_mode = PredictMode.SINGLE
        else:
            effective_mode = PredictMode.RANKING
    else:
        effective_mode = request.mode

    if effective_mode == PredictMode.SINGLE and n_candidates > 1:
        # Fallback: multiple candidates with mode='single' -> treat as ranking
        logger.warning(
            "PredictRequest mode='single' with %d candidates; falling back to 'ranking' mode",
            n_candidates,
        )
        effective_mode = PredictMode.RANKING

    if effective_mode == PredictMode.SINGLE:
        winner_index = 0
        preds = above_threshold.copy()
        ranks = [None] * n_candidates
    else:
        winner_index = int(np.argmax(probs))
        preds = np.zeros_like(above_threshold, dtype=int)
        preds[winner_index] = 1

        # Compute 1-based ranks by probability (descending)
        sorted_indices = np.argsort(-probs)
        rank_by_index: Dict[int, int] = {}
        for rank, idx in enumerate(sorted_indices, start=1):
            rank_by_index[int(idx)] = rank
        ranks = [rank_by_index[int(i)] for i in range(n_candidates)]

    results: List[TrackPrediction] = []
    for i in range(n_candidates):
        candidate = candidates[i]
        results.append(
            TrackPrediction(
                index=int(i),
                candidate_id=candidate.candidate_id,
                probability=float(probs[i]),
                prediction=int(preds[i]),
                above_threshold=int(above_threshold[i]),
                rank=ranks[i],
            )
        )

    winner_candidate_id: Optional[str] = None
    try:
        winner_candidate_id = candidates[winner_index].candidate_id
    except (IndexError, AttributeError):
        winner_candidate_id = None

    return PredictionResponse(
        n_candidates=n_candidates,
        threshold=THRESHOLD,
        mode=effective_mode,
        winner_index=winner_index,
        winner_candidate_id=winner_candidate_id,
        results=results,
    )

def _row_to_example_candidate(
    row: pd.Series,
    source: CandidateSource,
) -> ExampleCandidate:
    """
    Convert a weekly dataset row into an ExampleCandidate.

    - Uses MODEL_INPUT_FIELDS to populate the Candidate feature values.
    - Uses common metadata columns when present:
      track_name, artist_name, week_start, is_week_favorite, spotify_track_id.
    """
    feature_kwargs: Dict[str, Any] = {}
    missing_features: List[str] = []

    for field in MODEL_INPUT_FIELDS:
        if field in row:
            feature_kwargs[field] = row[field]
        else:
            missing_features.append(field)

    if missing_features:
        raise KeyError(
            f"Missing required feature columns in weekly dataset row: {missing_features}"
        )

    candidate = Candidate(
        source=source,
        track_name=row.get("track_name"),
        artist_name=row.get("artist_name"),
        week_start=row.get("week_start"),
        spotify_track_id=row.get("spotify_track_id"),
        **feature_kwargs,
    )

    metadata = ExampleMetadata(
        track_name=candidate.track_name,
        artist_name=candidate.artist_name,
        week_start=candidate.week_start,
        is_week_favorite=bool(row["is_week_favorite"])
        if "is_week_favorite" in row and row["is_week_favorite"] is not None
        else None,
        spotify_track_id=candidate.spotify_track_id,
    )

    return ExampleCandidate(candidate=candidate, metadata=metadata)


@app.get(
    "/examples/random",
    response_model=ExamplesResponse,
    summary="Return random model-ready examples from the weekly dataset",
)
def get_random_examples(count: int = 1) -> ExamplesResponse:
    """
    Sample `count` random rows from the weekly model-ready dataset.
    """
    if count <= 0:
        raise HTTPException(status_code=400, detail="'count' must be positive")

    available = len(EXAMPLES_DF)
    if available == 0:
        raise HTTPException(status_code=500, detail="Weekly dataset is empty")

    n = min(count, available)
    sample_df = EXAMPLES_DF.sample(n=n, replace=False, random_state=None)

    examples = [
        _row_to_example_candidate(row, source=CandidateSource.RANDOM_EXAMPLE)
        for _, row in sample_df.iterrows()
    ]

    return ExamplesResponse(candidates=examples)


@app.get(
    "/examples/favorites",
    response_model=ExamplesResponse,
    summary="Return random favourite examples from the weekly dataset",
)
def get_favorite_examples(count: int = 1) -> ExamplesResponse:
    """
    Sample `count` random rows where is_week_favorite == 1.
    """
    if "is_week_favorite" not in EXAMPLES_DF.columns:
        raise HTTPException(
            status_code=500,
            detail="Weekly dataset does not contain 'is_week_favorite' column",
        )

    fav_df = EXAMPLES_DF[EXAMPLES_DF["is_week_favorite"] == 1]
    available = len(fav_df)

    if available == 0:
        raise HTTPException(
            status_code=404,
            detail="No favourite examples available in the weekly dataset",
        )

    if count <= 0:
        raise HTTPException(status_code=400, detail="'count' must be positive")

    n = min(count, available)
    sample_df = fav_df.sample(n=n, replace=False, random_state=None)

    examples = [
        _row_to_example_candidate(row, source=CandidateSource.FAVOURITE_EXAMPLE)
        for _, row in sample_df.iterrows()
    ]

    return ExamplesResponse(candidates=examples)

def _extract_spotify_track_id(raw_url: str) -> str:
    """
    Extract Spotify track ID from a URL or URI.

    Supported formats (examples):
      - https://open.spotify.com/track/{id}
      - https://open.spotify.com/intl-pt/track/{id}
      - spotify:track:{id}
    """
    raw_url = raw_url.strip()

    # URI form: spotify:track:{id}
    if raw_url.startswith("spotify:track:"):
        return raw_url.split("spotify:track:", 1)[1]

    parsed = urlparse(raw_url)
    if "open.spotify.com" not in parsed.netloc:
        raise ValueError("URL does not look like a Spotify track URL")

    # Split path into segments and look for "track" anywhere
    parts = [p for p in parsed.path.split("/") if p]

    try:
        track_index = parts.index("track")
    except ValueError as exc:
        # No "track" segment found
        raise ValueError("URL path does not look like a Spotify track URL") from exc

    if track_index == len(parts) - 1:
        # "track" present but no ID after it
        raise ValueError("Spotify track URL is missing the track ID")

    return parts[track_index + 1]


def _normalize_spotify_track(payload: Dict[str, Any]) -> SpotifyTrack:
    """
    Normalize a Spotify track JSON payload into SpotifyTrack.
    """
    track_id = payload.get("id")
    if not track_id:
        raise ValueError("Spotify track payload missing 'id'")

    name = payload.get("name") or ""
    artists = payload.get("artists") or []
    artist_name = ", ".join(a.get("name", "") for a in artists if a.get("name"))

    album = payload.get("album") or {}
    album_name = album.get("name")

    popularity = payload.get("popularity")
    duration_ms = payload.get("duration_ms")
    duration_sec = duration_ms / 1000.0 if duration_ms is not None else None

    external_urls = payload.get("external_urls") or {}
    external_url = external_urls.get("spotify")

    preview_url = payload.get("preview_url")

    return SpotifyTrack(
        track_id=track_id,
        track_name=name,
        artist_name=artist_name,
        album_name=album_name,
        popularity=popularity,
        duration_ms=duration_ms,
        duration_sec=duration_sec,
        external_url=external_url,
        preview_url=preview_url,
    )


@app.get(
    "/spotify/track_from_url",
    response_model=SpotifyTrack,
    summary="Resolve a Spotify track from URL or URI",
)
def spotify_track_from_url(url: str) -> SpotifyTrack:
    """
    Given a Spotify track URL/URI, fetch normalized track info from Spotify.
    """
    try:
        track_id = _extract_spotify_track_id(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        payload = _spotify_api_get(f"/tracks/{track_id}")
        track = _normalize_spotify_track(payload)
    except Exception as exc:  # noqa: BLE001 – surface as HTTP error
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch track from Spotify: {exc}",
        ) from exc

    return track


@app.get(
    "/spotify/search_track",
    response_model=SpotifySearchResponse,
    summary="Search for Spotify tracks by artist and track name",
)
def spotify_search_track(
    artist: str,
    track: str,
    limit: int = 5,
) -> SpotifySearchResponse:
    """
    Search Spotify tracks using artist + track name.
    """
    if not artist.strip() or not track.strip():
        raise HTTPException(
            status_code=400,
            detail="'artist' and 'track' query parameters are required",
        )

    limit = max(1, min(limit, 10))
    query = f"artist:{artist} track:{track}"

    try:
        payload = _spotify_api_get(
            "/search",
            params={"q": query, "type": "track", "limit": limit},
        )
        items = payload.get("tracks", {}).get("items", []) or []
        tracks = [_normalize_spotify_track(item) for item in items]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Failed to search tracks on Spotify: {exc}",
        ) from exc

    return SpotifySearchResponse(items=tracks)


@app.get(
    "/spotify/candidate_from_url",
    response_model=Candidate,
    summary="Create a Candidate template from a Spotify URL",
)
def spotify_candidate_from_url(url: str) -> Candidate:
    """
    Resolve a Spotify URL to a Candidate template.
    """
    try:
        track_id = _extract_spotify_track_id(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        payload = _spotify_api_get(f"/tracks/{track_id}")
        track = _normalize_spotify_track(payload)
        candidate = spotify_track_to_candidate_template(track)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch/convert track from Spotify: {exc}",
        ) from exc

    return candidate


@app.get(
    "/spotify/candidate_from_id",
    response_model=Candidate,
    summary="Create a Candidate template from a Spotify Track ID",
)
def spotify_candidate_from_id(track_id: str) -> Candidate:
    """
    Resolve a Spotify Track ID to a Candidate template.
    """
    if not track_id:
        raise HTTPException(status_code=400, detail="track_id is required")

    try:
        payload = _spotify_api_get(f"/tracks/{track_id}")
        track = _normalize_spotify_track(payload)
        candidate = spotify_track_to_candidate_template(track)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch/convert track from Spotify: {exc}",
        ) from exc

    return candidate
