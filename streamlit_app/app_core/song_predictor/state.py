import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import streamlit as st

# --- Constants ---
GENRE_BUCKET_OPTIONS: List[str] = [
    "hip_hop_rap", "rnb_soul", "electronic_dance", "jazz", "classical_art",
    "folk_country_americana", "metal_hard", "rock", "pop", "latin",
    "world_regional", "experimental_avant", "unknown",
]
COOLDOWN_SECONDS: float = 1.0

# --- State Management ---

def init_session_state() -> None:
    if "backend_warmed_up" not in st.session_state:
        st.session_state["backend_warmed_up"] = False
    if "candidates" not in st.session_state:
        st.session_state["candidates"] = []
    if "selected_candidate_id" not in st.session_state:
        st.session_state["selected_candidate_id"] = None
    if "last_threshold" not in st.session_state:
        st.session_state["last_threshold"] = 0.5
    if "last_prediction_mode" not in st.session_state:
        st.session_state["last_prediction_mode"] = None

def can_call(name: str) -> bool:
    key = f"last_call_{name}"
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if now - last < COOLDOWN_SECONDS:
        st.warning("Please wait a moment before making another request.")
        return False
    st.session_state[key] = now
    return True

# --- Candidate Accessors ---

def get_candidates() -> List[Dict[str, Any]]:
    return st.session_state["candidates"]

def get_selected_candidate() -> Optional[Dict[str, Any]]:
    cid = st.session_state.get("selected_candidate_id")
    if cid:
        return get_candidate_by_id(cid)
    return None

def get_selected_candidate_id() -> Optional[str]:
    return st.session_state.get("selected_candidate_id")
    
def get_candidate_by_id(cid: str) -> Optional[Dict[str, Any]]:
    for cand in st.session_state["candidates"]:
        if cand.get("candidate_id") == cid:
            return cand
    return None

def get_last_prediction_mode() -> Optional[str]:
    return st.session_state.get("last_prediction_mode")

def get_last_threshold() -> float:
    return st.session_state.get("last_threshold", 0.5)

# --- Candidate Mutators ---

def add_candidate(candidate: Dict[str, Any]) -> None:
    if not candidate.get("candidate_id"):
        candidate["candidate_id"] = str(uuid4())
    st.session_state["candidates"].append(candidate)
    st.session_state["selected_candidate_id"] = candidate["candidate_id"]

def remove_candidate(cid: str) -> None:
    st.session_state["candidates"] = [
        c for c in st.session_state["candidates"] 
        if c.get("candidate_id") != cid
    ]
    if st.session_state["selected_candidate_id"] == cid:
        st.session_state["selected_candidate_id"] = None

def remove_all_candidates() -> None:
    st.session_state["candidates"] = []
    st.session_state["selected_candidate_id"] = None
    st.session_state["last_prediction_mode"] = None

def duplicate_candidate(candidate: Dict[str, Any]) -> None:
    new_candidate = {**candidate}
    new_candidate["candidate_id"] = str(uuid4())
    new_candidate["source"] = "manual"
    if new_candidate.get("track_name"):
        new_candidate["track_name"] = f"{new_candidate['track_name']} (copy)"
    # Reset prediction results on duplicate
    for k in ["probability", "rank", "prediction", "above_threshold"]:
        new_candidate[k] = None
    add_candidate(new_candidate)

def build_manual_template() -> Dict[str, Any]:
    """Returns a candidate with all feature fields initialized to safe defaults."""
    return {
        "candidate_id": str(uuid4()),
        "source": "manual",
        "track_name": "New Track",
        "artist_name": "New Artist",
        "week_start": None,
        # Features
        "spotify_popularity": 50,
        "track_duration": 180,
        "genre_bucket": "pop",
        "scrobbles_week": 0,
        "unique_days_week": 0,
        "scrobbles_last_fri_sat": 0,
        "scrobbles_saturday": 0,
        "last_scrobble_gap_days": 0,
        "within_week_rank_by_scrobbles": 5,
        "scrobbles_prev_1w": 0,
        "scrobbles_prev_4w": 0,
        "week_over_week_change": 0,
        "momentum_4w_ratio": 0.0,
        "prior_scrobbles_all_time": 50,
        "first_seen_week": 0,
        "days_since_release": 100,
        "released_within_28d": 0,
        # Outputs
        "probability": None,
        "rank": None,
        "prediction": None,
    }

def update_results(response: Dict[str, Any], mode: str) -> None:
    """Merges the API prediction results back into the session candidates."""
    results = response.get("results", [])
    threshold = response.get("threshold", 0.5)
    st.session_state["last_threshold"] = threshold
    st.session_state["last_prediction_mode"] = mode

    res_map = {r.get("candidate_id"): r for r in results if r.get("candidate_id")}
    res_list = results 

    for idx, cand in enumerate(st.session_state["candidates"]):
        cid = cand.get("candidate_id")
        match = res_map.get(cid)
        
        # In Ranking mode, if IDs fail, we fallback to list index
        if not match and idx < len(res_list) and mode == "ranking":
            match = res_list[idx]

        if match:
            cand["probability"] = match.get("probability")
            cand["prediction"] = match.get("prediction")
            cand["above_threshold"] = match.get("above_threshold")
            cand["rank"] = match.get("rank")
            # Store threshold locally in candidate for Inspector display
            cand["_threshold_at_prediction"] = threshold