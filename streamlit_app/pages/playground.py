
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st

from utils import api_client


# ------------- Session state helpers ------------- #


def _init_session_state() -> None:
    if "backend_warmed_up" not in st.session_state:
        st.session_state["backend_warmed_up"] = False

    if "candidates" not in st.session_state:
        # List of dicts matching the Candidate model from the backend
        st.session_state["candidates"] = []

    if "selected_candidate_id" not in st.session_state:
        st.session_state["selected_candidate_id"] = None


def _ensure_backend_warmup() -> None:
    if st.session_state.get("backend_warmed_up"):
        return

    with st.spinner("Warming up backend (healthcheck)..."):
        try:
            api_client.healthcheck()
            st.session_state["backend_warmed_up"] = True
        except api_client.APIClientError as exc:
            st.error(f"Failed to reach backend: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error while warming up backend: {exc}")


def _can_call(name: str, cooldown_seconds: float = 1.0) -> bool:
    key = f"last_call_{name}"
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if now - last < cooldown_seconds:
        return False
    st.session_state[key] = now
    return True


# ------------- Candidate helpers ------------- #


def _add_candidate(candidate: Dict[str, Any]) -> None:
    if not candidate.get("candidate_id"):
        candidate["candidate_id"] = str(uuid4())
    st.session_state["candidates"].append(candidate)
    st.session_state["selected_candidate_id"] = candidate["candidate_id"]


def _get_selected_candidate() -> Optional[Dict[str, Any]]:
    cid = st.session_state.get("selected_candidate_id")
    if not cid:
        return None
    for cand in st.session_state["candidates"]:
        if cand.get("candidate_id") == cid:
            return cand
    return None


def _duplicate_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    new_candidate = {**candidate}
    new_candidate["candidate_id"] = str(uuid4())
    # Optionally tweak the track_name to indicate duplication
    if new_candidate.get("track_name"):
        new_candidate["track_name"] = f"{new_candidate['track_name']} (copy)"
    return new_candidate

def _clear_candidates() -> None:
    """
    Remove all candidates and reset the current selection.
    """
    st.session_state["candidates"] = []
    st.session_state["selected_candidate_id"] = None

def _merge_predictions_into_candidates(prediction_response: Dict[str, Any]) -> None:
    # Results indexed by candidate_id if available, otherwise by index
    results = prediction_response.get("results", [])
    by_id: Dict[str, Dict[str, Any]] = {}
    by_index: Dict[int, Dict[str, Any]] = {}

    for res in results:
        cid = res.get("candidate_id")
        idx = int(res["index"])
        if cid:
            by_id[cid] = res
        by_index[idx] = res

    for idx, cand in enumerate(st.session_state["candidates"]):
        cid = cand.get("candidate_id")
        res = by_id.get(cid) or by_index.get(idx)
        if not res:
            continue

        cand["probability"] = res.get("probability")
        cand["prediction"] = res.get("prediction")
        cand["above_threshold"] = res.get("above_threshold")
        cand["rank"] = res.get("rank")


# ------------- UI building blocks ------------- #


def _render_mode_selector() -> str:
    mode_label = st.radio(
        "Prediction mode",
        options=["Ranking (Mode A)", "Single-song check (Mode B)"],
        index=0,
        help=(
            "Mode A: rank all candidates and pick a winner. "
            "Mode B: check whether a single candidate would be a favourite."
        ),
    )
    if mode_label.startswith("Ranking"):
        return "ranking"
    return "single"


def _render_candidates_table() -> None:
    candidates = st.session_state["candidates"]
    if not candidates:
        st.info("No candidates yet. Add one using the actions on the right.")
        return

    display_rows: List[Dict[str, Any]] = []
    for cand in candidates:
        display_rows.append(
            {
                "candidate_id": cand.get("candidate_id"),
                "source": cand.get("source"),
                "track_name": cand.get("track_name"),
                "artist_name": cand.get("artist_name"),
                "probability": cand.get("probability"),
                "rank": cand.get("rank"),
            }
        )

    df = pd.DataFrame(display_rows)

    # If ranks exist, sort by rank; otherwise leave order as-is
    if "rank" in df.columns and df["rank"].notna().any():
        df = df.sort_values(["rank", "probability"], ascending=[True, False])

    st.subheader("Candidates")
    st.dataframe(df, width="stretch")

    # Selection widget
    candidate_map = {
        cand.get("candidate_id"): (
            f"{cand.get('track_name') or 'Unknown track'}"
            f" - {cand.get('artist_name') or 'Unknown artist'}"
        )
        for cand in candidates
    }
    candidate_ids = list(candidate_map.keys())

    if not candidate_ids:
        return

    selected_id = st.selectbox(
        "Selected candidate (for editing and Mode B)",
        options=candidate_ids,
        format_func=lambda cid: candidate_map.get(cid, cid),
        index=0 if st.session_state.get("selected_candidate_id") not in candidate_ids else
        candidate_ids.index(st.session_state["selected_candidate_id"]),
    )
    st.session_state["selected_candidate_id"] = selected_id


def _render_candidate_details() -> None:
    cand = _get_selected_candidate()
    st.subheader("Candidate details")

    if not cand:
        st.info("Select a candidate to see details.")
        return

    # For now, just show the raw dict. We can later replace this with a proper form.
    st.json(cand)


def _handle_add_from_spotify_url() -> None:
    st.subheader("Add candidate from Spotify URL")

    spotify_url = st.text_input(
        "Spotify track URL or URI",
        placeholder="https://open.spotify.com/track/...",
        key="spotify_url_input",
    )
    if st.button("Fetch from Spotify"):
        if not spotify_url.strip():
            st.warning("Please enter a Spotify URL first.")
            return

        if not _can_call("spotify", cooldown_seconds=1.0):
            st.warning("Please wait a moment before making another Spotify request.")
            return

        try:
            candidate = api_client.get_spotify_candidate_from_url(spotify_url)
        except api_client.APIClientError as exc:
            st.error(f"Spotify API error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while calling Spotify: {exc}")
            return

        _add_candidate(candidate)
        st.success("Candidate added from Spotify.")


def _handle_add_random_example() -> None:
    st.subheader("Add random candidate from dataset")

    if st.button("Add random candidate"):
        if not _can_call("examples", cooldown_seconds=1.0):
            st.warning("Please wait a moment before requesting more examples.")
            return

        try:
            resp = api_client.get_random_examples(count=1)
        except api_client.APIClientError as exc:
            st.error(f"Examples API error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while fetching examples: {exc}")
            return

        items = resp.get("candidates", [])
        if not items:
            st.warning("No examples returned by the backend.")
            return

        example = items[0]
        candidate = example.get("candidate") or {}
        _add_candidate(candidate)
        st.success("Random candidate added from dataset.")

def _handle_add_favorite_example() -> None:
    st.subheader("Add candidate from favourite songs")

    if st.button("Add favourite candidate"):
        if not _can_call("examples", cooldown_seconds=1.0):
            st.warning("Please wait a moment before requesting more examples.")
            return

        try:
            resp = api_client.get_favorite_examples(count=1)
        except api_client.APIClientError as exc:
            st.error(f"Examples API error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while fetching favourite examples: {exc}")
            return

        items = resp.get("candidates", [])
        if not items:
            st.warning("No favourite examples returned by the backend.")
            return

        example = items[0]
        candidate = example.get("candidate") or {}
        _add_candidate(candidate)
        st.success("Favourite candidate added from dataset.")


def _handle_duplicate_candidate() -> None:
    st.subheader("What-if: duplicate candidate")

    cand = _get_selected_candidate()
    if cand is None:
        st.info("Select a candidate first to duplicate it.")
        return

    if st.button("Duplicate selected candidate"):
        new_cand = _duplicate_candidate(cand)
        _add_candidate(new_cand)
        st.success("Candidate duplicated. You can now tweak it and re-run predictions.")


def _handle_predict(mode: str) -> None:
    candidates = st.session_state["candidates"]
    if not candidates:
        st.warning("Add at least one candidate before predicting.")
        return

    if mode == "single":
        cand = _get_selected_candidate()
        if cand is None:
            st.warning("Select a candidate first for Mode B (single-song check).")
            return
        payload_candidates = [cand]
    else:
        payload_candidates = candidates

    with st.spinner("Running prediction..."):
        try:
            resp = api_client.predict_candidates(
                candidates=payload_candidates,
                mode=mode,
            )
        except api_client.APIClientError as exc:
            st.error(f"Prediction API error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error during prediction: {exc}")
            return

    # Merge predictions back into the full candidate list.
    # For single mode, the backend returns a response for the one candidate only.
    if mode == "single":
        _merge_predictions_into_candidates_single(resp)
    else:
        _merge_predictions_into_candidates(resp)

    # Show a short summary for Mode B
    if mode == "single":
        _render_single_mode_summary(resp)
    else:
        st.success("Predictions updated. Check the candidates table for ranks and probabilities.")


def _merge_predictions_into_candidates_single(prediction_response: Dict[str, Any]) -> None:
    results = prediction_response.get("results", [])
    if not results:
        return

    res = results[0]
    cid = res.get("candidate_id")
    idx = int(res["index"])

    # Update matching candidate (by id or by index)
    for i, cand in enumerate(st.session_state["candidates"]):
        if (cid and cand.get("candidate_id") == cid) or (i == idx):
            cand["probability"] = res.get("probability")
            cand["prediction"] = res.get("prediction")
            cand["above_threshold"] = res.get("above_threshold")
            cand["rank"] = res.get("rank")
            break


def _render_single_mode_summary(prediction_response: Dict[str, Any]) -> None:
    results = prediction_response.get("results", [])
    if not results:
        return

    res = results[0]
    threshold = prediction_response.get("threshold")
    prob = res.get("probability")
    is_fav = bool(res.get("prediction"))

    if is_fav:
        st.success(
            f"✅ Model would treat this song as a *favourite* "
            f"(p = {prob:.3f}, threshold = {threshold:.3f})."
        )
    else:
        st.info(
            f"ℹ️ Model would **not** treat this song as a favourite "
            f"(p = {prob:.3f}, threshold = {threshold:.3f})."
        )


# ------------- Main page ------------- #


def main() -> None:
    st.title("SOTW Playground - Candidate Builder")

    _init_session_state()
    _ensure_backend_warmup()

    mode = _render_mode_selector()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        _render_candidates_table()
        _render_candidate_details()

    with col_right:
        st.header("Add candidates")
        _handle_add_from_spotify_url()
        st.divider()
        _handle_add_random_example()
        st.divider()
        _handle_add_favorite_example()
        st.divider()
        _handle_duplicate_candidate()
        st.divider()

        st.header("Predict")
        if st.button(
            "Predict",
            help=(
                "In Ranking mode, scores all candidates and picks a winner. "
                "In Single mode, checks only the selected candidate."
            ),
        ):
            _handle_predict(mode)

        # Clear all candidates in the current session
        st.button(
            "Clear candidates",
            help="Remove all candidates and reset selection.",
            on_click=_clear_candidates,
        )


if __name__ == "__main__":
    main()
