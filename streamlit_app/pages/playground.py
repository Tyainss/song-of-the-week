
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st

from utils import api_client

GENRE_BUCKET_OPTIONS: List[str] = [
    "hip_hop_rap",
    "rnb_soul",
    "electronic_dance",
    "jazz",
    "classical_art",
    "folk_country_americana",
    "metal_hard",
    "rock",
    "pop",
    "latin",
    "world_regional",
    "experimental_avant",
    "unknown",
]

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
    """
    Duplicate an existing candidate for what-if experimentation.

    - Assigns a new candidate_id.
    - Marks the duplicate as a manual candidate so all fields are editable.
    - Optionally appends "(copy)" to the track name for clarity.
    """
    new_candidate = {**candidate}
    new_candidate["candidate_id"] = str(uuid4())
    new_candidate["source"] = "manual"
    if new_candidate.get("track_name"):
        new_candidate["track_name"] = f"{new_candidate['track_name']} (copy)"
    return new_candidate

def _build_manual_candidate_template() -> Dict[str, Any]:
    """
    Create a blank/manual candidate with all required model fields.

    The user can then edit this via the Candidate details form.
    """
    return {
        "candidate_id": str(uuid4()),
        "source": "manual",
        "track_name": "",
        "artist_name": "",
        "spotify_track_id": None,
        "week_start": None,
        # Model features (pre-DV/OHE)
        "spotify_popularity": 0.0,
        "track_duration": 0.0,
        "scrobbles_week": 0.0,
        "unique_days_week": 0.0,
        "scrobbles_last_fri_sat": 0.0,
        "scrobbles_saturday": 0.0,
        "last_scrobble_gap_days": 0.0,
        "within_week_rank_by_scrobbles": 1.0,
        "scrobbles_prev_1w": 0.0,
        "scrobbles_prev_4w": 0.0,
        "week_over_week_change": 0.0,
        "momentum_4w_ratio": 0.0,
        "prior_scrobbles_all_time": 0.0,
        "first_seen_week": 0.0,
        "days_since_release": 0.0,
        "released_within_28d": 0.0,
        "genre_bucket": "unknown",
        # Optional prediction fields
        "probability": None,
        "rank": None,
        "prediction": None,
        "above_threshold": None,
    }

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
                "source": cand.get("source"),
                "track_name": cand.get("track_name"),
                "artist_name": cand.get("artist_name"),
                "probability": cand.get("probability"),
                "rank": cand.get("rank"),
                "prediction": cand.get("prediction"),
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
        index=0
        if st.session_state.get("selected_candidate_id") not in candidate_ids
        else candidate_ids.index(st.session_state["selected_candidate_id"]),
    )
    st.session_state["selected_candidate_id"] = selected_id


def _render_candidate_details() -> None:
    cand = _get_selected_candidate()
    st.subheader("Candidate details")

    if not cand:
        st.info("Select a candidate to see details.")
        return

    source = cand.get("source") or "manual"
    is_manual = source == "manual"
    # Originals from Spotify / examples keep identity fields fixed
    identity_locked = source in {"spotify", "random_example", "favourite_example"} and not is_manual

    track_label = cand.get("track_name") or "Unknown track"
    artist_label = cand.get("artist_name") or "Unknown artist"

    # Summary title
    st.markdown(f"**{track_label}** - {artist_label}")

    # ---------------- Prediction KPIs ---------------- #
    st.markdown("**Prediction summary**")

    cols_meta = st.columns(3)
    probability = cand.get("probability")
    rank = cand.get("rank")
    prediction = cand.get("prediction")
    above_threshold = cand.get("above_threshold")

    with cols_meta[0]:
        if probability is not None:
            st.metric("Probability", f"{float(probability):.1%}")
        else:
            st.caption("Probability: N/A")

    with cols_meta[1]:
        if rank is not None:
            st.metric("Rank", f"{int(rank)}")
        else:
            st.caption("Rank: N/A")

    with cols_meta[2]:
        if prediction is not None and above_threshold is not None:
            label = "Favourite" if int(prediction) == 1 else "Not favourite"
            st.metric("Prediction", label)
        else:
            st.caption("Prediction: N/A")

    # Visual separator between model output vs raw features
    st.divider()

    # ---------------- Behaviour KPIs ---------------- #
    st.markdown("**Behaviour snapshot**")

    # Row 1: popularity + core weekly intensity
    cols_feats_top = st.columns(3)
    with cols_feats_top[0]:
        val = cand.get("spotify_popularity")
        st.metric(
            "Spotify popularity",
            f"{int(val)}" if val is not None else "N/A",
        )
    with cols_feats_top[1]:
        val = cand.get("scrobbles_week")
        st.metric(
            "Scrobbles this week",
            f"{int(val)}" if val is not None else "N/A",
        )
    with cols_feats_top[2]:
        val = cand.get("unique_days_week")
        st.metric(
            "Unique days this week",
            f"{int(val)}" if val is not None else "N/A",
        )

    # Row 2: weekend focus + within-week rank + long-term familiarity
    cols_feats_bottom = st.columns(3)
    with cols_feats_bottom[0]:
        val = cand.get("scrobbles_last_fri_sat")
        st.metric(
            "Scrobbles last Fri+Sat",
            f"{int(val)}" if val is not None else "N/A",
        )
    with cols_feats_bottom[1]:
        val = cand.get("within_week_rank_by_scrobbles")
        st.metric(
            "Within-week rank by scrobbles",
            f"{int(val)}" if val is not None else "N/A",
        )
    with cols_feats_bottom[2]:
        val = cand.get("prior_scrobbles_all_time")
        st.metric(
            "Prior scrobbles (all time)",
            f"{int(val)}" if val is not None else "N/A",
        )

    # ---------------- Edit form ---------------- #
    expand_default = is_manual and probability is None
    with st.expander("Edit candidate", expanded=expand_default):
        form_key = f"candidate_form_{cand.get('candidate_id', 'unknown')}"
        with st.form(key=form_key):
            # --- Metadata ---
            if identity_locked:
                st.markdown("**Metadata (fixed from source)**")
                st.caption(f"Track: **{track_label}**")
                st.caption(f"Artist: **{artist_label}**")
                track_name = cand.get("track_name") or ""
                artist_name = cand.get("artist_name") or ""
            else:
                st.markdown("**Metadata**")
                track_name = st.text_input(
                    "Track name",
                    value=cand.get("track_name") or "",
                )
                artist_name = st.text_input(
                    "Artist name",
                    value=cand.get("artist_name") or "",
                )

            # --- Track attributes ---
            st.markdown("**Track attributes**")
            col_attr1, col_attr2, col_attr3 = st.columns(3)
            with col_attr1:
                spotify_popularity = st.number_input(
                    "Spotify popularity",
                    min_value=0,
                    max_value=100,
                    value=int(cand.get("spotify_popularity") or 0),
                    step=1,
                )
            with col_attr2:
                track_duration = st.number_input(
                    "Track duration (seconds)",
                    min_value=0,
                    max_value=2000,
                    value=int(cand.get("track_duration") or 0.0),
                    step=1,
                )
            with col_attr3:
                current_genre = cand.get("genre_bucket") or "unknown"
                options = list(GENRE_BUCKET_OPTIONS)
                if current_genre not in options:
                    options = [current_genre] + [g for g in options if g != current_genre]

                genre_bucket = st.selectbox(
                    "Genre bucket",
                    options=options,
                    index=options.index(current_genre),
                )

            # --- Current week behaviour ---
            st.markdown("**Current week behaviour**")
            col_cw1, col_cw2, col_cw3 = st.columns(3)
            with col_cw1:
                scrobbles_week = st.number_input(
                    "Scrobbles this week",
                    min_value=0,
                    value=int(cand.get("scrobbles_week") or 0),
                    step=1,
                )
                unique_days_week = st.number_input(
                    "Unique days this week",
                    min_value=0,
                    value=int(cand.get("unique_days_week") or 0),
                    step=1,
                )
            with col_cw2:
                scrobbles_last_fri_sat = st.number_input(
                    "Scrobbles last Fri+Sat",
                    min_value=0,
                    value=int(cand.get("scrobbles_last_fri_sat") or 0),
                    step=1,
                )
                scrobbles_saturday = st.number_input(
                    "Scrobbles on Saturday",
                    min_value=0,
                    value=int(cand.get("scrobbles_saturday") or 0),
                    step=1,
                )
            with col_cw3:
                last_scrobble_gap_days = st.number_input(
                    "Gap since last scrobble (days)",
                    min_value=0.0,
                    value=float(cand.get("last_scrobble_gap_days") or 0.0),
                    step=1.0,
                )
                within_week_rank_by_scrobbles = st.number_input(
                    "Within-week rank by scrobbles",
                    min_value=1,
                    value=int(cand.get("within_week_rank_by_scrobbles") or 1),
                    step=1,
                )

            # --- Recent history (1-4 weeks) ---
            st.markdown("**Recent history (1-4 weeks)**")
            col_rh1, col_rh2 = st.columns(2)
            with col_rh1:
                scrobbles_prev_1w = st.number_input(
                    "Scrobbles previous 1 week",
                    min_value=0,
                    value=int(cand.get("scrobbles_prev_1w") or 0),
                    step=1,
                )
                momentum_4w_ratio = st.number_input(
                    "Momentum (4w ratio)",
                    min_value=0.0,
                    value=float(cand.get("momentum_4w_ratio") or 0.0),
                    step=0.1,
                )
            with col_rh2:
                scrobbles_prev_4w = st.number_input(
                    "Scrobbles previous 4 weeks",
                    min_value=0,
                    value=int(cand.get("scrobbles_prev_4w") or 0),
                    step=1,
                )
                week_over_week_change = st.number_input(
                    "Week-over-week change",
                    value=float(cand.get("week_over_week_change") or 0.0),
                    step=1.0,
                )

            # --- Long-term history and freshness ---
            st.markdown("**Long-term history and freshness**")
            col_lt1, col_lt2 = st.columns(2)
            with col_lt1:
                prior_scrobbles_all_time = st.number_input(
                    "Prior scrobbles (all time)",
                    min_value=0,
                    value=int(cand.get("prior_scrobbles_all_time") or 0),
                    step=1,
                )
                released_within_28d = st.number_input(
                    "Released within 28 days (0/1)",
                    min_value=0,
                    max_value=1,
                    value=int(cand.get("released_within_28d") or 0),
                    step=1,
                )
            with col_lt2:
                first_seen_week = st.number_input(
                    "First seen week (index)",
                    min_value=0,
                    value=int(cand.get("first_seen_week") or 0),
                    step=1,
                )
                days_since_release = st.number_input(
                    "Days since release",
                    min_value=0,
                    value=int(cand.get("days_since_release") or 0),
                    step=1,
                )

            submitted = st.form_submit_button("Save candidate")

            if submitted:
                cand.update(
                    {
                        "track_name": track_name or "",
                        "artist_name": artist_name or "",
                        "spotify_popularity": spotify_popularity,
                        "track_duration": track_duration,
                        "scrobbles_week": scrobbles_week,
                        "unique_days_week": unique_days_week,
                        "scrobbles_last_fri_sat": scrobbles_last_fri_sat,
                        "scrobbles_saturday": scrobbles_saturday,
                        "last_scrobble_gap_days": last_scrobble_gap_days,
                        "within_week_rank_by_scrobbles": within_week_rank_by_scrobbles,
                        "scrobbles_prev_1w": scrobbles_prev_1w,
                        "scrobbles_prev_4w": scrobbles_prev_4w,
                        "week_over_week_change": week_over_week_change,
                        "momentum_4w_ratio": momentum_4w_ratio,
                        "prior_scrobbles_all_time": prior_scrobbles_all_time,
                        "first_seen_week": first_seen_week,
                        "days_since_release": days_since_release,
                        "released_within_28d": released_within_28d,
                        "genre_bucket": genre_bucket or "unknown",
                    }
                )
                st.success("Candidate updated. Re-run predictions to see the impact.")




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


def _handle_add_manual_candidate() -> None:
    st.subheader("Add manual candidate")

    if st.button("Add blank candidate"):
        candidate = _build_manual_candidate_template()
        _add_candidate(candidate)
        st.success(
            "Manual candidate added. Use the details panel on the left to edit it."
        )


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

    # IMPORTANT: handle actions first (right column),
    # then render the table (left column) so new candidates show up immediately.
    with col_right:
        st.header("Add candidates")
        _handle_add_from_spotify_url()
        st.divider()
        _handle_add_random_example()
        st.divider()
        _handle_add_favorite_example()
        st.divider()
        _handle_add_manual_candidate()
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

    with col_left:
        _render_candidates_table()
        _render_candidate_details()


if __name__ == "__main__":
    main()
