import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st

from utils import api_client

# --- Constants ---
GENRE_BUCKET_OPTIONS: List[str] = [
    "hip_hop_rap", "rnb_soul", "electronic_dance", "jazz", "classical_art",
    "folk_country_americana", "metal_hard", "rock", "pop", "latin",
    "world_regional", "experimental_avant", "unknown",
]

# --- State Management ---

def _init_session_state() -> None:
    if "backend_warmed_up" not in st.session_state:
        st.session_state["backend_warmed_up"] = False
    if "candidates" not in st.session_state:
        st.session_state["candidates"] = []
    if "selected_candidate_id" not in st.session_state:
        st.session_state["selected_candidate_id"] = None
    if "last_threshold" not in st.session_state:
        st.session_state["last_threshold"] = 0.5
    # Tracks which mode was run last to control visual visibility
    if "last_prediction_mode" not in st.session_state:
        st.session_state["last_prediction_mode"] = None  # Options: 'ranking', 'single', None

def _ensure_backend_warmup() -> None:
    if st.session_state.get("backend_warmed_up"):
        return
    try:
        api_client.healthcheck()
        st.session_state["backend_warmed_up"] = True
    except Exception:
        pass

def _can_call(name: str, cooldown_seconds: float = 1.0) -> bool:
    key = f"last_call_{name}"
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if now - last < cooldown_seconds:
        return False
    st.session_state[key] = now
    return True

# --- Candidate Logic ---

def _add_candidate(candidate: Dict[str, Any]) -> None:
    if not candidate.get("candidate_id"):
        candidate["candidate_id"] = str(uuid4())
    st.session_state["candidates"].append(candidate)
    st.session_state["selected_candidate_id"] = candidate["candidate_id"]

def _get_candidate_by_id(cid: str) -> Optional[Dict[str, Any]]:
    for cand in st.session_state["candidates"]:
        if cand.get("candidate_id") == cid:
            return cand
    return None

def _remove_candidate(cid: str) -> None:
    st.session_state["candidates"] = [
        c for c in st.session_state["candidates"] 
        if c.get("candidate_id") != cid
    ]
    if st.session_state["selected_candidate_id"] == cid:
        st.session_state["selected_candidate_id"] = None

def _remove_all_candidates() -> None:
    st.session_state["candidates"] = []
    st.session_state["selected_candidate_id"] = None
    st.session_state["last_prediction_mode"] = None # Reset mode on clear

def _duplicate_candidate(candidate: Dict[str, Any]) -> None:
    new_candidate = {**candidate}
    new_candidate["candidate_id"] = str(uuid4())
    new_candidate["source"] = "manual"
    if new_candidate.get("track_name"):
        new_candidate["track_name"] = f"{new_candidate['track_name']} (copy)"
    # Reset prediction results on duplicate
    for k in ["probability", "rank", "prediction", "above_threshold"]:
        new_candidate[k] = None
    _add_candidate(new_candidate)

def _build_manual_template() -> Dict[str, Any]:
    return {
        "candidate_id": str(uuid4()),
        "source": "manual",
        "track_name": "New Track",
        "artist_name": "New Artist",
        "week_start": None,
        "spotify_popularity": 50.0,
        "track_duration": 180.0,
        "genre_bucket": "pop",
        "scrobbles_week": 10.0,
        "unique_days_week": 3.0,
        "scrobbles_last_fri_sat": 5.0,
        "scrobbles_saturday": 2.0,
        "last_scrobble_gap_days": 0.0,
        "within_week_rank_by_scrobbles": 5.0,
        "scrobbles_prev_1w": 0.0,
        "scrobbles_prev_4w": 0.0,
        "week_over_week_change": 0.0,
        "momentum_4w_ratio": 0.0,
        "prior_scrobbles_all_time": 50.0,
        "first_seen_week": 0.0,
        "days_since_release": 100.0,
        "released_within_28d": 0.0,
        "probability": None,
        "rank": None,
        "prediction": None,
    }

def _merge_predictions(response: Dict[str, Any], mode: str) -> None:
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
            cand["_threshold_at_prediction"] = threshold

# --- UI Renderers ---

def render_add_candidate_section():
    st.markdown("### 1. Draft Candidates")
    
    with st.container(border=True):
        tab_spotify, tab_random, tab_fav, tab_manual = st.tabs([
            "🟢 Spotify URL", "🎲 Random Dataset", "❤️ Favorites", "📝 Manual"
        ])

        with tab_spotify:
            col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
            url = col1.text_input("Track URL", placeholder="http://spotify.com/...", label_visibility="collapsed")
            if col2.button("Fetch", key="btn_add_spotify", use_container_width=True):
                if url:
                    try:
                        c = api_client.get_spotify_candidate_from_url(url)
                        _add_candidate(c)
                        st.toast(f"Added: {c.get('track_name')}", icon="✅")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with tab_random:
            st.caption("Pull a random non-favorite song from your listening history.")
            if st.button("Add Random Track", key="btn_add_rnd"):
                if _can_call("ex_rnd"):
                    resp = api_client.get_random_examples(1)
                    if resp.get("candidates"):
                        c = resp["candidates"][0]["candidate"]
                        _add_candidate(c)
                        st.toast("Random candidate added", icon="🎲")

        with tab_fav:
            st.caption("Pull a historical 'Song of the Week' from your dataset.")
            if st.button("Add Favorite Track", key="btn_add_fav"):
                if _can_call("ex_fav"):
                    resp = api_client.get_favorite_examples(1)
                    if resp.get("candidates"):
                        c = resp["candidates"][0]["candidate"]
                        _add_candidate(c)
                        st.toast("Favorite candidate added", icon="❤️")

        with tab_manual:
            st.caption("Start from scratch.")
            if st.button("Create Blank Candidate", key="btn_add_man"):
                _add_candidate(_build_manual_template())
                st.toast("Manual candidate created", icon="📝")

def render_main_workspace():
    col_list, col_inspector = st.columns([1.8, 1.2], gap="large")

    with col_list:
        render_candidate_list()
        st.divider()
        render_controls()

    with col_inspector:
        render_inspector_panel()

def render_candidate_list():
    st.markdown("### 2. Candidate List")
    candidates = st.session_state["candidates"]
    last_mode = st.session_state.get("last_prediction_mode")
    
    if not candidates:
        st.info("List is empty. Add candidates above.")
        return

    # Build DF
    df_data = []
    for c in candidates:
        df_data.append({
            "id": c.get("candidate_id"),
            "Track": c.get("track_name", "Unknown"),
            "Artist": c.get("artist_name", "Unknown"),
            "Prob": c.get("probability", 0.0) if c.get("probability") is not None else None,
            "Rank": c.get("rank"),
        })
    
    df = pd.DataFrame(df_data)

    # Sort logic matches visual logic:
    # If Ranking Mode -> Sort by Rank
    # If Check Mode -> Sort by Prob (desc)
    if last_mode == "ranking" and "Rank" in df.columns and df["Rank"].notna().any():
        df = df.sort_values(by=["Rank", "Prob"], ascending=[True, False])
    
    # Configure Columns based on Mode
    column_config = {
        "id": None, 
        # Display with 2 decimal places
        "Prob": st.column_config.ProgressColumn(
            "Confidence", format="%.2f%%", min_value=0, max_value=1
        ),
    }

    # Only show Rank column if we are in Ranking mode
    if last_mode == "ranking":
        column_config["Rank"] = st.column_config.NumberColumn("Rank", format="%d")
    else:
        column_config["Rank"] = None # Hide column

    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row"
    )

    if event.selection.rows:
        selected_index = event.selection.rows[0]
        selected_row_id = df.iloc[selected_index]["id"]
        st.session_state["selected_candidate_id"] = selected_row_id

    if candidates:
        if st.button("Remove All", type="secondary"):
            _remove_all_candidates()
            st.rerun()

def render_controls():
    candidates = st.session_state["candidates"]
    if not candidates:
        return

    col_mode, col_btn = st.columns([1, 2], vertical_alignment="bottom")
    
    with col_mode:
        mode_select = st.selectbox(
            "Analysis Scope",
            options=["Rank All", "Check Selected"],
            help="Rank All: Compare all songs to find a winner. Check Selected: See if the selected song passes the threshold."
        )
    
    with col_btn:
        is_ranking = mode_select == "Rank All"
        btn_label = "🏆 RANK ALL" if is_ranking else "🔎 CHECK SELECTED"
        
        if st.button(btn_label, type="primary", use_container_width=True):
            _handle_prediction(mode_select)

def _handle_prediction(mode_ui_label: str):
    mode_api = "ranking" if mode_ui_label == "Rank All" else "single"
    candidates = st.session_state["candidates"]

    if mode_api == "single" and not st.session_state["selected_candidate_id"]:
        st.error("Select a candidate in the table first.")
        return

    if mode_api == "single":
        selected = _get_candidate_by_id(st.session_state["selected_candidate_id"])
        if not selected: 
            return
        payload = [selected]
    else:
        payload = candidates

    with st.spinner("Asking the model..."):
        try:
            resp = api_client.predict_candidates(payload, mode=mode_api)
            _merge_predictions(resp, mode_api)
            
            if mode_api == "ranking":
                st.toast("Ranking complete!", icon="🏆")
            else:
                st.toast("Check complete!", icon="🔎")
                
            st.rerun()
        except Exception as e:
            st.error(f"Prediction error: {e}")

def render_inspector_panel():
    st.markdown("### 3. Inspector")
    
    cid = st.session_state.get("selected_candidate_id")
    if not cid:
        st.caption("Select a song to view details.")
        return

    cand = _get_candidate_by_id(cid)
    if not cand:
        return

    # Header & Actions (Standard buttons)
    with st.container(border=True):
        st.subheader(cand.get("track_name", "Unknown"))
        st.caption(cand.get("artist_name", "Unknown"))
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("Duplicate", use_container_width=True):
                _duplicate_candidate(cand)
                st.rerun()
        with col_act2:
            if st.button("Remove", use_container_width=True, type="primary"):
                _remove_candidate(cid)
                st.rerun()

    # Result Card
    prob = cand.get("probability")
    last_mode = st.session_state.get("last_prediction_mode")

    if prob is not None:
        threshold = cand.get("_threshold_at_prediction") or st.session_state.get("last_threshold", 0.5)
        
        # Scenario A: Check Mode -> Show Pass/Fail Status
        if last_mode == "single":
            is_fav = prob >= threshold
            if is_fav:
                bg_color = "rgba(40, 167, 69, 0.1)" # Light Green
                border_color = "green"
                icon = "✅"
                status = "Predicted Favorite"
            else:
                bg_color = "rgba(108, 117, 125, 0.1)" # Light Gray
                border_color = "gray"
                icon = "⚪"
                status = "Not Favorite"

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="margin:0; padding:0;">{icon} {status}</h4>
                    <p style="margin:5px 0 0 0; font-size: 0.9em;">
                        Probability: <b>{prob*100:.2f}%</b> <span style="color:gray">vs Threshold: {threshold*100:.2f}%</span>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        
        # Scenario B: Ranking Mode -> Show Rank (if available)
        elif last_mode == "ranking":
            rank = cand.get("rank")
            rank_display = f"#{rank}" if rank else "N/A"
            # Highlight Winner
            if rank == 1:
                bg_color = "rgba(255, 193, 7, 0.1)" # Gold
                border_color = "gold"
                icon = "🏆"
                status = "Winner (Rank #1)"
            else:
                bg_color = "rgba(240, 242, 246, 0.5)"
                border_color = "lightgray"
                icon = "📊"
                status = f"Rank {rank_display}"

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="margin:0; padding:0;">{icon} {status}</h4>
                    <p style="margin:5px 0 0 0; font-size: 0.9em;">
                        Confidence: <b>{prob*100:.2f}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )

    # Edit Features (Grouped)
    st.markdown("**Feature Editor**")
    is_manual = cand.get("source") == "manual"
    
    # Track Properties
    with st.expander("Track Properties", expanded=True):
        if is_manual:
            cand["track_name"] = st.text_input("Track Name", cand.get("track_name"))
            cand["artist_name"] = st.text_input("Artist Name", cand.get("artist_name"))
        
        c1, c2 = st.columns(2)
        with c1:
            cand["spotify_popularity"] = st.slider("Popularity", 0, 100, int(cand.get("spotify_popularity", 0)))
            cand["track_duration"] = st.number_input("Duration (s)", value=int(cand.get("track_duration", 180)))
        with c2:
            cand["days_since_release"] = st.number_input("Days Since Release", value=int(cand.get("days_since_release", 0)))
            
            curr_genre = cand.get("genre_bucket", "unknown")
            opts = GENRE_BUCKET_OPTIONS if curr_genre in GENRE_BUCKET_OPTIONS else [curr_genre] + GENRE_BUCKET_OPTIONS
            cand["genre_bucket"] = st.selectbox("Genre", opts, index=opts.index(curr_genre))

    # Weekly Intensity
    with st.expander("Weekly Intensity", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            cand["scrobbles_week"] = st.number_input("Scrobbles (Week)", value=int(cand.get("scrobbles_week", 0)))
            cand["unique_days_week"] = st.number_input("Unique Days", 0, 7, int(cand.get("unique_days_week", 1)))
        with c2:
            cand["within_week_rank_by_scrobbles"] = st.number_input("Rank (in week)", min_value=1, value=int(cand.get("within_week_rank_by_scrobbles", 10)))
            cand["scrobbles_saturday"] = st.number_input("Scrobbles (Sat)", value=int(cand.get("scrobbles_saturday", 0)))
        
        cand["scrobbles_last_fri_sat"] = st.number_input("Scrobbles (Fri+Sat)", value=int(cand.get("scrobbles_last_fri_sat", 0)))

    # History & Trend
    with st.expander("History & Trends", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            cand["scrobbles_prev_1w"] = st.number_input("Scrobbles (Prev 1w)", value=int(cand.get("scrobbles_prev_1w", 0)))
            cand["scrobbles_prev_4w"] = st.number_input("Scrobbles (Prev 4w)", value=int(cand.get("scrobbles_prev_4w", 0)))
        with c2:
            cand["prior_scrobbles_all_time"] = st.number_input("All Time Scrobbles", value=int(cand.get("prior_scrobbles_all_time", 0)))
            cand["momentum_4w_ratio"] = st.number_input("Momentum (4w)", value=float(cand.get("momentum_4w_ratio", 0.0)))

        cand["week_over_week_change"] = st.number_input("WoW Change", value=float(cand.get("week_over_week_change", 0.0)))
        cand["last_scrobble_gap_days"] = st.number_input("Days since last play", value=float(cand.get("last_scrobble_gap_days", 0.0)))

def main():
    _init_session_state()
    _ensure_backend_warmup()

    st.title("Song Predictor 🔮")
    st.markdown("Build a weekly playlist, simulate features, and predict the favorite.")

    render_add_candidate_section()
    st.divider()
    render_main_workspace()

if __name__ == "__main__":
    main()