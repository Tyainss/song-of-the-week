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

def _ensure_backend_warmup() -> None:
    if st.session_state.get("backend_warmed_up"):
        return
    try:
        api_client.healthcheck()
        st.session_state["backend_warmed_up"] = True
    except Exception:
        # Fail silently or show a small toast, don't block UI
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
    st.session_state["selected_candidate_id"] = candidate["candidate_id"] # Auto-select new

def _get_candidate_by_id(cid: str) -> Optional[Dict[str, Any]]:
    for cand in st.session_state["candidates"]:
        if cand.get("candidate_id") == cid:
            return cand
    return None

def _delete_candidate(cid: str) -> None:
    st.session_state["candidates"] = [
        c for c in st.session_state["candidates"] 
        if c.get("candidate_id") != cid
    ]
    if st.session_state["selected_candidate_id"] == cid:
        st.session_state["selected_candidate_id"] = None

def _duplicate_candidate(candidate: Dict[str, Any]) -> None:
    new_candidate = {**candidate}
    new_candidate["candidate_id"] = str(uuid4())
    new_candidate["source"] = "manual"
    if new_candidate.get("track_name"):
        new_candidate["track_name"] = f"{new_candidate['track_name']} (copy)"
    _add_candidate(new_candidate)

def _build_manual_template() -> Dict[str, Any]:
    return {
        "candidate_id": str(uuid4()),
        "source": "manual",
        "track_name": "New Track",
        "artist_name": "New Artist",
        "spotify_popularity": 50.0,
        "genre_bucket": "pop",
        # Default zeroed features
        "track_duration": 180.0,
        "scrobbles_week": 0.0, "unique_days_week": 0.0,
        "scrobbles_last_fri_sat": 0.0, "scrobbles_saturday": 0.0,
        "last_scrobble_gap_days": 0.0, "within_week_rank_by_scrobbles": 10.0,
        "scrobbles_prev_1w": 0.0, "scrobbles_prev_4w": 0.0,
        "week_over_week_change": 0.0, "momentum_4w_ratio": 0.0,
        "prior_scrobbles_all_time": 0.0, "first_seen_week": 0.0,
        "days_since_release": 0.0, "released_within_28d": 0.0,
    }

def _merge_predictions(response: Dict[str, Any], mode: str) -> None:
    results = response.get("results", [])
    
    # Map results by ID for easy lookup
    res_map = {r.get("candidate_id"): r for r in results if r.get("candidate_id")}
    # Fallback by index if IDs missing
    res_list = results 

    for idx, cand in enumerate(st.session_state["candidates"]):
        cid = cand.get("candidate_id")
        
        # If in single mode, we only update the specific candidate that was sent
        # The backend usually returns just that one result.
        
        match = res_map.get(cid)
        if not match and idx < len(res_list):
            # Only fallback to index if we are confident the list order matches (Ranking mode)
            if mode == "ranking":
                match = res_list[idx]

        if match:
            cand["probability"] = match.get("probability")
            cand["prediction"] = match.get("prediction")
            cand["above_threshold"] = match.get("above_threshold")
            cand["rank"] = match.get("rank")

# --- UI Components ---

def render_add_candidate_section():
    """A clean, tabbed interface for adding candidates."""
    st.markdown("### 1. Draft Candidates")
    
    with st.container(border=True):
        tab_spotify, tab_random, tab_fav, tab_manual = st.tabs([
            "🟢 Spotify URL", "🎲 Random Dataset", "❤️ Favorites", "📝 Manual"
        ])

        with tab_spotify:
            col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
            url = col1.text_input("Track URL", placeholder="https://open.spotify.com/track/...", label_visibility="collapsed")
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
    """Split view: Dataframe (Left) + Inspector (Right)."""
    
    col_list, col_inspector = st.columns([1.8, 1.2], gap="large")

    with col_list:
        st.markdown("### 2. Candidate List")
        
        candidates = st.session_state["candidates"]
        
        if not candidates:
            st.info("The list is empty. Add candidates above to get started.")
        else:
            # Prepare DataFrame for display
            df_data = []
            for c in candidates:
                df_data.append({
                    "id": c.get("candidate_id"),
                    "Track": c.get("track_name", "Unknown"),
                    "Artist": c.get("artist_name", "Unknown"),
                    "Prob": c.get("probability", 0.0) if c.get("probability") is not None else None,
                    "Rank": c.get("rank"),
                    "Fav?": "⭐" if c.get("prediction") == 1 else ""
                })
            
            df = pd.DataFrame(df_data)

            # Sort: Ranks first, then nulls
            if "Rank" in df.columns and df["Rank"].notna().any():
                df = df.sort_values(by=["Rank", "Prob"], ascending=[True, False])

            # Configure Columns
            column_config = {
                "id": None, # Hide ID
                "Prob": st.column_config.ProgressColumn(
                    "Confidence", 
                    help="Model probability", 
                    format="%.1f%%",
                    min_value=0, 
                    max_value=1
                ),
                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            }

            event = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
                on_select="rerun", # Interactive selection!
                selection_mode="single-row"
            )

            # Sync selection from DF to Session State
            if event.selection.rows:
                selected_index = event.selection.rows[0]
                # Map back to ID - careful with sorted DF
                # We need to find the ID in the dataframe row that was selected
                selected_row_id = df.iloc[selected_index]["id"]
                st.session_state["selected_candidate_id"] = selected_row_id
            
            # --- Prediction Control Bar ---
            st.markdown("---")
            render_prediction_controls(candidates)


    with col_inspector:
        render_inspector_panel()

def render_prediction_controls(candidates: List[Dict]):
    """The central action area."""
    
    col_mode, col_btn = st.columns([1, 2], vertical_alignment="bottom")
    
    with col_mode:
        # Simplified mode selector
        # Ranking = Mode A, Single Check = Mode B
        mode_select = st.selectbox(
            "Analysis Scope",
            options=["Rank All", "Check Selected"],
            help="Rank All: Compare everyone. Check Selected: Only analyze the track highlighted in the list."
        )
    
    with col_btn:
        # Visual prominence
        is_ranking = mode_select == "Rank All"
        
        if st.button(
            "✨ RUN PREDICTIONS" if is_ranking else "🔎 CHECK SELECTED", 
            type="primary", 
            use_container_width=True
        ):
            mode_api = "ranking" if is_ranking else "single"
            
            # Validation
            if mode_api == "single" and not st.session_state["selected_candidate_id"]:
                st.error("Please select a candidate in the table first.")
                return

            candidates_to_send = candidates
            if mode_api == "single":
                selected = _get_candidate_by_id(st.session_state["selected_candidate_id"])
                if not selected:
                    st.error("Selected candidate not found.")
                    return
                candidates_to_send = [selected]

            with st.spinner("Asking the model..."):
                try:
                    resp = api_client.predict_candidates(candidates_to_send, mode=mode_api)
                    _merge_predictions(resp, mode_api)
                    st.rerun() # Refresh table to show results
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Top Winner Highlight (Only if Ranking was run and we have a rank 1)
    # Check if we have results
    best_cand = None
    for c in candidates:
        if c.get("rank") == 1:
            best_cand = c
            break
            
    if best_cand:
        st.success(f"🏆 Current Winner: **{best_cand['track_name']}** by {best_cand['artist_name']} ({float(best_cand.get('probability',0))*100:.1f}%)")


def render_inspector_panel():
    """Right-hand side detail view."""
    st.markdown("### 3. Inspector")
    
    cid = st.session_state.get("selected_candidate_id")
    if not cid:
        st.caption("Select a song from the list to view details, edit features, or delete it.")
        return

    cand = _get_candidate_by_id(cid)
    if not cand:
        return # Should not happen

    # Header Card
    with st.container(border=True):
        st.subheader(cand.get("track_name", "Unknown"))
        st.markdown(f"**{cand.get('artist_name', 'Unknown')}**")
        
        # Toolbar
        col_t1, col_t2 = st.columns(2)
        if col_t1.button("Duplicate", use_container_width=True):
            _duplicate_candidate(cand)
            st.rerun()
            
        if col_t2.button("Delete", use_container_width=True, type="primary"):
            _delete_candidate(cid)
            st.rerun()

    # Edit Form
    with st.expander("Edit Features", expanded=True):
        is_manual = cand.get("source") == "manual"
        
        # Metadata
        if is_manual:
            cand["track_name"] = st.text_input("Track", cand.get("track_name"))
            cand["artist_name"] = st.text_input("Artist", cand.get("artist_name"))
        
        # Main Features
        cand["spotify_popularity"] = st.slider("Popularity", 0, 100, int(cand.get("spotify_popularity", 0)))
        cand["scrobbles_week"] = st.number_input("Scrobbles (Week)", value=int(cand.get("scrobbles_week", 0)))
        
        # We can add more fields here, but keeping it cleaner for the redesign.
        # Let's add the Genre bucket as it's important
        curr_genre = cand.get("genre_bucket", "unknown")
        if curr_genre not in GENRE_BUCKET_OPTIONS:
             # handle case where backend might return something odd
             options = [curr_genre] + GENRE_BUCKET_OPTIONS
        else:
             options = GENRE_BUCKET_OPTIONS
             
        cand["genre_bucket"] = st.selectbox("Genre", options, index=options.index(curr_genre))
        
        st.caption("Changes are saved automatically to memory. Re-run prediction to update scores.")


def main():
    _init_session_state()
    _ensure_backend_warmup()

    st.title("Song Predictor 🔮")
    st.markdown("Build a weekly playlist and predict which song will be your favorite.")

    # 1. Draft
    render_add_candidate_section()
    
    st.divider()
    
    # 2. Main Workspace (List + Inspector + Predict)
    render_main_workspace()
    
    # Session Reset
    if st.session_state["candidates"]:
        st.markdown("---")
        if st.button("Clear All Candidates", type="secondary"):
            st.session_state["candidates"] = []
            st.session_state["selected_candidate_id"] = None
            st.rerun()

if __name__ == "__main__":
    main()