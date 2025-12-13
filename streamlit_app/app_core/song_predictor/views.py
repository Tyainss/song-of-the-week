from typing import Any, Dict
import pandas as pd
import streamlit as st

from . import state
from . import actions

# --- UI Renderers ---

def render_add_candidate_section():
    """Tabbed interface for adding candidates."""
    st.markdown("### Draft Candidates")
    
    with st.container(border=True):
        tab_spotify, tab_random, tab_fav, tab_manual = st.tabs([
            "🟢 Spotify URL", "🎲 Random Dataset", "❤️ Favorites", "📝 Manual"
        ])

        with tab_spotify:
            col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
            url = col1.text_input(
                "Track URL", 
                placeholder="http://spotify.com/...", 
                label_visibility="collapsed",
                key="spotify_url_input"
            )
            if col2.button("Fetch", key="btn_add_spotify", width='stretch'):
                actions.handle_fetch_spotify_candidate(url)

        with tab_random:
            st.caption("Pull a random non-favorite song from your listening history.")
            if st.button("Add Random Track", key="btn_add_rnd"):
                actions.handle_add_random_example()

        with tab_fav:
            st.caption("Pull a historical 'Song of the Week' from your dataset.")
            if st.button("Add Favorite Track", key="btn_add_fav"):
                actions.handle_add_favorite_example()

        with tab_manual:
            st.caption("Start from scratch.")
            if st.button("Create Blank Candidate", key="btn_add_man"):
                actions.handle_add_manual_candidate()

def render_main_workspace():
    """Main two-column layout."""
    col_list, col_inspector = st.columns([1.8, 1.2], gap="large")

    with col_list:
        render_candidate_list()
        st.divider()
        render_controls()

    with col_inspector:
        render_inspector_panel()

def render_candidate_list():
    st.markdown("### Candidate List")
    candidates = state.get_candidates()
    last_mode = state.get_last_prediction_mode()
    
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
            "Prob": c.get("probability", 0.0) * 100 if c.get("probability") is not None else None,
            "Rank": c.get("rank"),
        })
    
    df = pd.DataFrame(df_data)

    # Sort if Ranks exist (Ranking Mode result)
    if last_mode == "ranking" and "Rank" in df.columns and df["Rank"].notna().any():
        df = df.sort_values(by=["Rank", "Prob"], ascending=[True, False])
    
    # Configure Columns
    column_config = {
        "id": None, 
        # Display with 2 decimal places
        "Prob": st.column_config.ProgressColumn(
            "Confidence", format="%.1f%%", 
            min_value=0, max_value=100
        ),
    }

    # Only show Rank column if we are in Ranking mode
    if last_mode == "ranking":
        column_config["Rank"] = st.column_config.NumberColumn("Rank", format="%d")
    else:
        column_config["Rank"] = None

    event = st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key="candidate_list_df"
    )

    if event.selection.rows:
        selected_index = event.selection.rows[0]
        selected_row_id = df.iloc[selected_index]["id"]
        # Update selected ID in state based on dataframe selection
        st.session_state["selected_candidate_id"] = selected_row_id


    if candidates:
        if st.button("Remove All", type="secondary"):
            state.remove_all_candidates()
            st.rerun()

def render_controls():
    """The 'Predict' Action Bar."""
    candidates = state.get_candidates()
    if not candidates:
        return

    col_mode, col_btn = st.columns([1, 2], vertical_alignment="bottom")
    
    with col_mode:
        mode_select = st.selectbox(
            "Analysis Scope",
            options=["Rank All", "Check Selected"],
            key="analysis_mode_select"
        )
    
    with col_btn:
        is_ranking = mode_select == "Rank All"
        btn_label = "🏆 RANK ALL" if is_ranking else "🔎 CHECK SELECTED"
        
        if st.button(btn_label, type="primary", width='stretch'):
            actions.handle_prediction(mode_select)

def render_inspector_panel():
    st.markdown("### Inspector")
    
    cand = state.get_selected_candidate()
    cid = state.get_selected_candidate_id()
    
    if not cand:
        st.caption("Select a song to view details.")
        return

    # Header & Actions
    with st.container(border=True):
        st.subheader(cand.get("track_name", "Unknown"))
        st.caption(cand.get("artist_name", "Unknown"))
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("Duplicate", width='stretch', key="btn_insp_duplicate"):
                state.duplicate_candidate(cand)
                st.rerun()
        with col_act2:
            if st.button("Remove", width='stretch', type="primary", key="btn_insp_remove"):
                state.remove_candidate(cid)
                st.rerun()

    # Result Card - VISUAL SEPARATION BASED ON MODE
    prob = cand.get("probability")
    last_mode = state.get_last_prediction_mode()

    if prob is not None:
        threshold = cand.get("_threshold_at_prediction") or state.get_last_threshold()
        
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
        
        elif last_mode == "ranking":
            rank = cand.get("rank")
            rank_display = f"#{rank}" if rank else "N/A"
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
    
    # Track Properties - Expanded by default
    with st.expander("Track Properties", expanded=True):
        if is_manual:
            cand["track_name"] = st.text_input("Track Name", cand.get("track_name"))
            cand["artist_name"] = st.text_input("Artist Name", cand.get("artist_name"))
        
        c1, c2 = st.columns(2)
        with c1:
            cand["spotify_popularity"] = st.slider(
                "Popularity", 
                0, 
                100, 
                int(cand.get("spotify_popularity", 0))
            )
            cand["track_duration"] = st.number_input(
                "Duration (s)", 
                min_value=0, 
                value=int(cand.get("track_duration", 180)),
                step=1,
                key=f"dur_{cid}"
            )
        with c2:
            cand["days_since_release"] = st.number_input(
                "Days Since Release", 
                min_value=0, 
                value=int(cand.get("days_since_release", 100)),
                step=1,
                key=f"dsr_{cid}"
            )
            
            curr_genre = cand.get("genre_bucket", "unknown")
            opts = state.GENRE_BUCKET_OPTIONS if curr_genre in state.GENRE_BUCKET_OPTIONS else [curr_genre] + state.GENRE_BUCKET_OPTIONS
            cand["genre_bucket"] = st.selectbox("Genre", opts, index=opts.index(curr_genre))

    # Weekly Intensity - Expanded by default
    with st.expander("Weekly Intensity", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            cand["scrobbles_week"] = st.number_input(
                "Scrobbles (Week)", 
                min_value=0, 
                value=int(cand.get("scrobbles_week", 0)),
                step=1, 
                key=f"scrw_{cid}"
            )
            cand["unique_days_week"] = st.number_input(
                "Unique Days", 
                min_value=0, 
                max_value=7, 
                value=int(cand.get("unique_days_week", 0)),
                step=1,
                key=f"udw_{cid}"
            )
        with c2:
            cand["within_week_rank_by_scrobbles"] = st.number_input(
                "Rank (in week)", 
                min_value=1, 
                value=int(cand.get("within_week_rank_by_scrobbles", 10)),
                step=1,
                key=f"wwr_{cid}"
            )
            cand["scrobbles_saturday"] = st.number_input(
                "Scrobbles (Sat)", 
                min_value=0, 
                value=int(cand.get("scrobbles_saturday", 0)),
                step=1,
                key=f"scrsat_{cid}"
            )
        
        cand["scrobbles_last_fri_sat"] = st.number_input(
            "Scrobbles (Fri+Sat)", 
            min_value=0, 
            value=int(cand.get("scrobbles_last_fri_sat", 0)),
            step=1,
            key=f"scrfs_{cid}"
        )

    # History & Trends - Collapsed by default
    with st.expander("History & Trends", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            cand["scrobbles_prev_1w"] = st.number_input(
                "Scrobbles (Prev 1w)", 
                min_value=0, 
                value=int(cand.get("scrobbles_prev_1w", 0)),
                step=1,
                key=f"scrp1_{cid}"
            )
            cand["scrobbles_prev_4w"] = st.number_input(
                "Scrobbles (Prev 4w)", 
                min_value=0, 
                value=int(cand.get("scrobbles_prev_4w", 0)),
                step=1,
                key=f"scrp4_{cid}"
            )
        with c2:
            cand["prior_scrobbles_all_time"] = st.number_input(
                "All Time Scrobbles", 
                min_value=0, 
                value=int(cand.get("prior_scrobbles_all_time", 0)),
                step=1,
                key=f"scrat_{cid}"
            )
            # Retain float behavior for ratios originally set to float
            cand["momentum_4w_ratio"] = st.number_input(
                "Momentum (4w)", 
                value=float(cand.get("momentum_4w_ratio", 0.0)),
                step=0.1,
                key=f"mom4w_{cid}"
            )

        cand["week_over_week_change"] = st.number_input(
            "WoW Change", 
            value=int(cand.get("week_over_week_change", 0)),
            step=1,
            key=f"wowc_{cid}"
        )
        cand["last_scrobble_gap_days"] = st.number_input(
            "Days since last play", 
            min_value=0,
            value=int(cand.get("last_scrobble_gap_days", 0)),
            step=1,
            key=f"lsgd_{cid}"
        )