
import streamlit as st

from streamlit_app.utils import api_client
from . import state

def handle_fetch_spotify_candidate(url: str) -> None:
    if not url:
        return
        
    if not state.can_call("spotify"):
        return
        
    try:
        c = api_client.get_spotify_candidate_from_url(url)
        
        state.add_candidate(c)
        st.toast(f"Added: {c.get('track_name')}", icon="✅")
        st.rerun()
    except Exception as e:
        st.error(f"Error fetching Spotify candidate: {e}")

def handle_add_random_example() -> None:
    if not state.can_call("examples"):
        return
        
    try:
        resp = api_client.get_random_examples(1)
        if resp.get("candidates"):
            c = resp["candidates"][0]["candidate"]
            state.add_candidate(c)
            st.toast("Random candidate added", icon="🎲")
            st.rerun()
    except Exception as e:
        st.error(f"Error fetching random example: {e}")

def handle_add_favorite_example() -> None:
    if not state.can_call("examples"):
        return
        
    try:
        resp = api_client.get_favorite_examples(1)
        if resp.get("candidates"):
            c = resp["candidates"][0]["candidate"]
            state.add_candidate(c)
            st.toast("Favorite candidate added", icon="❤️")
            st.rerun()
    except Exception as e:
        st.error(f"Error fetching favorite example: {e}")

def handle_add_manual_candidate() -> None:
    state.add_candidate(state.build_manual_template())
    st.toast("Manual candidate created", icon="📝")
    st.rerun()
    
def handle_prediction(mode_ui_label: str) -> None:
    mode_api = "ranking" if mode_ui_label == "Rank All" else "single"
    candidates = state.get_candidates()

    if mode_api == "single":
        selected = state.get_selected_candidate()
        if not selected:
            st.error("Select a candidate in the table first.")
            return
        payload = [selected]
    else:
        payload = candidates

    with st.spinner("Analyzing..."):
        try:
            resp = api_client.predict_candidates(payload, mode=mode_api)
            state.update_results(resp, mode_api)
            
            if mode_api == "ranking":
                st.toast("Ranking complete!", icon="🏆")
            else:
                st.toast("Check complete!", icon="🔎")
                
            st.rerun()
        except Exception as e:
            st.error(f"Prediction error: {e}")