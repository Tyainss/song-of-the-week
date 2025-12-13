import streamlit as st

from streamlit_app.app_core.song_predictor import state, views
from streamlit_app.utils import api_client

# --- Startup Helpers ---

def _ensure_backend_warmup() -> None:
    """Checks the health of the prediction API on startup."""
    if st.session_state.get("backend_warmed_up"):
        return
    try:
        api_client.healthcheck()
        st.session_state["backend_warmed_up"] = True
    except Exception:
        # Fail silently or show a small toast, don't block UI
        pass

# ------------- Main page ------------- #

def main():
    state.init_session_state()
    _ensure_backend_warmup()

    st.title("Song Predictor 🔮")
    st.markdown("Build a weekly playlist, simulate features, and predict the favorite.")

    views.render_add_candidate_section()
    st.divider()
    views.render_main_workspace()

if __name__ == "__main__":
    main()