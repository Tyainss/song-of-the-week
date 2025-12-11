
import streamlit as st


def define_pages():
    song_predictor_page = st.Page(
        "pages/song_predictor.py", 
        title="Song Predictor", 
        icon="🔮", 
        default=True 
    )

    pg = st.navigation(
        [
            song_predictor_page,
        ]
    )

    st.set_page_config(
        layout="wide",
        page_title="SOTW - Song Predictor",
        page_icon="🎵",
    )

    return pg


pg = define_pages()
st.logo("streamlit_app/logo/personal_mark.png")
pg.run()

