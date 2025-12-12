
import streamlit as st


def define_pages():
    song_predictor_page = st.Page(
        "pages/song_predictor.py", 
        title="Song Predictor", 
        icon="🔮", 
        default=True 
    )
    about_page = st.Page(
        "pages/about.py",
        title="About",
    )

    pg = st.navigation(
        [
            song_predictor_page,
            about_page,
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

