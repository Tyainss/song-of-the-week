
import streamlit as st


def define_pages():
    song_predictor_page = st.Page(
        "pages/song_predictor.py", 
        title="Song Predictor", 
        icon="🔮", 
        default=True 
    )
    playground_page = st.Page(
        "pages/playground.py",
        title="Playground",
        icon="🎵",
    )
    model_insights_page = st.Page(
        "pages/model_insights.py",
        title="Model Insights",
        icon="📈",
    )
    data_explorer_page = st.Page(
        "pages/data_explorer.py",
        title="Data Explorer",
        icon="📂",
    )

    pg = st.navigation(
        [
            song_predictor_page,
            playground_page,
            model_insights_page,
            data_explorer_page,
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

