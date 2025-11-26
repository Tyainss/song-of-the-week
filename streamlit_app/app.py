
import streamlit as st
# import os


def define_pages():
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
            playground_page,
            model_insights_page,
            data_explorer_page,
        ]
    )

    st.set_page_config(
        layout="wide",
        page_title="Song of the Week - Playground",
        page_icon="🎵",
    )

    return pg


pg = define_pages()

# Optional: uncomment and point to a logo file when you have one
st.logo("streamlit_app/logo/personal_mark.png")

pg.run()

