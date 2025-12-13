import streamlit as st


def main():
    st.title("About Song Predictor")

    st.markdown(
        """
    Every week, I listen to dozens of new tracks. On Saturdays, I select exactly **one** track 
    as my favorite. 
    
    **Song Predictor** is an ML-powered tool designed to predict which songs are likely to be chosen as favorites.
    """
    )

    st.divider()

    st.header("Workflow")

    # Using columns for a step-by-step visual flow
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True, height="stretch"):
            st.subheader("1. Draft")
            st.markdown(
                "Add songs from Spotify URLs or pull random examples from history."
            )

    with c2:
        with st.container(border=True, height="stretch"):
            st.subheader("2. Inspect")
            st.markdown("Click a song in the list to inspect its features.")

    with c3:
        with st.container(border=True, height="stretch"):
            st.subheader("3. Edit")
            st.markdown("Play around with the features as you wish.")

    with c4:
        with st.container(border=True, height="stretch"):
            st.subheader("4. Predict")
            st.markdown(
                "Run the model to see if the track ranks as a **Weekly Winner** or is likely to be a **Favorite**."
            )

    st.divider()

    st.header("Under the Hood")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True, height="stretch"):
            st.subheader("Data Collection")
            st.markdown(
                """
            The system ingests data from two primary sources:
            * **[Last.fm](https://www.last.fm/user/Tyains):** My scrobble history (play counts, timestamps).
            * **Spotify API:** Track features (popularity, duration, release date).
            """
            )

    with col2:
        with st.container(border=True, height="stretch"):
            st.subheader("The Model")
            st.markdown(
                """
            At the core is a **Logistic Regression** classifier.
            
            It looks at features like *weekly play counts*, *momentum* (4-week trends), and *novelty* to assign a **probability score (0-100%)** to each candidate.
            """
            )

    st.divider()

    st.header("Prediction Modes")
    st.caption("Choose the scope that fits your question.")

    m1, m2 = st.columns(2, gap="large")
    with m1:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 🏆 Rank All")
            st.markdown('**Question:** *"Who wins this week?"*')
            st.write(
                "Simulates a competitive week. It compares all drafted candidates against one another and highlights the single track with the highest probability."
            )

    with m2:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 🔎 Check Selected")
            st.markdown('**Question:** *"Is this song good enough?"*')
            st.write(
                "Analyzes a single song in isolation. It checks if the song's probability score crosses the model's confidence threshold, regardless of other tracks."
            )


if __name__ == "__main__":
    main()
