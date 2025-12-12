import itertools

from streamlit_app.app_core.song_predictor import state


def test_duplicate_candidate_creates_clean_copy_and_selects_it(monkeypatch, fake_st):
    # Arrange: replace streamlit dependency with our fake
    monkeypatch.setattr(state, "st", fake_st)

    # Arrange: initialize required session state keys
    state.init_session_state()

    # Arrange: control uuid4 so the test is deterministic
    ids = itertools.count(1)
    monkeypatch.setattr(state, "uuid4", lambda: f"id-{next(ids)}")

    original = {
        "candidate_id": "orig-1",
        "source": "spotify",
        "track_name": "My Track",
        "artist_name": "My Artist",
        # Simulate that it already had prediction results
        "probability": 0.91,
        "rank": 1,
        "prediction": 1,
        "above_threshold": True,
    }

    # Put original in session state
    fake_st.session_state["candidates"] = [original]
    fake_st.session_state["selected_candidate_id"] = "orig-1"

    # Act: duplicate the candidate
    state.duplicate_candidate(original)

    # Assert: we now have two candidates
    assert len(fake_st.session_state["candidates"]) == 2

    new_cand = fake_st.session_state["candidates"][1]

    # Assert: new candidate has a NEW id and becomes selected
    assert new_cand["candidate_id"] != original["candidate_id"]
    assert fake_st.session_state["selected_candidate_id"] == new_cand["candidate_id"]

    # Assert: duplication rules
    assert new_cand["source"] == "manual"
    assert new_cand["track_name"] == "My Track (copy)"

    # Assert: prediction fields are cleared
    assert new_cand["probability"] is None
    assert new_cand["rank"] is None
    assert new_cand["prediction"] is None
    assert new_cand["above_threshold"] is None
