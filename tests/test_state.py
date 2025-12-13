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

def test_add_candidate_assigns_id_and_selects(monkeypatch, fake_st):
    # Arrange
    monkeypatch.setattr(state, "st", fake_st)
    state.init_session_state()

    # Make uuid deterministic (so the test is stable)
    monkeypatch.setattr(state, "uuid4", lambda: "fixed-id-123")

    candidate = {
        "candidate_id": None,  # intentionally missing
        "track_name": "Track A",
        "artist_name": "Artist A",
        "source": "spotify",
    }

    # Act
    state.add_candidate(candidate)

    # Assert
    assert len(fake_st.session_state["candidates"]) == 1
    assert fake_st.session_state["candidates"][0]["candidate_id"] == "fixed-id-123"
    assert fake_st.session_state["selected_candidate_id"] == "fixed-id-123"


def test_update_results_updates_candidates_by_id(monkeypatch, fake_st):
    # Arrange
    monkeypatch.setattr(state, "st", fake_st)
    state.init_session_state()

    c1 = {"candidate_id": "c1", "track_name": "T1", "artist_name": "A1"}
    c2 = {"candidate_id": "c2", "track_name": "T2", "artist_name": "A2"}
    fake_st.session_state["candidates"] = [c1, c2]

    response = {
        "threshold": 0.7,
        "results": [
            {"candidate_id": "c2", "probability": 0.91, "prediction": 1, "above_threshold": True, "rank": 1},
            {"candidate_id": "c1", "probability": 0.12, "prediction": 0, "above_threshold": False, "rank": 2},
        ],
    }

    # Act
    state.update_results(response, mode="ranking")

    # Assert: session-level fields
    assert fake_st.session_state["last_threshold"] == 0.7
    assert fake_st.session_state["last_prediction_mode"] == "ranking"

    # Assert: c1 updated correctly (matched by id)
    assert c1["probability"] == 0.12
    assert c1["prediction"] == 0
    assert c1["above_threshold"] is False
    assert c1["rank"] == 2
    assert c1["_threshold_at_prediction"] == 0.7

    # Assert: c2 updated correctly (matched by id)
    assert c2["probability"] == 0.91
    assert c2["prediction"] == 1
    assert c2["above_threshold"] is True
    assert c2["rank"] == 1
    assert c2["_threshold_at_prediction"] == 0.7

def test_update_results_ranking_falls_back_to_index_when_ids_missing(monkeypatch, fake_st):
    monkeypatch.setattr(state, "st", fake_st)
    state.init_session_state()

    c1 = {"candidate_id": "c1", "track_name": "T1", "artist_name": "A1"}
    c2 = {"candidate_id": "c2", "track_name": "T2", "artist_name": "A2"}
    fake_st.session_state["candidates"] = [c1, c2]

    response = {
        "threshold": 0.6,
        # candidate_id values do NOT match c1/c2 on purpose
        "results": [
            {"candidate_id": "x", "probability": 0.9, "prediction": 1, "above_threshold": True, "rank": 1},
            {"candidate_id": "y", "probability": 0.1, "prediction": 0, "above_threshold": False, "rank": 2},
        ],
    }

    state.update_results(response, mode="ranking")

    # Fallback-by-index should apply:
    assert c1["probability"] == 0.9
    assert c1["rank"] == 1
    assert c2["probability"] == 0.1
    assert c2["rank"] == 2

def test_can_call_enforces_cooldown(monkeypatch, fake_st):
    monkeypatch.setattr(state, "st", fake_st)
    state.init_session_state()

    # time starts at 1000
    t = {"now": 1000.0}
    monkeypatch.setattr(state.time, "time", lambda: t["now"])

    assert state.can_call("predict") is True
    assert state.can_call("predict") is False  # immediately blocked

    # advance beyond cooldown window
    t["now"] = 1000.0 + state.COOLDOWN_SECONDS + 0.01
    assert state.can_call("predict") is True
