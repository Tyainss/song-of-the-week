from streamlit_app.app_core.song_predictor import actions, state


def test_handle_prediction_single_requires_selected_candidate(monkeypatch, fake_st):
    # Arrange: use fake Streamlit in both actions and state
    monkeypatch.setattr(actions, "st", fake_st)
    monkeypatch.setattr(state, "st", fake_st)
    monkeypatch.setattr(actions, "state", state)

    state.init_session_state()
    fake_st.session_state["candidates"] = []
    fake_st.session_state["selected_candidate_id"] = None

    # Act
    actions.handle_prediction(mode_ui_label="Single")

    # Assert: showed error, did not rerun
    assert len(fake_st.errors) == 1
    assert fake_st.reruns == 0


def test_handle_prediction_ranking_calls_api_and_reruns(monkeypatch, fake_st):
    # Arrange
    monkeypatch.setattr(actions, "st", fake_st)
    monkeypatch.setattr(state, "st", fake_st)
    monkeypatch.setattr(actions, "state", state)

    state.init_session_state()
    fake_st.session_state["candidates"] = [
        {"candidate_id": "c1", "track_name": "T1", "artist_name": "A1"},
        {"candidate_id": "c2", "track_name": "T2", "artist_name": "A2"},
    ]
    fake_st.session_state["selected_candidate_id"] = "c1"

    # Disable cooldown gating
    monkeypatch.setattr(actions.state, "can_call", lambda name: True)

    # Fake API client
    class FakeAPI:
        called = False
        last_payload = None
        last_mode = None

        @staticmethod
        def predict_candidates(candidates, mode):
            FakeAPI.called = True
            FakeAPI.last_payload = candidates
            FakeAPI.last_mode = mode
            return {
                "threshold": 0.5,
                "results": [
                    {
                        "candidate_id": "c1",
                        "probability": 0.8,
                        "prediction": 1,
                        "above_threshold": True,
                        "rank": 1,
                    },
                    {
                        "candidate_id": "c2",
                        "probability": 0.2,
                        "prediction": 0,
                        "above_threshold": False,
                        "rank": 2,
                    },
                ],
            }

    monkeypatch.setattr(actions, "api_client", FakeAPI)

    # Act
    actions.handle_prediction(mode_ui_label="Rank All")

    # Assert: API called with correct mode and payload size
    assert FakeAPI.called is True
    assert FakeAPI.last_mode == "ranking"
    assert len(FakeAPI.last_payload) == 2

    # Assert: state updated and rerun triggered
    assert fake_st.session_state["last_prediction_mode"] == "ranking"
    assert fake_st.reruns == 1
