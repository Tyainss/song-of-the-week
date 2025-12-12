import types
import pytest

class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.warnings = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

@pytest.fixture()
def fake_st():
    return FakeStreamlit()
