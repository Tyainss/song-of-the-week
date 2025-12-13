import contextlib
import pytest


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}

        self.warnings = []
        self.errors = []
        self.toasts = []
        self.reruns = 0

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def toast(self, msg: str, icon=None) -> None:
        self.toasts.append((msg, icon))

    def rerun(self) -> None:
        self.reruns += 1

    @contextlib.contextmanager
    def spinner(self, msg: str):
        yield


@pytest.fixture()
def fake_st():
    return FakeStreamlit()
