import requests
import pytest

from streamlit_app.utils import api_client


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", raise_http_error=False):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self._raise_http_error = raise_http_error

    def raise_for_status(self):
        if self._raise_http_error or self.status_code >= 400:
            raise requests.HTTPError("http error")

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


def test_build_url_normalizes_slashes(monkeypatch):
    monkeypatch.setattr(api_client, "BASE_URL", "http://example.com/")
    assert api_client._build_url("/health") == "http://example.com/health"
    assert api_client._build_url("health") == "http://example.com/health"


def test_handle_response_success_non_json_raises():
    resp = DummyResponse(status_code=200, json_data=ValueError("not json"), text="OK")
    with pytest.raises(api_client.APIClientError) as e:
        api_client._handle_response(resp)
    assert "non-JSON" in str(e.value)


def test_handle_response_http_error_uses_detail_if_available():
    resp = DummyResponse(
        status_code=400,
        json_data={"detail": "Bad request"},
        text="fallback",
        raise_http_error=True,
    )
    with pytest.raises(api_client.APIClientError) as e:
        api_client._handle_response(resp)
    assert "Bad request" in str(e.value)
