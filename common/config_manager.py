
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

class ConfigManager:
    """
    Minimal, strict config loader.
    """

    def __init__(self, repo_root: Path) -> None:
        self.root = Path(repo_root).resolve()
        self.config_dir = self.root / "configs"
        # Load .env from repo root (OS env vars keep precedence)
        load_dotenv(self.root / ".env", override=False)
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Configs directory not found: {self.config_dir}")

    # ---------------------------
    # Public API
    # ---------------------------

    def project(self) -> Dict[str, Any]:
        """Return configs/project.yaml as a dict."""
        return self._load_yaml(self.config_dir / "project.yaml")

    def lastfm(self) -> Dict[str, Any]:
        """Return configs/lastfm.yaml as a dict."""
        return self._load_yaml(self.config_dir / "lastfm.yaml")

    def musicbrainz(self) -> Dict[str, Any]:
        """Return configs/musicbrainz.yaml as a dict."""
        return self._load_yaml(self.config_dir / "musicbrainz.yaml")
    
    def spotify(self) -> Dict[str, Any]:
        """Return configs/spotify.yaml as a dict."""
        return self._load_yaml(self.config_dir / "spotify.yaml")

    def load(self, name: str) -> Dict[str, Any]:
        """
        Generic loader for any other YAML in configs/.
        Example: cfg.load("something.yaml")
        """
        return self._load_yaml(self.config_dir / name)

    def env(self, key: str, default: str | None = None, *, required: bool = False) -> str | None:
        """
        Read environment variables (for secrets).
        - If required=True and the variable is missing, raises RuntimeError.
        - No defaults are injected unless you pass one explicitly.
        """
        val = os.getenv(key, default)
        if required and val is None:
            raise RuntimeError(f"Missing required environment variable: {key}")
        return val

    @staticmethod
    def require_keys(config: Dict[str, Any], keys: list[str]) -> None:
        """
        Assert that a list of keys exists in a given config dict.
        Raises KeyError on the first missing key.
        """
        for k in keys:
            if k not in config:
                raise KeyError(f"Missing required key '{k}' in config")

    # ---------------------------
    # Internals
    # ---------------------------

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a YAML mapping (dict): {path}")
        
        return data
