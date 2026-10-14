"""Where Destiny Bot keeps its files, and the user's settings.

Everything Destiny Bot stores lives under one folder (:func:`data_dir`) so that
deleting that folder removes every trace of it. Set ``DESTINY_HOME`` to put the
folder somewhere specific (the launcher scripts point it inside the app folder,
so deleting the app folder removes the data too).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .models import ReadingOptions

APP_DIR_NAME = "DestinyBot"
DEFAULT_AI_MODEL = "claude-opus-5-5"


def data_dir() -> Path:
    """Root folder for all Destiny Bot data (created on first use)."""
    override = os.environ.get("DESTINY_HOME")
    if override:
        root = Path(override).expanduser()
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        root = Path(base) / APP_DIR_NAME
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        root = Path(base) / "destiny-bot"
    root.mkdir(parents=True, exist_ok=True)
    return root


def library_dir() -> Path:
    path = data_dir() / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def voices_dir() -> Path:
    path = data_dir() / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def exports_dir() -> Path:
    path = data_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    return data_dir() / "settings.json"


class Settings(BaseModel):
    # "auto" = best available *offline* engine: piper (if a voice is installed) > system > browser.
    tts_engine: str = "auto"
    voices: dict[str, str] = Field(default_factory=dict)  # engine name -> chosen voice id
    speed: float = Field(default=1.0, ge=0.5, le=3.0)  # playback speed in the reader
    reading: ReadingOptions = Field(default_factory=ReadingOptions)

    # Privacy switches. Both are OFF until the user turns them on.
    online_tts_allowed: bool = False  # allow engines that send text to an online service
    ai_enabled: bool = False  # allow Claude study helpers (sends text to Anthropic)
    ai_model: str = DEFAULT_AI_MODEL
    # Stored only in the local settings file (owner-only permissions). Never sent to the browser.
    anthropic_api_key: str | None = None

    theme: Literal["system", "light", "dark"] = "system"
    font_scale: float = Field(default=1.0, ge=0.7, le=2.0)

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to show in the UI: the API key is replaced by a yes/no flag."""
        data = self.model_dump(mode="json", exclude={"anthropic_api_key"})
        data["anthropic_api_key_set"] = bool(self.resolved_api_key())
        return data

    def resolved_api_key(self) -> str | None:
        """The saved key, falling back to the ANTHROPIC_API_KEY environment variable."""
        return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or None


_settings_lock = threading.Lock()


def load_settings() -> Settings:
    path = settings_path()
    if not path.exists():
        return Settings()
    try:
        return Settings.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError):
        # A corrupt settings file must never stop the app from starting.
        return Settings()


def save_settings(settings: Settings) -> None:
    """Atomically write settings with owner-only permissions (it may hold an API key)."""
    path = settings_path()
    payload = json.dumps(settings.model_dump(mode="json"), indent=2)
    with _settings_lock:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".settings-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            if os.name == "posix":
                os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def update_settings(changes: dict[str, Any]) -> Settings:
    """Merge a partial update (nested ``reading`` dict allowed) into the saved settings."""
    current = load_settings().model_dump(mode="python")
    for key, value in changes.items():
        if key == "reading" and isinstance(value, dict):
            current["reading"] = {**current["reading"], **value}
        elif key == "voices" and isinstance(value, dict):
            current["voices"] = {**current["voices"], **value}
        elif key == "anthropic_api_key":
            current[key] = (value or "").strip() or None
        else:
            current[key] = value
    settings = Settings.model_validate(current)
    save_settings(settings)
    return settings
