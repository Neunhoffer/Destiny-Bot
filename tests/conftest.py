"""Shared pytest fixtures.

Every test runs with its own empty data folder (DESTINY_HOME), so tests never
touch a real library, settings file, or API key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    home = tmp_path / "destiny-home"
    home.mkdir()
    monkeypatch.setenv("DESTINY_HOME", str(home))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return home


@pytest.fixture
def piper_voice_path() -> Path:
    """Path to a real Piper voice (.onnx with .onnx.json next to it), or skip.

    Set DESTINY_TEST_PIPER_VOICE=/path/to/voice.onnx to run the real-synthesis tests.
    """
    value = os.environ.get("DESTINY_TEST_PIPER_VOICE")
    if not value or not Path(value).is_file():
        pytest.skip("set DESTINY_TEST_PIPER_VOICE to a Piper .onnx voice to run this test")
    return Path(value)
