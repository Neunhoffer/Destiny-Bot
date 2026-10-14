"""End-to-end fixtures: the real Athena server with real voices, driven by Chromium.

These tests start ``python -m destiny_bot serve`` with its own data folder holding
the real Piper and Kokoro voices, and drive the web UI with Playwright. They skip
cleanly when Playwright, Chromium or the voices are missing. Set:

- ``DESTINY_TEST_PIPER_VOICE``: a Piper ``.onnx`` voice (with its ``.onnx.json``)
- ``DESTINY_TEST_KOKORO_DIR``: a folder with a Kokoro model and ``voices-v1.0.bin``
- ``DESTINY_E2E_CHROMIUM`` (optional): the Chromium executable to use

The page is opened with ``?debug``, which turns on the app's timing hooks
(``window.athenaDebug``): scheduler events and a snapshot of playback state.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from .harness import AUTOPLAY, App, Server, chromium_executable


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def chromium_path() -> str:
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        pytest.skip("Playwright isn't installed (pip install playwright)")
    path = chromium_executable()
    if not path:
        pytest.skip("no Chromium found: set DESTINY_E2E_CHROMIUM or run `playwright install chromium`")
    return path


@pytest.fixture(scope="session")
def server(tmp_path_factory: pytest.TempPathFactory, chromium_path: str) -> Iterator[Server]:
    """Athena with both Kokoro and Piper installed (Kokoro is the default voice)."""
    srv = Server(tmp_path_factory.mktemp("athena-home"))
    yield srv
    srv.stop()


@pytest.fixture(scope="session")
def browser(chromium_path: str) -> Iterator[Any]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium_path, args=[AUTOPLAY, "--enable-precise-memory-info"])
        yield b
        b.close()


@pytest.fixture
def open_app(browser: Any) -> Iterator[Any]:
    apps: list[App] = []

    def make(base: str, **kwargs: Any) -> App:
        app = App(browser, base, **kwargs)
        apps.append(app)
        return app

    yield make
    for app in apps:
        app.close()
