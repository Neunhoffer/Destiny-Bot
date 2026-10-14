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

import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

AUTOPLAY = "--autoplay-policy=no-user-gesture-required"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/e2e" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.e2e)


# --------------------------------------------------------------------------- environment


def _chromium() -> str | None:
    """A Chromium executable: $DESTINY_E2E_CHROMIUM, a preinstalled build, or Playwright's own."""
    wanted = os.environ.get("DESTINY_E2E_CHROMIUM")
    if wanted:
        return wanted if Path(wanted).is_file() else None
    for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",):
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            path = p.chromium.executable_path
        return path if path and Path(path).is_file() else None
    except Exception:
        return None


def _kokoro_files() -> tuple[Path, Path] | None:
    value = os.environ.get("DESTINY_TEST_KOKORO_DIR")
    if not value:
        return None
    from destiny_bot.tts import catalog

    return catalog.find_kokoro_model(Path(value))


def _piper_voice() -> Path | None:
    value = os.environ.get("DESTINY_TEST_PIPER_VOICE")
    if not value:
        return None
    model = Path(value)
    if not model.is_file() or not model.with_name(model.name + ".json").is_file():
        return None
    return model


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    """A running ``destiny serve`` process and a tiny JSON client for it."""

    def __init__(self, home: Path, *, kokoro: bool = True) -> None:
        self.home = home
        piper = _piper_voice()
        kokoro_files = _kokoro_files()
        if piper is None:
            pytest.skip("set DESTINY_TEST_PIPER_VOICE to a Piper .onnx voice to run the end-to-end tests")
        if kokoro and kokoro_files is None:
            pytest.skip("set DESTINY_TEST_KOKORO_DIR to a Kokoro model folder to run the end-to-end tests")
        for src in (piper, piper.with_name(piper.name + ".json")):
            _link_or_copy(src, home / "voices" / "piper" / src.name)
        if kokoro and kokoro_files:
            for src in kokoro_files:
                _link_or_copy(src, home / "voices" / "kokoro" / src.name)
        env = {k: v for k, v in os.environ.items() if k not in ("DESTINY_EXTRA_VOICE_DIRS", "ANTHROPIC_API_KEY")}
        env.update(DESTINY_HOME=str(home), PYTHONUNBUFFERED="1")
        self.log = open(home / "server.log", "w+", encoding="utf-8")  # noqa: SIM115 - closed in stop()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "destiny_bot", "serve", "--no-browser", "--port", str(_free_port())],
            stdout=self.log, stderr=subprocess.STDOUT, env=env,
        )
        self.base = self._wait_for_url()
        html = urllib.request.urlopen(self.base + "/", timeout=10).read().decode()
        self.token = re.search(r'name="destiny-token" content="([^"]+)"', html).group(1)

    def _wait_for_url(self) -> str:
        deadline = time.time() + 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.log.seek(0)
                raise RuntimeError(f"Athena exited early:\n{self.log.read()}")
            self.log.seek(0)
            match = re.search(r"(http://127\.0\.0\.1:\d+)/", self.log.read())
            if match:
                try:
                    urllib.request.urlopen(match.group(1) + "/", timeout=2)
                    return match.group(1)
                except OSError:
                    pass
            time.sleep(0.1)
        raise RuntimeError("Athena didn't start within 60 seconds")

    def api(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "Content-Type": "application/json", "X-Destiny-Token": self.token,
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read() or b"null")

    def paste(self, title: str, text: str) -> dict[str, Any]:
        return self.api("POST", "/api/library/paste", {"title": title, "text": text})["book"]

    def upload(self, path: Path) -> dict[str, Any]:
        """Import a file through the API and wait for the job."""
        boundary = uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(self.base + "/api/library/import", data=body, method="POST", headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}", "X-Destiny-Token": self.token,
        })
        job_id = json.loads(urllib.request.urlopen(req, timeout=60).read())["job_id"]
        for _ in range(600):
            job = self.api("GET", f"/api/jobs/{job_id}")
            if job["status"] in ("done", "error", "cancelled"):
                return job
            time.sleep(0.2)
        raise TimeoutError(path.name)

    def settings(self, **changes: Any) -> dict[str, Any]:
        return self.api("PUT", "/api/settings", changes)

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def chromium_path() -> str:
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        pytest.skip("Playwright isn't installed (pip install playwright)")
    path = _chromium()
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


class App:
    """One browser tab on the Athena UI, with console errors collected."""

    def __init__(self, browser: Any, base: str, *, book_id: str | None = None, debug: bool = True,
                 **context: Any) -> None:
        context.setdefault("viewport", {"width": 1366, "height": 768})
        self.ctx = browser.new_context(accept_downloads=True, **context)
        if book_id:
            self.ctx.add_init_script(f"localStorage.setItem('destiny.lastBook', {json.dumps(json.dumps(book_id))})")
        # Record every /api/tts request so tests can check their order and overlap.
        self.ctx.add_init_script("""(() => {
          const log = window.__ttsLog = [];
          const original = window.fetch;
          window.fetch = function (input, init) {
            const url = typeof input === 'string' ? input : input.url;
            if (!url.includes('/api/tts')) return original.call(this, input, init);
            const rec = { text: JSON.parse(init.body).text, t0: performance.now(), t1: null };
            log.push(rec);
            const done = () => { rec.t1 = performance.now(); };
            return original.call(this, input, init).then((res) => { res.clone().blob().then(done, done); return res; },
              (err) => { done(); throw err; });
          };
        })();""")
        self.page = self.ctx.new_page()
        self.errors: list[str] = []
        self.page.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.base = base
        self.page.goto(base + ("/?debug" if debug else "/"))

    def wait(self, js: str, timeout: float = 30, interval: float = 0.1) -> Any:
        """Poll a JS expression until it's truthy (the app's CSP blocks wait_for_function's eval)."""
        deadline = time.time() + timeout
        while True:
            value = self.page.evaluate(js)
            if value:
                return value
            if time.time() > deadline:
                raise AssertionError(f"timed out after {timeout}s waiting for: {js}")
            time.sleep(interval)

    def state(self) -> dict[str, Any]:
        return self.page.evaluate("athenaDebug.state()")

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        events = self.page.evaluate("athenaDebug.events")
        return [e for e in events if kind is None or e["type"] == kind]

    def now(self) -> float:
        return self.page.evaluate("performance.now()")

    def close(self) -> None:
        self.ctx.close()


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


def sound_time(event: dict[str, Any]) -> float:
    """performance.now() at which a scheduled sentence starts sounding."""
    return event["perf"] + (event["start"] - event["now"]) * 1000


def book_text(chapters: int, paragraphs: int, sentences: int, *, words: str = "") -> str:
    """Plain text with ``# Chapter`` headings and numbered, distinct sentences of similar length."""
    filler = words or "about how living cells move water and salts across their thin outer membranes"
    out = []
    for c in range(1, chapters + 1):
        out.append(f"# Chapter {c}: Membranes")
        for p in range(1, paragraphs + 1):
            out.append(" ".join(f"Sentence {s} of paragraph {p} in chapter {c} is {filler}."
                                for s in range(1, sentences + 1)))
    return "\n\n".join(out)
