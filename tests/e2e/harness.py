"""Helpers for the end-to-end tests: the real server, a browser tab, timing maths."""

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
from pathlib import Path
from typing import Any

import pytest

AUTOPLAY = "--autoplay-policy=no-user-gesture-required"


# --------------------------------------------------------------------------- environment


def chromium_executable() -> str | None:
    """A Chromium executable: $DESTINY_E2E_CHROMIUM, Playwright's own, or a preinstalled build."""
    wanted = os.environ.get("DESTINY_E2E_CHROMIUM")
    if wanted:
        return wanted if Path(wanted).is_file() else None
    path = ""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            path = p.chromium.executable_path
    except Exception:  # noqa: BLE001 - any Playwright problem means "look elsewhere"
        path = ""
    if path and Path(path).is_file():
        return path
    found = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    return found[-1] if found else None


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
            const body = JSON.parse(init.body);
            const rec = { text: body.text, normalize: !!body.normalize, t0: performance.now(), t1: null };
            log.push(rec);
            // The server answers once the clip is made, so the response time is the synthesis time.
            const done = () => { rec.t1 = performance.now(); };
            return original.call(this, input, init).then((res) => { done(); return res; }, (err) => { done(); throw err; });
          };
        })();""")
        self.page = self.ctx.new_page()
        override = os.environ.get("DESTINY_E2E_APP_JS")  # compare another build of app.js (A/B checks)
        if override:
            body = Path(override).read_text(encoding="utf-8")
            self.page.route("**/static/app.js", lambda route: route.fulfill(
                status=200, content_type="text/javascript", body=body))
        self.errors: list[str] = []
        self.page.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.base = base
        self.page.goto(base + ("/?debug" if debug else "/"))

    def js(self, expression: str, timeout: float = 10) -> Any:
        """Evaluate a JS expression; raises instead of hanging if the page's script is stuck."""
        return self.page.locator("html").evaluate(f"() => ({expression})", timeout=timeout * 1000)

    def wait(self, js: str, timeout: float = 30, interval: float = 0.1) -> Any:
        """Poll a JS expression until it's truthy (the app's CSP blocks wait_for_function's eval)."""
        deadline = time.time() + timeout
        while True:
            value = self.js(js)
            if value:
                return value
            if time.time() > deadline:
                raise AssertionError(f"timed out after {timeout}s waiting for: {js}")
            time.sleep(interval)

    def state(self) -> dict[str, Any]:
        return self.js("athenaDebug.state()")

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        events = self.js("athenaDebug.events")
        return [e for e in events if kind is None or e["type"] == kind]

    def now(self) -> float:
        return self.js("performance.now()")

    def tts_log(self) -> list[dict[str, Any]]:
        return self.js("window.__ttsLog")

    def set_speed(self, value: float) -> None:
        """Move the speed slider the way a person drags it."""
        self.page.evaluate("""(v) => { const s = document.querySelector('#speed');
          s.value = String(v); s.dispatchEvent(new Event('input', { bubbles: true })); }""", value)

    def open_book(self) -> None:
        self.wait("document.querySelector('.article .section-title')")

    def check_no_errors(self, allowed: tuple[str, ...] = ()) -> None:
        bad = [e for e in self.errors if not any(a in e for a in allowed)]
        assert not bad, bad

    def close(self) -> None:
        self.ctx.close()


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


def max_overlap(log: list[dict[str, Any]]) -> int:
    """Most /api/tts requests that were waiting on the server at the same moment."""
    most = 0
    for rec in log:
        at = rec["t0"]
        most = max(most, sum(1 for o in log if o["t0"] <= at and (o["t1"] is None or o["t1"] > at)))
    return most


def pause_seconds(utterance: dict[str, Any], rate: float) -> float:
    """The app's planned pause after a sentence (app.js pauseSeconds)."""
    return utterance["pause_after_ms"] / max(1.0, rate * 0.8) / 1000
