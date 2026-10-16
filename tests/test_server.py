"""Tests for the local web server (destiny_bot.server) and background jobs (destiny_bot.jobs).

The voice engines, speech planner, AI helper, exporter and voice catalog are replaced
with small fakes, so these tests only exercise the server's own behaviour. A few
integration tests at the end use the real modules when they're available.
"""

from __future__ import annotations

import errno
import io
import re
import struct
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from destiny_bot import config, ingest, server
from destiny_bot.ingest import IngestError
from destiny_bot.ingest.common import doc_id_for_file
from destiny_bot.jobs import Job, JobCancelled, JobError, JobManager, friendly_error
from destiny_bot.library import Library
from destiny_bot.models import Block, BlockKind, Document, Section, Utterance
from destiny_bot.server import CSP, ONLINE_TTS_OFF, content_disposition, create_app, host_policy
from destiny_bot.tts.base import AudioClip, EngineUnavailable, TTSEngine, TTSError, Voice

TOKEN = "test-token-123"
AUTH = {"X-Destiny-Token": TOKEN}


# -- fakes


def tiny_wav(n_samples: int = 160, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n_samples}h", *([1000] * n_samples)))
    return buf.getvalue()


class FakeEngine(TTSEngine):
    def __init__(self, name: str, *, online: bool = False, available: bool = True, voices=("v1", "v2"),
                 error: Exception | None = None, broken: bool = False) -> None:
        self.name = name
        self.label = f"Fake {name}"
        self.online = online
        self.available = available
        self.voices = list(voices)
        self.error = error
        self.broken = broken
        self.calls: list[tuple[str, str | None, float]] = []

    def is_available(self):
        if self.broken:
            raise RuntimeError("engine exploded")
        return (self.available, "" if self.available else f"{self.name} isn't installed")

    def list_voices(self):
        return [Voice(id=v, name=v.upper(), engine=self.name, online=self.online) for v in self.voices] + [
            Voice(id="not-installed", name="X", engine=self.name, installed=False)
        ]

    def default_voice(self):
        return self.voices[0] if self.voices else None

    def synthesize(self, text, voice=None, rate=1.0):
        self.calls.append((text, voice, rate))
        if self.error is not None:
            raise self.error
        return AudioClip(tiny_wav(), "audio/wav", 16000)


class FakeTTS:
    ENGINE_NAMES = ("piper", "system", "edge", "browser")

    def __init__(self) -> None:
        self.engines = {
            "piper": FakeEngine("piper"),
            "system": FakeEngine("system", voices=("sys1",)),
            "edge": FakeEngine("edge", online=True, voices=("en-US-AriaNeural",)),
            "browser": FakeEngine("browser", voices=()),
        }

    def get_engine(self, name):
        return self.engines[name]

    def all_engines(self):
        return list(self.engines.values())

    def resolve_engine(self, preferred, *, online_allowed):
        if preferred in self.engines and preferred != "auto":
            engine = self.engines[preferred]
            if not engine.online or online_allowed:
                return engine
        for name in ("piper", "system", "browser"):
            if self.engines[name].available:
                return self.engines[name]
        return self.engines["browser"]


class FakeCache:
    def __init__(self, root) -> None:
        self.root = Path(root)
        self.store: dict[tuple[str, ...], AudioClip] = {}
        self.keys: list[tuple[str, ...]] = []

    def get_or_create(self, key_parts, make):
        self.keys.append(key_parts)
        if key_parts not in self.store:
            self.store[key_parts] = make()
        return self.store[key_parts]


def fake_plan_section(section: Section, options=None) -> list[Utterance]:
    plan = []
    for i, block in enumerate(section.blocks):
        if block.text.strip():
            plan.append(
                Utterance(index=len(plan), block=i, start=0, end=len(block.text), kind=block.kind,
                          speech=block.text, pause_after_ms=700 if block.kind == BlockKind.HEADING else 350)
            )
    return plan


def fake_section_plain_text(section: Section, *, max_chars=None) -> str:
    text = "\n".join(b.text for b in section.blocks)
    return text[:max_chars] if max_chars else text


class AIUnavailable(RuntimeError):
    pass


class AIError(RuntimeError):
    pass


class QuizQuestion(BaseModel):
    question: str
    choices: list[str]
    answer_index: int
    explanation: str


class FakeAssistant:
    mode = "ok"
    calls: list[tuple[str, tuple, dict]] = []

    def __init__(self, settings, *, client=None) -> None:
        self.settings = settings

    def _do(self, name, args, kwargs):
        FakeAssistant.calls.append((name, args, kwargs))
        if FakeAssistant.mode == "unavailable":
            raise AIUnavailable("Add your Claude API key in Settings.")
        if FakeAssistant.mode == "error":
            raise AIError("Claude is busy right now. Try again in a minute.")
        return f"{name} answer"

    def explain(self, passage, **kw):
        return self._do("explain", (passage,), kw)

    def ask(self, question, **kw):
        return self._do("ask", (question,), kw)

    def summarize(self, section_text, **kw):
        return self._do("summarize", (section_text,), kw)

    def narrate_block(self, block, **kw):
        return self._do("narrate_block", (block,), kw)

    def describe_image(self, png, **kw):
        return self._do("describe_image", (png,), kw)

    def quiz(self, section_text, **kw):
        self._do("quiz", (section_text,), kw)
        return [QuizQuestion(question=f"Q{i}", choices=["a", "b"], answer_index=0, explanation="e")
                for i in range(kw["count"])]


class CatalogVoice(BaseModel):
    id: str
    name: str
    language: str
    quality: str
    size_mb: float
    description: str


class DownloadCancelled(RuntimeError):
    pass


class FakeCatalog:
    RECOMMENDED_VOICE = "en_US-lessac-medium"

    def __init__(self) -> None:
        self.CATALOG = [
            CatalogVoice(id="en_US-lessac-medium", name="Lessac", language="en-US", quality="medium",
                         size_mb=63.2, description="Clear US English"),
            CatalogVoice(id="en_GB-alba-medium", name="Alba", language="en-GB", quality="medium",
                         size_mb=63.1, description="Scottish English"),
        ]
        self.block = threading.Event()
        self.should_block = False

    def installed_piper_voices(self, voices_dir=None):
        folder = config.voices_dir() / "piper"
        return sorted(folder.glob("*.onnx")) if folder.is_dir() else []

    def download_piper_voice(self, voice_id, *, voices_dir=None, progress=None, base_url=None, cancel=None):
        if progress:
            progress(0.5, "Downloading")
        if self.should_block:
            while not cancel.is_set():
                time.sleep(0.01)
            raise DownloadCancelled("Download cancelled.")
        folder = config.voices_dir() / "piper"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{voice_id}.onnx.json").write_text("{}")
        path = folder / f"{voice_id}.onnx"
        path.write_bytes(b"onnx")
        return path


class FakeExporter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.mode = "ok"

    def __call__(self, doc, *, engine, voice, options, out_dir, section_ids=None, fmt="mp3", rate=1.0,
                 progress=None, cancel=None):
        self.calls.append(dict(doc=doc, engine=engine, voice=voice, out_dir=out_dir, section_ids=section_ids,
                               fmt=fmt, rate=rate))
        if progress:
            progress(0.5, "Reading section 1 of 2")
        if self.mode == "block":
            out_dir.joinpath("partial.mp3").write_bytes(b"partial")
            while not cancel.is_set():
                time.sleep(0.01)
            raise RuntimeError("export cancelled")
        if self.mode == "tts_error":
            raise TTSError("The voice stopped working.")
        if self.mode == "crash":
            raise ValueError("bad thing\nwith a second line")
        names = ["01 - Cells.mp3", "02 - Membranes & Transport ü.mp3", "playlist.m3u"]
        paths = []
        for i, name in enumerate(names):
            path = out_dir / name
            path.write_bytes(f"audio {i}".encode())
            paths.append(path)
        outside = out_dir.parent / "outside.mp3"
        outside.write_bytes(b"x")
        return paths + [outside]


class FakeAudioModule:
    def __init__(self) -> None:
        self.trimmed = 0

    def trim_silence(self, clip, **kwargs):
        self.trimmed += 1
        return AudioClip(clip.data + b"", clip.mime, clip.sample_rate)


# -- documents


def make_doc(path: Path, *, source_type: str, page_count: int | None = None, source_name: str | None = None) -> Document:
    return Document(
        id=doc_id_for_file(path),
        title="Biology Basics",
        source_name=source_name or path.name,
        source_type=source_type,
        page_count=page_count,
        warnings=["Page 2 was read with OCR"],
        sections=[
            Section(id="s0", title="Cells", page_start=1, page_end=1, blocks=[
                Block(kind=BlockKind.HEADING, text="Cells", level=1, page=1),
                Block(kind=BlockKind.PARAGRAPH, text="Cells are the unit of life.", page=1),
                Block(kind=BlockKind.TABLE, text="A B", rows=[["A", "B"], ["1", "2"]], page=1),
            ]),
            Section(id="s1", title="Membranes", level=2, page_start=2, page_end=2, blocks=[
                Block(kind=BlockKind.HEADING, text="Membranes", level=2, page=2),
                Block(kind=BlockKind.PARAGRAPH, text="Notes for the speaker.", page=2,
                      meta={"speaker_notes": True}),
            ]),
            Section(id="s2", title="Transport", page_start=3, page_end=3, blocks=[
                Block(kind=BlockKind.PARAGRAPH, text="Diffusion moves molecules.", page=3),
            ]),
        ],
    )


def fake_load_document(path, *, source_name=None, ocr=True, progress=None):
    path = Path(path)
    name = source_name or path.name
    kind = ingest.source_type_for(name)
    if kind is None:
        raise IngestError("unsupported")
    if path.read_bytes().startswith(b"BROKEN"):
        raise IngestError(f"'{name}' is damaged and can't be opened.")
    if progress:
        progress(0.5, "Reading page 1 of 2")
    page_count = None
    if kind == "pdf":
        import pymupdf

        with pymupdf.open(path) as pdf:
            page_count = pdf.page_count
    return make_doc(path, source_type=kind, page_count=page_count, source_name=name)


def make_pdf(path: Path, pages: int = 3) -> Path:
    import pymupdf

    pdf = pymupdf.open()
    for i in range(pages):
        page = pdf.new_page(width=300, height=400)
        page.insert_text((40, 60), f"Page {i + 1}: cells and membranes", fontsize=12)
    pdf.save(path)
    pdf.close()
    return path


# -- fixtures


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    tts = FakeTTS()
    caches: list[FakeCache] = []
    exporter = FakeExporter()
    catalog = FakeCatalog()
    audio_module = FakeAudioModule()
    FakeAssistant.mode = "ok"
    FakeAssistant.calls = []
    ai = SimpleNamespace(Assistant=FakeAssistant, AIUnavailable=AIUnavailable, AIError=AIError,
                         QuizQuestion=QuizQuestion)
    speech = SimpleNamespace(plan_section=fake_plan_section, section_plain_text=fake_section_plain_text)

    def cache_factory(root):
        cache = FakeCache(root)
        caches.append(cache)
        return cache

    monkeypatch.setattr(server, "_tts", lambda: tts)
    monkeypatch.setattr(server, "_audio_cache_class", lambda: cache_factory)
    monkeypatch.setattr(server, "_speech", lambda: speech)
    monkeypatch.setattr(server, "_ai", lambda: ai)
    monkeypatch.setattr(server, "_export_audiobook", lambda: exporter)
    monkeypatch.setattr(server, "_catalog", lambda: catalog)
    monkeypatch.setattr(server, "_audio", lambda: audio_module)
    monkeypatch.setattr(server, "_ocr_status", lambda: (False, "Install Tesseract to read scanned pages."))
    monkeypatch.setattr(ingest, "load_document", fake_load_document)
    return SimpleNamespace(tts=tts, caches=caches, exporter=exporter, catalog=catalog, ai=ai, audio=audio_module)


@pytest.fixture
def web_dir(tmp_path) -> Path:
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text(
        '<!doctype html><html><head><meta name="destiny-token" content="__DESTINY_TOKEN__">'
        '<script src="/static/app.js"></script></head><body>Destiny</body></html>',
        encoding="utf-8",
    )
    (web / "app.js").write_text("console.log('hi');", encoding="utf-8")
    (web / "styles.css").write_text("body{}", encoding="utf-8")
    (web / "sub").mkdir()
    (web / "sub" / "icon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    return web


@pytest.fixture
def app_factory(fakes, web_dir):
    made: list[JobManager] = []

    def make(**kwargs):
        jobs = kwargs.pop("jobs", None) or JobManager()
        made.append(jobs)
        kwargs.setdefault("token", TOKEN)
        kwargs.setdefault("web_dir", web_dir)
        return create_app(library=kwargs.pop("library", None) or Library(), jobs=jobs, **kwargs)

    yield make
    for jobs in made:
        jobs.shutdown(wait=True)


@pytest.fixture
def client(app_factory):
    with TestClient(app_factory(), headers=AUTH) as c:
        yield c


def wait_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] != "running":
            return data
        if time.monotonic() > deadline:
            raise AssertionError(f"job didn't finish: {data}")
        time.sleep(0.02)


def upload(client: TestClient, name: str, data: bytes, **kwargs):
    return client.post("/api/library/import", files={"file": (name, data, "application/octet-stream")}, **kwargs)


def import_book(client: TestClient, tmp_path: Path, name: str = "bio.md", data: bytes | None = None) -> dict:
    if data is None:
        data = make_pdf(tmp_path / name).read_bytes() if name.endswith(".pdf") else f"# Notes {name}\n".encode()
    response = upload(client, name, data)
    assert response.status_code == 200, response.text
    job = wait_job(client, response.json()["job_id"])
    assert job["status"] == "done", job
    return job["result"]["book"]


def set_settings(**changes):
    config.update_settings(changes)


# -- security


def test_api_requires_the_token(app_factory):
    app = app_factory()
    with TestClient(app) as c:
        missing = c.get("/api/status")
        assert missing.status_code == 401
        assert "token" in missing.json()["detail"]
        assert c.get("/api/status", headers={"X-Destiny-Token": "wrong"}).status_code == 401
        assert c.get("/api/status", headers={"X-Destiny-Token": ""}).status_code == 401
        assert c.get("/api/status", headers={"X-Destiny-Token": TOKEN + "x"}).status_code == 401
        assert c.post("/api/wipe", json={"confirm": "DELETE"}).status_code == 401
        assert c.get("/api/status", headers=AUTH).status_code == 200
        # ?token= only works for job file downloads.
        assert c.get(f"/api/status?token={TOKEN}").status_code == 401
        assert c.get(f"/api/library?token={TOKEN}").status_code == 401
        # The UI page itself doesn't need the token.
        assert c.get("/").status_code == 200


def test_random_token_per_app(app_factory):
    a = app_factory(token=None)
    b = app_factory(token=None)
    assert a.state.token != b.state.token and len(a.state.token) >= 32


@pytest.mark.parametrize("host", ["evil.com", "evil.com:8765", "127.0.0.1.evil.com", "", "attacker.localhost:80"])
def test_bad_host_is_rejected(client, host):
    for path in ("/", "/api/status", "/static/app.js"):
        response = client.get(path, headers={"Host": host})
        assert response.status_code == 403, (path, host)
        assert response.headers["content-security-policy"] == CSP


def test_loopback_hosts_are_accepted(client):
    for host in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765", "LOCALHOST:9000", "testserver"):
        assert client.get("/api/status", headers={"Host": host}).status_code == 200, host


def test_host_policy_with_a_port():
    allowed = host_policy(8765)
    for good in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765", "LocalHost:8765"):
        assert allowed(good), good
    for bad in ("127.0.0.1:8766", "127.0.0.1", "localhost", "testserver", "evil.com:8765", "[::2]:8765",
                "10.0.0.5:8765", "127.0.0.1:8765@evil.com", "user@127.0.0.1:8765", ""):
        assert not allowed(bad), bad

    lan = host_policy(8765, "0.0.0.0")
    assert lan("192.168.1.20:8765") and lan("[fe80::1]:8765") and lan("localhost:8765")
    assert not lan("evil.com:8765") and not lan("192.168.1.20:9999")

    named = host_policy(8765, "192.168.1.20")
    assert named("192.168.1.20:8765") and not named("192.168.1.21:8765")


def test_origin_check(client):
    assert client.get("/api/status", headers={"Origin": "http://testserver"}).status_code == 200
    for origin in ("http://evil.com", "null", "https://testserver", "http://testserver.evil.com", "http://localhost:8765"):
        response = client.post("/api/library/paste", json={"title": "x", "text": "y"}, headers={"Origin": origin})
        assert response.status_code == 403, origin
    with_host = client.get(
        "/api/status", headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
    )
    assert with_host.status_code == 200
    assert client.get("/", headers={"Origin": "http://evil.com"}).status_code == 403


def test_security_headers_everywhere(client):
    responses = {
        "/": client.get("/"),
        "/api/status": client.get("/api/status"),
        "/static/app.js": client.get("/static/app.js"),
        "/missing": client.get("/missing"),
        "/api/nothing": client.get("/api/nothing"),
        "401": client.get("/api/status", headers={"X-Destiny-Token": "nope"}),
    }
    for name, response in responses.items():
        headers = response.headers
        assert headers["content-security-policy"] == CSP, name
        assert headers["x-content-type-options"] == "nosniff", name
        assert headers["referrer-policy"] == "no-referrer", name
    for name in ("/", "/api/status", "/api/nothing", "401"):
        assert responses[name].headers["cache-control"] == "no-store", name
    assert responses["/static/app.js"].headers["cache-control"] == "no-cache"


def test_csp_is_exactly_as_documented():
    assert CSP == (
        "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )


def test_upload_cap(app_factory):
    with TestClient(app_factory(max_upload_bytes=1000), headers=AUTH) as c:
        too_big = upload(c, "big.txt", b"a" * 2000)
        assert too_big.status_code == 413
        assert "too big" in too_big.json()["detail"]
        assert upload(c, "ok.txt", b"b" * 900).status_code == 200

        # A chunked upload has no Content-Length to trust: the cap holds while streaming.
        boundary = "XyZ123"
        head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.txt\"\r\n"
                "Content-Type: text/plain\r\n\r\n").encode()

        def body():
            yield head
            for _ in range(10):
                yield b"c" * 500
            yield f"\r\n--{boundary}--\r\n".encode()

        chunked = c.post("/api/library/import", content=body(),
                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert chunked.status_code == 413
    # Rejected uploads leave nothing behind (the accepted one is removed when its job ends).
    leftovers = [f for f in (config.data_dir() / "tmp").rglob("*") if f.is_file() and f.stat().st_size > 1000]
    assert leftovers == []


def test_upload_lying_content_length_is_rejected_early(app_factory):
    with TestClient(app_factory(max_upload_bytes=1000), headers=AUTH) as c:
        response = c.post(
            "/api/library/import",
            content=b"x" * 10,
            headers={"Content-Type": "multipart/form-data; boundary=abc", "Content-Length": str(50 * 1024 * 1024)},
        )
        assert response.status_code == 413


def test_upload_errors(client):
    unsupported = upload(client, "setup.exe", b"MZ....")
    assert unsupported.status_code == 400
    assert "can't open '.exe'" in unsupported.json()["detail"]
    assert upload(client, "empty.txt", b"").status_code == 400
    assert client.post("/api/library/import", json={"file": "x"}).status_code == 400
    no_file = client.post("/api/library/import", data={"other": "x"}, files={"note": ("a.txt", b"hi")})
    assert no_file.status_code == 400
    assert "No file" in no_file.json()["detail"]
    damaged = client.post("/api/library/import", content=b"--abc\r\nbroken",
                          headers={"Content-Type": "multipart/form-data; boundary=abc"})
    assert damaged.status_code == 400
    assert not [p for p in (config.data_dir() / "tmp").iterdir()]


def test_upload_filename_paths_are_stripped(client, tmp_path):
    response = upload(client, "C:\\Users\\me\\Desktop\\notes.md", b"# Notes\n")
    job = wait_job(client, response.json()["job_id"])
    assert job["result"]["book"]["source_name"] == "notes.md"


def test_upload_unicode_filename(client):
    name = "Biología – capítulo 1.md"
    job = wait_job(client, upload(client, name, b"# Capitulo\n").json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["book"]["source_name"] == name


# -- UI files


def test_index_has_token(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert f'content="{TOKEN}"' in response.text
    assert "__DESTINY_TOKEN__" not in response.text


def test_index_token_is_escaped(app_factory):
    with TestClient(app_factory(token='a"<b>'), headers={"X-Destiny-Token": 'a"<b>'}) as c:
        assert 'content="a&quot;&lt;b&gt;"' in c.get("/").text


def test_missing_ui_is_friendly(app_factory, tmp_path):
    with TestClient(app_factory(web_dir=tmp_path / "nowhere")) as c:
        response = c.get("/")
        assert response.status_code == 500
        assert "missing" in response.text


def test_real_index_html_gets_token(app_factory):
    if not (server.WEB_DIR / "index.html").is_file():
        pytest.skip("the real web UI isn't there yet")
    with TestClient(app_factory(web_dir=server.WEB_DIR)) as c:
        page = c.get("/").text
        assert TOKEN in page and "__DESTINY_TOKEN__" not in page


def test_static_files(client):
    js = client.get("/static/app.js")
    assert js.status_code == 200 and js.text == "console.log('hi');"
    assert js.headers["content-type"].startswith("text/javascript")
    assert client.get("/static/styles.css").headers["content-type"].startswith("text/css")
    assert client.get("/static/sub/icon.svg").headers["content-type"] == "image/svg+xml"
    for path in ("/static/missing.js", "/static/", "/static/sub", "/static/%2e%2e/secret.txt",
                 "/static/..%2Fsecret.txt", "/static/sub/%2e%2e/%2e%2e/secret.txt", "/static/%2Fetc%2Fpasswd"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert "top secret" not in response.text


# -- status and settings


def test_status(client, fakes):
    fakes.tts.engines["system"].broken = True
    fakes.tts.engines["piper"].available = False
    data = client.get("/api/status").json()
    assert set(data) == {"version", "engines", "resolved_engine", "ai", "ocr", "data_dir"}
    engines = {e["name"]: e for e in data["engines"]}
    assert set(engines) == {"piper", "system", "edge", "browser"}
    assert engines["piper"] == {"name": "piper", "label": "Fake piper", "online": False, "available": False,
                                "reason": "piper isn't installed", "default_voice": None}
    assert engines["system"]["available"] is False and "exploded" in engines["system"]["reason"]
    # Online engines aren't asked for voices while online voices are off.
    assert engines["edge"]["available"] is True and engines["edge"]["default_voice"] is None
    assert engines["edge"]["online"] is True
    assert data["resolved_engine"] == "system"
    assert data["ai"] == {"enabled": False, "configured": False, "model": config.DEFAULT_AI_MODEL}
    assert data["ocr"] == {"available": False, "reason": "Install Tesseract to read scanned pages."}
    assert data["data_dir"] == str(config.data_dir())

    set_settings(online_tts_allowed=True, anthropic_api_key="sk-x", ai_enabled=True)
    fakes.tts.engines["piper"].available = True
    data = client.get("/api/status").json()
    engines = {e["name"]: e for e in data["engines"]}
    assert engines["edge"]["default_voice"] == "en-US-AriaNeural"
    assert engines["piper"]["default_voice"] == "v1"
    assert data["resolved_engine"] == "piper"
    assert data["ai"]["enabled"] is True and data["ai"]["configured"] is True


def test_status_survives_broken_tts_module(client, monkeypatch):
    def broken():
        raise ImportError("no tts")

    monkeypatch.setattr(server, "_tts", broken)
    data = client.get("/api/status").json()
    assert data["engines"] == [] and data["resolved_engine"] == "browser"


def test_settings_roundtrip(client):
    data = client.get("/api/settings").json()
    assert data["anthropic_api_key_set"] is False
    assert "anthropic_api_key" not in data

    updated = client.put("/api/settings", json={"speed": 1.5, "reading": {"tables": "skip"}, "voices": {"piper": "v2"}})
    assert updated.status_code == 200
    body = updated.json()
    assert body["speed"] == 1.5
    assert body["reading"]["tables"] == "skip" and body["reading"]["read_headings"] is True
    assert body["voices"] == {"piper": "v2"}
    assert config.load_settings().speed == 1.5

    keyed = client.put("/api/settings", json={"anthropic_api_key": "sk-ant-secret", "anthropic_api_key_set": False})
    assert keyed.json()["anthropic_api_key_set"] is True
    assert "sk-ant-secret" not in keyed.text
    assert "sk-ant-secret" not in client.get("/api/settings").text
    assert "sk-ant-secret" not in client.get("/api/status").text
    assert config.load_settings().anthropic_api_key == "sk-ant-secret"
    cleared = client.put("/api/settings", json={"anthropic_api_key": ""})
    assert cleared.json()["anthropic_api_key_set"] is False


@pytest.mark.parametrize(
    "changes",
    [{"speed": 10}, {"reading": {"tables": "dance"}}, {"theme": "neon"}, {"reading": None},
     {"anthropic_api_key": 5}, {"tts_engine": "robot"}, {"font_scale": "big"}],
)
def test_settings_validation(client, changes):
    response = client.put("/api/settings", json=changes)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)
    assert response.json()["detail"].startswith("Invalid setting")
    assert config.load_settings() == config.Settings()


def test_settings_body_must_be_an_object(client):
    response = client.put("/api/settings", json=[1, 2])
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


# -- library and books


def test_import_job_and_library(client, tmp_path):
    response = upload(client, "Biology Notes.md", b"# Notes\nCells.\n")
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert re.fullmatch(r"j_[0-9a-f]+", job_id)
    job = wait_job(client, job_id)
    assert set(job) == {"id", "kind", "status", "progress", "message", "error", "result"}
    assert job["kind"] == "import" and job["status"] == "done" and job["progress"] == 1.0
    assert job["error"] is None
    book = job["result"]["book"]
    assert job["result"]["warnings"] == ["Page 2 was read with OCR"]
    assert set(book) == {"id", "title", "author", "source_name", "source_type", "added_at", "section_count",
                         "page_count", "progress"}
    assert book["source_name"] == "Biology Notes.md" and book["source_type"] == "md"
    assert book["section_count"] == 3 and book["progress"] is None
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", book["added_at"])

    books = client.get("/api/library").json()["books"]
    assert [b["id"] for b in books] == [book["id"]]
    # The temp upload is gone once the job ends.
    assert not [p for p in (config.data_dir() / "tmp").iterdir()]
    # Same file again: same book.
    again = wait_job(client, upload(client, "copy.md", b"# Notes\nCells.\n").json()["job_id"])
    assert again["result"]["book"]["id"] == book["id"]
    assert len(client.get("/api/library").json()["books"]) == 1


def test_import_job_error_is_friendly(client):
    job = wait_job(client, upload(client, "bad.md", b"BROKEN file").json()["job_id"])
    assert job["status"] == "error"
    assert job["error"] == "'bad.md' is damaged and can't be opened."
    assert job["result"] is None


def test_paste(client):
    response = client.post("/api/library/paste", json={"title": "Lecture", "text": "# Hello\nSome text."})
    assert response.status_code == 200, response.text
    book = response.json()["book"]
    assert book["source_type"] == "paste"
    assert client.get("/api/library").json()["books"][0]["id"] == book["id"]
    empty = client.post("/api/library/paste", json={"title": "x", "text": "   "})
    assert empty.status_code == 400 and isinstance(empty.json()["detail"], str)
    assert client.post("/api/library/paste", json={"title": "x"}).status_code == 422


def test_delete_book(client, tmp_path):
    book = import_book(client, tmp_path)
    assert client.delete(f"/api/library/{book['id']}").json() == {"ok": True}
    assert client.delete(f"/api/library/{book['id']}").status_code == 404
    assert client.get(f"/api/books/{book['id']}").status_code == 404
    assert client.get("/api/library").json() == {"books": []}


def test_open_book(client, tmp_path):
    first = import_book(client, tmp_path, "a.md")
    second = import_book(client, tmp_path, "b.md")
    assert [b["id"] for b in client.get("/api/library").json()["books"]] == [second["id"], first["id"]]
    data = client.get(f"/api/books/{first['id']}").json()
    assert set(data) == {"book", "toc", "warnings"}
    assert data["book"]["id"] == first["id"]
    assert data["warnings"] == ["Page 2 was read with OCR"]
    entry = data["toc"][0]
    assert entry == {"id": "s0", "title": "Cells", "level": 1, "page_start": 1, "page_end": 1,
                     "block_count": 3, "char_count": entry["char_count"]}
    assert entry["char_count"] > 0
    assert [t["id"] for t in data["toc"]] == ["s0", "s1", "s2"]
    # Opening it moved it to the top.
    assert [b["id"] for b in client.get("/api/library").json()["books"]] == [first["id"], second["id"]]


@pytest.mark.parametrize("bad", ["0123456789abcdef", "..", "..%2F..%2Fsettings", "ZZZZ", "0123456789ABCDEF"])
def test_unknown_or_malicious_book_ids(client, bad):
    for path in (f"/api/books/{bad}", f"/api/books/{bad}/sections/s0", f"/api/books/{bad}/progress",
                 f"/api/books/{bad}/bookmarks", f"/api/books/{bad}/pages/1.png"):
        response = client.get(path)
        assert response.status_code == 404, path
    assert client.delete(f"/api/library/{bad}").status_code == 404


def test_section(client, tmp_path):
    book = import_book(client, tmp_path)
    data = client.get(f"/api/books/{book['id']}/sections/s0").json()
    assert set(data) == {"section", "blocks", "utterances", "prev_id", "next_id"}
    assert data["section"]["id"] == "s0"
    assert data["prev_id"] is None and data["next_id"] == "s1"
    assert data["blocks"][0] == {"kind": "heading", "text": "Cells", "level": 1, "rows": None, "page": 1,
                                 "notes": False}
    assert data["blocks"][2]["rows"] == [["A", "B"], ["1", "2"]]
    assert all("meta" not in b for b in data["blocks"])
    assert data["utterances"][1] == {"index": 1, "block": 1, "start": 0, "end": 27, "kind": "paragraph",
                                     "speech": "Cells are the unit of life.", "pause_after_ms": 350}
    middle = client.get(f"/api/books/{book['id']}/sections/s1").json()
    assert middle["prev_id"] == "s0" and middle["next_id"] == "s2"
    assert [b["notes"] for b in middle["blocks"]] == [False, True]
    last = client.get(f"/api/books/{book['id']}/sections/s2").json()
    assert last["prev_id"] == "s1" and last["next_id"] is None
    assert client.get(f"/api/books/{book['id']}/sections/s9").status_code == 404


def test_section_uses_saved_reading_options(client, tmp_path, monkeypatch, fakes):
    seen = []

    def plan(section, options=None):
        seen.append(options)
        return []

    monkeypatch.setattr(server, "_speech", lambda: SimpleNamespace(plan_section=plan))
    set_settings(reading={"tables": "skip"})
    book = import_book(client, tmp_path)
    client.get(f"/api/books/{book['id']}/sections/s0")
    assert seen[0].tables == "skip"


def test_unexpected_errors_are_json_with_headers(client, tmp_path, monkeypatch):
    book = import_book(client, tmp_path)

    def broken(section, options=None):
        raise RuntimeError("planner blew up")

    monkeypatch.setattr(server, "_speech", lambda: SimpleNamespace(plan_section=broken))
    response = client.get(f"/api/books/{book['id']}/sections/s0")
    assert response.status_code == 500
    assert response.json() == {"detail": "Something went wrong: planner blew up"}
    assert response.headers["content-security-policy"] == CSP


def test_progress(client, tmp_path):
    book = import_book(client, tmp_path)
    url = f"/api/books/{book['id']}/progress"
    assert client.get(url).json() == {"progress": None}
    saved = client.put(url, json={"section_id": "s1", "utterance": 4}).json()["progress"]
    assert saved["section_id"] == "s1" and saved["utterance"] == 4 and saved["updated_at"].endswith("Z")
    assert client.get(url).json() == {"progress": saved}
    assert client.get("/api/library").json()["books"][0]["progress"] == saved
    assert client.put(url, json={"section_id": "s9", "utterance": 0}).status_code == 404
    negative = client.put(url, json={"section_id": "s0", "utterance": -1})
    assert negative.status_code == 422 and isinstance(negative.json()["detail"], str)


def test_bookmarks(client, tmp_path):
    book = import_book(client, tmp_path)
    url = f"/api/books/{book['id']}/bookmarks"
    assert client.get(url).json() == {"bookmarks": []}
    mark = client.post(url, json={"section_id": "s0", "utterance": 2, "label": "Key idea"}).json()["bookmark"]
    assert set(mark) == {"id", "section_id", "utterance", "label", "created_at"}
    assert mark["label"] == "Key idea" and re.fullmatch(r"[0-9a-f]{16}", mark["id"])
    assert client.get(url).json() == {"bookmarks": [mark]}
    assert client.post(url, json={"section_id": "s7", "utterance": 0, "label": "x"}).status_code == 404
    assert client.delete(f"{url}/{mark['id']}").json() == {"ok": True}
    assert client.delete(f"{url}/{mark['id']}").status_code == 404
    assert client.delete(f"{url}/..").status_code in (404, 405)
    assert client.get(url).json() == {"bookmarks": []}


def test_page_images(client, tmp_path):
    pdf_book = import_book(client, tmp_path, "slides.pdf")
    assert pdf_book["page_count"] == 3
    page = client.get(f"/api/books/{pdf_book['id']}/pages/2.png")
    assert page.status_code == 200
    assert page.headers["content-type"] == "image/png"
    assert page.content.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", page.content[16:24])
    assert abs(width - 300 * 110 / 72) <= 1.5 and abs(height - 400 * 110 / 72) <= 1.5
    for bad in ("0", "4", "999999", "-1"):
        assert client.get(f"/api/books/{pdf_book['id']}/pages/{bad}.png").status_code == 404, bad
    assert client.get(f"/api/books/{pdf_book['id']}/pages/abc.png").status_code == 422
    md_book = import_book(client, tmp_path, "notes.md")
    assert client.get(f"/api/books/{md_book['id']}/pages/1.png").status_code == 404


def test_giant_pdf_pages_are_scaled_down(client, tmp_path):
    import pymupdf

    path = tmp_path / "poster.pdf"
    pdf = pymupdf.open()
    pdf.new_page(width=72 * 100, height=72 * 50)  # 100 x 50 inches
    pdf.save(path)
    pdf.close()
    book = import_book(client, tmp_path, "poster.pdf", path.read_bytes())
    png = client.get(f"/api/books/{book['id']}/pages/1.png").content
    width, height = struct.unpack(">II", png[16:24])
    assert max(width, height) <= server.MAX_RENDER_SIDE_PX


# -- voices


def test_voices(client, fakes):
    voices = client.get("/api/voices").json()["voices"]
    assert {(v["engine"], v["id"]) for v in voices} == {("piper", "v1"), ("piper", "v2"), ("system", "sys1")}
    assert all(v["installed"] for v in voices)
    assert set(voices[0]) == {"id", "name", "engine", "language", "gender", "online", "installed", "size_mb"}
    piper = client.get("/api/voices?engine=piper").json()["voices"]
    assert [v["id"] for v in piper] == ["v1", "v2"]
    assert client.get("/api/voices?engine=edge").json() == {"voices": []}
    set_settings(online_tts_allowed=True)
    assert [v["id"] for v in client.get("/api/voices?engine=edge").json()["voices"]] == ["en-US-AriaNeural"]
    assert client.get("/api/voices?engine=robot").status_code == 400


def test_voice_catalog_and_download(client, fakes):
    catalog = client.get("/api/voices/catalog").json()["voices"]
    assert [v["id"] for v in catalog] == ["en_US-lessac-medium", "en_GB-alba-medium"]
    assert catalog[0] == {"id": "en_US-lessac-medium", "name": "Lessac", "language": "en-US", "quality": "medium",
                          "size_mb": 63.2, "description": "Clear US English", "installed": False}
    response = client.post("/api/voices/download", json={"voice_id": "en_US-lessac-medium"})
    job = wait_job(client, response.json()["job_id"])
    assert job["kind"] == "voice_download" and job["status"] == "done"
    assert job["result"] == {"voice_id": "en_US-lessac-medium"}
    catalog = client.get("/api/voices/catalog").json()["voices"]
    assert [v["installed"] for v in catalog] == [True, False]
    bad = client.post("/api/voices/download", json={"voice_id": "../../evil"})
    assert bad.status_code == 400 and "isn't a valid voice name" in bad.json()["detail"]
    assert client.post("/api/voices/download", json={}).status_code == 422
    # Any well-formed Piper voice can be downloaded, not only the curated ones.
    fakes.catalog.is_valid_voice_id = lambda voice_id: voice_id == "de_DE-thorsten-medium"
    other = wait_job(client, client.post("/api/voices/download", json={"voice_id": "de_DE-thorsten-medium"}).json()["job_id"])
    assert other["status"] == "done" and other["result"] == {"voice_id": "de_DE-thorsten-medium"}


def test_voice_download_cancel(client, fakes):
    fakes.catalog.should_block = True
    job_id = client.post("/api/voices/download", json={"voice_id": "en_GB-alba-medium"}).json()["job_id"]
    deadline = time.monotonic() + 5
    while client.get(f"/api/jobs/{job_id}").json()["progress"] < 0.5 and time.monotonic() < deadline:
        time.sleep(0.01)
    cancelled = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["id"] == job_id
    assert wait_job(client, job_id)["status"] == "cancelled"
    assert client.post("/api/jobs/j_nope/cancel").status_code == 404


# -- text to speech


def test_tts_and_cache(client, fakes):
    piper = fakes.tts.engines["piper"]
    response = client.post("/api/tts", json={"text": "  Hello there.  "})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert response.headers["cache-control"] == "no-store"
    assert piper.calls == [("Hello there.", "v1", 1.0)]
    assert fakes.audio.trimmed == 1
    again = client.post("/api/tts", json={"text": "Hello there."})
    assert again.content == response.content
    assert len(piper.calls) == 1  # served from the cache
    assert fakes.caches[-1].keys[-1] == ("piper", "v1", "1.00", "Hello there.")


def test_tts_rate_reaches_the_engine_and_is_cached_separately(client, fakes):
    piper = fakes.tts.engines["piper"]
    client.post("/api/tts", json={"text": "Faster please.", "rate": 1.5})
    client.post("/api/tts", json={"text": "Faster please.", "rate": 1.25})
    client.post("/api/tts", json={"text": "Faster please.", "rate": 1.5})
    assert piper.calls == [("Faster please.", "v1", 1.5), ("Faster please.", "v1", 1.25)]
    keys = set(fakes.caches[-1].store)
    assert keys == {("piper", "v1", "1.50", "Faster please."), ("piper", "v1", "1.25", "Faster please.")}
    for bad in (0.2, 3.5, "fast"):
        response = client.post("/api/tts", json={"text": "x", "rate": bad})
        assert response.status_code == 422, bad
        assert isinstance(response.json()["detail"], str)


def test_tts_voice_choice(client, fakes):
    piper = fakes.tts.engines["piper"]
    client.post("/api/tts", json={"text": "a", "voice": "v2"})
    set_settings(voices={"piper": "v2", "system": "sys1"})
    client.post("/api/tts", json={"text": "b"})
    client.post("/api/tts", json={"text": "c", "engine": "system"})
    assert [c[1] for c in piper.calls] == ["v2", "v2"]
    assert fakes.tts.engines["system"].calls == [("c", "sys1", 1.0)]
    # A voice that isn't installed (any more) falls back to the saved one, then the default.
    client.post("/api/tts", json={"text": "d", "voice": "not-installed"})
    set_settings(voices={"piper": "deleted-voice"})
    client.post("/api/tts", json={"text": "e", "voice": "gone"})
    assert [c[1] for c in piper.calls[2:]] == ["v2", "v1"]


def test_tts_without_any_voice(client, fakes):
    fakes.tts.engines["piper"].voices = []
    response = client.post("/api/tts", json={"text": "hello"})
    assert response.status_code == 503 and "no voices installed" in response.json()["detail"]


def test_tts_validation_and_gates(client, fakes):
    assert client.post("/api/tts", json={"text": "   "}).status_code == 400
    too_long = client.post("/api/tts", json={"text": "a" * 5001})
    assert too_long.status_code == 400 and "5,000" in too_long.json()["detail"]
    assert client.post("/api/tts", json={"text": "a" * 5000}).status_code == 200
    assert client.post("/api/tts", json={"text": "hi", "engine": "browser"}).status_code == 400
    assert client.post("/api/tts", json={"text": "hi", "engine": "robot"}).status_code == 400
    online = client.post("/api/tts", json={"text": "hi", "engine": "edge"})
    assert online.status_code == 403
    assert online.json()["detail"] == ONLINE_TTS_OFF
    assert fakes.tts.engines["edge"].calls == []
    set_settings(online_tts_allowed=True)
    assert client.post("/api/tts", json={"text": "hi", "engine": "edge"}).status_code == 200
    assert fakes.tts.engines["edge"].calls == [("hi", "en-US-AriaNeural", 1.0)]


def test_tts_auto_resolves_to_browser(client, fakes):
    fakes.tts.engines["piper"].available = False
    fakes.tts.engines["system"].available = False
    response = client.post("/api/tts", json={"text": "hi", "engine": "auto"})
    assert response.status_code == 400 and "browser" in response.json()["detail"]


def test_tts_engine_errors(client, fakes):
    fakes.tts.engines["piper"].error = EngineUnavailable("Download a voice first.")
    unavailable = client.post("/api/tts", json={"text": "one"})
    assert unavailable.status_code == 503 and unavailable.json()["detail"] == "Download a voice first."
    fakes.tts.engines["piper"].error = TTSError("The voice crashed.")
    failed = client.post("/api/tts", json={"text": "two"})
    assert failed.status_code == 502 and failed.json()["detail"] == "The voice crashed."


def test_tts_without_trim_silence(client, fakes, monkeypatch):
    monkeypatch.setattr(server, "_audio", lambda: SimpleNamespace())
    assert client.post("/api/tts", json={"text": "no trimming"}).status_code == 200


# -- AI


def enable_ai():
    set_settings(ai_enabled=True, anthropic_api_key="sk-test")


AI_REQUESTS = [
    ("/api/ai/explain", {"section_id": "s0", "passage": "unit of life"}, "text"),
    ("/api/ai/ask", {"section_id": "s0", "question": "What is a cell?",
                     "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
     "text"),
    ("/api/ai/summarize", {"section_id": "s0", "length": "detailed"}, "text"),
    ("/api/ai/quiz", {"section_id": "s0", "count": 3}, "questions"),
    ("/api/ai/narrate", {"section_id": "s0", "block": 2}, "text"),
]


def test_ai_off_returns_409_without_calling_claude(client, tmp_path):
    book = import_book(client, tmp_path)
    for path, body, _ in AI_REQUESTS:
        response = client.post(path, json={"book_id": book["id"], **body})
        assert response.status_code == 409, path
        assert "turned off" in response.json()["detail"]
    assert FakeAssistant.calls == []


@pytest.mark.parametrize("mode,status", [("unavailable", 409), ("error", 502)])
def test_ai_errors(client, tmp_path, mode, status):
    enable_ai()
    book = import_book(client, tmp_path)
    FakeAssistant.mode = mode
    for path, body, _ in AI_REQUESTS:
        response = client.post(path, json={"book_id": book["id"], **body})
        assert response.status_code == status, path
        assert response.json()["detail"] in ("Add your Claude API key in Settings.",
                                              "Claude is busy right now. Try again in a minute.")


def test_ai_success(client, tmp_path):
    enable_ai()
    book = import_book(client, tmp_path)
    for path, body, key in AI_REQUESTS:
        response = client.post(path, json={"book_id": book["id"], **body})
        assert response.status_code == 200, (path, response.text)
        assert set(response.json()) == {key}
    calls = {name: (args, kwargs) for name, args, kwargs in FakeAssistant.calls}
    context = "Cells\nCells are the unit of life.\nA B"
    assert calls["explain"] == (("unit of life",), {"context": context, "book_title": "Biology Basics",
                                                     "section_title": "Cells"})
    assert calls["ask"][1]["history"] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert calls["ask"][1]["section_title"] == "Cells"
    assert calls["summarize"] == ((context,), {"section_title": "Cells", "book_title": "Biology Basics",
                                               "length": "detailed"})
    assert calls["quiz"][1] == {"section_title": "Cells", "count": 3, "book_title": "Biology Basics"}
    block, = calls["narrate_block"][0]
    assert block.kind == BlockKind.TABLE
    assert calls["narrate_block"][1]["book_title"] == "Biology Basics"
    quiz = client.post("/api/ai/quiz", json={"book_id": book["id"], "section_id": "s0", "count": 3}).json()
    assert quiz["questions"][0] == {"question": "Q0", "choices": ["a", "b"], "answer_index": 0, "explanation": "e"}


def test_ai_context_is_capped(client, tmp_path, monkeypatch, fakes):
    seen = []

    def plain(section, *, max_chars=None):
        seen.append(max_chars)
        return "text"

    monkeypatch.setattr(server, "_speech", lambda: SimpleNamespace(section_plain_text=plain))
    enable_ai()
    book = import_book(client, tmp_path)
    client.post("/api/ai/summarize", json={"book_id": book["id"], "section_id": "s0"})
    assert seen == [150_000]


def test_ai_quiz_count_is_clamped(client, tmp_path):
    enable_ai()
    book = import_book(client, tmp_path)
    many = client.post("/api/ai/quiz", json={"book_id": book["id"], "section_id": "s0", "count": 50}).json()
    none = client.post("/api/ai/quiz", json={"book_id": book["id"], "section_id": "s0", "count": 0}).json()
    default = client.post("/api/ai/quiz", json={"book_id": book["id"], "section_id": "s0"}).json()
    assert (len(many["questions"]), len(none["questions"]), len(default["questions"])) == (10, 1, 5)


def test_ai_request_validation(client, tmp_path):
    enable_ai()
    book = import_book(client, tmp_path)
    bid = book["id"]
    assert client.post("/api/ai/summarize", json={"book_id": bid, "section_id": "s0", "length": "huge"}).status_code == 422
    assert client.post("/api/ai/explain", json={"book_id": bid, "section_id": "s0", "passage": " "}).status_code == 400
    assert client.post("/api/ai/ask", json={"book_id": bid, "section_id": "s0", "question": ""}).status_code == 400
    bad_role = client.post("/api/ai/ask", json={"book_id": bid, "section_id": "s0", "question": "q",
                                                "history": [{"role": "system", "content": "x"}]})
    assert bad_role.status_code == 422
    assert client.post("/api/ai/narrate", json={"book_id": bid, "section_id": "s0", "block": 3}).status_code == 404
    assert client.post("/api/ai/narrate", json={"book_id": bid, "section_id": "s0", "block": -1}).status_code == 404
    assert client.post("/api/ai/explain", json={"book_id": bid, "section_id": "s5", "passage": "x"}).status_code == 404
    assert client.post("/api/ai/explain", json={"book_id": "0" * 16, "section_id": "s0", "passage": "x"}).status_code == 404


def test_ai_describe_page(client, tmp_path):
    enable_ai()
    pdf = import_book(client, tmp_path, "lecture.pdf")
    response = client.post("/api/ai/describe-page", json={"book_id": pdf["id"], "page": 2})
    assert response.status_code == 200 and response.json() == {"text": "describe_image answer"}
    (png,), kwargs = next((a, k) for n, a, k in FakeAssistant.calls if n == "describe_image")
    assert png.startswith(b"\x89PNG")
    width = struct.unpack(">I", png[16:20])[0]
    assert abs(width - 300 * 150 / 72) <= 1.5
    assert kwargs["context"] == "Membranes Notes for the speaker."
    assert kwargs["book_title"] == "Biology Basics" and kwargs["section_title"] == "Membranes"
    assert client.post("/api/ai/describe-page", json={"book_id": pdf["id"], "page": 9}).status_code == 404
    md = import_book(client, tmp_path, "notes.md")
    assert client.post("/api/ai/describe-page", json={"book_id": md["id"], "page": 1}).status_code == 404


# -- export


def test_export_job_and_download(client, tmp_path, fakes):
    book = import_book(client, tmp_path)
    set_settings(voices={"piper": "v2"})
    response = client.post("/api/export", json={"book_id": book["id"], "section_ids": ["s0", "s2", "s0"],
                                                "format": "wav", "rate": 1.25})
    assert response.status_code == 200, response.text
    job = wait_job(client, response.json()["job_id"])
    assert job["kind"] == "export" and job["status"] == "done", job
    call = fakes.exporter.calls[0]
    assert call["engine"] is fakes.tts.engines["piper"] and call["voice"] == "v2"
    assert call["section_ids"] == ["s0", "s2"] and call["fmt"] == "wav" and call["rate"] == 1.25
    assert call["out_dir"] == config.exports_dir() / job["id"]

    files = job["result"]["files"]
    assert [f["name"] for f in files] == ["01 - Cells.mp3", "02 - Membranes & Transport ü.mp3", "playlist.m3u"]
    assert files[1]["url"] == f"/api/jobs/{job['id']}/files/02%20-%20Membranes%20%26%20Transport%20%C3%BC.mp3"
    assert files[0]["size"] == len(b"audio 0")

    with TestClient(client.app) as plain:  # no token header, like an <a href> download
        for f in files:
            download = plain.get(f"{f['url']}?token={TOKEN}")
            assert download.status_code == 200, f
            assert download.headers["content-disposition"].startswith("attachment;")
            assert "filename*=UTF-8''" + f["url"].rsplit("/", 1)[1] in download.headers["content-disposition"]
        first = plain.get(f"{files[0]['url']}?token={TOKEN}")
        assert first.content == b"audio 0" and first.headers["content-type"] == "audio/mpeg"
        assert plain.get(files[0]["url"]).status_code == 401
        assert plain.get(f"{files[0]['url']}?token=wrong").status_code == 401
    assert client.get(files[0]["url"]).status_code == 200

    base = f"/api/jobs/{job['id']}/files/"
    for attack in ("..%2F..%2Fsettings.json", "%2e%2e", "%2E%2E%2Fx", "outside.mp3", "..%5C..%5Csettings.json",
                   "%2Fetc%2Fpasswd", "missing.mp3"):
        assert client.get(base + attack).status_code == 404, attack
    assert client.get("/api/jobs/j_unknown/files/01%20-%20Cells.mp3").status_code == 404


def test_export_defaults_to_all_sections_and_mp3(client, tmp_path, fakes):
    book = import_book(client, tmp_path)
    job = wait_job(client, client.post("/api/export", json={"book_id": book["id"]}).json()["job_id"])
    assert job["status"] == "done"
    call = fakes.exporter.calls[0]
    assert call["section_ids"] is None and call["fmt"] == "mp3" and call["rate"] == 1.0 and call["voice"] == "v1"


def test_export_validation(client, tmp_path, fakes):
    book = import_book(client, tmp_path)
    bid = book["id"]
    assert client.post("/api/export", json={"book_id": "0" * 16}).status_code == 404
    assert client.post("/api/export", json={"book_id": bid, "section_ids": ["s9"]}).status_code == 404
    assert client.post("/api/export", json={"book_id": bid, "section_ids": []}).status_code == 400
    assert client.post("/api/export", json={"book_id": bid, "format": "ogg"}).status_code == 422
    assert client.post("/api/export", json={"book_id": bid, "rate": 9}).status_code == 422
    assert client.post("/api/export", json={"book_id": bid, "engine": "browser"}).status_code == 400
    online = client.post("/api/export", json={"book_id": bid, "engine": "edge"})
    assert online.status_code == 403 and online.json()["detail"] == ONLINE_TTS_OFF
    fakes.tts.engines["system"].available = False
    assert client.post("/api/export", json={"book_id": bid, "engine": "system"}).status_code == 503
    fakes.tts.engines["piper"].available = False
    auto = client.post("/api/export", json={"book_id": bid, "engine": "auto"})
    assert auto.status_code == 400 and "can't make audio files" in auto.json()["detail"]
    assert fakes.exporter.calls == []


@pytest.mark.parametrize("mode,error", [
    ("tts_error", "The voice stopped working."),
    ("crash", "Something went wrong: bad thing"),
])
def test_export_failures(client, tmp_path, fakes, mode, error):
    book = import_book(client, tmp_path)
    fakes.exporter.mode = mode
    job = wait_job(client, client.post("/api/export", json={"book_id": book["id"]}).json()["job_id"])
    assert job["status"] == "error" and job["error"] == error
    assert not (config.exports_dir() / job["id"]).exists()


def test_export_cancel(client, tmp_path, fakes):
    book = import_book(client, tmp_path)
    fakes.exporter.mode = "block"
    job_id = client.post("/api/export", json={"book_id": book["id"]}).json()["job_id"]
    deadline = time.monotonic() + 5
    while not (config.exports_dir() / job_id / "partial.mp3").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    client.post(f"/api/jobs/{job_id}/cancel")
    job = wait_job(client, job_id)
    assert job["status"] == "cancelled" and job["error"] is None
    assert not (config.exports_dir() / job_id).exists()
    assert client.get(f"/api/jobs/{job_id}/files/partial.mp3").status_code == 404


def test_unknown_job(client):
    assert client.get("/api/jobs/j_123").status_code == 404


# -- wipe


def test_wipe(client, tmp_path, fakes):
    book = import_book(client, tmp_path)
    enable_ai()
    client.post("/api/tts", json={"text": "cached"})
    voice = config.voices_dir() / "piper" / "keep.onnx"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"voice")
    (config.cache_dir() / "x.wav").write_bytes(b"RIFF")

    wrong = client.post("/api/wipe", json={"confirm": "delete"})
    assert wrong.status_code == 400
    assert client.get("/api/library").json()["books"]

    assert client.post("/api/wipe", json={"confirm": "DELETE"}).json() == {"ok": True}
    assert client.get("/api/library").json() == {"books": []}
    assert client.get(f"/api/books/{book['id']}").status_code == 404
    assert client.get("/api/settings").json()["anthropic_api_key_set"] is False
    assert not config.settings_path().exists()
    assert list(config.cache_dir().iterdir()) == []
    assert voice.read_bytes() == b"voice"
    # The in-memory audio cache was reset too: the same text is synthesized again.
    calls = len(fakes.tts.engines["piper"].calls)
    client.post("/api/tts", json={"text": "cached"})
    assert len(fakes.tts.engines["piper"].calls) == calls + 1


class _QuickClearJobs(JobManager):
    """Wipe normally waits 10 s for jobs to stop; these tests don't need to wait that long."""

    def clear(self, *, wait: float = 0.0) -> None:
        super().clear(wait=min(wait, 0.2))


def test_wipe_during_a_slow_import_does_not_bring_the_book_back(app_factory, monkeypatch):
    """A loader that never reports progress outlives the wipe's wait; it must not save the book afterwards."""
    started, release = threading.Event(), threading.Event()

    def slow_loader(path, *, source_name=None, ocr=True, progress=None):
        data = Path(path).read_bytes()
        started.set()
        release.wait(10)
        # On Windows the loader keeps the upload open, so the wipe can't delete it.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
        return fake_load_document(path, source_name=source_name, ocr=ocr)

    monkeypatch.setattr(ingest, "load_document", slow_loader)
    with TestClient(app_factory(jobs=_QuickClearJobs()), headers=AUTH) as client:
        job_id = upload(client, "Private diary.md", b"# Dear diary\nSecrets.\n").json()["job_id"]
        assert started.wait(5)
        assert client.post("/api/wipe", json={"confirm": "DELETE"}).json() == {"ok": True}
        release.set()
        assert wait_job(client, job_id)["status"] == "cancelled"
        assert client.get("/api/library").json() == {"books": []}
        assert not any(config.library_dir().iterdir())


def test_wipe_erases_the_reader_window_browser_data(client):
    root = config.data_dir()
    for name in ("webview", "browser-profile"):
        storage = root / name / "Default" / "Local Storage" / "leveldb"
        storage.mkdir(parents=True)
        (storage / "000003.log").write_bytes(b"destiny.lastBook")
    assert client.post("/api/wipe", json={"confirm": "DELETE"}).json() == {"ok": True}
    assert not (root / "webview").exists() and not (root / "browser-profile").exists()


def test_startup_finishes_an_interrupted_erase(app_factory):
    root = config.data_dir()
    leftover = root / "webview" / "EBWebView" / "Default" / "LOCK"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"")
    (root / ".erase-pending").write_text("x", encoding="utf-8")
    app_factory()
    assert not (root / "webview").exists() and not (root / ".erase-pending").exists()
    # Without the marker, the window's data is left alone.
    leftover.parent.mkdir(parents=True)
    app_factory()
    assert leftover.parent.is_dir()


def test_settings_save_waits_for_a_briefly_locked_file(client, monkeypatch):
    """On Windows, replacing settings.json fails while another thread or a virus scanner reads it."""
    real_replace, attempts = config.os.replace, []

    def replace(src, dst):
        attempts.append(Path(dst).name)
        if len(attempts) <= 2:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        real_replace(src, dst)

    monkeypatch.setattr(config, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(config.os, "replace", replace)
    monkeypatch.setattr(config.time, "sleep", lambda seconds: None)
    response = client.put("/api/settings", json={"speed": 1.25})
    assert response.status_code == 200, response.text
    assert attempts == ["settings.json"] * 3
    assert config.load_settings().speed == 1.25
    assert [p.name for p in config.data_dir().glob(".settings-*")] == []


@pytest.mark.parametrize("key", ["sk-ant-api03-AB CD", "sk-ant-api03-AB​CD", "sk-ant-api03-AB\nCD"])
def test_api_key_with_invisible_characters_is_refused(client, key):
    response = client.put("/api/settings", json={"anthropic_api_key": key})
    assert response.status_code == 422
    assert "spaces or invisible characters" in response.json()["detail"]
    assert "AB" not in response.json()["detail"]  # the key itself is never echoed back
    assert client.get("/api/settings").json()["anthropic_api_key_set"] is False


def test_disk_full_is_explained_plainly(client, tmp_path, monkeypatch):
    book = import_book(client, tmp_path)

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device", str(config.data_dir() / "library" / "x"))

    monkeypatch.setattr(config.os, "replace", full)
    response = client.put(f"/api/books/{book['id']}/progress", json={"section_id": "s1", "utterance": 2})
    assert response.status_code == 500
    assert "disk is full" in response.json()["detail"].lower()
    assert str(config.data_dir()) not in response.json()["detail"]
    assert client.get(f"/api/books/{book['id']}/progress").json() == {"progress": None}


def test_friendly_error_for_a_full_disk():
    message = friendly_error(OSError(errno.ENOSPC, "No space left on device", "/home/me/.local/share/athena/x"))
    assert "disk is full" in message.lower() and "/home/me" not in message
    assert friendly_error(OSError(errno.EACCES, "Permission denied")).startswith("Something went wrong")


def test_uvicorn_starts_without_a_console(monkeypatch):
    """A windowed app (pythonw) has no stdout/stderr; uvicorn's default log config would crash on that."""
    jobs = JobManager(max_workers=1)
    try:
        app = create_app(token=TOKEN, jobs=jobs)
        with monkeypatch.context() as m:
            m.setattr("sys.stdout", None)
            m.setattr("sys.stderr", None)
            uv = server._uvicorn_server(app, "127.0.0.1", 8765)
    finally:
        jobs.shutdown(wait=True)
    assert uv.config.log_config is None


def test_validation_errors_are_strings(client):
    response = client.post("/api/library/paste", content=b"{not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid request: the request wasn't valid JSON"


def test_content_disposition():
    assert content_disposition("01 - Cells.mp3") == (
        "attachment; filename=\"01 - Cells.mp3\"; filename*=UTF-8''01%20-%20Cells.mp3"
    )
    header = content_disposition('Ch "1" ü.mp3')
    assert header.startswith('attachment; filename="Ch _1_ _.mp3"; filename*=UTF-8\'\'Ch%20%221%22%20%C3%BC.mp3')
    assert content_disposition("日本.mp3").startswith('attachment; filename="download.mp3"')
    header.encode("latin-1")


# -- running in a background thread (for the desktop window)


def _get(port: int, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path, headers={"Host": f"127.0.0.1:{port}", **(headers or {})})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_start_server_in_thread(fakes):
    import socket

    port = _free_port()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        url, stop = server.start_server_in_thread(port=port)
    finally:
        blocker.close()
    try:
        match = re.fullmatch(r"http://127\.0\.0\.1:(\d+)/", url)
        assert match, url
        actual = int(match.group(1))
        assert port < actual < port + server.PORT_ATTEMPTS  # the busy port was skipped
        status, body = _get(actual, "/")
        assert status in (200, 500)  # 500 only while the web UI files are missing
        assert _get(actual, "/api/status")[0] == 401
        assert _get(actual, "/", {"Host": "evil.com"})[0] == 403
        if status == 200:
            token = re.search(rb'name="destiny-token" content="([^"]+)"', body).group(1).decode()
            ok, data = _get(actual, "/api/settings", {"X-Destiny-Token": token})
            assert ok == 200 and b"anthropic_api_key_set" in data
    finally:
        stop()
    stop()  # a second call does nothing
    with pytest.raises(OSError):  # stopped: nothing listens any more
        _get(actual, "/")
    assert not [t for t in threading.enumerate() if t.name == "destiny-server"]
    with pytest.raises(ValueError):
        server.start_server_in_thread(host="0.0.0.0", port=port)


# -- jobs


def test_job_manager_result_and_shape():
    jobs = JobManager()
    try:
        def work(job: Job):
            job.update(0.3, "Halfway")
            job.update(7, None)  # clamped
            return {"answer": 42}

        job = jobs.submit("import", work)
        assert job.wait(5)
        assert job.to_dict() == {"id": job.id, "kind": "import", "status": "done", "progress": 1.0,
                                 "message": "Done", "error": None, "result": {"answer": 42}}
        assert jobs.get(job.id) is job
    finally:
        jobs.shutdown(wait=True)


def test_job_errors_are_friendly(caplog):
    jobs = JobManager()
    try:
        cases = [
            (IngestError("This PDF is password-protected."), "This PDF is password-protected."),
            (TTSError("No voice installed."), "No voice installed."),
            (JobError("Book not found."), "Book not found."),
            (KeyError("x"), "Something went wrong: 'x'"),
            (RuntimeError(""), "Something went wrong: RuntimeError"),
            (ValueError("x" * 500), "Something went wrong: " + "x" * 199 + "…"),
        ]
        for exc, message in cases:
            def fail(job, exc=exc):
                raise exc

            job = jobs.submit("export", fail)
            assert job.wait(5)
            assert job.status == "error" and job.error == message and job.result is None
        assert "Traceback" in caplog.text or any(r.exc_info for r in caplog.records)
    finally:
        jobs.shutdown(wait=True)


def test_job_cancel_before_start_runs_cleanup(tmp_path):
    jobs = JobManager(max_workers=1)
    gate = threading.Event()
    cleaned = []
    try:
        blocker = jobs.submit("export", lambda job: gate.wait(5))
        queued = jobs.submit("import", lambda job: "never", cleanup=lambda: cleaned.append(True),
                             files_root=tmp_path)
        assert queued.files_dir == tmp_path / queued.id and queued.files_dir.is_dir()
        jobs.cancel(queued.id)
        assert queued.status == "cancelled" and cleaned == [True]
        assert not queued.files_dir.exists()
        gate.set()
        assert blocker.wait(5) and blocker.status == "done"
    finally:
        jobs.shutdown(wait=True)


def test_job_cooperative_cancel():
    jobs = JobManager()
    try:
        started = threading.Event()

        def work(job: Job):
            started.set()
            while True:
                job.raise_if_cancelled()
                time.sleep(0.005)

        job = jobs.submit("voice_download", work)
        assert started.wait(5)
        jobs.cancel(job.id)
        assert job.wait(5)
        assert job.status == "cancelled" and job.cancelled
        with pytest.raises(JobCancelled):
            job.raise_if_cancelled()
    finally:
        jobs.shutdown(wait=True)


def test_job_pruning_removes_files(tmp_path):
    jobs = JobManager(retention_seconds=3600)
    try:
        job = jobs.submit("export", lambda j: j.files_dir.joinpath("a.mp3").write_bytes(b"x"), files_root=tmp_path)
        assert job.wait(5)
        jobs.prune(now=time.time() + 60)
        assert jobs.get(job.id) is job and job.files_dir.exists()
        jobs.prune(now=time.time() + 3601)
        with pytest.raises(KeyError):
            jobs.get(job.id)
        assert not job.files_dir.exists()
    finally:
        jobs.shutdown(wait=True)


def test_shutdown_cancels_running_jobs():
    jobs = JobManager()
    started = threading.Event()

    def work(job: Job):
        started.set()
        while not job.cancelled:
            time.sleep(0.005)

    job = jobs.submit("export", work)
    assert started.wait(5)
    jobs.shutdown(wait=True)
    assert job.status == "cancelled"
    late = jobs.submit("export", lambda j: 1)
    assert late.status == "cancelled"


def test_export_cancelled_exception_means_cancelled():
    export = pytest.importorskip("destiny_bot.export")
    jobs = JobManager()
    try:
        def work(job):
            raise export.ExportCancelled("Export cancelled.")

        def fail(job):
            raise export.ExportError("There's nothing to read in these sections.")

        cancelled = jobs.submit("export", work)
        failed = jobs.submit("export", fail)
        assert cancelled.wait(5) and failed.wait(5)
        assert cancelled.status == "cancelled"
        assert failed.status == "error" and failed.error == "There's nothing to read in these sections."
    finally:
        jobs.shutdown(wait=True)


def test_friendly_error_for_ai_errors_when_loaded():
    ai = pytest.importorskip("destiny_bot.ai")
    assert friendly_error(ai.AIError("Claude is overloaded.")) == "Claude is overloaded."


# -- integration with the real modules (skipped while they aren't available)


def _module_ready(name: str, *attrs: str) -> bool:
    try:
        module = __import__(name, fromlist=["_"])
    except Exception:
        return False
    return all(hasattr(module, a) for a in attrs)


@pytest.mark.skipif(not _module_ready("destiny_bot.speech", "plan_section", "section_plain_text"),
                    reason="destiny_bot.speech isn't available yet")
@pytest.mark.skipif(not _module_ready("destiny_bot.ingest.text", "load_text_file"),
                    reason="destiny_bot.ingest.text isn't available yet")
def test_integration_import_and_read_section(web_dir):
    jobs = JobManager()
    try:
        with TestClient(create_app(token=TOKEN, jobs=jobs, web_dir=web_dir), headers=AUTH) as c:
            text = b"# Cells\n\nCells are the basic unit of life, e.g. bacteria. They divide.\n\n## Membranes\n\nMembranes control transport.\n"
            job = wait_job(c, upload(c, "biology.md", text).json()["job_id"])
            assert job["status"] == "done", job["error"]
            book = job["result"]["book"]
            data = c.get(f"/api/books/{book['id']}").json()
            first = data["toc"][0]["id"]
            section = c.get(f"/api/books/{book['id']}/sections/{first}").json()
            assert section["utterances"]
            for u in section["utterances"]:
                assert u["speech"].strip()
                if u["block"] >= 0 and section["blocks"][u["block"]]["kind"] != "table":
                    assert 0 <= u["start"] <= u["end"] <= len(section["blocks"][u["block"]]["text"])
    finally:
        jobs.shutdown(wait=True)


def _system_voice_ready() -> bool:
    if not _module_ready("destiny_bot.tts", "get_engine"):
        return False
    try:
        from destiny_bot.tts import get_engine

        return bool(get_engine("system").is_available()[0])
    except Exception:
        return False


@pytest.mark.skipif(not _system_voice_ready(), reason="no real system voice available")
def test_integration_real_tts_with_cache(web_dir):
    jobs = JobManager()
    try:
        with TestClient(create_app(token=TOKEN, jobs=jobs, web_dir=web_dir), headers=AUTH) as c:
            first = c.post("/api/tts", json={"text": "Hello from Destiny Bot.", "engine": "system"})
            assert first.status_code == 200, first.text
            assert first.headers["content-type"] in ("audio/wav", "audio/mpeg")
            assert len(first.content) > 1000
            second = c.post("/api/tts", json={"text": "Hello from Destiny Bot.", "engine": "system"})
            assert second.content == first.content
            assert any(config.cache_dir().iterdir())
            status = c.get("/api/status").json()
            assert {e["name"] for e in status["engines"]} >= {"piper", "system", "edge", "browser"}
    finally:
        jobs.shutdown(wait=True)


@pytest.mark.skipif(not _module_ready("destiny_bot.tts.catalog", "CATALOG", "installed_piper_voices"),
                    reason="destiny_bot.tts.catalog isn't available yet")
def test_integration_real_catalog(web_dir):
    jobs = JobManager()
    try:
        with TestClient(create_app(token=TOKEN, jobs=jobs, web_dir=web_dir), headers=AUTH) as c:
            voices = c.get("/api/voices/catalog").json()["voices"]
            assert voices and all(v["installed"] is False for v in voices)
            assert any(v["id"] == "en_US-lessac-medium" for v in voices)
    finally:
        jobs.shutdown(wait=True)


@pytest.mark.skipif(not (_module_ready("destiny_bot.ai", "Assistant", "AIUnavailable")
                         and _module_ready("destiny_bot.speech", "section_plain_text")),
                    reason="destiny_bot.ai or destiny_bot.speech isn't available yet")
def test_integration_real_ai_without_key(web_dir, tmp_path, monkeypatch):
    set_settings(ai_enabled=True)
    monkeypatch.setattr(ingest, "load_document", fake_load_document)
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nSomething.\n", encoding="utf-8")
    lib = Library()
    doc = lib.import_file(path)
    jobs = JobManager()
    try:
        with TestClient(create_app(library=lib, token=TOKEN, jobs=jobs, web_dir=web_dir), headers=AUTH) as c:
            response = c.post("/api/ai/explain", json={"book_id": doc.id, "section_id": "s0", "passage": "Cells"})
            assert response.status_code == 409, response.text
            assert isinstance(response.json()["detail"], str)
    finally:
        jobs.shutdown(wait=True)


@pytest.mark.skipif(not _module_ready("destiny_bot.ingest.pdf", "load_pdf"), reason="the PDF loader isn't available yet")
def test_integration_real_pdf_import_and_page_image(web_dir, tmp_path):
    pdf = make_pdf(tmp_path / "lecture.pdf", pages=2)
    jobs = JobManager()
    try:
        with TestClient(create_app(token=TOKEN, jobs=jobs, web_dir=web_dir), headers=AUTH) as c:
            job = wait_job(c, upload(c, "Lecture 1.pdf", pdf.read_bytes()).json()["job_id"], timeout=60)
            assert job["status"] == "done", job["error"]
            book = job["result"]["book"]
            assert book["source_type"] == "pdf" and book["page_count"] == 2
            assert book["source_name"] == "Lecture 1.pdf"
            page = c.get(f"/api/books/{book['id']}/pages/2.png")
            assert page.status_code == 200 and page.content.startswith(b"\x89PNG")
            assert c.get(f"/api/books/{book['id']}/pages/3.png").status_code == 404
    finally:
        jobs.shutdown(wait=True)


@pytest.mark.skipif(not _module_ready("destiny_bot.tts.cache", "AudioCache"), reason="the audio cache isn't available yet")
def test_integration_real_audio_cache(app_factory, fakes, monkeypatch):
    from destiny_bot.tts.cache import AudioCache

    monkeypatch.setattr(server, "_audio_cache_class", lambda: AudioCache)
    with TestClient(app_factory(), headers=AUTH) as c:
        first = c.post("/api/tts", json={"text": "Cached on disk.", "rate": 1.1})
        second = c.post("/api/tts", json={"text": "Cached on disk.", "rate": 1.1})
        assert first.status_code == second.status_code == 200
        assert first.content == second.content
        assert len(fakes.tts.engines["piper"].calls) == 1
        assert [p.suffix for p in config.cache_dir().iterdir()] == [".wav"]


@pytest.mark.skipif(
    not (_module_ready("destiny_bot.export", "export_audiobook") and _module_ready("destiny_bot.speech", "plan_section")
         and _system_voice_ready()),
    reason="the real exporter, speech planner or a system voice isn't available",
)
def test_integration_real_export(web_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "load_document", fake_load_document)
    jobs = JobManager()
    try:
        with TestClient(create_app(token=TOKEN, jobs=jobs, web_dir=web_dir), headers=AUTH) as c:
            book = import_book(c, tmp_path, "bio.md")
            response = c.post("/api/export", json={"book_id": book["id"], "section_ids": ["s2"], "engine": "system",
                                                   "format": "mp3"})
            assert response.status_code == 200, response.text
            job = wait_job(c, response.json()["job_id"], timeout=120)
            assert job["status"] == "done", job["error"]
            files = job["result"]["files"]
            audio = [f for f in files if f["name"].endswith(".mp3")]
            assert audio and audio[0]["size"] > 0
            download = c.get(audio[0]["url"])
            assert download.status_code == 200 and len(download.content) == audio[0]["size"]
    finally:
        jobs.shutdown(wait=True)


# -- voice warm-up


def test_warm_up_loads_default_offline_voice(fakes):
    server._warm_up_voice()
    assert fakes.tts.engines["piper"].calls == [("Ready.", None, 1.0)]


def test_warm_up_uses_saved_voice_when_installed(fakes):
    config.update_settings({"voices": {"piper": "v2"}})
    server._warm_up_voice()
    assert fakes.tts.engines["piper"].calls == [("Ready.", "v2", 1.0)]


def test_warm_up_never_touches_online_voices(fakes):
    config.update_settings({"tts_engine": "edge", "online_tts_allowed": True})
    for name in ("piper", "system"):
        fakes.tts.engines[name].available = False
    server._warm_up_voice()
    assert all(not engine.calls for engine in fakes.tts.engines.values())


def test_warm_up_failure_is_harmless(fakes):
    fakes.tts.engines["piper"].error = RuntimeError("model missing")
    server._warm_up_voice()  # must not raise


def test_catalog_marks_kokoro_pack_installed(client, fakes):
    fakes.catalog.installed_catalog_ids = lambda: {"en_GB-alba-medium"}
    voices = {v["id"]: v["installed"] for v in client.get("/api/voices/catalog").json()["voices"]}
    assert voices == {"en_US-lessac-medium": False, "en_GB-alba-medium": True}


# -- reading text that didn't come from a reading plan


def test_tts_normalizes_when_asked(client, fakes, monkeypatch):
    calls = []

    def normalize(text, options=None):
        calls.append(text)
        return text.replace("$3.50", "3 dollars and 50 cents")

    monkeypatch.setattr(server, "_speech", lambda: SimpleNamespace(normalize_for_speech=normalize))
    r = client.post("/api/tts", json={"text": "It costs $3.50.", "engine": "piper", "normalize": True})
    assert r.status_code == 200
    assert calls == ["It costs $3.50."]
    assert fakes.tts.engines["piper"].calls[-1][0] == "It costs 3 dollars and 50 cents."


def test_tts_reads_plan_text_as_is_by_default(client, fakes, monkeypatch):
    monkeypatch.setattr(server, "_speech", lambda: SimpleNamespace(normalize_for_speech=lambda t, o=None: "changed"))
    assert client.post("/api/tts", json={"text": "It costs $3.50.", "engine": "piper"}).status_code == 200
    assert fakes.tts.engines["piper"].calls[-1][0] == "It costs $3.50."


def test_tts_with_nothing_speakable_after_normalizing(client, fakes, monkeypatch):
    monkeypatch.setattr(server, "_speech", lambda: SimpleNamespace(normalize_for_speech=lambda t, o=None: " "))
    r = client.post("/api/tts", json={"text": "[12]", "engine": "piper", "normalize": True})
    assert r.status_code == 400
    assert not fakes.tts.engines["piper"].calls


def test_tts_normalize_with_the_real_speech_layer(client, fakes, monkeypatch):
    from destiny_bot import speech

    monkeypatch.setattr(server, "_speech", lambda: speech)
    r = client.post("/api/tts", json={"text": "In 1990 it cost $3.50.", "engine": "piper", "normalize": True})
    assert r.status_code == 200
    spoken = fakes.tts.engines["piper"].calls[-1][0]
    assert "$" not in spoken and "dollars" in spoken
