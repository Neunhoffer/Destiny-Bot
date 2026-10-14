"""The local web server: the reader UI plus the JSON API described in docs/API.md.

The server only listens on 127.0.0.1. Defences against other websites and other
programs reaching it:

- A ``Host`` header allowlist blocks DNS-rebinding attacks.
- Every ``/api`` request must carry the per-process token that is embedded in
  ``index.html`` (header ``X-Destiny-Token``).
- Requests that carry an ``Origin`` header must come from the server's own origin.
- Uploads are capped while they stream in, never trusting ``Content-Length`` alone.
- Strict security headers (CSP, nosniff, no referrer) go on every response.
"""

from __future__ import annotations

import html
import ipaddress
import logging
import mimetypes
import os
import re
import secrets
import shutil
import socket
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence
from urllib.parse import quote

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError
from python_multipart.multipart import MultipartParseError, MultipartParser, parse_options_header
from starlette.datastructures import Headers, MutableHeaders, QueryParams
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__, config
from .ingest import SUPPORTED_EXTENSIONS, IngestError, source_type_for
from .jobs import Job, JobManager, friendly_error
from .library import Library, temp_dir, wipe_user_data
from .models import Block, Document, Section
from .tts.base import AudioClip, EngineUnavailable, TTSEngine, TTSError

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PORT_ATTEMPTS = 10
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_TTS_CHARS = 5000
MAX_PASTE_CHARS = 5_000_000
MAX_PASSAGE_CHARS = 20_000
AI_CONTEXT_CHARS = 150_000
PAGE_CONTEXT_CHARS = 20_000
PAGE_DPI = 110
DESCRIBE_DPI = 150
MAX_RENDER_SIDE_PX = 4000  # keeps a giant PDF page (posters, maps) from eating memory
WEB_DIR = Path(__file__).parent / "web"
TOKEN_PLACEHOLDER = "__DESTINY_TOKEN__"

CSP = (
    "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; "
    "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
_SECURITY_HEADERS = (
    ("Content-Security-Policy", CSP),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)

ONLINE_TTS_OFF = "Online voices are turned off in Settings (they send text to Microsoft)."
AI_OFF = "AI study helpers are turned off. Turn them on in Settings (they send text to Anthropic)."
BOOK_NOT_FOUND = "That book isn't in your library anymore."
SECTION_NOT_FOUND = "That part of the book couldn't be found."
UPLOAD_TOO_BIG = "That file is too big to import (the limit is {limit} MB)."

_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "[::1]"})
_HOST_RE = re.compile(r"(\[[0-9a-f:.]+\]|[^:\[\]/@\s]+)(?::(\d{1,5}))?")
_FILE_DOWNLOAD_RE = re.compile(r"/api/jobs/[^/]+/files/[^/]+")
_MULTIPART_SLACK = 1024 * 1024  # room for the multipart envelope around the file

_MEDIA_TYPES = {
    # Explicit, because Windows registry entries often map .js to text/plain, which
    # browsers refuse to run when nosniff is set.
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".webmanifest": "application/manifest+json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m3u": "audio/x-mpegurl",
}

_MISSING_UI_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Athena</title></head>
<body><h1>Athena</h1>
<p>The reader's web page files are missing from this copy of Athena.
Download Athena again, or reinstall it, then restart it.</p></body></html>
"""


# -- lazily imported modules (heavy or optional). Tests replace these accessors with fakes.


def _tts() -> Any:
    from . import tts

    return tts


def _audio_cache_class() -> Any:
    from .tts.cache import AudioCache

    return AudioCache


def _speech() -> Any:
    from . import speech

    return speech


def _ai() -> Any:
    from . import ai

    return ai


def _audio() -> Any:
    from . import audio

    return audio


def _export_audiobook() -> Callable[..., list[Path]]:
    from .export import export_audiobook

    return export_audiobook


def _catalog() -> Any:
    from .tts import catalog

    return catalog


def _ocr_status() -> tuple[bool, str]:
    try:
        from .ingest.ocr import ocr_available
    except ImportError as exc:
        return False, f"Reading scanned pages isn't installed in this copy of Athena ({exc})."
    try:
        ok, reason = ocr_available()
    except Exception as exc:  # status must never fail because of one optional feature
        return False, f"Couldn't check for Tesseract: {exc}"
    return bool(ok), str(reason or "")


def _render_pdf_page(path: Path, page: int, dpi: int) -> bytes:
    """PNG of the 1-based ``page`` of a PDF. Raises IndexError if the page doesn't exist."""
    import pymupdf

    with _RENDER_LOCK, pymupdf.open(path) as pdf:
        if not 1 <= page <= pdf.page_count:
            raise IndexError(page)
        pdf_page = pdf.load_page(page - 1)
        zoom = dpi / 72.0
        longest = max(pdf_page.rect.width, pdf_page.rect.height) * zoom
        if longest > MAX_RENDER_SIDE_PX:
            zoom *= MAX_RENDER_SIDE_PX / longest
        pixmap = pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")


_RENDER_LOCK = threading.Lock()


# -- request bodies


class PasteBody(BaseModel):
    title: str = Field(default="", max_length=1000)
    text: str = Field(max_length=MAX_PASTE_CHARS)


class ProgressBody(BaseModel):
    section_id: str = Field(max_length=100)
    utterance: int = Field(default=0, ge=0)


class BookmarkBody(BaseModel):
    section_id: str = Field(max_length=100)
    utterance: int = Field(default=0, ge=0)
    label: str = Field(default="", max_length=1000)


class TTSBody(BaseModel):
    text: str
    engine: str | None = Field(default=None, max_length=50)
    voice: str | None = Field(default=None, max_length=300)
    rate: float = Field(default=1.0, ge=0.5, le=3.0)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_PASSAGE_CHARS)


class ExplainBody(BaseModel):
    book_id: str
    section_id: str
    passage: str = Field(max_length=MAX_PASSAGE_CHARS)


class AskBody(BaseModel):
    book_id: str
    section_id: str
    question: str = Field(max_length=MAX_PASSAGE_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=100)


class SummarizeBody(BaseModel):
    book_id: str
    section_id: str
    length: Literal["short", "detailed"] = "short"


class QuizBody(BaseModel):
    book_id: str
    section_id: str
    count: int = 5


class NarrateBody(BaseModel):
    book_id: str
    section_id: str
    block: int


class DescribePageBody(BaseModel):
    book_id: str
    page: int


class ExportBody(BaseModel):
    book_id: str
    section_ids: list[str] | None = None
    engine: str | None = Field(default=None, max_length=50)
    voice: str | None = Field(default=None, max_length=300)
    format: Literal["mp3", "wav"] = "mp3"
    rate: float = Field(default=1.0, ge=0.5, le=3.0)


class VoiceDownloadBody(BaseModel):
    voice_id: str = Field(min_length=1, max_length=200)


class WipeBody(BaseModel):
    confirm: str = ""


# -- security middleware


def _split_host(value: str) -> tuple[str, int | None]:
    match = _HOST_RE.fullmatch(value.strip().lower())
    if not match:
        return "", None
    port = match.group(2)
    return match.group(1), int(port) if port else None


def _is_loopback(host: str) -> bool:
    name = host.strip().lower().strip("[]")
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _is_ip_literal(name: str) -> bool:
    try:
        ipaddress.ip_address(name.strip("[]"))
        return True
    except ValueError:
        return False


def host_policy(port: int | None, bind_host: str | None = None) -> Callable[[str], bool]:
    """Which ``Host`` headers to accept.

    With a port: exactly ``127.0.0.1:<port>``, ``localhost:<port>`` and ``[::1]:<port>``
    (plus the explicitly chosen non-loopback host, if any). Without a port (an app
    created for tests): loopback names on any port and ``testserver``.
    """
    names = set(_LOOPBACK_NAMES)
    allow_ip_literals = False
    if bind_host and not _is_loopback(bind_host):
        bare = bind_host.strip().lower().strip("[]")
        if bare in ("0.0.0.0", "::"):
            # Any address of this machine. DNS rebinding needs a host *name*, so IP literals are safe.
            allow_ip_literals = True
        else:
            names.add(f"[{bare}]" if ":" in bare else bare)

    def allowed(header: str) -> bool:
        name, header_port = _split_host(header)
        if not name:
            return False
        if port is None:
            return name == "testserver" or name in names
        if header_port != port:
            return False
        return name in names or (allow_ip_literals and _is_ip_literal(name))

    return allowed


def _is_api_path(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


async def _send_json(send: Send, status: int, detail: str) -> None:
    response = JSONResponse({"detail": detail}, status_code=status)
    await send({"type": "http.response.start", "status": status, "headers": response.raw_headers})
    await send({"type": "http.response.body", "body": response.body})


class SecurityMiddleware:
    """Host / Origin / token checks plus security headers, as a plain ASGI middleware."""

    def __init__(self, app: ASGIApp, *, token: str, host_allowed: Callable[[str], bool]) -> None:
        self.app = app
        self._token = token.encode("utf-8")
        self._host_allowed = host_allowed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return

        path: str = scope.get("path", "")
        started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = MutableHeaders(scope=message)
                for name, value in _SECURITY_HEADERS:
                    headers[name] = value
                if path == "/" or _is_api_path(path):
                    headers["Cache-Control"] = "no-store"
                elif "cache-control" not in headers:
                    headers["Cache-Control"] = "no-cache"
            await send(message)

        headers = Headers(scope=scope)
        host = headers.get("host", "")
        if not self._host_allowed(host):
            await _send_json(
                send_with_headers, 403, "Athena only answers requests addressed to 127.0.0.1 or localhost."
            )
            return
        origin = headers.get("origin")
        if origin is not None and origin.strip().lower().rstrip("/") != f"http://{host.strip().lower()}":
            await _send_json(send_with_headers, 403, "Requests from other websites are blocked.")
            return
        if _is_api_path(path) and not self._token_ok(scope, headers, path):
            await _send_json(send_with_headers, 401, "Missing or wrong access token. Reload the page and try again.")
            return
        try:
            await self.app(scope, receive, send_with_headers)
        except Exception as exc:
            if started:
                raise
            log.error("Unexpected error for %s %s", scope.get("method"), path, exc_info=exc)
            await _send_json(send_with_headers, 500, friendly_error(exc))

    def _token_ok(self, scope: Scope, headers: Headers, path: str) -> bool:
        supplied = headers.get("x-destiny-token")
        if supplied is None and scope.get("method") in ("GET", "HEAD") and _FILE_DOWNLOAD_RE.fullmatch(path):
            # Plain <a href> downloads can't send headers.
            supplied = QueryParams(scope.get("query_string", b"")).get("token")
        if not supplied:
            return False
        return secrets.compare_digest(supplied.encode("utf-8", "replace"), self._token)


# -- uploads


@dataclass
class _Upload:
    folder: Path
    path: Path
    filename: str
    size: int


class _UploadRejected(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _unsupported_message(name: str) -> str:
    return (
        f"Athena can't open '{Path(name).suffix or name}' files yet. "
        "Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
    )


def _clean_filename(raw: bytes) -> str:
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError:
        name = raw.decode("latin-1")
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch >= " " and ch != "\x7f").strip()
    return name[:255]


class _UploadSink:
    """Streams the multipart part named ``file`` to disk, enforcing the size limit."""

    def __init__(self, folder: Path, max_bytes: int, limit_message: str) -> None:
        self.folder = folder
        self.max_bytes = max_bytes
        self.limit_message = limit_message
        self.path: Path | None = None
        self.filename = ""
        self.size = 0
        self._fh: Any = None
        self._in_file = False
        self._headers: dict[bytes, bytes] = {}
        self._field = b""
        self._value = b""

    def callbacks(self) -> dict[str, Callable[..., None]]:
        return {
            "on_part_begin": self._part_begin,
            "on_header_field": self._header_field,
            "on_header_value": self._header_value,
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
        }

    def _part_begin(self) -> None:
        self._headers = {}
        self._in_file = False

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._field += data[start:end]

    def _header_value(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def _header_end(self) -> None:
        self._headers[self._field.strip().lower()] = self._value.strip()
        self._field = self._value = b""

    def _headers_finished(self) -> None:
        _, options = parse_options_header(self._headers.get(b"content-disposition", b"").decode("latin-1"))
        if options.get(b"name") != b"file" or b"filename" not in options or self.path is not None:
            return  # other form fields (and any second file) are ignored
        name = _clean_filename(options[b"filename"])
        if not name:
            raise _UploadRejected(400, "No file was chosen. Choose a file and try again.")
        if source_type_for(name) is None:
            raise _UploadRejected(400, _unsupported_message(name))
        self.filename = name
        self.path = self.folder / f"upload{Path(name).suffix.lower()}"
        self._fh = open(self.path, "wb")
        self._in_file = True

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._in_file:
            return
        self.size += end - start
        if self.size > self.max_bytes:
            raise _UploadRejected(413, self.limit_message)
        self._fh.write(data[start:end])

    def _part_end(self) -> None:
        if self._in_file:
            self._fh.close()
            self._fh = None
            self._in_file = False

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def result(self) -> _Upload:
        if self.path is None:
            raise _UploadRejected(400, "No file was uploaded. Choose a file and try again.")
        if self.size == 0:
            raise _UploadRejected(400, f"'{self.filename}' is empty.")
        return _Upload(self.folder, self.path, self.filename, self.size)


async def _receive_upload(request: Request, max_bytes: int) -> _Upload:
    """Stream a multipart upload (field ``file``) into a private temp folder."""
    limit_message = UPLOAD_TOO_BIG.format(limit=max(1, round(max_bytes / (1024 * 1024))))
    ctype, params = parse_options_header(request.headers.get("content-type", ""))
    if ctype != b"multipart/form-data" or not params.get(b"boundary"):
        raise HTTPException(400, "Upload the file as multipart/form-data in a field named 'file'.")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes + _MULTIPART_SLACK:
        raise HTTPException(413, limit_message)

    folder = Path(tempfile.mkdtemp(prefix="upload-", dir=temp_dir()))
    sink = _UploadSink(folder, max_bytes, limit_message)
    parser = MultipartParser(params[b"boundary"], sink.callbacks())  # type: ignore[arg-type]
    received = 0
    ok = False
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > max_bytes + _MULTIPART_SLACK:
                raise _UploadRejected(413, limit_message)
            if chunk:
                parser.write(chunk)
        parser.finalize()
        sink.close()
        upload = sink.result()
        ok = True
        return upload
    except _UploadRejected as exc:
        raise HTTPException(exc.status, exc.detail) from None
    except MultipartParseError:
        raise HTTPException(400, "The upload was damaged on the way. Please try again.") from None
    except ClientDisconnect:
        raise HTTPException(400, "The upload was interrupted. Please try again.") from None
    finally:
        sink.close()
        if not ok:
            shutil.rmtree(folder, ignore_errors=True)


# -- helpers


def _describe_errors(errors: Sequence[Any]) -> str:
    parts: list[str] = []
    for err in list(errors)[:3]:
        if err.get("type") == "json_invalid":
            parts.append("the request wasn't valid JSON")
            continue
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path", "header")]
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{'.'.join(loc)}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid value"


def _toc_entry(section: Section) -> dict[str, Any]:
    return {
        "id": section.id,
        "title": section.title,
        "level": section.level,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "block_count": len(section.blocks),
    }


def _block_dict(block: Block) -> dict[str, Any]:
    data = block.model_dump(mode="json", exclude={"meta"})
    data["notes"] = bool(block.meta.get("speaker_notes"))
    return data


def _section_index(doc: Document, section_id: str) -> int:
    for i, section in enumerate(doc.sections):
        if section.id == section_id:
            return i
    raise HTTPException(404, SECTION_NOT_FOUND)


def _media_type(path: Path) -> str:
    ext = path.suffix.lower()
    return _MEDIA_TYPES.get(ext) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _static_file(web_dir: Path, relative: str) -> Path | None:
    if not relative or "\x00" in relative:
        return None
    try:
        root = web_dir.resolve()
        target = (root / relative).resolve()
    except (OSError, ValueError):
        return None
    if target == root or not target.is_relative_to(root) or not target.is_file():
        return None
    return target


def content_disposition(name: str) -> str:
    """``attachment`` with an ASCII fallback name and the exact UTF-8 name (RFC 5987/6266)."""
    ext = Path(name).suffix
    fallback = "".join(ch if " " <= ch < "\x7f" and ch not in '"\\' else "_" for ch in name).strip()
    if not any(ch.isalnum() for ch in fallback[: len(fallback) - len(ext)]):
        fallback = "download" + (ext if ext.isascii() and ext[1:].isalnum() else "")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(name, safe='')}"


def _page_text(doc: Document, page: int) -> str:
    text = " ".join(b.text for s in doc.sections for b in s.blocks if b.page == page and b.text)
    return text[:PAGE_CONTEXT_CHARS]


def _section_for_page(doc: Document, page: int) -> Section | None:
    for section in doc.sections:
        if any(b.page == page for b in section.blocks):
            return section
    for section in doc.sections:
        if section.page_start is not None and section.page_end is not None:
            if section.page_start <= page <= section.page_end:
                return section
    return None


def _remove_stale(folder: Path, prefix: str, max_age_s: float) -> None:
    """Delete leftovers of earlier runs (temp uploads, exports no longer reachable)."""
    cutoff = time.time() - max_age_s
    try:
        entries = list(folder.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.name.startswith(prefix) and entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


class _Caches:
    """In-memory caches shared by requests (reset when all data is erased)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._audio: Any = None

    def audio(self) -> Any:
        with self._lock:
            if self._audio is None:
                self._audio = _audio_cache_class()(config.cache_dir())
            return self._audio

    def reset(self) -> None:
        with self._lock:
            self._audio = None


# -- the app


def create_app(
    *,
    library: Library | None = None,
    token: str | None = None,
    jobs: JobManager | None = None,
    port: int | None = None,
    bind_host: str | None = None,
    web_dir: Path | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    warm_up: bool = False,
) -> FastAPI:
    """Build the app. ``port`` enables the strict Host check (``run()`` always passes it).

    ``warm_up`` loads the default offline voice in the background so the first
    sentence starts without a multi-second model load.
    """
    library = library if library is not None else Library()
    jobs = jobs if jobs is not None else JobManager()
    token = token or secrets.token_urlsafe(32)
    web_root = Path(web_dir) if web_dir is not None else WEB_DIR
    caches = _Caches()
    settings_lock = threading.Lock()

    _remove_stale(temp_dir(), "upload-", 6 * 3600)
    _remove_stale(config.exports_dir(), "j_", 3600)

    app = FastAPI(title="Athena", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.library = library
    app.state.jobs = jobs
    app.state.token = token
    app.add_middleware(SecurityMiddleware, token=token, host_allowed=host_policy(port, bind_host))
    if warm_up:
        threading.Thread(target=_warm_up_voice, name="athena-voice-warmup", daemon=True).start()

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": "Invalid request: " + _describe_errors(exc.errors())}, status_code=422)

    # -- shared lookups

    def doc_or_404(book_id: str) -> Document:
        try:
            return library.get(book_id)
        except KeyError:
            raise HTTPException(404, BOOK_NOT_FOUND) from None

    def doc_and_section(book_id: str, section_id: str) -> tuple[Document, Section]:
        doc = doc_or_404(book_id)
        return doc, doc.sections[_section_index(doc, section_id)]

    def summary_dict(book_id: str) -> dict[str, Any]:
        try:
            return library.summary(book_id).model_dump(mode="json")
        except KeyError:
            raise HTTPException(404, BOOK_NOT_FOUND) from None

    def job_or_404(job_id: str) -> Job:
        try:
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, "That task has finished long ago or never existed.") from None

    def pdf_source(doc: Document) -> Path:
        source = library.source_file(doc.id) if doc.source_type == "pdf" else None
        if source is None or source.suffix.lower() != ".pdf":
            raise HTTPException(404, "Page pictures are only available for PDF books.")
        return source

    def render_page(doc: Document, page: int, dpi: int) -> bytes:
        source = pdf_source(doc)
        if page < 1 or (doc.page_count is not None and page > doc.page_count):
            raise HTTPException(404, f"This book has no page {page}.")
        try:
            return _render_pdf_page(source, page, dpi)
        except IndexError:
            raise HTTPException(404, f"This book has no page {page}.") from None
        except Exception as exc:
            log.warning("Couldn't render page %s of %s: %s", page, doc.id, exc)
            raise HTTPException(500, f"Couldn't draw page {page} of this PDF.") from None

    def tts_module() -> Any:
        try:
            return _tts()
        except ImportError as exc:
            raise HTTPException(503, f"Voices aren't available in this copy of Athena ({exc}).") from None

    def choose_engine(name: str | None, settings: config.Settings, *, for_files: bool) -> TTSEngine:
        tts = tts_module()
        if not name or name == "auto":
            try:
                engine = tts.resolve_engine(settings.tts_engine, online_allowed=settings.online_tts_allowed)
            except Exception as exc:
                raise HTTPException(503, f"No voice is available right now: {exc}") from None
        else:
            try:
                engine = tts.get_engine(name)
            except KeyError:
                raise HTTPException(400, f"There's no voice engine called '{name}'.") from None
        if engine.name == "browser":
            if for_files:
                raise HTTPException(
                    400,
                    "The browser's voice can't make audio files. Download an offline voice in "
                    "Settings → Voices (or choose another voice), then try again.",
                )
            raise HTTPException(400, "The browser's voice speaks inside the page; the server has no audio for it.")
        if engine.online and not settings.online_tts_allowed:
            raise HTTPException(403, ONLINE_TTS_OFF)
        return engine

    def pick_voice(engine: TTSEngine, requested: str | None, settings: config.Settings) -> str:
        """The requested (or saved) voice when it's installed, else the engine's default."""
        wanted = [v for v in (requested, settings.voices.get(engine.name)) if v]
        if wanted:
            try:
                installed = {v.id for v in engine.list_voices() if v.installed}
            except Exception as exc:
                log.warning("Couldn't list %s voices: %s", engine.name, exc)
                installed = set()
            for voice in wanted:
                if voice in installed:
                    return voice
        try:
            default = engine.default_voice()
        except Exception as exc:
            log.warning("Couldn't get the default %s voice: %s", engine.name, exc)
            default = None
        if not default:
            raise HTTPException(503, f"{engine.label} has no voices installed. Download one in Settings → Voices.")
        return default

    def run_ai(call: Callable[[Any], Any]) -> Any:
        settings = config.load_settings()
        if not settings.ai_enabled:  # checked here too, so no text can leave unless the switch is on
            raise HTTPException(409, AI_OFF)
        try:
            ai = _ai()
        except ImportError:
            raise HTTPException(409, "AI study helpers aren't installed in this copy of Athena.") from None
        try:
            return call(ai.Assistant(settings))
        except ai.AIUnavailable as exc:
            raise HTTPException(409, str(exc) or AI_OFF) from None
        except ai.AIError as exc:
            raise HTTPException(502, str(exc) or "Claude couldn't answer right now. Please try again.") from None

    def ai_context(section: Section) -> str:
        return _speech().section_plain_text(section, max_chars=AI_CONTEXT_CHARS)

    # -- UI

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        try:
            page = (web_root / "index.html").read_text(encoding="utf-8")
        except OSError:
            return HTMLResponse(_MISSING_UI_HTML, status_code=500)
        return HTMLResponse(page.replace(TOKEN_PLACEHOLDER, html.escape(token, quote=True)))

    @app.get("/static/{path:path}", include_in_schema=False)
    def static_file(path: str) -> FileResponse:
        target = _static_file(web_root, path)
        if target is None:
            raise HTTPException(404, "Not found")
        return FileResponse(target, media_type=_media_type(target))

    # -- status and settings

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        settings = config.load_settings()
        engines: list[dict[str, Any]] = []
        resolved = "browser"
        try:
            tts = _tts()
        except Exception as exc:
            log.warning("Voice engines unavailable: %s", exc)
            tts = None
        if tts is not None:
            try:
                all_engines = list(tts.all_engines())
            except Exception as exc:
                log.warning("Couldn't list voice engines: %s", exc)
                all_engines = []
            engines = [_engine_info(engine, settings) for engine in all_engines]
            try:
                resolved = tts.resolve_engine(settings.tts_engine, online_allowed=settings.online_tts_allowed).name
            except Exception as exc:
                log.warning("Couldn't pick a voice engine: %s", exc)
        ocr_ok, ocr_reason = _ocr_status()
        return {
            "version": __version__,
            "engines": engines,
            "resolved_engine": resolved,
            "ai": {
                "enabled": settings.ai_enabled,
                "configured": bool(settings.resolved_api_key()),
                "model": settings.ai_model,
            },
            "ocr": {"available": ocr_ok, "reason": ocr_reason},
            "data_dir": str(config.data_dir()),
        }

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return config.load_settings().public_dict()

    @app.put("/api/settings")
    def put_settings(changes: dict[str, Any] = Body(...)) -> dict[str, Any]:
        changes = dict(changes)
        changes.pop("anthropic_api_key_set", None)
        key = changes.get("anthropic_api_key")
        if key is not None and not isinstance(key, str):
            raise HTTPException(422, "Invalid setting: anthropic_api_key must be text.")
        if "tts_engine" in changes:
            engine_names = _engine_names()
            if engine_names and changes["tts_engine"] not in ("auto", *engine_names):
                raise HTTPException(422, f"Invalid setting: tts_engine must be one of auto, {', '.join(engine_names)}.")
        with settings_lock:
            try:
                settings = config.update_settings(changes)
            except ValidationError as exc:
                raise HTTPException(422, "Invalid setting: " + _describe_errors(exc.errors())) from None
            except (TypeError, ValueError, AttributeError) as exc:
                raise HTTPException(422, f"Invalid setting: {exc}") from None
        return settings.public_dict()

    # -- library

    @app.get("/api/library")
    def list_books() -> dict[str, Any]:
        return {"books": [book.model_dump(mode="json") for book in library.list()]}

    @app.post("/api/library/import")
    async def import_book(request: Request) -> dict[str, Any]:
        upload = await _receive_upload(request, max_upload_bytes)

        def run(job: Job) -> dict[str, Any]:
            def progress(fraction: float, message: str) -> None:
                job.update(fraction, message)
                job.raise_if_cancelled()  # loaders have no cancel flag; stopping at a progress report works

            job.update(0.0, f"Opening {upload.filename}")
            doc = library.import_file(upload.path, original_name=upload.filename, progress=progress)
            book = library.summary(doc.id).model_dump(mode="json")
            return {"book": book, "warnings": list(doc.warnings)}

        job = jobs.submit(
            "import",
            run,
            message=f"Waiting to open {upload.filename}",
            cleanup=lambda: shutil.rmtree(upload.folder, ignore_errors=True),
        )
        return {"job_id": job.id}

    @app.post("/api/library/paste")
    def paste_text(body: PasteBody) -> dict[str, Any]:
        try:
            doc = library.import_text(body.title, body.text)
        except IngestError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"book": summary_dict(doc.id)}

    @app.delete("/api/library/{book_id}")
    def delete_book(book_id: str) -> dict[str, Any]:
        try:
            library.delete(book_id)
        except KeyError:
            raise HTTPException(404, BOOK_NOT_FOUND) from None
        except OSError as exc:
            raise HTTPException(500, str(exc)) from None
        return {"ok": True}

    # -- books

    @app.get("/api/books/{book_id}")
    def open_book(book_id: str) -> dict[str, Any]:
        doc = doc_or_404(book_id)
        try:
            library.mark_opened(book_id)
        except (KeyError, OSError) as exc:
            log.warning("Couldn't record that %s was opened: %s", book_id, exc)
        return {
            "book": summary_dict(book_id),
            "toc": [_toc_entry(section) for section in doc.sections],
            "warnings": list(doc.warnings),
        }

    @app.get("/api/books/{book_id}/sections/{section_id}")
    def get_section(book_id: str, section_id: str) -> dict[str, Any]:
        doc = doc_or_404(book_id)
        index = _section_index(doc, section_id)
        section = doc.sections[index]
        settings = config.load_settings()
        utterances = _speech().plan_section(section, settings.reading)
        return {
            "section": _toc_entry(section),
            "blocks": [_block_dict(block) for block in section.blocks],
            "utterances": [u.model_dump(mode="json") for u in utterances],
            "prev_id": doc.sections[index - 1].id if index > 0 else None,
            "next_id": doc.sections[index + 1].id if index + 1 < len(doc.sections) else None,
        }

    @app.get("/api/books/{book_id}/progress")
    def get_progress(book_id: str) -> dict[str, Any]:
        try:
            progress = library.get_progress(book_id)
        except KeyError:
            raise HTTPException(404, BOOK_NOT_FOUND) from None
        return {"progress": progress.model_dump(mode="json") if progress else None}

    @app.put("/api/books/{book_id}/progress")
    def put_progress(book_id: str, body: ProgressBody) -> dict[str, Any]:
        doc_and_section(book_id, body.section_id)
        try:
            progress = library.set_progress(book_id, body.section_id, body.utterance)
        except KeyError:
            raise HTTPException(404, BOOK_NOT_FOUND) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"progress": progress.model_dump(mode="json")}

    @app.get("/api/books/{book_id}/bookmarks")
    def list_bookmarks(book_id: str) -> dict[str, Any]:
        try:
            marks = library.list_bookmarks(book_id)
        except KeyError:
            raise HTTPException(404, BOOK_NOT_FOUND) from None
        return {"bookmarks": [mark.model_dump(mode="json") for mark in marks]}

    @app.post("/api/books/{book_id}/bookmarks")
    def add_bookmark(book_id: str, body: BookmarkBody) -> dict[str, Any]:
        doc_and_section(book_id, body.section_id)
        try:
            mark = library.add_bookmark(book_id, body.section_id, body.utterance, body.label)
        except KeyError:
            raise HTTPException(404, BOOK_NOT_FOUND) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"bookmark": mark.model_dump(mode="json")}

    @app.delete("/api/books/{book_id}/bookmarks/{bookmark_id}")
    def delete_bookmark(book_id: str, bookmark_id: str) -> dict[str, Any]:
        doc_or_404(book_id)
        try:
            library.delete_bookmark(book_id, bookmark_id)
        except KeyError:
            raise HTTPException(404, "That bookmark doesn't exist anymore.") from None
        return {"ok": True}

    @app.get("/api/books/{book_id}/pages/{page}.png")
    def page_image(book_id: str, page: int) -> Response:
        doc = doc_or_404(book_id)
        return Response(render_page(doc, page, PAGE_DPI), media_type="image/png")

    # -- voices

    @app.get("/api/voices")
    def list_voices(engine: str | None = None) -> dict[str, Any]:
        settings = config.load_settings()
        tts = tts_module()
        if engine:
            try:
                engines = [tts.get_engine(engine)]
            except KeyError:
                raise HTTPException(400, f"There's no voice engine called '{engine}'.") from None
        else:
            engines = list(tts.all_engines())
        voices: list[dict[str, Any]] = []
        for eng in engines:
            if eng.online and not settings.online_tts_allowed:
                continue  # listing online voices may itself contact the service
            try:
                found = eng.list_voices()
            except Exception as exc:
                if engine:
                    raise HTTPException(503, friendly_error(exc)) from None
                log.warning("Couldn't list %s voices: %s", eng.name, exc)
                continue
            voices.extend(v.model_dump(mode="json") for v in found if v.installed)
        return {"voices": voices}

    @app.get("/api/voices/catalog")
    def voice_catalog() -> dict[str, Any]:
        catalog = _catalog()
        installed = {path.stem for path in catalog.installed_piper_voices()}
        return {
            "voices": [
                {**voice.model_dump(mode="json"), "installed": voice.id in installed} for voice in catalog.CATALOG
            ]
        }

    @app.post("/api/voices/download")
    def download_voice(body: VoiceDownloadBody) -> dict[str, Any]:
        catalog = _catalog()
        voice_id = body.voice_id.strip()
        entry = next((v for v in catalog.CATALOG if v.id == voice_id), None)
        if entry is None and not _valid_piper_voice_id(catalog, voice_id):
            raise HTTPException(
                400, f"'{voice_id}' isn't a valid voice name (expected something like en_US-lessac-medium)."
            )
        label = entry.name if entry is not None else voice_id

        def run(job: Job) -> dict[str, Any]:
            job.update(0.0, f"Downloading {label}")
            catalog.download_piper_voice(voice_id, progress=job.update, cancel=job.cancel_event)
            return {"voice_id": voice_id}

        job = jobs.submit("voice_download", run, message=f"Waiting to download {label}")
        return {"job_id": job.id}

    @app.post("/api/tts")
    def synthesize(body: TTSBody) -> Response:
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "There's no text to read.")
        if len(text) > MAX_TTS_CHARS:
            raise HTTPException(400, f"That's too much text to read at once ({MAX_TTS_CHARS:,} characters at most).")
        settings = config.load_settings()
        engine = choose_engine(body.engine, settings, for_files=False)
        voice = pick_voice(engine, body.voice, settings)
        rate = _clamp_rate(body.rate)
        try:
            clip = caches.audio().get_or_create(
                (engine.name, voice or "", f"{rate:.2f}", text),
                lambda: _trimmed(engine.synthesize(text, voice, rate)),
            )
        except EngineUnavailable as exc:
            raise HTTPException(503, str(exc) or "That voice isn't available right now.") from None
        except TTSError as exc:
            raise HTTPException(502, str(exc) or "The voice couldn't read this text.") from None
        return Response(clip.data, media_type=clip.mime)

    # -- AI study helpers

    @app.post("/api/ai/explain")
    def ai_explain(body: ExplainBody) -> dict[str, Any]:
        doc, section = doc_and_section(body.book_id, body.section_id)
        passage = body.passage.strip()
        if not passage:
            raise HTTPException(400, "Select some text to explain first.")
        context = ai_context(section)
        text = run_ai(
            lambda a: a.explain(passage, context=context, book_title=doc.title, section_title=section.title)
        )
        return {"text": text}

    @app.post("/api/ai/ask")
    def ai_ask(body: AskBody) -> dict[str, Any]:
        doc, section = doc_and_section(body.book_id, body.section_id)
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "Type a question first.")
        context = ai_context(section)
        history = [turn.model_dump() for turn in body.history]
        text = run_ai(
            lambda a: a.ask(
                question, context=context, book_title=doc.title, history=history, section_title=section.title
            )
        )
        return {"text": text}

    @app.post("/api/ai/summarize")
    def ai_summarize(body: SummarizeBody) -> dict[str, Any]:
        doc, section = doc_and_section(body.book_id, body.section_id)
        section_text = ai_context(section)
        text = run_ai(
            lambda a: a.summarize(
                section_text, section_title=section.title, book_title=doc.title, length=body.length
            )
        )
        return {"text": text}

    @app.post("/api/ai/quiz")
    def ai_quiz(body: QuizBody) -> dict[str, Any]:
        doc, section = doc_and_section(body.book_id, body.section_id)
        count = max(1, min(10, body.count))
        section_text = ai_context(section)
        questions = run_ai(
            lambda a: a.quiz(section_text, section_title=section.title, count=count, book_title=doc.title)
        )
        return {"questions": [q.model_dump(mode="json") for q in questions]}

    @app.post("/api/ai/narrate")
    def ai_narrate(body: NarrateBody) -> dict[str, Any]:
        doc, section = doc_and_section(body.book_id, body.section_id)
        if not 0 <= body.block < len(section.blocks):
            raise HTTPException(404, "That part of the section couldn't be found.")
        block = section.blocks[body.block]
        context = ai_context(section)
        text = run_ai(
            lambda a: a.narrate_block(block, context=context, book_title=doc.title, section_title=section.title)
        )
        return {"text": text}

    @app.post("/api/ai/describe-page")
    def ai_describe_page(body: DescribePageBody) -> dict[str, Any]:
        doc = doc_or_404(body.book_id)
        png = render_page(doc, body.page, DESCRIBE_DPI)
        section = _section_for_page(doc, body.page)
        context = _page_text(doc, body.page)
        text = run_ai(
            lambda a: a.describe_image(
                png, context=context, book_title=doc.title, section_title=section.title if section else ""
            )
        )
        return {"text": text}

    # -- export

    @app.post("/api/export")
    def start_export(body: ExportBody) -> dict[str, Any]:
        doc = doc_or_404(body.book_id)
        section_ids: list[str] | None = None
        if body.section_ids is not None:
            section_ids = list(dict.fromkeys(body.section_ids))
            if not section_ids:
                raise HTTPException(400, "Choose at least one section to export.")
            for section_id in section_ids:
                _section_index(doc, section_id)
        settings = config.load_settings()
        engine = choose_engine(body.engine, settings, for_files=True)
        try:
            available, reason = engine.is_available()
        except Exception as exc:
            available, reason = False, friendly_error(exc)
        if not available:
            raise HTTPException(503, reason or "That voice isn't available right now.")
        voice = pick_voice(engine, body.voice, settings)
        rate = _clamp_rate(body.rate)
        options = settings.reading
        try:
            export_audiobook = _export_audiobook()
        except ImportError as exc:
            message = f"Saving audio files isn't available in this copy of Athena ({exc})."
            raise HTTPException(503, message) from None

        def run(job: Job) -> dict[str, Any]:
            assert job.files_dir is not None
            out_dir = job.files_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = export_audiobook(
                doc,
                engine=engine,
                voice=voice,
                options=options,
                out_dir=out_dir,
                section_ids=section_ids,
                fmt=body.format,
                rate=rate,
                progress=job.update,
                cancel=job.cancel_event,
            )
            root = out_dir.resolve()
            files: list[dict[str, Any]] = []
            seen: set[str] = set()
            for path in paths:
                resolved = Path(path).resolve()
                if resolved.parent != root or not resolved.is_file() or resolved.name in seen:
                    continue
                seen.add(resolved.name)
                files.append(
                    {
                        "name": resolved.name,
                        "url": f"/api/jobs/{job.id}/files/{quote(resolved.name, safe='')}",
                        "size": resolved.stat().st_size,
                    }
                )
            return {"files": files}

        job = jobs.submit("export", run, files_root=config.exports_dir(), message="Waiting to start the export")
        return {"job_id": job.id}

    # -- jobs

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return job_or_404(job_id).to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id).to_dict()
        except KeyError:
            raise HTTPException(404, "That task has finished long ago or never existed.") from None

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str) -> FileResponse:
        job = job_or_404(job_id)
        if job.files_dir is None or job.status != "done":
            raise HTTPException(404, "That file isn't ready.")
        try:
            root = job.files_dir.resolve()
            target = (root / name).resolve()
        except (OSError, ValueError):
            raise HTTPException(404, "That file doesn't exist.") from None
        if target.parent != root or not target.is_file():
            raise HTTPException(404, "That file doesn't exist.")
        return FileResponse(
            target,
            media_type=_media_type(target),
            headers={"Content-Disposition": content_disposition(target.name)},
        )

    # -- privacy

    @app.post("/api/wipe")
    def wipe(body: WipeBody) -> dict[str, Any]:
        if body.confirm != "DELETE":
            raise HTTPException(400, "To erase everything, type DELETE to confirm.")
        jobs.clear(wait=10.0)
        problems: list[str] = []
        for step in (library.wipe, wipe_user_data):
            try:
                step()
            except OSError as exc:
                problems.append(str(exc))
        library.clear_cache()
        caches.reset()
        if problems:
            raise HTTPException(500, problems[0])
        return {"ok": True}

    return app


def _valid_piper_voice_id(catalog: Any, voice_id: str) -> bool:
    """Any well-formed Piper voice id can be downloaded, not only the curated ones."""
    check = getattr(catalog, "is_valid_voice_id", None)
    try:
        return bool(check(voice_id)) if check is not None else False
    except Exception:
        return False


def _clamp_rate(rate: float) -> float:
    try:
        clamp = _tts().clamp_rate
    except (ImportError, AttributeError):
        return rate
    return float(clamp(rate))


def _trimmed(clip: AudioClip) -> AudioClip:
    """Drop the engine's leading/trailing silence from WAV clips: the UI times the pauses itself."""
    if clip.mime != "audio/wav":
        return clip
    try:
        trim = getattr(_audio(), "trim_silence", None)
    except ImportError:
        return clip
    if trim is None:
        return clip
    try:
        return trim(clip)
    except Exception as exc:  # an untrimmed clip is better than no audio
        log.warning("Couldn't trim silence: %s", exc)
        return clip


def _engine_names() -> tuple[str, ...]:
    try:
        return tuple(_tts().ENGINE_NAMES)
    except Exception:  # voices not installed: nothing to check against
        return ()


def _engine_info(engine: TTSEngine, settings: config.Settings) -> dict[str, Any]:
    name = str(getattr(engine, "name", "unknown"))
    info: dict[str, Any] = {
        "name": name,
        "label": str(getattr(engine, "label", name)),
        "online": bool(getattr(engine, "online", False)),
        "available": False,
        "reason": "",
        "default_voice": None,
    }
    try:
        available, reason = engine.is_available()
        info["available"] = bool(available)
        info["reason"] = str(reason or "")
    except Exception as exc:
        info["reason"] = f"This voice engine couldn't start: {exc}"
        return info
    # Asking an online engine for voices may contact the service, so only when allowed.
    if info["available"] and (not info["online"] or settings.online_tts_allowed):
        try:
            info["default_voice"] = engine.default_voice()
        except Exception as exc:
            log.warning("Couldn't get the default %s voice: %s", name, exc)
    return info


# -- running the server


def _bind_free_port(host: str, port: int, attempts: int = PORT_ATTEMPTS) -> tuple[socket.socket, int]:
    """Bind a listening socket on the first free port from ``port`` on."""
    bind_host = "127.0.0.1" if host.lower() == "localhost" else host.strip("[]")
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    last_error: OSError | None = None
    for candidate in range(port, min(port + attempts, 65536)):
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            if os.name != "nt":
                # Lets a restart reuse a port still in TIME_WAIT. (On Windows this flag would
                # let two servers share a port, so it's left off there.)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, candidate))
            sock.listen(128)
            return sock, candidate
        except OSError as exc:
            sock.close()
            last_error = exc
    raise OSError(
        f"Couldn't start Athena: ports {port} to {port + attempts - 1} are all in use. "
        "Close the other program, or choose another port with --port."
    ) from last_error


def _warm_up_voice() -> None:
    """Load the default offline voice so pressing play is instant. Never touches online voices."""
    try:
        settings = config.load_settings()
        engine = _tts().resolve_engine(settings.tts_engine, online_allowed=False)
        if engine.online or engine.name == "browser":
            return
        voice = settings.voices.get(engine.name)
        installed = {v.id for v in engine.list_voices() if v.installed}
        engine.synthesize("Ready.", voice if voice in installed else None, 1.0)
    except Exception as exc:  # warming up is only an optimisation
        log.info("Voice warm-up skipped: %s", exc)


def _browser_url(host: str, port: int) -> str:
    bare = host.strip("[]").lower()
    if bare in ("0.0.0.0", "::", "localhost") or _is_loopback(bare):
        name = "[::1]" if bare == "::1" else "127.0.0.1"
    else:
        name = f"[{bare}]" if ":" in bare else bare
    return f"http://{name}:{port}/"


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception as exc:  # no browser (e.g. a server without a desktop): the banner has the URL
        log.info("Couldn't open a browser: %s", exc)


def _uvicorn_server(app: FastAPI, host: str, port: int) -> Any:
    import uvicorn

    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
        ws="none",
        server_header=False,
        proxy_headers=False,
        timeout_graceful_shutdown=5,
    )
    return uvicorn.Server(server_config)


def _wait_until_serving(server: Any, thread: threading.Thread, host: str, port: int, timeout: float) -> bool:
    """True once uvicorn has started and answers an HTTP request."""
    import http.client

    deadline = time.monotonic() + timeout
    connect_host = "127.0.0.1" if host.strip("[]") in ("0.0.0.0", "::", "localhost") else host.strip("[]")
    while time.monotonic() < deadline and thread.is_alive():
        if getattr(server, "started", False):
            conn = http.client.HTTPConnection(connect_host, port, timeout=2)
            try:
                conn.request("GET", "/", headers={"Host": f"127.0.0.1:{port}"})
                conn.getresponse().read()
                return True
            except OSError:
                pass
            finally:
                conn.close()
        time.sleep(0.02)
    return False


def start_server_in_thread(
    *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 10.0
) -> tuple[str, Callable[[], None]]:
    """Start the server in a background thread, for a native-window wrapper.

    Uses the first free port from ``port`` on, waits until the server answers
    requests, and returns ``(base_url, stop)``. ``stop()`` shuts uvicorn down
    cleanly and waits for its thread; calling it again does nothing. Only
    loopback addresses are allowed here.
    """
    host = (host or DEFAULT_HOST).strip()
    if not _is_loopback(host):
        raise ValueError(f"Refusing to listen on {host}: the app window only needs this computer (127.0.0.1).")
    sock, actual_port = _bind_free_port(host, port)
    jobs = JobManager()
    try:
        server = _uvicorn_server(
            create_app(jobs=jobs, port=actual_port, bind_host=host, warm_up=True), host, actual_port
        )
    except BaseException:
        jobs.shutdown(wait=False)
        sock.close()
        raise
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name="destiny-server", daemon=True)
    lock = threading.Lock()
    stopped = False

    def stop() -> None:
        nonlocal stopped
        with lock:
            if stopped:
                return
            stopped = True
        server.should_exit = True
        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=15)
        jobs.shutdown(wait=False)
        sock.close()

    thread.start()
    if not _wait_until_serving(server, thread, host, actual_port, timeout):
        stop()
        raise RuntimeError("Athena's reader didn't start in time. Close other copies of it and try again.")
    return _browser_url(host, actual_port), stop


def run(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    *,
    allow_remote: bool = False,
) -> None:
    """Start the server and block until Ctrl+C.

    Listening on anything but a loopback address needs ``allow_remote=True`` (the CLI
    passes it when the user explicitly gives ``--host``) and prints a privacy warning.
    """
    host = (host or DEFAULT_HOST).strip()
    loopback = _is_loopback(host)
    if not loopback and not allow_remote:
        raise ValueError(
            f"Refusing to listen on {host}: Athena only listens on this computer (127.0.0.1) "
            "unless you explicitly choose another address."
        )
    sock, actual_port = _bind_free_port(host, port)
    jobs = JobManager()
    try:
        app = create_app(jobs=jobs, port=actual_port, bind_host=host, warm_up=True)
        url = _browser_url(host, actual_port)
        if actual_port != port:
            print(f"Port {port} is busy, so Athena is using port {actual_port} instead.")
        if not loopback:
            print(
                f"WARNING: Athena is listening on {host}, so other devices on your network can open it\n"
                "and read your books, notes and settings. Only do this on a network you trust.\n"
                "Use the default (127.0.0.1) to keep everything private to this computer."
            )
        print(f"Athena is running at {url} (press Ctrl+C to stop)", flush=True)
        if open_browser:
            timer = threading.Timer(1.0, _open_browser, args=(url,))
            timer.daemon = True
            timer.start()
        server = _uvicorn_server(app, host, actual_port)
        try:
            server.run(sockets=[sock])
        except KeyboardInterrupt:  # uvicorn re-raises the Ctrl+C it already handled gracefully
            pass
    finally:
        jobs.shutdown(wait=False)
        sock.close()
    print("Athena stopped.")
