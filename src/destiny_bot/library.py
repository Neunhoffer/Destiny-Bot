"""The local library: imported books, reading progress and bookmarks.

Every book lives in its own folder, ``library/<doc_id>/``:

- ``document.json``: the parsed :class:`~destiny_bot.models.Document`, written once at import.
- ``meta.json``: added_at, last_opened, progress and bookmarks, plus a cached copy of the
  summary fields so listing the library never has to parse every book.
- ``source<ext>``: a copy of the original file (used, for example, to draw PDF pages).

Every write goes to a temp file first and is then moved into place with ``os.replace``,
so a crash never leaves a half-written file behind.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from . import config, ingest
from .ingest import IngestError, ProgressFn
from .ingest.common import doc_id_for_bytes, doc_id_for_file
from .models import Document

log = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"[0-9a-f]{16}")
_BOOKMARK_ID_RE = re.compile(r"[0-9a-f]{16}")
_TEMP_PREFIX = ".tmp-"
_TRASH_PREFIX = ".deleted-"
_DOC_CACHE_SIZE = 4
MAX_TITLE_CHARS = 200
MAX_LABEL_CHARS = 200


def utc_now() -> str:
    """The current UTC time as ISO-8601 with milliseconds and a ``Z`` suffix (like JS ``toISOString``)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _iso_from_timestamp(ts: float) -> str:
    when = datetime.fromtimestamp(ts, timezone.utc)
    return when.strftime("%Y-%m-%dT%H:%M:%S.") + f"{when.microsecond // 1000:03d}Z"


def is_valid_doc_id(value: object) -> bool:
    """True for a well-formed book id (16 lowercase hex characters)."""
    return isinstance(value, str) and _DOC_ID_RE.fullmatch(value) is not None


def temp_dir() -> Path:
    """Scratch folder for uploads in progress (inside the data folder, so a wipe removes it)."""
    path = config.data_dir() / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Progress(BaseModel):
    section_id: str
    utterance: int = 0
    updated_at: str


class Bookmark(BaseModel):
    id: str
    section_id: str
    utterance: int
    label: str
    created_at: str


class BookSummary(BaseModel):
    id: str
    title: str
    author: str | None = None
    source_name: str
    source_type: str
    added_at: str
    section_count: int
    page_count: int | None = None
    progress: Progress | None = None


class Library:
    """Books the user has imported, stored under ``root`` (default ``config.library_dir()``).

    Safe to share between threads. Documents returned by :meth:`get` are cached and
    shared, so callers must treat them as read-only.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else None
        self._meta_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._docs: OrderedDict[str, tuple[tuple[int, int], Document]] = OrderedDict()

    @property
    def root(self) -> Path:
        root = self._root if self._root is not None else config.library_dir()
        root.mkdir(parents=True, exist_ok=True)
        return root

    # -- importing

    def import_file(
        self,
        path: Path,
        *,
        original_name: str | None = None,
        progress: ProgressFn | None = None,
    ) -> Document:
        """Import a file (or return the existing copy if the same file was imported before).

        ``original_name`` is the name the user knows the file by (uploads arrive as temp
        files). Raises :class:`~destiny_bot.ingest.IngestError` with a friendly message.
        """
        path = Path(path)
        name = original_name or path.name
        if ingest.source_type_for(name) is None and ingest.source_type_for(path.name) is None:
            # Let the dispatcher produce its standard "can't open these files" message.
            return ingest.load_document(path, source_name=name, progress=progress)
        if not path.is_file():
            raise IngestError(f"File not found: {path}")
        try:
            doc_id = doc_id_for_file(path)
        except OSError as exc:
            raise IngestError(f"Couldn't open '{name}': {exc.strerror or exc}") from exc

        existing = self._try_get(doc_id)
        if existing is not None:
            self.mark_opened(doc_id)
            if progress is not None:
                progress(1.0, "This file is already in your library")
            return existing

        doc = ingest.load_document(path, source_name=name, progress=progress)
        if doc.id != doc_id:
            doc = doc.model_copy(update={"id": doc_id})
        self._store(doc, lambda dest_dir: _store_source_copy(path, dest_dir, _source_extension(name, path)))
        return doc

    def import_text(self, title: str, text: str) -> Document:
        """Add pasted text as a book (Markdown headings become sections)."""
        if not text or not text.strip():
            raise IngestError("There's no text to read. Paste some text first.")
        title = " ".join((title or "").split())[:MAX_TITLE_CHARS] or "Pasted text"
        doc = ingest.document_from_text(text, title)
        doc_id = doc.id if is_valid_doc_id(doc.id) else doc_id_for_bytes(text.encode("utf-8"))
        if doc.id != doc_id:
            doc = doc.model_copy(update={"id": doc_id})

        existing = self._try_get(doc_id)
        if existing is not None:
            self.mark_opened(doc_id)
            return existing
        payload = text.encode("utf-8")
        self._store(doc, lambda dest_dir: _atomic_write(dest_dir / "source.txt", payload))
        return doc

    def _store(self, doc: Document, write_source: Callable[[Path], None]) -> None:
        book_dir = self.root / doc.id
        book_dir.mkdir(parents=True, exist_ok=True)
        write_source(book_dir)
        doc_path = book_dir / "document.json"
        _atomic_write(doc_path, doc.model_dump_json().encode("utf-8"))
        self._remember(doc, doc_path)
        now = utc_now()
        with self._meta_lock:
            meta = self._read_meta(doc.id) or {}
            meta.setdefault("added_at", now)
            meta["last_opened"] = now
            meta.setdefault("progress", None)
            meta.setdefault("bookmarks", [])
            _attach_summary(meta, doc, doc_path)
            self._write_meta(doc.id, meta)

    # -- reading

    def list(self) -> list[BookSummary]:
        """Every readable book, most recently opened (or added) first. Broken entries are skipped."""
        entries: list[tuple[str, BookSummary]] = []
        try:
            children = sorted(self.root.iterdir())
        except OSError:
            return []
        for child in children:
            if not is_valid_doc_id(child.name) or not child.is_dir():
                continue
            try:
                summary, meta = self._summary_and_meta(child.name)
            except (KeyError, OSError, ValueError) as exc:
                log.info("Skipping unreadable library entry %s: %s", child.name, exc)
                continue
            sort_key = meta.get("last_opened") or summary.added_at
            entries.append((str(sort_key), summary))
        entries.sort(key=lambda e: (e[0], e[1].added_at, e[1].id), reverse=True)
        return [summary for _, summary in entries]

    def get(self, doc_id: str) -> Document:
        """The full document. Raises KeyError if it's missing, damaged or the id is invalid."""
        doc_path = self._book_dir(doc_id) / "document.json"
        try:
            st = doc_path.stat()
        except OSError:
            raise KeyError(doc_id) from None
        stamp = (st.st_size, st.st_mtime_ns)
        with self._cache_lock:
            hit = self._docs.get(doc_id)
            if hit is not None and hit[0] == stamp:
                self._docs.move_to_end(doc_id)
                return hit[1]
        try:
            doc = Document.model_validate_json(doc_path.read_bytes())
        except (OSError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
            log.info("Couldn't read %s: %s", doc_path, exc)
            raise KeyError(doc_id) from None
        if doc.id != doc_id:
            raise KeyError(doc_id)
        self._remember(doc, doc_path, stamp)
        return doc

    def summary(self, doc_id: str) -> BookSummary:
        """Summary of one book. Raises KeyError if it's missing or damaged."""
        try:
            return self._summary_and_meta(doc_id)[0]
        except (OSError, ValueError):
            raise KeyError(doc_id) from None

    def source_file(self, doc_id: str) -> Path | None:
        """The stored copy of the original file, if there is one."""
        book_dir = self._book_dir(doc_id)
        try:
            candidates = sorted(book_dir.iterdir())
        except OSError:
            return None
        for path in candidates:
            if (path.name == "source" or path.name.startswith("source.")) and path.is_file():
                return path
        return None

    def mark_opened(self, doc_id: str) -> None:
        """Record that the book was just opened (it moves to the top of :meth:`list`)."""
        self._update_meta(doc_id, lambda meta: meta.__setitem__("last_opened", utc_now()))

    # -- removing

    def delete(self, doc_id: str) -> None:
        """Remove a book with its progress and bookmarks. Raises KeyError if it isn't there."""
        book_dir = self._book_dir(doc_id)
        if not book_dir.is_dir():
            raise KeyError(doc_id)
        with self._cache_lock:
            self._docs.pop(doc_id, None)
        # Move it out of the way first so a half-deleted book never shows up in list().
        trash = self.root / f"{_TRASH_PREFIX}{doc_id}-{secrets.token_hex(4)}"
        with self._meta_lock:
            try:
                _replace(book_dir, trash)
            except OSError:
                trash = book_dir
        _rmtree(trash)
        if book_dir.exists():
            raise OSError(f"Couldn't delete every file of this book. Close any program using {book_dir} and try again.")

    def wipe(self) -> None:
        """Delete every book (used by "Erase all my data")."""
        with self._cache_lock:
            self._docs.clear()
        root = self.root
        problems: list[str] = []
        with self._meta_lock:
            _rmtree(root, problems)
        root.mkdir(parents=True, exist_ok=True)
        if problems:
            raise OSError(_wipe_problem_message(problems))

    def clear_cache(self) -> None:
        """Forget cached documents (after files were changed behind the library's back)."""
        with self._cache_lock:
            self._docs.clear()

    # -- progress and bookmarks

    def get_progress(self, doc_id: str) -> Progress | None:
        meta = self._read_meta_checked(doc_id)
        return _parse_progress(meta.get("progress"))

    def set_progress(self, doc_id: str, section_id: str, utterance: int) -> Progress:
        """Save the reading position. ValueError for an unknown section or a negative utterance."""
        doc = self.get(doc_id)
        _check_position(doc, section_id, utterance)
        progress = Progress(section_id=section_id, utterance=int(utterance), updated_at=utc_now())
        self._update_meta(doc_id, lambda meta: meta.__setitem__("progress", progress.model_dump()))
        return progress

    def list_bookmarks(self, doc_id: str) -> list[Bookmark]:
        meta = self._read_meta_checked(doc_id)
        return sorted(_parse_bookmarks(meta.get("bookmarks")), key=lambda b: (b.created_at, b.id))

    def add_bookmark(self, doc_id: str, section_id: str, utterance: int, label: str) -> Bookmark:
        """Bookmark a position. A blank label becomes the section title."""
        doc = self.get(doc_id)
        _check_position(doc, section_id, utterance)
        clean = " ".join((label or "").split())[:MAX_LABEL_CHARS] or doc.section(section_id).title
        bookmark = Bookmark(
            id=secrets.token_hex(8),
            section_id=section_id,
            utterance=int(utterance),
            label=clean,
            created_at=utc_now(),
        )

        def add(meta: dict[str, Any]) -> None:
            marks = [b.model_dump() for b in _parse_bookmarks(meta.get("bookmarks"))]
            marks.append(bookmark.model_dump())
            meta["bookmarks"] = marks

        self._update_meta(doc_id, add)
        return bookmark

    def delete_bookmark(self, doc_id: str, bookmark_id: str) -> None:
        """Remove one bookmark. Raises KeyError if the book or bookmark doesn't exist."""
        if not isinstance(bookmark_id, str) or not _BOOKMARK_ID_RE.fullmatch(bookmark_id):
            raise KeyError(bookmark_id)

        def remove(meta: dict[str, Any]) -> None:
            marks = _parse_bookmarks(meta.get("bookmarks"))
            kept = [b for b in marks if b.id != bookmark_id]
            if len(kept) == len(marks):
                raise KeyError(bookmark_id)
            meta["bookmarks"] = [b.model_dump() for b in kept]

        self._update_meta(doc_id, remove)

    # -- internals

    def _book_dir(self, doc_id: str) -> Path:
        if not is_valid_doc_id(doc_id):
            raise KeyError(doc_id)
        return self.root / doc_id

    def _try_get(self, doc_id: str) -> Document | None:
        try:
            return self.get(doc_id)
        except KeyError:
            return None

    def _remember(self, doc: Document, doc_path: Path, stamp: tuple[int, int] | None = None) -> None:
        if stamp is None:
            try:
                st = doc_path.stat()
            except OSError:
                return
            stamp = (st.st_size, st.st_mtime_ns)
        with self._cache_lock:
            self._docs[doc.id] = (stamp, doc)
            self._docs.move_to_end(doc.id)
            while len(self._docs) > _DOC_CACHE_SIZE:
                self._docs.popitem(last=False)

    def _read_meta(self, doc_id: str) -> dict[str, Any] | None:
        """meta.json as a dict, or None when it's missing or damaged."""
        path = self._book_dir(doc_id) / "meta.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.info("Ignoring damaged %s: %s", path, exc)
            return None
        return data if isinstance(data, dict) else None

    def _default_meta(self, doc_id: str) -> dict[str, Any]:
        doc_path = self._book_dir(doc_id) / "document.json"
        try:
            added = _iso_from_timestamp(doc_path.stat().st_mtime)
        except OSError:
            raise KeyError(doc_id) from None
        return {"added_at": added, "last_opened": None, "progress": None, "bookmarks": []}

    def _read_meta_checked(self, doc_id: str) -> dict[str, Any]:
        """meta.json for an existing book (defaults if it's missing or damaged); KeyError if no book."""
        if not (self._book_dir(doc_id) / "document.json").is_file():
            raise KeyError(doc_id)
        meta = self._read_meta(doc_id)
        return meta if meta is not None else self._default_meta(doc_id)

    def _write_meta(self, doc_id: str, meta: dict[str, Any]) -> None:
        path = self._book_dir(doc_id) / "meta.json"
        _atomic_write(path, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))

    def _update_meta(self, doc_id: str, change: Callable[[dict[str, Any]], None]) -> None:
        with self._meta_lock:
            meta = self._read_meta_checked(doc_id)
            change(meta)
            self._write_meta(doc_id, meta)

    def _summary_and_meta(self, doc_id: str) -> tuple[BookSummary, dict[str, Any]]:
        doc_path = self._book_dir(doc_id) / "document.json"
        try:
            st = doc_path.stat()
        except OSError:
            raise KeyError(doc_id) from None
        meta = self._read_meta(doc_id)
        if meta is not None and meta.get("doc_stat") == [st.st_size, st.st_mtime_ns]:
            summary = self._summary_from_meta(doc_id, meta)
            if summary is not None:
                return summary, meta
        # First time, or document.json changed behind our back: rebuild from the document.
        doc = self.get(doc_id)
        with self._meta_lock:
            meta = self._read_meta(doc_id) or self._default_meta(doc_id)
            meta.setdefault("progress", None)
            meta.setdefault("bookmarks", [])
            _attach_summary(meta, doc, doc_path)
            try:
                self._write_meta(doc_id, meta)
            except OSError as exc:  # a read-only folder still lists the book
                log.warning("Couldn't update %s/meta.json: %s", doc_id, exc)
        summary = self._summary_from_meta(doc_id, meta)
        if summary is None:
            raise KeyError(doc_id)
        return summary, meta

    def _summary_from_meta(self, doc_id: str, meta: dict[str, Any]) -> BookSummary | None:
        cached = meta.get("summary")
        if not isinstance(cached, dict):
            return None
        try:
            return BookSummary.model_validate(
                {
                    **cached,
                    "id": doc_id,
                    "added_at": str(meta.get("added_at") or self._default_meta(doc_id)["added_at"]),
                    "progress": _parse_progress(meta.get("progress")),
                }
            )
        except ValidationError:
            return None


def _attach_summary(meta: dict[str, Any], doc: Document, doc_path: Path) -> None:
    st = doc_path.stat()
    meta["summary"] = {
        "title": doc.title,
        "author": doc.author,
        "source_name": doc.source_name,
        "source_type": doc.source_type,
        "section_count": len(doc.sections),
        "page_count": doc.page_count,
    }
    meta["doc_stat"] = [st.st_size, st.st_mtime_ns]


def _parse_progress(value: Any) -> Progress | None:
    if not isinstance(value, dict):
        return None
    try:
        return Progress.model_validate(value)
    except ValidationError:
        return None


def _parse_bookmarks(value: Any) -> list[Bookmark]:
    marks: list[Bookmark] = []
    for item in value if isinstance(value, list) else []:
        try:
            marks.append(Bookmark.model_validate(item))
        except ValidationError:
            continue
    return marks


def _check_position(doc: Document, section_id: str, utterance: int) -> None:
    if not any(sec.id == section_id for sec in doc.sections):
        raise ValueError(f"This book has no section '{section_id}'.")
    if utterance < 0:
        raise ValueError("The reading position can't be negative.")


def _source_extension(name: str, path: Path) -> str:
    for candidate in (Path(name).suffix.lower(), path.suffix.lower()):
        if candidate in ingest.SUPPORTED_EXTENSIONS:
            return candidate
    return ""


def _store_source_copy(src: Path, book_dir: Path, ext: str) -> None:
    target = book_dir / f"source{ext}"
    for old in book_dir.glob("source*"):
        if old.name != target.name and (old.name == "source" or old.name.startswith("source.")):
            _unlink(old)
    fd, tmp = tempfile.mkstemp(dir=book_dir, prefix=_TEMP_PREFIX, suffix=ext)
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out, 1 << 20)
            out.flush()
            os.fsync(out.fileno())
        _replace(Path(tmp), target)
    except BaseException:
        _unlink(Path(tmp))
        raise


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=_TEMP_PREFIX, suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _replace(Path(tmp), path)
    except BaseException:
        _unlink(Path(tmp))
        raise


def _replace(src: Path, dst: Path) -> None:
    """``os.replace`` that retries briefly on Windows, where a reader or virus scanner may hold the file."""
    attempts = 8 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("Couldn't delete %s: %s", path, exc)


def _rmtree(path: Path, problems: list[str] | None = None) -> None:
    """Delete a folder tree, clearing read-only flags (Windows) and collecting failures."""

    def on_error(func: Any, failed: str, exc: BaseException) -> None:
        if isinstance(exc, FileNotFoundError):
            return
        if func in (os.unlink, os.remove, os.rmdir):
            try:
                os.chmod(failed, 0o700)  # read-only files can't be deleted on Windows
                func(failed)
                return
            except OSError:
                pass
        if problems is not None:
            problems.append(str(failed))
        log.warning("Couldn't delete %s: %s", failed, exc)

    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=on_error)
    else:
        shutil.rmtree(path, onerror=lambda func, failed, info: on_error(func, failed, info[1]))


def _wipe_problem_message(problems: list[str]) -> str:
    shown = ", ".join(problems[:3]) + (" ..." if len(problems) > 3 else "")
    return f"Some files couldn't be deleted ({shown}). Close other programs using them and try again."


def wipe_user_data() -> None:
    """Delete the library, audio cache, exports, temp files and settings (with the API key).

    Downloaded voices are kept. Empty folders are recreated afterwards. Raises OSError
    with a friendly message if some files couldn't be deleted.
    """
    root = config.data_dir()
    problems: list[str] = []
    for name in ("library", "cache", "exports", "tmp"):
        _rmtree(root / name, problems)
    for path in [config.settings_path(), *root.glob(".settings-*")]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            problems.append(str(path))
    config.library_dir()
    config.cache_dir()
    config.exports_dir()
    if problems:
        raise OSError(_wipe_problem_message(problems))
