"""On-disk LRU cache of synthesized audio.

Each clip is stored as ``<sha256 of the key parts><ext>`` in one folder. A hit
touches the file's modification time, and the least recently used files are
deleted once the folder grows past ``max_bytes``.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Callable

from .base import AudioClip

_EXTENSIONS = {".wav": "audio/wav", ".mp3": "audio/mpeg"}
_NAME_RE = re.compile(r"^[0-9a-f]{64}\.(wav|mp3)$")
_TEMP_PREFIX = ".tmp-"


class AudioCache:
    def __init__(self, root: Path, max_bytes: int = 500 * 1024 * 1024) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._key_locks: dict[str, list] = {}  # digest -> [lock, number of waiters]
        self._total: int | None = None  # bytes on disk; computed on first write

    @staticmethod
    def key_for(key_parts: tuple[str, ...]) -> str:
        return hashlib.sha256("\x1f".join(key_parts).encode("utf-8")).hexdigest()

    def get(self, key_parts: tuple[str, ...]) -> AudioClip | None:
        """The cached clip, or None (missing, empty or unreadable files count as misses)."""
        return self._read(self.key_for(key_parts))

    def get_or_create(self, key_parts: tuple[str, ...], make: Callable[[], AudioClip]) -> AudioClip:
        """Return the cached clip or call ``make()`` and store its result.

        Concurrent callers asking for the same key wait for one ``make()`` call.
        """
        digest = self.key_for(key_parts)
        clip = self._read(digest)
        if clip is not None:
            return clip
        with self._lock:
            slot = self._key_locks.setdefault(digest, [threading.Lock(), 0])
            slot[1] += 1
        try:
            with slot[0]:
                clip = self._read(digest)
                if clip is not None:
                    return clip
                clip = make()
                self._write(digest, clip)
        finally:
            with self._lock:
                slot[1] -= 1
                if slot[1] == 0:
                    self._key_locks.pop(digest, None)
        self._evict()
        return clip

    def clear(self) -> None:
        """Delete every cached clip."""
        with self._lock:
            for entry in self._entries():
                _unlink(entry.path)
            self._total = 0

    def size_bytes(self) -> int:
        with self._lock:
            return sum(e.size for e in self._entries())

    # -- internals

    def _read(self, digest: str) -> AudioClip | None:
        for ext, mime in _EXTENSIONS.items():
            path = self.root / f"{digest}{ext}"
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                continue
            except OSError:
                return None
            clip = _decode(data, mime)
            if clip is None:  # corrupt: drop it so it gets re-created
                self._remove(path, len(data))
                return None
            try:
                os.utime(path, None)
            except OSError:
                pass
            return clip
        return None

    def _write(self, digest: str, clip: AudioClip) -> None:
        if not clip.data:
            return
        ext = clip.extension
        path = self.root / f"{digest}{ext}"
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=_TEMP_PREFIX, suffix=ext)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(clip.data)
            os.replace(tmp, path)
        except OSError:
            # A full disk or a file locked by a reader (Windows) only costs a cache miss.
            _unlink(Path(tmp))
            return
        other = ".mp3" if ext == ".wav" else ".wav"
        _unlink(self.root / f"{digest}{other}")
        with self._lock:
            if self._total is not None:
                self._total += len(clip.data)

    def _remove(self, path: Path, size: int) -> None:
        if _unlink(path):
            with self._lock:
                if self._total is not None:
                    self._total = max(0, self._total - size)

    def _entries(self) -> list[_Entry]:
        entries: list[_Entry] = []
        try:
            scan = list(os.scandir(self.root))
        except OSError:
            return entries
        for item in scan:
            if not (_NAME_RE.match(item.name) or item.name.startswith(_TEMP_PREFIX)):
                continue
            try:
                st = item.stat()
            except OSError:
                continue
            entries.append(_Entry(Path(item.path), st.st_size, st.st_mtime))
        return entries

    def _evict(self) -> None:
        with self._lock:
            if self._total is not None and self._total <= self.max_bytes:
                return
            entries = [e for e in self._entries() if not e.path.name.startswith(_TEMP_PREFIX)]
            total = sum(e.size for e in entries)
            if total > self.max_bytes:
                # Trim to 90% so we don't rescan on every new clip.
                target = int(self.max_bytes * 0.9)
                for entry in sorted(entries, key=lambda e: e.mtime):
                    if total <= target:
                        break
                    if _unlink(entry.path):
                        total -= entry.size
            self._total = total


class _Entry:
    __slots__ = ("path", "size", "mtime")

    def __init__(self, path: Path, size: int, mtime: float) -> None:
        self.path, self.size, self.mtime = path, size, mtime


def _decode(data: bytes, mime: str) -> AudioClip | None:
    """Validate cached bytes and rebuild the clip (None if the file is damaged)."""
    if not data:
        return None
    if mime == "audio/mpeg":
        from ..audio import mp3_info

        try:
            info = mp3_info(data)
        except ValueError:
            return None
        return AudioClip(data, mime, info.sample_rate)
    from ..audio import read_wav

    try:
        pcm, rate, _, _ = read_wav(data)
    except ValueError:
        return None
    if not pcm:
        return None
    return AudioClip(data, mime, rate)


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:  # in use on Windows; try again next time
        return False
