"""Export a book (or some of its sections) as audiobook files: one track per section.

Each section is planned with the speech layer, spoken utterance by utterance,
and streamed straight into its file, so memory stays small even for long
chapters. Each utterance's own leading/trailing silence is trimmed, so the
pacing comes entirely from the plan's ``pause_after_ms``. MP3 tracks get ID3
tags, and a ``playlist.m3u`` lists them in order.
"""

from __future__ import annotations

import io
import os
import re
import struct
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Callable, Literal

from .audio import ClipJoiner, trim_silence
from .models import Document, ReadingOptions, Section, Utterance
from .tts.base import AudioClip, EngineUnavailable, TTSEngine, TTSError

LEAD_SILENCE_MS = 500
TRAIL_SILENCE_MS = 500
MP3_BITRATE_KBPS = 64
PLAYLIST_NAME = "playlist.m3u"

ProgressFn = Callable[[float, str], None]


class ExportCancelled(RuntimeError):
    """The user cancelled the export; files it had started were removed."""


class ExportError(RuntimeError):
    """The export can't be done; the message is safe to show to the user."""


def export_audiobook(
    doc: Document,
    *,
    engine: TTSEngine,
    voice: str | None,
    options: ReadingOptions,
    out_dir: Path,
    section_ids: list[str] | None = None,
    fmt: Literal["mp3", "wav"] = "mp3",
    rate: float = 1.0,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> list[Path]:
    """Write one audio file per section plus ``playlist.m3u`` into ``out_dir``.

    Returns the track paths in reading order followed by the playlist. Sections
    with nothing to read are skipped. Raises :class:`ExportCancelled` when
    ``cancel`` is set (all files written by this call are removed),
    :class:`ExportError` for impossible requests, and ``TTSError`` /
    ``EngineUnavailable`` from the voice.
    """
    if fmt not in ("mp3", "wav"):
        raise ExportError("Audio can be saved as MP3 or WAV.")
    sections = _select_sections(doc, section_ids)
    plans = [(sec, plan) for sec in sections if (plan := _plan(sec, options))]
    if not plans:
        raise ExportError("There's nothing to read aloud in the chosen sections.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_utts = sum(len(plan) for _, plan in plans)
    width = max(2, len(str(len(plans))))
    report = progress or (lambda fraction, message: None)
    written: list[Path] = []
    tracks: list[tuple[Path, float, str]] = []
    done = 0

    try:
        for number, (section, plan) in enumerate(plans, start=1):
            title = section.title.strip() or f"Section {number}"

            def on_utterance(i: int, n: int, title: str = title, before: int = done) -> None:
                report((before + i) / total_utts * 0.99, f"Recording {title} ({i}/{n})")

            path, seconds = _record_section(
                section_title=title,
                plan=plan,
                number=number,
                total_tracks=len(plans),
                width=width,
                doc=doc,
                engine=engine,
                voice=voice,
                rate=rate,
                fmt=fmt,
                out_dir=out_dir,
                written=written,
                cancel=cancel,
                on_utterance=on_utterance,
            )
            done += len(plan)
            tracks.append((path, seconds, title))

        _check_cancel(cancel)
        playlist = out_dir / PLAYLIST_NAME
        _atomic_write(playlist, _playlist_text(tracks).encode("utf-8"))
        written.append(playlist)
    except BaseException as exc:
        for path in written:
            _unlink(path)
        if isinstance(exc, ExportCancelled):
            report(done / total_utts, "Export cancelled")
        raise

    report(1.0, f"Saved {len(tracks)} audio file{'s' if len(tracks) != 1 else ''}")
    return [path for path, _, _ in tracks] + [playlist]


def _plan(section: Section, options: ReadingOptions) -> list[Utterance]:
    from .speech import plan_section  # imported lazily: the speech layer is optional at import time

    return plan_section(section, options)


def _select_sections(doc: Document, section_ids: list[str] | None) -> list[Section]:
    if section_ids is None:
        return list(doc.sections)
    wanted = set(section_ids)
    known = {sec.id for sec in doc.sections}
    missing = [sid for sid in section_ids if sid not in known]
    if missing:
        raise ExportError("Some of the chosen sections aren't in this book any more. Please choose again.")
    return [sec for sec in doc.sections if sec.id in wanted]


def _record_section(
    *,
    section_title: str,
    plan: list[Utterance],
    number: int,
    total_tracks: int,
    width: int,
    doc: Document,
    engine: TTSEngine,
    voice: str | None,
    rate: float,
    fmt: str,
    out_dir: Path,
    written: list[Path],
    cancel: threading.Event | None,
    on_utterance: Callable[[int, int], None],
) -> tuple[Path, float]:
    """Speak one section into its own file. Returns (path, duration in seconds)."""
    _check_cancel(cancel)
    first = _speak(engine, plan[0].speech, voice, rate)
    # MP3-only engines can't produce WAV (we have no MP3 decoder); save those as MP3.
    ext = "mp3" if fmt == "mp3" or first.mime == "audio/mpeg" else "wav"
    prefix = f"{number:0{width}d} - "
    final = out_dir / f"{prefix}{safe_filename(section_title, _name_room(out_dir, prefix, ext))}.{ext}"
    fd, tmp_name = tempfile.mkstemp(dir=out_dir, prefix=".export-", suffix=f".{ext}.part")
    tmp = Path(tmp_name)
    written.append(tmp)
    with os.fdopen(fd, "wb") as fh:
        if ext == "mp3":
            fh.write(
                id3v2_tag(
                    title=section_title,
                    album=doc.title,
                    artist=doc.author,
                    track=f"{number}/{total_tracks}",
                )
            )
        joiner = ClipJoiner(fh, "mp3" if ext == "mp3" else "wav", bitrate_kbps=MP3_BITRATE_KBPS)
        joiner.add_silence(LEAD_SILENCE_MS)
        total = len(plan)
        for i, utt in enumerate(plan):
            if i:
                _check_cancel(cancel)
            clip = first if i == 0 else _speak(engine, utt.speech, voice, rate)
            try:
                joiner.add(clip, utt.pause_after_ms)
            except ValueError as exc:
                raise ExportError(f"Couldn't join the recorded audio: {exc}") from exc
            on_utterance(i + 1, total)
        joiner.add_silence(TRAIL_SILENCE_MS)
        seconds = joiner.close()
    _check_cancel(cancel)
    try:
        _replace(tmp, final)
    except PermissionError as exc:
        raise ExportError(
            f"Couldn't save \"{final.name}\" because another program (maybe a music player) is using a file "
            "with that name. Close it and try again."
        ) from exc
    written[written.index(tmp)] = final
    return final, seconds


def _speak(engine: TTSEngine, text: str, voice: str | None, rate: float) -> AudioClip:
    """One utterance with its edge silence trimmed, so pauses come only from the plan."""
    try:
        clip = engine.synthesize(text, voice, rate)
    except EngineUnavailable:
        raise
    except TTSError:
        clip = engine.synthesize(text, voice, rate)  # one retry for a passing hiccup
    return trim_silence(clip)


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise ExportCancelled("Export cancelled.")


# --------------------------------------------------------------------------- file names

_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
# Windows device names, also with any extension ("NUL.txt", "com1 .tar.gz") or a superscript digit.
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"{dev}{n}" for dev in ("COM", "LPT") for n in "0123456789¹²³")}


# Windows refuses full paths of 260 characters or more unless long paths are switched on.
_WINDOWS_MAX_PATH = 259
_MIN_NAME = 20


def _name_room(out_dir: Path, prefix: str, ext: str, max_len: int = 80) -> int:
    """How long a track's title part may be so its full path stays usable on this system."""
    if sys.platform != "win32":
        return max_len
    used = len(os.path.abspath(out_dir)) + 1 + len(prefix) + 1 + len(ext)
    return max(_MIN_NAME, min(max_len, _WINDOWS_MAX_PATH - used))


def safe_filename(name: str, max_len: int = 80) -> str:
    """A file name that is valid on Windows, macOS and Linux (no extension added)."""
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"[\t\n\r\f\v]", " ", name)
    name = "".join(ch for ch in name if unicodedata.category(ch) not in ("Cc", "Cf"))
    name = _FORBIDDEN.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        cut = name[:max_len]
        space = cut.rfind(" ")
        name = cut[:space] if space > max_len // 2 else cut
    name = name.rstrip(" .").lstrip(" ")
    if not name:
        name = "Untitled"
    stem, dot, rest = name.partition(".")
    if stem.rstrip(" ").upper() in _RESERVED:
        name = f"{stem.rstrip(' ')}_{dot}{rest}"
    return name


# --------------------------------------------------------------------------- ID3 + playlist


def _id3_text_frame(frame_id: str, text: str) -> bytes:
    try:
        payload = b"\x00" + text.encode("latin-1")
    except UnicodeEncodeError:
        payload = b"\x01" + text.encode("utf-16")  # with BOM, as ID3v2.3 requires
    return frame_id.encode("ascii") + struct.pack(">I", len(payload)) + b"\x00\x00" + payload


def _syncsafe(n: int) -> bytes:
    return bytes(((n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F))


def id3v2_tag(*, title: str, album: str | None = None, artist: str | None = None, track: str | None = None) -> bytes:
    """A minimal ID3v2.3 tag (TIT2, TALB, TPE1, TRCK) to put in front of MP3 audio."""
    frames = io.BytesIO()
    for frame_id, value in (("TIT2", title), ("TALB", album), ("TPE1", artist), ("TRCK", track)):
        if value and value.strip():
            frames.write(_id3_text_frame(frame_id, value.strip()))
    body = frames.getvalue()
    return b"ID3\x03\x00\x00" + _syncsafe(len(body)) + body


def _playlist_text(tracks: list[tuple[Path, float, str]]) -> str:
    lines = ["#EXTM3U"]
    for path, seconds, title in tracks:
        clean_title = " ".join(title.split())
        lines.append(f"#EXTINF:{max(1, round(seconds))},{clean_title}")
        lines.append(path.name)
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".export-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        _replace(Path(tmp), path)
    except BaseException:
        _unlink(Path(tmp))
        raise


def _replace(src: Path, dst: Path) -> None:
    """``os.replace`` that retries briefly on Windows, where a virus scanner or player may hold the file."""
    attempts = 8 if sys.platform == "win32" else 1
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
    except OSError:
        pass
