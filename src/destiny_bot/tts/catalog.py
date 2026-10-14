"""Curated downloadable Piper voices and a safe downloader for them.

Voices come from the official ``rhasspy/piper-voices`` repository on Hugging Face
and are saved as ``<voice_id>.onnx`` + ``<voice_id>.onnx.json`` pairs in
``config.voices_dir() / "piper"``. Downloading is the only network access in
this module and only happens when the user asks for a voice.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from .. import config

DEFAULT_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
RECOMMENDED_VOICE = "en_US-lessac-medium"

# <lang>_<REGION>-<name>-<quality>, e.g. en_US-lessac-medium. Strict on purpose: the id
# becomes part of a URL and a file name, so no dots, slashes or other surprises.
VOICE_ID_RE = re.compile(
    r"^(?P<family>[a-z]{2,3})_(?P<region>[A-Z0-9]{2,3})-(?P<name>[A-Za-z0-9_]{1,64})"
    r"-(?P<quality>x_low|low|medium|high)$"
)

_CHUNK = 256 * 1024
_TIMEOUT_S = 30
_NETWORK_ERROR = "Couldn't download the voice. Check your internet connection."


class CatalogVoice(BaseModel):
    id: str  # e.g. "en_US-lessac-medium"
    name: str  # e.g. "Lessac (US English)"
    language: str  # BCP-47-ish, e.g. "en-US"
    quality: str  # "x_low" | "low" | "medium" | "high"
    size_mb: float  # approximate download size
    description: str
    gender: str | None = None
    recommended: bool = False


class VoiceDownloadError(RuntimeError):
    """The voice could not be downloaded; the message is safe to show to the user."""


class DownloadCancelled(RuntimeError):
    """The user cancelled the download."""


# (voice id, display name, approx. download size in MB, gender, description)
_ENTRIES: list[tuple[str, str, float, str, str]] = [
    ("en_US-lessac-medium", "Lessac (US English)", 63.2, "female",
     "Clear, warm female voice. The best all-round choice for textbooks."),
    ("en_US-amy-medium", "Amy (US English)", 63.2, "female", "Friendly, bright female voice."),
    ("en_US-ryan-high", "Ryan (US English, high quality)", 120.8, "male",
     "Rich male voice with extra detail; larger download, a little slower."),
    ("en_US-hfc_female-medium", "HFC Female (US English)", 63.2, "female",
     "Calm, even-paced female voice, good for long listening."),
    ("en_US-hfc_male-medium", "HFC Male (US English)", 63.2, "male",
     "Calm, even-paced male voice, good for long listening."),
    ("en_US-joe-medium", "Joe (US English)", 63.2, "male", "Relaxed, conversational male voice."),
    ("en_US-kristin-medium", "Kristin (US English)", 63.2, "female", "Expressive female audiobook-style voice."),
    ("en_GB-alba-medium", "Alba (British English)", 63.2, "female", "Gentle Scottish-accented female voice."),
    ("en_GB-northern_english_male-medium", "Northern English Male (British English)", 63.2, "male",
     "Steady male voice with a Northern English accent."),
    ("en_GB-cori-high", "Cori (British English, high quality)", 114.2, "female",
     "Polished female British voice; larger download."),
    ("es_ES-davefx-medium", "Davefx (Spanish, Spain)", 63.2, "male", "Clear male voice for Spanish from Spain."),
    ("es_MX-ald-medium", "Ald (Spanish, Mexico)", 63.2, "male", "Natural male voice for Latin American Spanish."),
]  # fmt: skip

CATALOG: list[CatalogVoice] = [
    CatalogVoice(
        id=voice_id,
        name=name,
        language=voice_id.split("-", 1)[0].replace("_", "-"),
        quality=voice_id.rsplit("-", 1)[1],
        size_mb=size_mb,
        description=description,
        gender=gender,
        recommended=voice_id == RECOMMENDED_VOICE,
    )
    for voice_id, name, size_mb, gender, description in _ENTRIES
]

_BY_ID = {v.id: v for v in CATALOG}
_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def catalog_voice(voice_id: str) -> CatalogVoice | None:
    """The curated entry for ``voice_id``, if it is in the catalog."""
    return _BY_ID.get(voice_id)


def is_valid_voice_id(voice_id: str) -> bool:
    return bool(VOICE_ID_RE.match(voice_id or ""))


def piper_voices_dir(voices_dir: Path | None = None) -> Path:
    """Folder holding the Piper ``.onnx`` voices (created on first use)."""
    path = Path(voices_dir) if voices_dir is not None else config.voices_dir() / "piper"
    path.mkdir(parents=True, exist_ok=True)
    return path


def installed_piper_voices(voices_dir: Path | None = None) -> list[Path]:
    """Installed voice models: non-empty ``*.onnx`` files with a matching ``.onnx.json``."""
    folder = piper_voices_dir(voices_dir)
    found: list[Path] = []
    for model in sorted(folder.glob("*.onnx")):
        if model.name.startswith("."):
            continue
        cfg = model.with_name(model.name + ".json")
        try:
            if model.is_file() and cfg.is_file() and model.stat().st_size > 0 and cfg.stat().st_size > 0:
                found.append(model)
        except OSError:
            continue
    return found


def voice_urls(voice_id: str, base_url: str | None = None) -> tuple[str, str]:
    """(model URL, config URL), using the same layout as piper's own downloader."""
    match = VOICE_ID_RE.match(voice_id)
    if not match:
        raise ValueError(f"'{voice_id}' isn't a valid voice name (expected something like {RECOMMENDED_VOICE}).")
    family, region = match.group("family"), match.group("region")
    name, quality = match.group("name"), match.group("quality")
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    lang_code = f"{family}_{region}"
    stem = f"{base}/{family}/{lang_code}/{name}/{quality}/{voice_id}"
    return f"{stem}.onnx?download=true", f"{stem}.onnx.json?download=true"


def download_piper_voice(
    voice_id: str,
    *,
    voices_dir: Path | None = None,
    progress: Callable[[float, str], None] | None = None,
    base_url: str | None = None,
    cancel: threading.Event | None = None,
    force: bool = False,
) -> Path:
    """Download a Piper voice and return the path of its ``.onnx`` model.

    Already-installed voices are returned immediately unless ``force`` is set.
    Raises ``ValueError`` for an invalid id, :class:`VoiceDownloadError` with a
    friendly message on failure, and :class:`DownloadCancelled` if ``cancel`` is set.
    """
    voice_id = (voice_id or "").strip()
    model_url, config_url = voice_urls(voice_id, base_url)
    folder = piper_voices_dir(voices_dir)
    model_path = folder / f"{voice_id}.onnx"
    config_path = folder / f"{voice_id}.onnx.json"
    entry = catalog_voice(voice_id)
    label = entry.name if entry else voice_id

    with _lock_for(voice_id):
        if not force and model_path in installed_piper_voices(folder):
            _report(progress, 1.0, f"{label} is already downloaded")
            return model_path

        temps: list[Path] = []
        try:
            _report(progress, 0.0, f"Starting download of {label}")
            config_tmp = _fetch(config_url, folder, temps, cancel=cancel)
            try:
                parsed = json.loads(config_tmp.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, ValueError):
                parsed = None
            if not isinstance(parsed, dict):
                raise VoiceDownloadError("The downloaded voice settings file was damaged. Please try again.")

            expected = int(entry.size_mb * 1_000_000) if entry else None

            def on_bytes(done: int, total: int | None) -> None:
                size = total or expected
                frac = min(done / size, 0.99) if size else 0.0
                total_text = f" of {size / 1_000_000:.1f} MB" if size else " MB"
                _report(progress, 0.01 + 0.98 * frac, f"Downloading {label}: {done / 1_000_000:.1f}{total_text}")

            model_tmp = _fetch(model_url, folder, temps, cancel=cancel, on_bytes=on_bytes)
            if model_tmp.stat().st_size == 0:
                raise VoiceDownloadError("The downloaded voice file was empty. Please try again.")

            # The .json goes last: a voice counts as installed once both files exist.
            os.replace(model_tmp, model_path)
            os.replace(config_tmp, config_path)
        except DownloadCancelled:
            _report(progress, 0.0, "Download cancelled")
            raise
        except OSError as exc:
            if isinstance(exc, (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError)):
                raise VoiceDownloadError(_NETWORK_ERROR) from exc
            raise VoiceDownloadError(f"Couldn't save the voice on this computer: {exc.strerror or exc}") from exc
        finally:
            for tmp in temps:
                _unlink(tmp)

    _report(progress, 1.0, f"Downloaded {label}")
    return model_path


def _fetch(
    url: str,
    folder: Path,
    temps: list[Path],
    *,
    cancel: threading.Event | None,
    on_bytes: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Stream ``url`` into a new temp file in ``folder`` (appended to ``temps``)."""
    fd, name = tempfile.mkstemp(dir=folder, prefix=".download-", suffix=".part")
    tmp = Path(name)
    temps.append(tmp)
    request = urllib.request.Request(url, headers={"User-Agent": "DestinyBot voice downloader"})
    with os.fdopen(fd, "wb") as out:
        try:
            response = urllib.request.urlopen(request, timeout=_TIMEOUT_S)  # noqa: S310 (https / test URL only)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise VoiceDownloadError(
                    "Couldn't download the voice: the download server doesn't have it (error 404)."
                ) from exc
            raise VoiceDownloadError(
                f"Couldn't download the voice: the server answered with error {exc.code}."
            ) from exc
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
            raise VoiceDownloadError(_NETWORK_ERROR) from exc
        with response:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            done = 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled("Download cancelled.")
                try:
                    chunk = response.read(_CHUNK)
                except (OSError, http.client.HTTPException) as exc:
                    raise VoiceDownloadError(_NETWORK_ERROR) from exc
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if on_bytes is not None:
                    on_bytes(done, total)
            if total is not None and done < total:
                raise VoiceDownloadError(_NETWORK_ERROR)
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled("Download cancelled.")
    return tmp


def _lock_for(voice_id: str) -> threading.Lock:
    with _download_locks_guard:
        return _download_locks.setdefault(voice_id, threading.Lock())


def _report(progress: Callable[[float, str], None] | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, fraction)), message)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
