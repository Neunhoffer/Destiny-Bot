"""Curated downloadable voices (Piper voices and the Kokoro voice pack) and a safe downloader.

Piper voices come from the official ``rhasspy/piper-voices`` repository on Hugging
Face and are saved as ``<voice_id>.onnx`` + ``<voice_id>.onnx.json`` pairs in
``config.voices_dir() / "piper"``. The Kokoro model and its voice table come from
the kokoro-onnx GitHub release and go to ``config.voices_dir() / "kokoro"``.

Voices can also be bundled read-only with the desktop installer: every folder in
``DESTINY_EXTRA_VOICE_DIRS`` (separated by ``os.pathsep``) may hold ``piper/`` and
``kokoro/`` subfolders. Downloaded voices win over bundled ones with the same id.

Downloading is the only network access in this module and only happens when the
user asks for a voice.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .. import config

DEFAULT_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
KOKORO_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
RECOMMENDED_VOICE = "en_US-lessac-medium"
KOKORO_CATALOG_ID = "kokoro-v1.0"
EXTRA_DIRS_ENV = "DESTINY_EXTRA_VOICE_DIRS"

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
    id: str  # e.g. "en_US-lessac-medium", or "kokoro-v1.0" for the Kokoro voice pack
    name: str  # e.g. "Lessac (US English)"
    language: str  # BCP-47-ish, e.g. "en-US"
    quality: str  # "x_low" | "low" | "medium" | "high"
    size_mb: float  # approximate download size
    description: str
    gender: str | None = None
    recommended: bool = False
    engine: str = "piper"  # the engine that uses this download: "piper" or "kokoro"


@dataclass(frozen=True)
class KokoroFile:
    name: str
    size: int  # bytes
    sha256: str


# Files of the kokoro-onnx "model-files-v1.0" release. Downloads are checked against these hashes.
KOKORO_FILES: dict[str, KokoroFile] = {
    "voices": KokoroFile(
        "voices-v1.0.bin", 28_214_398, "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
    ),
    "int8": KokoroFile(
        "kokoro-v1.0.int8.onnx", 92_361_271, "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb"
    ),
    "fp32": KokoroFile(
        "kokoro-v1.0.onnx", 325_532_387, "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
    ),
}
KOKORO_VARIANTS = ("int8", "fp32")
# int8 sounds the same as fp32 (same noise floor, band energies within ~1 dB) at a
# quarter of the download, and runs about as fast on CPUs.
DEFAULT_KOKORO_VARIANT = "int8"


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

KOKORO_CATALOG_VOICE = CatalogVoice(
    id=KOKORO_CATALOG_ID,
    name="Kokoro voice pack (most natural)",
    language="en",
    quality="high",
    size_mb=round((KOKORO_FILES["voices"].size + KOKORO_FILES[DEFAULT_KOKORO_VARIANT].size) / 1_000_000, 1),
    description=(
        "The most natural-sounding offline voices: 13 US and British English voices in one download. "
        "Needs a bit more computer power than the other voices."
    ),
    recommended=True,
    engine="kokoro",
)

CATALOG: list[CatalogVoice] = [
    KOKORO_CATALOG_VOICE,
    *(
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
    ),
]

_BY_ID = {v.id: v for v in CATALOG}
_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def catalog_voice(voice_id: str) -> CatalogVoice | None:
    """The curated entry for ``voice_id``, if it is in the catalog."""
    return _BY_ID.get(voice_id)


def is_valid_voice_id(voice_id: str) -> bool:
    """True for Piper voice ids (en_US-lessac-medium) and the Kokoro voice pack id."""
    return voice_id == KOKORO_CATALOG_ID or bool(VOICE_ID_RE.match(voice_id or ""))


def extra_voice_dirs() -> list[Path]:
    """Read-only voice folders bundled with the app (``DESTINY_EXTRA_VOICE_DIRS``)."""
    raw = os.environ.get(EXTRA_DIRS_ENV, "")
    return [Path(part.strip()).expanduser() for part in raw.split(os.pathsep) if part.strip()]


def piper_voices_dir(voices_dir: Path | None = None) -> Path:
    """Folder holding downloaded Piper ``.onnx`` voices (created on first use)."""
    path = Path(voices_dir) if voices_dir is not None else config.voices_dir() / "piper"
    path.mkdir(parents=True, exist_ok=True)
    return path


def kokoro_dir(voices_dir: Path | None = None) -> Path:
    """Folder holding the downloaded Kokoro model and voice table (created on first use)."""
    path = Path(voices_dir) if voices_dir is not None else config.voices_dir() / "kokoro"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def installed_piper_voices(voices_dir: Path | None = None) -> list[Path]:
    """Installed voice models: non-empty ``*.onnx`` files with a matching ``.onnx.json``.

    With no ``voices_dir`` this is the downloaded voices followed by bundled ones
    (``DESTINY_EXTRA_VOICE_DIRS``) whose id wasn't downloaded too.
    """
    if voices_dir is not None:
        folders = [piper_voices_dir(voices_dir)]
    else:
        folders = [piper_voices_dir(), *(extra / "piper" for extra in extra_voice_dirs())]
    found: list[Path] = []
    seen: set[str] = set()
    for folder in folders:
        try:
            models = sorted(folder.glob("*.onnx"))
        except OSError:
            continue
        for model in models:
            if model.name.startswith(".") or model.stem in seen:
                continue
            if _nonempty(model) and _nonempty(model.with_name(model.name + ".json")):
                found.append(model)
                seen.add(model.stem)
    return found


def find_kokoro_model(voices_dir: Path | None = None, variant: str | None = None) -> tuple[Path, Path] | None:
    """``(model, voices table)`` of an installed Kokoro model, or None.

    Downloaded files win over bundled ones; within a folder ``variant`` (or the
    default variant, then any other) is preferred.
    """
    if voices_dir is not None:
        folders = [Path(voices_dir)]
    else:
        folders = [config.voices_dir() / "kokoro", *(extra / "kokoro" for extra in extra_voice_dirs())]
    others = [v for v in KOKORO_VARIANTS if v != DEFAULT_KOKORO_VARIANT]
    order = [variant] if variant else [DEFAULT_KOKORO_VARIANT, *others]
    for folder in folders:
        voices = folder / KOKORO_FILES["voices"].name
        if not _nonempty(voices):
            continue
        for name in order:
            model = folder / KOKORO_FILES[name].name
            if _nonempty(model):
                return model, voices
    return None


def installed_catalog_ids() -> set[str]:
    """Ids of catalog entries that are installed (downloaded or bundled)."""
    ids = {path.stem for path in installed_piper_voices()}
    if find_kokoro_model() is not None:
        ids.add(KOKORO_CATALOG_ID)
    return ids


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

    Already-installed voices (downloaded or bundled) are returned immediately
    unless ``force`` is set. The catalog's Kokoro entry (``kokoro-v1.0``) is
    handed to :func:`download_kokoro_model` (into the ``kokoro`` folder next to
    ``voices_dir``), so one download flow serves the whole catalog.
    Raises ``ValueError`` for an invalid id, :class:`VoiceDownloadError` with a
    friendly message on failure, and :class:`DownloadCancelled` if ``cancel`` is set.
    """
    voice_id = (voice_id or "").strip()
    if voice_id == KOKORO_CATALOG_ID:
        kokoro_folder = Path(voices_dir).parent / "kokoro" if voices_dir is not None else None
        return download_kokoro_model(
            voices_dir=kokoro_folder, progress=progress, cancel=cancel, base_url=base_url, force=force
        )
    model_url, config_url = voice_urls(voice_id, base_url)
    folder = piper_voices_dir(voices_dir)
    model_path = folder / f"{voice_id}.onnx"
    config_path = folder / f"{voice_id}.onnx.json"
    entry = catalog_voice(voice_id)
    label = entry.name if entry else voice_id

    with _lock_for(voice_id):
        if not force:
            existing = next((p for p in installed_piper_voices(voices_dir) if p.stem == voice_id), None)
            if existing is not None:
                _report(progress, 1.0, f"{label} is already downloaded")
                return existing

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
            _replace(model_tmp, model_path)
            _replace(config_tmp, config_path)
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


def download_kokoro_model(
    *,
    voices_dir: Path | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
    base_url: str | None = None,
    variant: str = DEFAULT_KOKORO_VARIANT,
    force: bool = False,
) -> Path:
    """Download the Kokoro model (``variant`` "int8" or "fp32") and its voice table.

    Returns the model path. Files are checked against known SHA-256 hashes and
    moved into place atomically; an already-installed model returns at once.
    Raises like :func:`download_piper_voice`.
    """
    if variant not in KOKORO_VARIANTS:
        raise ValueError(f"Unknown Kokoro model variant '{variant}' (expected one of {', '.join(KOKORO_VARIANTS)}).")
    label = KOKORO_CATALOG_VOICE.name.split(" (")[0]
    base = (base_url or KOKORO_BASE_URL).rstrip("/")
    with _lock_for(KOKORO_CATALOG_ID):
        if not force:
            existing = find_kokoro_model(voices_dir, variant=variant)
            if existing is not None:
                _report(progress, 1.0, f"{label} is already downloaded")
                return existing[0]
        folder = kokoro_dir(voices_dir)
        wanted = [KOKORO_FILES["voices"], KOKORO_FILES[variant]]
        if not force and _nonempty(folder / wanted[0].name):
            wanted = wanted[1:]  # e.g. adding fp32 next to int8: the voice table is shared
        total = sum(f.size for f in wanted)
        temps: list[Path] = []
        fetched: list[tuple[Path, KokoroFile]] = []
        try:
            _report(progress, 0.0, f"Starting download of {label}")
            before = 0
            for item in wanted:

                def on_bytes(done: int, _total: int | None, before: int = before) -> None:
                    got = before + done
                    message = f"Downloading {label}: {got / 1e6:.1f} of {total / 1e6:.1f} MB"
                    _report(progress, 0.99 * got / total, message)

                digest = hashlib.sha256()
                tmp = _fetch(f"{base}/{item.name}", folder, temps, cancel=cancel, on_bytes=on_bytes, digest=digest)
                if digest.hexdigest() != item.sha256:
                    raise VoiceDownloadError("The downloaded voice file was damaged. Please try again.")
                fetched.append((tmp, item))
                before += item.size
            # The model goes last: Kokoro counts as installed once both files exist.
            for tmp, item in fetched:
                _replace(tmp, folder / item.name)
        except DownloadCancelled:
            _report(progress, 0.0, "Download cancelled")
            raise
        except OSError as exc:
            raise VoiceDownloadError(f"Couldn't save the voice on this computer: {exc.strerror or exc}") from exc
        finally:
            for tmp in temps:
                _unlink(tmp)
    _report(progress, 1.0, f"Downloaded {label}")
    return folder / KOKORO_FILES[variant].name


def download_voice(
    voice_id: str,
    *,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
    base_url: str | None = None,
    force: bool = False,
) -> Path:
    """Download any catalog entry (a Piper voice id or ``kokoro-v1.0``) to the default folders."""
    if voice_id.strip() == KOKORO_CATALOG_ID:
        return download_kokoro_model(progress=progress, cancel=cancel, base_url=base_url, force=force)
    return download_piper_voice(voice_id, progress=progress, cancel=cancel, base_url=base_url, force=force)


def _fetch(
    url: str,
    folder: Path,
    temps: list[Path],
    *,
    cancel: threading.Event | None,
    on_bytes: Callable[[int, int | None], None] | None = None,
    digest: Any = None,
) -> Path:
    """Stream ``url`` into a new temp file in ``folder`` (appended to ``temps``).

    ``digest`` (a hashlib object) is fed the bytes as they arrive.
    """
    fd, name = tempfile.mkstemp(dir=folder, prefix=".download-", suffix=".part")
    tmp = Path(name)
    temps.append(tmp)
    request = urllib.request.Request(url, headers={"User-Agent": "DestinyBot voice downloader"})
    with os.fdopen(fd, "wb") as out:
        try:
            response = urllib.request.urlopen(  # noqa: S310 (https / test URL only)
                request, timeout=_TIMEOUT_S, context=_ssl_context()
            )
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
                if digest is not None:
                    digest.update(chunk)
                done += len(chunk)
                if on_bytes is not None:
                    on_bytes(done, total)
            if total is not None and done < total:
                raise VoiceDownloadError(_NETWORK_ERROR)
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled("Download cancelled.")
    return tmp


def _ssl_context() -> ssl.SSLContext:
    """The computer's trusted certificates, plus certifi's when Python found none.

    Python from python.org on macOS ships without any (until "Install Certificates"
    is run), which made every download fail as if the internet were down.
    """
    context = ssl.create_default_context()
    if not context.cert_store_stats().get("x509_ca"):
        try:
            import certifi

            context.load_verify_locations(cafile=certifi.where())
        except (ImportError, OSError, ssl.SSLError):
            pass
    return context


def _replace(src: Path, dst: Path) -> None:
    """``os.replace`` that waits for a virus scanner to let go of a just-downloaded file (Windows)."""
    attempts = 10 if sys.platform == "win32" else 1
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.1 * 2**attempt, 2.0))


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
