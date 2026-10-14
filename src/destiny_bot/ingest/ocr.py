"""Optical character recognition (OCR) with Tesseract, for scanned pages and photos.

Tesseract is a separate free program; ``pytesseract`` and Pillow are its Python
helpers. All three are optional: when any is missing, :func:`ocr_available`
reports why and how to install it, and :func:`ocr_image_bytes` raises
:class:`IngestError` with the same friendly instructions. Importing this module
never fails.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import IngestError

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

# How long a failed check is remembered. A short time lets the user install
# Tesseract and try again without restarting Destiny Bot.
_NEGATIVE_TTL_SECONDS = 30.0
# Seconds Tesseract may spend on one image before we give up.
_OCR_TIMEOUT_SECONDS = 300
# Tesseract reads best when letters are roughly 20-40 px tall: small images
# (screenshots, cropped photos) are enlarged, huge ones are shrunk for speed.
_MIN_LONG_SIDE = 1600
_MAX_UPSCALE = 3.0
_MAX_LONG_SIDE = 6000

_lock = threading.Lock()
_status: tuple[bool, str] | None = None
_status_time = 0.0


def install_instructions() -> str:
    """How to install the Tesseract program on this operating system."""
    if sys.platform == "win32":
        return (
            "To read scanned pages and photos, install the free Tesseract OCR program for Windows: "
            "download the installer from UB Mannheim (https://github.com/UB-Mannheim/tesseract/wiki), "
            "run it with the default options, then try again."
        )
    if sys.platform == "darwin":
        return (
            "To read scanned pages and photos, install the free Tesseract OCR program: open Terminal and "
            "run 'brew install tesseract' (get Homebrew from https://brew.sh if you don't have it), "
            "then try again."
        )
    return (
        "To read scanned pages and photos, install the free Tesseract OCR program with your package "
        "manager, for example 'sudo apt install tesseract-ocr' (Ubuntu, Debian) or "
        "'sudo dnf install tesseract' (Fedora), then try again."
    )


def _candidate_paths() -> list[Path]:
    """Usual install locations, for when ``tesseract`` is not on PATH."""
    if sys.platform == "win32":
        roots = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramW6432"),
            os.environ.get("ProgramFiles(x86)"),
            r"C:\Program Files",
            r"C:\Program Files (x86)",
        ]
        paths = [Path(root) / "Tesseract-OCR" / "tesseract.exe" for root in roots if root]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            paths.append(Path(local) / "Programs" / "Tesseract-OCR" / "tesseract.exe")
            paths.append(Path(local) / "Tesseract-OCR" / "tesseract.exe")
        return paths
    if sys.platform == "darwin":
        # Apps started from Finder may not see Homebrew / MacPorts on PATH.
        return [Path("/opt/homebrew/bin/tesseract"), Path("/usr/local/bin/tesseract"), Path("/opt/local/bin/tesseract")]
    return [Path("/usr/bin/tesseract"), Path("/usr/local/bin/tesseract"), Path("/snap/bin/tesseract")]


def _locate_tesseract(pytesseract_module) -> None:
    """Point pytesseract at an installed tesseract binary that is not on PATH."""
    current = pytesseract_module.pytesseract.tesseract_cmd
    if current != "tesseract" or shutil.which(current):
        return  # a custom path was configured, or PATH already works
    for candidate in _candidate_paths():
        if candidate.is_file():
            pytesseract_module.pytesseract.tesseract_cmd = str(candidate)
            return


def _check() -> tuple[bool, str]:
    try:
        import pytesseract
    except ImportError:
        return False, (
            "The OCR helper isn't installed (run 'pip install pytesseract pillow'). " + install_instructions()
        )
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False, "The Pillow image library isn't installed (run 'pip install pillow'). " + install_instructions()

    _locate_tesseract(pytesseract)
    try:
        version = pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        return False, "Tesseract OCR isn't installed. " + install_instructions()
    except Exception as exc:  # broken install, permission problem, ...
        return False, f"Tesseract OCR is installed but doesn't work ({exc}). " + install_instructions()

    try:
        languages = set(pytesseract.get_languages(config=""))
    except Exception:
        languages = set()
    if languages and "eng" not in languages:
        return False, (
            "Tesseract OCR is installed but its English language data is missing. "
            "Reinstall Tesseract and include the English language pack."
        )
    return True, f"Tesseract {version} is installed."


def ocr_available() -> tuple[bool, str]:
    """``(usable, reason)``. The reason explains how to install Tesseract when it's missing."""
    global _status, _status_time
    with _lock:
        now = time.monotonic()
        if _status is not None and (_status[0] or now - _status_time < _NEGATIVE_TTL_SECONDS):
            return _status
        _status, _status_time = _check(), now
        return _status


def reset_ocr_cache() -> None:
    """Forget the last availability check (e.g. after the user installs Tesseract)."""
    global _status
    with _lock:
        _status = None


def _tesseract_environment() -> dict[str, str]:
    env = dict(os.environ)
    # Tesseract's OpenMP worker threads spin while waiting; when the computer is busy
    # (e.g. a voice is synthesizing) one thread is many times faster than several.
    env.setdefault("OMP_THREAD_LIMIT", "1")
    return env


def _require() -> None:
    ok, reason = ocr_available()
    if not ok:
        raise IngestError(reason)


def _prepare(image: Image) -> Image:
    """Upright, grayscale, sensibly sized copy of ``image`` for Tesseract."""
    from PIL import Image as PILImage
    from PIL import ImageOps

    try:
        image = ImageOps.exif_transpose(image) or image  # phone photos are often stored sideways
    except Exception:
        pass
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        # Transparent pixels are usually black underneath; put the image on white paper.
        rgba = image.convert("RGBA")
        paper = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
        paper.alpha_composite(rgba)
        image = paper
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        image = image.convert("I").point(lambda v: v / 256).convert("L")  # 16-bit scans
    else:
        image = image.convert("L")

    long_side = max(image.size)
    if long_side and long_side < _MIN_LONG_SIDE:
        factor = min(_MAX_UPSCALE, _MIN_LONG_SIDE / long_side)
        if factor > 1.05:
            size = (round(image.width * factor), round(image.height * factor))
            image = image.resize(size, PILImage.Resampling.LANCZOS)
    elif long_side > _MAX_LONG_SIDE:
        factor = _MAX_LONG_SIDE / long_side
        size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        image = image.resize(size, PILImage.Resampling.LANCZOS)
    return ImageOps.autocontrast(image)


def ocr_image(image: Image, *, lang: str = "eng") -> str:
    """Recognise the text in a Pillow image. Raises :class:`IngestError`."""
    _require()
    import pytesseract

    if image.width < 3 or image.height < 3:
        return ""
    prepared = _prepare(image)
    if hasattr(pytesseract.pytesseract, "environ"):  # the environment pytesseract starts tesseract with
        pytesseract.pytesseract.environ = _tesseract_environment()
    try:
        text = pytesseract.image_to_string(prepared, lang=lang, config="--psm 3", timeout=_OCR_TIMEOUT_SECONDS)
    except pytesseract.TesseractNotFoundError as exc:
        reset_ocr_cache()
        raise IngestError("Tesseract OCR isn't installed. " + install_instructions()) from exc
    except pytesseract.TesseractError as exc:
        message = str(getattr(exc, "message", "") or exc)
        if "too small" in message.lower():
            return ""
        if "Failed loading language" in message or "Error opening data file" in message:
            raise IngestError(
                f"Tesseract OCR doesn't have the '{lang}' language installed. "
                "Reinstall Tesseract and include that language pack."
            ) from exc
        raise IngestError(f"Tesseract OCR couldn't read this image ({message.strip() or 'unknown error'}).") from exc
    except RuntimeError as exc:  # pytesseract reports its timeout as a bare RuntimeError
        if "timeout" in str(exc).lower():
            raise IngestError(
                "Reading the text in this image took too long. Try a smaller or clearer picture."
            ) from exc
        raise
    return text.replace("\f", "\n").replace("\r\n", "\n").strip()


def ocr_image_bytes(data: bytes, *, lang: str = "eng") -> str:
    """Recognise the text in an encoded image (PNG, JPEG, TIFF, ...).

    Only the first frame of a multi-page image is read. Raises :class:`IngestError`
    when OCR isn't available or the image can't be opened.
    """
    _require()
    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError

    try:
        with PILImage.open(io.BytesIO(data)) as image:
            image.load()
            return ocr_image(image, lang=lang)
    except UnidentifiedImageError as exc:
        raise IngestError("This picture couldn't be opened. It may be damaged or in an unsupported format.") from exc
    except PILImage.DecompressionBombError as exc:
        raise IngestError("This picture is too large to read. Try a smaller copy of it.") from exc
    except OSError as exc:
        raise IngestError(f"This picture couldn't be read ({exc}). It may be damaged or incomplete.") from exc
