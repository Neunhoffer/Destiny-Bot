"""Text-to-speech engines and the registry that picks one.

Engines are created lazily, one instance per name, so importing this package
doesn't load Piper/onnxruntime or touch the network.
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

from .base import AudioClip, EngineUnavailable, TTSEngine, TTSError, Voice

ENGINE_NAMES = ("kokoro", "piper", "system", "edge", "browser")
# What "auto" tries, best first; an engine counts only when it is installed and usable
# (Kokoro needs its voice pack, Piper a downloaded voice). Online engines are never picked automatically.
AUTO_ORDER = ("kokoro", "piper", "system", "browser")

_engines: dict[str, TTSEngine] = {}
_lock = threading.Lock()


def clamp_rate(rate: float | None, low: float = 0.25, high: float = 4.0) -> float:
    """A usable speed multiplier: 1.0 for missing/invalid values, else clamped to [low, high]."""
    try:
        value = float(rate) if rate is not None else 1.0
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 0:
        return 1.0
    return min(max(value, low), high)


def disable_onnx_telemetry() -> None:
    """Turn off onnxruntime's telemetry events (on by default in Windows builds).

    Call before creating an inference session; it is process-wide and harmless to repeat.
    """
    try:
        import onnxruntime

        onnxruntime.disable_telemetry_events()
    except Exception:  # not installed, or an onnxruntime without this switch
        pass


def engine_start_problem(label: str, exc: BaseException) -> str:
    """Why a voice engine's packages failed to load (``label`` e.g. "The Kokoro voice engine")."""
    reason = f"{label} couldn't start: {exc}"
    if "DLL load failed" in str(exc):  # Windows, usually onnxruntime without the Visual C++ runtime
        reason += ". Installing Microsoft's Visual C++ Redistributable (x64) usually fixes this."
    return reason


def espeak_path_reason(path: Path | str) -> str:
    """Why Piper/Kokoro can't start when :func:`espeak_data_path` gives None."""
    example = "C:\\Athena" if sys.platform == "win32" else "~/Athena"
    return (
        f"The natural voices can't start from the folder Athena is installed in ({path}): its name is too long "
        f"or has letters they can't read. Reinstall Athena in a folder with a short, plain name, such as {example}."
    )


def espeak_data_path(path: Path | str) -> str | None:
    """``path`` spelled so espeak-ng can open it, or None when it can't be.

    Piper and Kokoro find phonemes with espeak-ng, which keeps its data folder in a
    fixed buffer (230 bytes on Windows, 160 elsewhere, in espeak-ng 1.52) and, on
    Windows, opens files through the C library, which reads paths in the ANSI code page
    while Piper and phonemizer pass UTF-8. A folder that is too long, or one such as
    C:\\Users\\Zoë\\..., is then "missing", and espeak-ng ends the whole program (exit 1)
    instead of raising an error. Windows' short 8.3 names (C:\\Users\\ZO~1\\...) are
    plain ASCII and shorter, so they are used when needed.
    """
    text = str(path)
    windows = sys.platform == "win32"
    limit = 229 if windows else 159

    # The packaged Windows app runs with a UTF-8 code page (see packaging/athena.spec), so
    # espeak-ng reads non-ASCII paths there; elsewhere on Windows they need the 8.3 name.
    readable_non_ascii = not windows or _ansi_code_page() == 65001

    def usable(candidate: str) -> bool:
        return len(candidate.encode("utf-8")) <= limit and (candidate.isascii() or readable_non_ascii)

    if usable(text):
        return text
    short = _short_path_name(text) if windows else None
    return short if short and usable(short) else None


def _ansi_code_page() -> int | None:
    """This process's ANSI code page on Windows (65001 means UTF-8), else None."""
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetACP())  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return None


def _short_path_name(path: str) -> str | None:
    """The 8.3 short form of an existing ``path`` (Windows), or None."""
    try:
        import ctypes
        from ctypes import wintypes

        get = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
        get.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get.restype = wintypes.DWORD
        size = get(path, None, 0)
        if not size:
            return None
        buf = ctypes.create_unicode_buffer(size)
        return buf.value if 0 < get(path, buf, size) < size else None
    except (AttributeError, OSError, ValueError):
        return None


def _create(name: str) -> TTSEngine:
    if name == "kokoro":
        from .kokoro_engine import KokoroEngine

        return KokoroEngine()
    if name == "piper":
        from .piper_engine import PiperEngine

        return PiperEngine()
    if name == "system":
        from .system_engine import SystemEngine

        return SystemEngine()
    if name == "edge":
        from .edge_engine import EdgeEngine

        return EdgeEngine()
    from .browser_engine import BrowserEngine

    return BrowserEngine()


def get_engine(name: str) -> TTSEngine:
    """The shared engine called ``name``. Raises ``KeyError`` for unknown names."""
    if name not in ENGINE_NAMES:
        raise KeyError(name)
    with _lock:
        engine = _engines.get(name)
        if engine is None:
            engine = _engines[name] = _create(name)
        return engine


def all_engines() -> list[TTSEngine]:
    return [get_engine(name) for name in ENGINE_NAMES]


def _usable(engine: TTSEngine, online_allowed: bool) -> bool:
    if engine.online and not online_allowed:
        return False
    try:
        return engine.is_available()[0]
    except Exception:  # a broken engine must not stop the app from choosing another
        return False


def resolve_engine(preferred: str, *, online_allowed: bool) -> TTSEngine:
    """The engine to use for ``preferred`` ("auto" or an engine name).

    A requested engine that is unknown, unavailable, or online while online voices
    are off falls back to "auto": the first available of kokoro (with its voice
    pack), piper (with a voice installed), system, browser.
    """
    if preferred and preferred != "auto" and preferred in ENGINE_NAMES:
        engine = get_engine(preferred)
        if _usable(engine, online_allowed):
            return engine
    for name in AUTO_ORDER:
        engine = get_engine(name)
        if _usable(engine, online_allowed):
            return engine
    return get_engine("browser")


__all__ = [
    "AUTO_ORDER",
    "ENGINE_NAMES",
    "AudioClip",
    "EngineUnavailable",
    "TTSEngine",
    "TTSError",
    "Voice",
    "all_engines",
    "clamp_rate",
    "disable_onnx_telemetry",
    "engine_start_problem",
    "espeak_data_path",
    "espeak_path_reason",
    "get_engine",
    "resolve_engine",
]
