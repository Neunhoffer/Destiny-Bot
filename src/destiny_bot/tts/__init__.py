"""Text-to-speech engines and the registry that picks one.

Engines are created lazily, one instance per name, so importing this package
doesn't load Piper/onnxruntime or touch the network.
"""

from __future__ import annotations

import math
import threading

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
    "get_engine",
    "resolve_engine",
]
