"""Text-to-speech engines and the registry that picks one.

Engines are created lazily, one instance per name, so importing this package
doesn't load Piper/onnxruntime or touch the network.
"""

from __future__ import annotations

import threading

from .base import AudioClip, EngineUnavailable, TTSEngine, TTSError, Voice

ENGINE_NAMES = ("piper", "system", "edge", "browser")
# What "auto" tries, best first. Online engines are never picked automatically.
AUTO_ORDER = ("piper", "system", "browser")

_engines: dict[str, TTSEngine] = {}
_lock = threading.Lock()


def _create(name: str) -> TTSEngine:
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
    are off falls back to "auto": the first available of piper (with a voice
    installed), system, browser.
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
    "get_engine",
    "resolve_engine",
]
