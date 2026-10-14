"""Piper: natural-sounding neural voices that run entirely on this computer."""

from __future__ import annotations

import io
import json
import math
import threading
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import clamp_rate
from .base import AudioClip, EngineUnavailable, TTSEngine, TTSError, Voice
from .catalog import CATALOG, RECOMMENDED_VOICE, catalog_voice, installed_piper_voices, piper_voices_dir

NO_VOICE_REASON = "No natural voice downloaded yet. Open Settings → Voices and download one (about 60 MB)."
MAX_LOADED_VOICES = 2  # each loaded voice holds ~60-120 MB of RAM
# Piper's length_scale: < 1 is faster. Beyond this range speech turns garbled or droning.
MIN_LENGTH_SCALE = 0.3
MAX_LENGTH_SCALE = 2.5
_CATALOG_ORDER = {v.id: i for i, v in enumerate(CATALOG)}


@dataclass
class _Loaded:
    voice: Any  # piper.PiperVoice
    lock: threading.Lock  # one synthesis at a time per voice
    stamp: tuple[int, int]  # (mtime_ns, size) of the model when it was loaded


class PiperEngine(TTSEngine):
    name = "piper"
    label = "Piper (natural, offline)"
    online = False

    def __init__(self, voices_dir: Path | None = None) -> None:
        self._voices_dir = voices_dir
        self._lock = threading.Lock()
        self._loaded: OrderedDict[str, _Loaded] = OrderedDict()
        self._load_locks: dict[str, threading.Lock] = {}
        self._import_error: str | None = None
        self._imported = False
        self._meta_cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}

    # -- TTSEngine

    def is_available(self) -> tuple[bool, str]:
        problem = self._piper_problem()
        if problem:
            return False, problem
        if not self._installed():
            return False, NO_VOICE_REASON
        return True, ""

    def list_voices(self) -> list[Voice]:
        voices = [self._voice_info(path) for path in self._installed()]
        voices.sort(key=lambda v: (_CATALOG_ORDER.get(v.id, len(_CATALOG_ORDER)), v.name.lower()))
        return voices

    def default_voice(self) -> str | None:
        ids = [path.stem for path in self._installed()]
        if RECOMMENDED_VOICE in ids:
            return RECOMMENDED_VOICE
        voices = self.list_voices()
        english = [v.id for v in voices if (v.language or "").startswith("en")]
        return (english or [v.id for v in voices] or [None])[0]

    def synthesize(self, text: str, voice: str | None = None, rate: float = 1.0) -> AudioClip:
        if not text or not text.strip():
            raise TTSError("There's no text to read aloud.")
        problem = self._piper_problem()
        if problem:
            raise EngineUnavailable(problem)
        voice_id = voice or self.default_voice()
        if voice_id is None:
            raise EngineUnavailable(NO_VOICE_REASON)
        entry = self._get_voice(voice_id)
        length_scale = length_scale_for(entry.voice, rate)
        try:
            with entry.lock:
                return self._run(entry.voice, text, length_scale)
        except TTSError:
            raise
        except Exception as exc:  # onnxruntime / phonemizer failures
            raise TTSError(f"The natural voice couldn't read this text ({type(exc).__name__}: {exc}).") from exc

    # -- voices on disk

    def voices_dir(self) -> Path:
        return piper_voices_dir(self._voices_dir)

    def _installed(self) -> list[Path]:
        return installed_piper_voices(self.voices_dir())

    def _model_path(self, voice_id: str) -> Path:
        for path in self._installed():
            if path.stem == voice_id:
                return path
        raise EngineUnavailable(
            f"The natural voice '{voice_id}' isn't downloaded. "
            "Open Settings → Voices to download it or pick another voice."
        )

    def _meta(self, path: Path) -> dict[str, Any]:
        """The voice's .onnx.json (cached until the file changes)."""
        cfg = path.with_name(path.name + ".json")
        try:
            st = cfg.stat()
            stamp = (st.st_mtime_ns, st.st_size)
            cached = self._meta_cache.get(cfg)
            if cached and cached[0] == stamp:
                return cached[1]
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        data = data if isinstance(data, dict) else {}
        # Only keep the small fields we need; phoneme maps are large.
        slim = {k: data.get(k) for k in ("language", "espeak", "audio", "inference", "dataset", "quality")}
        self._meta_cache[cfg] = (stamp, slim)
        return slim

    def _voice_info(self, path: Path) -> Voice:
        voice_id = path.stem
        meta = self._meta(path)
        entry = catalog_voice(voice_id)
        language = _language(voice_id, meta)
        try:
            size_mb = round(path.stat().st_size / 1_000_000, 1)
        except OSError:
            size_mb = None
        if entry is not None:
            name, gender = entry.name, entry.gender
        else:
            name, gender = _derived_name(voice_id, meta, language), None
        return Voice(id=voice_id, name=name, engine=self.name, language=language, gender=gender, size_mb=size_mb)

    # -- loading

    def _piper_problem(self) -> str | None:
        if not self._imported:
            try:
                import piper  # noqa: F401
            except ImportError:
                self._import_error = (
                    "Natural voices need the piper-tts package, which isn't installed. "
                    "Reinstall Destiny Bot with natural voices included."
                )
            except Exception as exc:  # e.g. onnxruntime failing to load its DLLs
                self._import_error = f"The natural voice engine couldn't start: {exc}"
            self._imported = True
        return self._import_error

    def _get_voice(self, voice_id: str) -> _Loaded:
        path = self._model_path(voice_id)
        stamp = _stamp(path)
        with self._lock:
            entry = self._loaded.get(voice_id)
            if entry is not None and entry.stamp == stamp:
                self._loaded.move_to_end(voice_id)
                return entry
            load_lock = self._load_locks.setdefault(voice_id, threading.Lock())
        with load_lock:  # one thread loads; others asking for the same voice wait
            with self._lock:
                entry = self._loaded.get(voice_id)
                if entry is not None and entry.stamp == stamp:
                    self._loaded.move_to_end(voice_id)
                    return entry
            loaded = _Loaded(self._load(path), threading.Lock(), stamp)
            with self._lock:
                self._loaded[voice_id] = loaded
                self._loaded.move_to_end(voice_id)
                while len(self._loaded) > MAX_LOADED_VOICES:
                    self._loaded.popitem(last=False)
        return loaded

    def _load(self, path: Path) -> Any:
        from piper import PiperVoice

        config_path = path.with_name(path.name + ".json")
        try:
            try:
                return PiperVoice.load(str(path), config_path=str(config_path), download_dir=str(path.parent))
            except TypeError:  # older piper-tts without download_dir
                return PiperVoice.load(str(path), config_path=str(config_path))
        except Exception as exc:
            raise EngineUnavailable(
                f"The natural voice '{path.stem}' couldn't be loaded; it may be damaged. "
                "Delete it and download it again in Settings → Voices."
            ) from exc

    def _run(self, voice: Any, text: str, length_scale: float) -> AudioClip:
        sample_rate = int(voice.config.sample_rate)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            if hasattr(voice, "synthesize_wav"):  # piper-tts >= 1.3
                from piper import SynthesisConfig

                voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=length_scale))
            else:  # piper-tts 1.2
                voice.synthesize(text, wav, length_scale=length_scale)
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            if frames == 0:  # nothing speakable (e.g. only punctuation): a short pause
                wav.writeframes(b"\x00\x00" * (sample_rate // 5))
        return AudioClip(buf.getvalue(), "audio/wav", sample_rate)


def length_scale_for(voice: Any, rate: float) -> float:
    """Piper length_scale for a speed multiplier: the voice's own scale / rate, clamped."""
    try:
        base = float(getattr(voice.config, "length_scale", None) or 1.0)
    except (TypeError, ValueError):
        base = 1.0
    if not math.isfinite(base) or base <= 0:
        base = 1.0
    scale = base / clamp_rate(rate)
    return round(min(max(scale, MIN_LENGTH_SCALE), MAX_LENGTH_SCALE), 4)


def _stamp(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return 0, 0


def _bcp47(code: str) -> str:
    parts = code.replace("_", "-").split("-")
    if len(parts) >= 2 and len(parts[1]) == 2:
        parts[1] = parts[1].upper()
    parts[0] = parts[0].lower()
    return "-".join(p for p in parts if p)


def _language(voice_id: str, meta: dict[str, Any]) -> str | None:
    lang = meta.get("language")
    if isinstance(lang, dict) and isinstance(lang.get("code"), str) and lang["code"]:
        return _bcp47(lang["code"])
    prefix = voice_id.split("-", 1)[0]
    if "_" in prefix and len(prefix) <= 7:
        return _bcp47(prefix)
    espeak = meta.get("espeak")
    if isinstance(espeak, dict) and isinstance(espeak.get("voice"), str) and espeak["voice"]:
        return _bcp47(espeak["voice"])
    return None


def _derived_name(voice_id: str, meta: dict[str, Any], language: str | None) -> str:
    """'en_US-foo_bar-high' -> 'Foo Bar (en-US, high quality)'."""
    parts = voice_id.split("-")
    if len(parts) >= 3:
        speaker, quality = "-".join(parts[1:-1]), parts[-1]
    else:
        speaker, quality = (meta.get("dataset") or voice_id), meta.get("quality")
    speaker = str(speaker).replace("_", " ").strip().title() or voice_id
    details = [d for d in (language, f"{str(quality).replace('_', ' ')} quality" if quality else None) if d]
    return f"{speaker} ({', '.join(details)})" if details else speaker
