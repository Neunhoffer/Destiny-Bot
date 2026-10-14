"""Kokoro: the most natural-sounding offline voices (kokoro-onnx, an 82M-parameter model).

One model file serves every voice; a small table (``voices-v1.0.bin``) holds the
voice styles. Everything runs on this computer.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

from . import clamp_rate, disable_onnx_telemetry, espeak_data_path, espeak_path_reason
from .base import AudioClip, EngineUnavailable, TTSEngine, TTSError, Voice
from .catalog import find_kokoro_model

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
# Kokoro's own speed control (it refuses values outside this range). It sounds
# natural; beyond it the audio is time-stretched with pitch preserved.
NATIVE_MIN_SPEED = 0.5
NATIVE_MAX_SPEED = 2.0
NO_MODEL_REASON = "Download the Kokoro voice pack in Settings → Voices (about 115 MB)."
NO_PACKAGE_REASON = (
    "Kokoro voices need the kokoro-onnx package, which isn't installed. "
    "Reinstall Destiny Bot with natural voices included."
)

# Best English voices, best first: (id, display name, gender).
CURATED_VOICES: list[tuple[str, str, str]] = [
    ("af_heart", "Heart (US English, female)", "female"),
    ("af_bella", "Bella (US English, female)", "female"),
    ("am_michael", "Michael (US English, male)", "male"),
    ("bf_emma", "Emma (British English, female)", "female"),
    ("bm_george", "George (British English, male)", "male"),
    ("af_nicole", "Nicole (US English, female, soft)", "female"),
    ("am_fenrir", "Fenrir (US English, male)", "male"),
    ("am_puck", "Puck (US English, male)", "male"),
    ("af_aoede", "Aoede (US English, female)", "female"),
    ("af_kore", "Kore (US English, female)", "female"),
    ("af_sarah", "Sarah (US English, female)", "female"),
    ("bf_isabella", "Isabella (British English, female)", "female"),
    ("bm_fable", "Fable (British English, male)", "male"),
]
# Voice id prefix -> (espeak language for phonemes, BCP-47 language).
_LANGUAGES = {
    "a": ("en-us", "en-US"),
    "b": ("en-gb", "en-GB"),
    "e": ("es", "es"),
    "f": ("fr-fr", "fr-FR"),
    "h": ("hi", "hi"),
    "i": ("it", "it"),
    "p": ("pt-br", "pt-BR"),
    "j": ("ja", "ja"),
    "z": ("cmn", "zh"),
}


def physical_cores() -> int:
    """Number of physical CPU cores (hyper-threads don't speed up neural inference)."""
    logical = os.cpu_count() or 2
    try:
        if sys.platform.startswith("linux"):
            cores: set[tuple[str, str]] = set()
            phys = core = ""
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    key, _, value = line.partition(":")
                    key, value = key.strip(), value.strip()
                    if key == "physical id":
                        phys = value
                    elif key == "core id":
                        core = value
                    elif not key and (phys or core):
                        cores.add((phys, core))
                        phys = core = ""
            if phys or core:
                cores.add((phys, core))
            if cores:
                return max(1, min(len(cores), logical))
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=5, check=False
            ).stdout.strip()
            if out.isdigit():
                return max(1, int(out))
        elif sys.platform == "win32":
            count = _windows_physical_cores()
            if count:
                return count
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    # Unknown: assume 2-way hyper-threading on machines with 4+ logical CPUs.
    return max(1, logical // 2 if logical >= 4 else logical)


def _windows_physical_cores() -> int | None:
    import ctypes

    class _Info(ctypes.Structure):  # SYSTEM_LOGICAL_PROCESSOR_INFORMATION
        _fields_ = [
            ("mask", ctypes.c_size_t),
            ("relationship", ctypes.c_int),
            ("union", ctypes.c_ulonglong * 2),
        ]

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    size = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(size))
    if not size.value:
        return None
    buf = (_Info * (size.value // ctypes.sizeof(_Info)))()
    if not kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
        return None
    cores = sum(1 for item in buf if item.relationship == 0)  # RelationProcessorCore
    return cores or None


def _espeak_data() -> tuple[str, str | None]:
    """(espeak-ng data folder kokoro-onnx uses, the spelling espeak-ng can open or None)."""
    import espeakng_loader

    data = espeakng_loader.get_data_path()
    return data, espeak_data_path(data)


def _check_espeak_data() -> str | None:
    try:
        data, usable = _espeak_data()
    except Exception:  # kokoro-onnx reports a missing espeak-ng itself
        return None
    return espeak_path_reason(data) if usable is None else None


def _use_short_espeak_path() -> None:
    """Point espeak-ng at an ASCII spelling of its data folder when the real one isn't.

    phonemizer resolves a folder it is given back to its long (non-ASCII) name, so the
    short name goes through ESPEAK_DATA_PATH, which espeak-ng reads itself when it is
    given no folder. Call after creating Kokoro (its tokenizer sets phonemizer's folder).
    """
    data, usable = _espeak_data()
    if usable is None or usable == data:
        return
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    os.environ["ESPEAK_DATA_PATH"] = usable
    os.environ.pop("PHONEMIZER_ESPEAK_DATA_PATH", None)
    EspeakWrapper.set_data_path(None)


def voice_table_ids(voices_path: Path) -> set[str]:
    """Voice ids stored in a Kokoro voices file (an .npz archive), without loading it."""
    with zipfile.ZipFile(voices_path) as archive:
        return {name[:-4] for name in archive.namelist() if name.endswith(".npy")}


class KokoroEngine(TTSEngine):
    name = "kokoro"
    label = "Kokoro (most natural, offline)"
    online = False

    def __init__(
        self, voices_dir: Path | None = None, *, variant: str | None = None, threads: int | None = None
    ) -> None:
        self._voices_dir = voices_dir  # None: downloaded + bundled folders
        self._variant = variant
        self._threads = threads
        self._load_lock = threading.Lock()
        self._synth_lock = threading.Lock()  # one synthesis at a time: it already uses every core
        self._kokoro: Any = None
        self._loaded_key: tuple[str, str, int, int] | None = None
        self._ids_cache: tuple[tuple[str, int, int], set[str]] | None = None
        self._package_problem: str | None = None
        self._package_checked = False

    # -- TTSEngine

    def is_available(self) -> tuple[bool, str]:
        problem = self._package_issue()
        if problem:
            return False, problem
        if self._files() is None:
            return False, NO_MODEL_REASON
        return True, ""

    def list_voices(self) -> list[Voice]:
        files = self._files()
        if files is None:
            return []
        available = self._voice_ids(files[1])
        return [
            Voice(id=vid, name=name, engine=self.name, language=_LANGUAGES[vid[0]][1], gender=gender)
            for vid, name, gender in CURATED_VOICES
            if vid in available
        ]

    def default_voice(self) -> str | None:
        voices = self.list_voices()
        ids = [v.id for v in voices]
        if DEFAULT_VOICE in ids:
            return DEFAULT_VOICE
        return ids[0] if ids else None

    def synthesize(self, text: str, voice: str | None = None, rate: float = 1.0) -> AudioClip:
        if not text or not text.strip():
            raise TTSError("There's no text to read aloud.")
        problem = self._package_issue()
        if problem:
            raise EngineUnavailable(problem)
        files = self._files()
        if files is None:
            raise EngineUnavailable(NO_MODEL_REASON)
        voice_id = voice or DEFAULT_VOICE
        if voice_id not in self._voice_ids(files[1]) or voice_id[0] not in _LANGUAGES:
            raise EngineUnavailable(
                f"The Kokoro voice '{voice_id}' isn't available. Pick another voice in Settings → Voices."
            )
        rate = clamp_rate(rate)
        native = min(max(rate, NATIVE_MIN_SPEED), NATIVE_MAX_SPEED)
        stretch = rate / native
        kokoro = self._get(files)
        try:
            with self._synth_lock:
                lang = _LANGUAGES[voice_id[0]][0]
                samples, sample_rate = kokoro.create(text, voice=voice_id, speed=native, lang=lang)
        except ValueError as exc:
            if "phonemes" in str(exc) or "Nothing to synthesize" in str(exc):
                return _silence_clip()  # nothing speakable (e.g. only punctuation)
            raise TTSError(f"The Kokoro voice couldn't read this text ({exc}).") from exc
        except Exception as exc:  # onnxruntime / phonemizer failures
            raise TTSError(f"The Kokoro voice couldn't read this text ({type(exc).__name__}: {exc}).") from exc
        return _to_clip(samples, int(sample_rate), stretch)

    # -- internals

    def _package_issue(self) -> str | None:
        if not self._package_checked:
            try:
                import kokoro_onnx  # noqa: F401
                import onnxruntime  # noqa: F401
            except ImportError:
                self._package_problem = NO_PACKAGE_REASON
            except Exception as exc:  # e.g. onnxruntime failing to load its DLLs
                self._package_problem = f"The Kokoro voice engine couldn't start: {exc}"
            else:
                self._package_problem = _check_espeak_data()
            self._package_checked = True
        return self._package_problem

    def _files(self) -> tuple[Path, Path] | None:
        return find_kokoro_model(self._voices_dir, variant=self._variant)

    def _voice_ids(self, voices_path: Path) -> set[str]:
        try:
            st = voices_path.stat()
            stamp = (str(voices_path), st.st_mtime_ns, st.st_size)
            if self._ids_cache is not None and self._ids_cache[0] == stamp:
                return self._ids_cache[1]
            ids = voice_table_ids(voices_path)
        except (OSError, zipfile.BadZipFile):
            return set()
        self._ids_cache = (stamp, ids)
        return ids

    def _get(self, files: tuple[Path, Path]) -> Any:
        model, voices = files
        try:
            m_st, v_st = model.stat(), voices.stat()
        except OSError as exc:
            raise EngineUnavailable(NO_MODEL_REASON) from exc
        key = (str(model), str(voices), m_st.st_mtime_ns ^ v_st.st_mtime_ns, m_st.st_size + v_st.st_size)
        with self._load_lock:
            if self._kokoro is not None and self._loaded_key == key:
                return self._kokoro
            self._kokoro = None  # free the old model before loading another
            self._kokoro = self._load(model, voices)
            self._loaded_key = key
            return self._kokoro

    def _load(self, model: Path, voices: Path) -> Any:
        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        disable_onnx_telemetry()  # privacy: nothing about this computer goes to Microsoft
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = self._threads or physical_cores()
        opts.inter_op_num_threads = 1
        opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        # Idle worker threads sleep instead of busy-waiting: measured ~45% less CPU for the
        # same speech (and faster when the browser or other apps need the CPU too).
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        try:
            session = rt.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
            kokoro = Kokoro.from_session(session, str(voices))
            _use_short_espeak_path()
            return kokoro
        except Exception as exc:
            raise EngineUnavailable(
                "The Kokoro voice pack couldn't be loaded; it may be damaged. "
                "Download it again in Settings → Voices."
            ) from exc


def _to_clip(samples: Any, sample_rate: int, stretch: float) -> AudioClip:
    import numpy as np

    from ..audio import time_stretch, wav_bytes

    pcm = (np.clip(np.asarray(samples, dtype=np.float32).ravel(), -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
    if not pcm:
        return _silence_clip()
    if abs(stretch - 1.0) > 1e-3:
        pcm = time_stretch(pcm, 1, sample_rate, stretch)
    return AudioClip(wav_bytes(pcm, sample_rate), "audio/wav", sample_rate)


def _silence_clip() -> AudioClip:
    from ..audio import silence_pcm, wav_bytes

    return AudioClip(wav_bytes(silence_pcm(200, SAMPLE_RATE), SAMPLE_RATE), "audio/wav", SAMPLE_RATE)
