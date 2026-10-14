"""Voice engines, the engine registry, the Piper voice catalog/downloader and the audio cache."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from destiny_bot import audio, config
from destiny_bot import tts as registry
from destiny_bot.tts import catalog
from destiny_bot.tts.base import AudioClip, EngineUnavailable, TTSError
from destiny_bot.tts.browser_engine import BrowserEngine
from destiny_bot.tts.cache import AudioCache
from destiny_bot.tts.edge_engine import NETWORK_ERROR, EdgeEngine
from destiny_bot.tts.kokoro_engine import NO_MODEL_REASON, NO_PACKAGE_REASON, KokoroEngine
from destiny_bot.tts.piper_engine import NO_VOICE_REASON, PiperEngine
from destiny_bot.tts.system_engine import SystemEngine

HAS_ESPEAK = bool(shutil.which("espeak-ng") or shutil.which("espeak"))


@pytest.fixture(autouse=True)
def no_bundled_voices(monkeypatch):
    """Tests opt in to bundled voice folders explicitly."""
    monkeypatch.delenv(catalog.EXTRA_DIRS_ENV, raising=False)


@pytest.fixture
def kokoro_model_dir() -> Path:
    """A folder with a real Kokoro model + voices-v1.0.bin, or skip.

    Set DESTINY_TEST_KOKORO_DIR to run the real Kokoro tests.
    """
    value = os.environ.get("DESTINY_TEST_KOKORO_DIR")
    if not value or catalog.find_kokoro_model(Path(value)) is None:
        pytest.skip("set DESTINY_TEST_KOKORO_DIR to a folder with a Kokoro model to run this test")
    return Path(value)


# --------------------------------------------------------------------------- helpers


def sine_wav(seconds: float = 0.5, rate: int = 22050, freq: float = 440.0, channels: int = 1) -> bytes:
    frames = int(seconds * rate)
    pcm = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) * channels for i in range(frames))
    return audio.wav_bytes(pcm, rate, channels)


def wav_seconds(data: bytes) -> float:
    pcm, rate, channels, width = audio.read_wav(data)
    return len(pcm) / (rate * channels * width)


def piper_dir() -> Path:
    path = config.voices_dir() / "piper"
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_fake_voice(voice_id: str, meta: dict | None = None, size: int = 1000) -> Path:
    folder = piper_dir()
    model = folder / f"{voice_id}.onnx"
    model.write_bytes(b"\x08" * size)
    (folder / f"{voice_id}.onnx.json").write_text(
        json.dumps(meta if meta is not None else {"audio": {"sample_rate": 22050}}), encoding="utf-8"
    )
    return model


def install_real_voice(src: Path) -> str:
    folder = piper_dir()
    for path in (src, src.with_name(src.name + ".json")):
        dest = folder / path.name
        try:
            dest.symlink_to(path)
        except OSError:
            shutil.copyfile(path, dest)
    return src.stem


class FakePiperVoice:
    """Stands in for piper.PiperVoice (1.3+ API) and records how it is used."""

    loads: list[str] = []

    def __init__(self, path: str, length_scale: float = 1.0, sample_rate: int = 22050) -> None:
        self.path = path
        self.config = SimpleNamespace(sample_rate=sample_rate, length_scale=length_scale)
        self.calls: list[float] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def synthesize_wav(self, text, wav_file, syn_config=None, set_wav_format=True):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        self.calls.append(syn_config.length_scale)
        wav_file.setframerate(self.config.sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(b"\x01\x00" * int(self.config.sample_rate * 0.1 * syn_config.length_scale))
        with self._guard:
            self.active -= 1


@pytest.fixture
def fake_piper(monkeypatch):
    """A PiperEngine whose voices load as FakePiperVoice objects."""
    engine = PiperEngine()
    loaded: list[str] = []

    def fake_load(self, path):
        loaded.append(path.stem)
        meta = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
        return FakePiperVoice(str(path), length_scale=meta.get("inference", {}).get("length_scale", 1.0))

    monkeypatch.setattr(PiperEngine, "_load", fake_load)
    engine.loaded_log = loaded  # type: ignore[attr-defined]
    return engine


# --------------------------------------------------------------------------- registry


def test_get_engine_singletons_and_unknown():
    assert registry.ENGINE_NAMES == ("kokoro", "piper", "system", "edge", "browser")
    assert registry.AUTO_ORDER == ("kokoro", "piper", "system", "browser")
    assert registry.get_engine("piper") is registry.get_engine("piper")
    assert [e.name for e in registry.all_engines()] == list(registry.ENGINE_NAMES)
    with pytest.raises(KeyError):
        registry.get_engine("nope")
    labels = {e.name: (e.label, e.online) for e in registry.all_engines()}
    assert labels == {
        "kokoro": ("Kokoro (most natural, offline)", False),
        "piper": ("Piper (natural, offline)", False),
        "system": ("System voice (offline)", False),
        "edge": ("Microsoft online voices", True),
        "browser": ("Browser voice", False),
    }


@pytest.fixture
def availability(monkeypatch):
    """Control each engine's is_available() through a dict."""
    state = {"kokoro": False, "piper": True, "system": True, "edge": True, "browser": True}
    for name in registry.ENGINE_NAMES:
        engine = registry.get_engine(name)
        monkeypatch.setattr(engine, "is_available", lambda n=name: (state[n], "" if state[n] else "nope"))
    return state


def test_resolve_auto_order(availability):
    assert registry.resolve_engine("auto", online_allowed=False).name == "piper"
    availability["piper"] = False
    assert registry.resolve_engine("auto", online_allowed=True).name == "system"
    availability["system"] = False
    # Auto never picks an online engine, even when allowed.
    assert registry.resolve_engine("auto", online_allowed=True).name == "browser"


def test_resolve_explicit_and_fallbacks(availability):
    assert registry.resolve_engine("system", online_allowed=False).name == "system"
    assert registry.resolve_engine("browser", online_allowed=False).name == "browser"
    assert registry.resolve_engine("edge", online_allowed=True).name == "edge"
    # Online engine while online voices are off -> behaves like auto.
    assert registry.resolve_engine("edge", online_allowed=False).name == "piper"
    availability["piper"] = False
    assert registry.resolve_engine("piper", online_allowed=False).name == "system"
    assert registry.resolve_engine("bogus", online_allowed=False).name == "system"
    assert registry.resolve_engine("", online_allowed=False).name == "system"
    availability["edge"] = False
    assert registry.resolve_engine("edge", online_allowed=True).name == "system"


def test_resolve_prefers_kokoro_when_installed(availability):
    availability["kokoro"] = True
    assert registry.resolve_engine("auto", online_allowed=False).name == "kokoro"
    assert registry.resolve_engine("piper", online_allowed=False).name == "piper"  # an explicit choice is kept
    assert registry.resolve_engine("edge", online_allowed=False).name == "kokoro"
    availability["kokoro"] = False
    assert registry.resolve_engine("kokoro", online_allowed=False).name == "piper"


def test_resolve_survives_broken_engine(availability, monkeypatch):
    def boom():
        raise RuntimeError("broken")

    monkeypatch.setattr(registry.get_engine("piper"), "is_available", boom)
    assert registry.resolve_engine("auto", online_allowed=False).name == "system"


def test_resolve_real_auto_without_voices():
    # No Kokoro pack or Piper voice in the fresh data folder: auto skips both.
    engine = registry.resolve_engine("auto", online_allowed=False)
    # Windows (SAPI) and macOS (say) always have a system voice; Linux needs espeak-ng.
    has_system_voice = HAS_ESPEAK or sys.platform in ("win32", "darwin")
    assert engine.name == ("system" if has_system_voice else "browser")


# --------------------------------------------------------------------------- piper


def test_piper_unavailable_without_voices():
    engine = PiperEngine()
    assert engine.is_available() == (False, NO_VOICE_REASON)
    assert engine.list_voices() == []
    assert engine.default_voice() is None
    with pytest.raises(EngineUnavailable, match="No natural voice"):
        engine.synthesize("Hello")


def test_piper_lists_installed_voices():
    install_fake_voice("en_US-amy-medium", size=2_000_000)
    install_fake_voice("en_GB-weird_voice-x_low", {"language": {"code": "en_GB"}, "audio": {"sample_rate": 16000}})
    install_fake_voice("en_US-lessac-medium")
    install_fake_voice("custom", {"espeak": {"voice": "de"}, "dataset": "my_voice", "quality": "high"})
    (piper_dir() / "orphan-model.onnx").write_bytes(b"x")  # no .json -> not installed
    engine = PiperEngine()
    voices = engine.list_voices()
    assert [v.id for v in voices] == ["en_US-lessac-medium", "en_US-amy-medium", "custom", "en_GB-weird_voice-x_low"]
    by_id = {v.id: v for v in voices}
    assert by_id["en_US-lessac-medium"].name == "Lessac (US English)"
    assert by_id["en_US-amy-medium"].size_mb == 2.0
    assert by_id["en_US-amy-medium"].language == "en-US"
    assert by_id["en_GB-weird_voice-x_low"].name == "Weird Voice (en-GB, x low quality)"
    assert by_id["custom"].language == "de"
    assert by_id["custom"].name == "My Voice (de, high quality)"
    assert all(v.engine == "piper" and v.installed and not v.online for v in voices)
    assert engine.default_voice() == "en_US-lessac-medium"


def test_piper_default_prefers_english_when_no_lessac():
    install_fake_voice("es_ES-davefx-medium")
    install_fake_voice("en_GB-alba-medium")
    assert PiperEngine().default_voice() == "en_GB-alba-medium"


def test_piper_rate_maps_to_length_scale_and_stretch(fake_piper):
    install_fake_voice("en_US-amy-medium", {"inference": {"length_scale": 1.2}})
    assert fake_piper.is_available() == (True, "")
    normal = fake_piper.synthesize("Hello", rate=1.0)
    slow = fake_piper.synthesize("Hello", "en_US-amy-medium", rate=0.5)
    fast = fake_piper.synthesize("Hello", "en_US-amy-medium", rate=2.0)
    voice = fake_piper._loaded["en_US-amy-medium"].voice
    # Slower: Piper's own length_scale. Faster: natural render, then time-stretched.
    assert voice.calls == [pytest.approx(1.2), pytest.approx(2.4), pytest.approx(1.2)]
    assert normal.mime == fast.mime == "audio/wav" and fast.sample_rate == 22050
    assert wav_seconds(fast.data) == pytest.approx(wav_seconds(normal.data) / 2, abs=0.001)
    assert wav_seconds(slow.data) == pytest.approx(wav_seconds(normal.data) * 2, abs=0.001)


def test_clamp_rate():
    assert registry.clamp_rate(1.5) == 1.5
    assert registry.clamp_rate(10) == 4.0 and registry.clamp_rate(0.1) == 0.25
    for bad in (None, float("nan"), float("inf"), 0, -2, "fast"):
        assert registry.clamp_rate(bad) == 1.0  # type: ignore[arg-type]
    assert registry.clamp_rate(3.0, 0.5, 2.0) == 2.0


def test_piper_speed_plan():
    from destiny_bot.tts.piper_engine import MAX_LENGTH_SCALE, speed_plan

    def voice(scale):
        return SimpleNamespace(config=SimpleNamespace(length_scale=scale))

    plans = [speed_plan(voice(1.0), r) for r in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0)]
    assert plans == [(2.0, 1.0), (1.25, 1.0), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0), (1.0, 3.0)]
    # Slower than length_scale allows: the rest is a (slowing) stretch.
    assert speed_plan(voice(1.2), 0.25) == (MAX_LENGTH_SCALE, pytest.approx(0.5208, abs=1e-4))
    assert speed_plan(voice(1.0), 9.0) == (1.0, 4.0)
    assert speed_plan(voice(None), float("nan")) == (1.0, 1.0)
    assert speed_plan(voice(-3), 2.0) == (1.0, 2.0)
    assert speed_plan(voice(40), 1.0) == (MAX_LENGTH_SCALE, 1.0)


def test_piper_lru_keeps_two_voices(fake_piper):
    for vid in ("en_US-amy-medium", "en_US-joe-medium", "en_US-ryan-high"):
        install_fake_voice(vid)
    fake_piper.synthesize("a", "en_US-amy-medium")
    fake_piper.synthesize("a", "en_US-joe-medium")
    fake_piper.synthesize("a", "en_US-amy-medium")  # amy is now most recent
    fake_piper.synthesize("a", "en_US-ryan-high")  # evicts joe
    assert list(fake_piper._loaded) == ["en_US-amy-medium", "en_US-ryan-high"]
    fake_piper.synthesize("a", "en_US-joe-medium")
    assert fake_piper.loaded_log == ["en_US-amy-medium", "en_US-joe-medium", "en_US-ryan-high", "en_US-joe-medium"]
    assert len(fake_piper._loaded) == 2


def test_piper_reloads_replaced_model(fake_piper):
    model = install_fake_voice("en_US-amy-medium")
    fake_piper.synthesize("a", "en_US-amy-medium")
    model.write_bytes(b"\x09" * 2000)  # re-downloaded: different size
    fake_piper.synthesize("a", "en_US-amy-medium")
    assert fake_piper.loaded_log == ["en_US-amy-medium", "en_US-amy-medium"]


def test_piper_serializes_per_voice_and_loads_once(fake_piper):
    install_fake_voice("en_US-amy-medium")
    errors: list[BaseException] = []

    def work():
        try:
            for _ in range(5):
                fake_piper.synthesize("hello", "en_US-amy-medium")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert fake_piper.loaded_log == ["en_US-amy-medium"]
    voice = fake_piper._loaded["en_US-amy-medium"].voice
    assert voice.max_active == 1 and len(voice.calls) == 30


def test_piper_old_api_and_unknown_voice(monkeypatch):
    class OldVoice:  # piper-tts 1.2: synthesize(text, wav_file, length_scale=...)
        config = SimpleNamespace(sample_rate=16000, length_scale=1.0)

        def synthesize(self, text, wav_file, length_scale=None):
            self.length_scale = length_scale
            wav_file.setframerate(16000)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(b"\x00\x01" * 1600)

    old = OldVoice()
    monkeypatch.setattr(PiperEngine, "_load", lambda self, path: old)
    install_fake_voice("en_US-amy-medium")
    engine = PiperEngine()
    clip = engine.synthesize("Hi", rate=0.8)
    assert old.length_scale == pytest.approx(1.25)
    assert clip.sample_rate == 16000 and wav_seconds(clip.data) == pytest.approx(0.1)
    with pytest.raises(EngineUnavailable, match="isn't downloaded"):
        engine.synthesize("Hi", "en_US-missing-medium")
    with pytest.raises(TTSError):
        engine.synthesize("   ")


def test_piper_damaged_model_is_friendly():
    install_fake_voice("en_US-amy-medium")  # not a real ONNX file
    with pytest.raises(EngineUnavailable, match="damaged"):
        PiperEngine().synthesize("Hello")


def test_piper_real_synthesis(piper_voice_path):
    voice_id = install_real_voice(piper_voice_path)
    engine = PiperEngine()
    assert engine.is_available() == (True, "")
    assert engine.default_voice() == voice_id
    clip = engine.synthesize("Photosynthesis turns light into chemical energy.")
    rate = json.loads(piper_voice_path.with_name(piper_voice_path.name + ".json").read_text(encoding="utf-8"))["audio"]["sample_rate"]
    assert clip.mime == "audio/wav" and clip.sample_rate == rate
    pcm, sr, channels, width = audio.read_wav(clip.data)
    assert (sr, channels, width) == (rate, 1, 2)
    normal = len(pcm) / (sr * 2)
    assert 1.0 < normal < 10.0
    assert max(abs(s) for s in struct.unpack(f"<{len(pcm) // 2}h", pcm)) > 1000  # not silence
    fast = wav_seconds(engine.synthesize("Photosynthesis turns light into chemical energy.", rate=2.0).data)
    assert fast < normal * 0.75
    # Only punctuation: a short pause rather than an error.
    assert 0 < wav_seconds(engine.synthesize("...").data) <= 0.5


def test_piper_real_reader_rates(piper_voice_path):
    """The web reader asks the engine for 0.5x-3x; each step must be faster and still clean audio."""
    install_real_voice(piper_voice_path)
    engine = PiperEngine()
    text = "Mitochondria release energy from food during cellular respiration."
    durations = []
    for rate in (0.5, 1.0, 1.5, 2.0, 3.0):
        clip = audio.trim_silence(engine.synthesize(text, rate=rate))
        pcm, sr, _, _ = audio.read_wav(clip.data)
        samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
        assert samples[0] == 0 and samples[-1] == 0  # trimmed and faded
        assert max(abs(s) for s in samples) > 1000
        durations.append(len(samples) / sr)
    assert durations == sorted(durations, reverse=True)
    # Native slow-down: natural, but sub-linear (tokens at Piper's 1-frame minimum don't lengthen).
    assert 1.3 < durations[0] / durations[1] < 2.5
    assert durations[1] / durations[4] == pytest.approx(3.0, rel=0.1)


def test_piper_real_concurrent(piper_voice_path):
    install_real_voice(piper_voice_path)
    engine = PiperEngine()
    results: list[AudioClip] = []
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            results.append(engine.synthesize(f"Sentence number {i}."))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 4
    assert all(wav_seconds(c.data) > 0.5 for c in results)


# --------------------------------------------------------------------------- kokoro


def write_fake_kokoro(folder: Path, voices: tuple[str, ...] = ("af_heart", "bm_george", "af_bella", "qq_odd"),
                      variant: str = "int8") -> tuple[Path, Path]:
    """A fake Kokoro install: a (non-ONNX) model file and a real .npz voice table."""
    np = pytest.importorskip("numpy")
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / catalog.KOKORO_FILES[variant].name
    model.write_bytes(b"fake onnx model")
    table = folder / catalog.KOKORO_FILES["voices"].name
    with open(table, "wb") as fh:
        np.savez(fh, **{v: np.zeros((510, 1, 256), dtype=np.float32) for v in voices})
    return model, table


class FakeKokoro:
    """Stands in for kokoro_onnx.Kokoro: speech is a 0.5 s / speed tone."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, str]] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def create(self, text, voice, speed=1.0, lang="en-us"):
        import numpy as np

        if not any(ch.isalnum() for ch in text):
            raise ValueError(f"Nothing to synthesize, {text!r} produced no phonemes")
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        self.calls.append((text, voice, speed, lang))
        n = int(24000 * 0.5 / speed)
        samples = 0.5 * np.sin(2 * np.pi * 200 * np.arange(n) / 24000)
        with self._guard:
            self.active -= 1
        return samples.astype(np.float32), 24000


@pytest.fixture
def fake_kokoro(monkeypatch):
    write_fake_kokoro(config.voices_dir() / "kokoro")
    fake = FakeKokoro()
    loads: list[tuple[Path, Path]] = []

    def load(self, model, voices):
        loads.append((model, voices))
        return fake

    monkeypatch.setattr(KokoroEngine, "_load", load)
    engine = KokoroEngine()
    engine.fake = fake  # type: ignore[attr-defined]
    engine.loads = loads  # type: ignore[attr-defined]
    return engine


def test_kokoro_unavailable_reasons(monkeypatch):
    engine = KokoroEngine()
    assert engine.is_available() == (False, NO_MODEL_REASON)
    assert "Settings → Voices" in NO_MODEL_REASON and "115 MB" in NO_MODEL_REASON
    assert engine.list_voices() == [] and engine.default_voice() is None
    with pytest.raises(EngineUnavailable, match="Kokoro voice pack"):
        engine.synthesize("Hello")
    write_fake_kokoro(config.voices_dir() / "kokoro")
    assert engine.is_available() == (True, "")
    monkeypatch.setitem(sys.modules, "kokoro_onnx", None)  # package missing
    fresh = KokoroEngine()
    assert fresh.is_available() == (False, NO_PACKAGE_REASON)
    with pytest.raises(EngineUnavailable, match="kokoro-onnx"):
        fresh.synthesize("Hello")


def test_kokoro_lists_curated_voices(fake_kokoro):
    voices = fake_kokoro.list_voices()
    assert [v.id for v in voices] == ["af_heart", "af_bella", "bm_george"]  # curated order, odd ones hidden
    assert voices[0].name == "Heart (US English, female)" and voices[0].language == "en-US"
    assert voices[2].language == "en-GB" and voices[2].gender == "male"
    assert all(v.engine == "kokoro" and not v.online for v in voices)
    assert fake_kokoro.default_voice() == "af_heart"


def test_kokoro_speed_and_languages(fake_kokoro):
    fake = fake_kokoro.fake
    normal = fake_kokoro.synthesize("Hello there.")
    assert normal.mime == "audio/wav" and normal.sample_rate == 24000
    assert wav_seconds(normal.data) == pytest.approx(0.5, abs=0.001)
    durations = {}
    for rate in (0.25, 0.5, 1.5, 2.0, 3.0, 4.0):
        durations[rate] = wav_seconds(fake_kokoro.synthesize("Hello there.", "bm_george", rate).data)
    speeds = [c[2] for c in fake.calls[1:]]
    assert speeds == [0.5, 0.5, 1.5, 2.0, 2.0, 2.0]  # Kokoro's own range is 0.5-2.0
    assert {c[3] for c in fake.calls[1:]} == {"en-gb"} and fake.calls[0][1:] == ("af_heart", 1.0, "en-us")
    # Outside 0.5-2.0 the rest is a pitch-preserving time-stretch, so speed stays exact.
    assert durations[0.25] == pytest.approx(2.0, abs=0.002)
    assert durations[3.0] == pytest.approx(0.5 / 3, abs=0.002)
    assert durations[4.0] == pytest.approx(0.125, abs=0.002)
    assert fake_kokoro.loads == [(config.voices_dir() / "kokoro" / "kokoro-v1.0.int8.onnx",
                                  config.voices_dir() / "kokoro" / "voices-v1.0.bin")]  # fmt: skip


def test_kokoro_errors(fake_kokoro):
    with pytest.raises(EngineUnavailable, match="isn't available"):
        fake_kokoro.synthesize("Hello", "af_missing")
    with pytest.raises(EngineUnavailable):
        fake_kokoro.synthesize("Hello", "qq_odd")  # in the table, but no known language
    with pytest.raises(TTSError):
        fake_kokoro.synthesize("   ")
    pause = fake_kokoro.synthesize("...")
    assert wav_seconds(pause.data) == pytest.approx(0.2, abs=0.001)


def test_kokoro_thread_safe_single_load(fake_kokoro):
    errors: list[BaseException] = []
    results: list[AudioClip] = []

    def work():
        try:
            for _ in range(4):
                results.append(fake_kokoro.synthesize("Hello there.", rate=1.0))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 20
    assert len(fake_kokoro.loads) == 1 and fake_kokoro.fake.max_active == 1


def test_kokoro_reloads_when_files_change(fake_kokoro):
    fake_kokoro.synthesize("Hi")
    model = config.voices_dir() / "kokoro" / "kokoro-v1.0.int8.onnx"
    model.write_bytes(b"a different fake model")
    fake_kokoro.synthesize("Hi")
    assert len(fake_kokoro.loads) == 2


def test_kokoro_session_settings(monkeypatch):
    import onnxruntime

    from destiny_bot.tts import kokoro_engine

    write_fake_kokoro(config.voices_dir() / "kokoro")
    seen = {}

    def fake_session(path, sess_options=None, providers=None):
        seen.update(path=path, opts=sess_options, providers=providers)
        raise RuntimeError("stop here")

    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)
    monkeypatch.setattr(kokoro_engine, "physical_cores", lambda: 3)
    with pytest.raises(EngineUnavailable):
        KokoroEngine().synthesize("Hello")
    opts = seen["opts"]
    assert seen["path"].endswith("kokoro-v1.0.int8.onnx") and seen["providers"] == ["CPUExecutionProvider"]
    assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == (3, 1)
    assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "0"


def test_onnx_telemetry_is_off_before_any_session(monkeypatch):
    # onnxruntime's Windows builds send telemetry events by default; turn them off first.
    import onnxruntime

    piper = pytest.importorskip("piper")
    events: list[str] = []
    monkeypatch.setattr(onnxruntime, "disable_telemetry_events", lambda: events.append("telemetry off"))

    def fake_session(*args, **kwargs):
        events.append("session")
        raise RuntimeError("stop here")

    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)
    write_fake_kokoro(config.voices_dir() / "kokoro")
    with pytest.raises(EngineUnavailable):
        KokoroEngine().synthesize("Hello")
    assert events == ["telemetry off", "session"]

    events.clear()
    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(fake_session))  # Piper makes its own session
    install_fake_voice("en_US-test-medium")
    with pytest.raises(EngineUnavailable):
        PiperEngine().synthesize("Hello")
    assert events == ["telemetry off", "session"]


# espeak-ng opens its data with the C library's narrow file functions: on Windows a UTF-8 path
# like C:\Users\Zoë\... is read in the ANSI code page, isn't found, and espeak-ng calls exit(1).
ZOE_DATA = r"C:\Users\Zoë\AppData\Local\Programs\Athena\_internal\espeak-ng-data"
ZOE_SHORT = r"C:\Users\ZO7E2~1\AppData\Local\Programs\Athena\_internal\ESPEAK~1"


def _windows_short_names(monkeypatch, short: str | None) -> list[str]:
    asked: list[str] = []
    monkeypatch.setattr(registry, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(registry, "_short_path_name", lambda path: asked.append(path) or short)
    return asked


def test_espeak_data_path_uses_windows_short_names(monkeypatch):
    assert registry.espeak_data_path(Path("/opt/Zoë/espeak-ng-data")) == "/opt/Zoë/espeak-ng-data"  # UTF-8 is fine here
    # espeak-ng 1.52 keeps the folder in a 160-byte buffer (230 on Windows): longer is "missing" + exit(1).
    deep = "/Users/zoe/" + "Study folder/" * 10 + "site-packages/espeakng_loader/espeak-ng-data"
    assert len(deep) > 159 and registry.espeak_data_path(deep) is None
    asked = _windows_short_names(monkeypatch, ZOE_SHORT)
    assert registry.espeak_data_path(r"C:\Athena\espeak-ng-data") == r"C:\Athena\espeak-ng-data" and asked == []
    assert registry.espeak_data_path(ZOE_DATA) == ZOE_SHORT and asked == [ZOE_DATA]
    _windows_short_names(monkeypatch, None)  # 8.3 names turned off on that drive
    assert registry.espeak_data_path(ZOE_DATA) is None
    _windows_short_names(monkeypatch, r"C:\Users\Zoë~1\x")
    assert registry.espeak_data_path(ZOE_DATA) is None
    long_ascii = "C:\\" + "Study folder\\" * 18 + "espeak-ng-data"
    assert 229 < len(long_ascii) and registry.espeak_data_path(long_ascii) is None
    _windows_short_names(monkeypatch, r"C:\STUDYF~1\STUDYF~1\ESPEAK~1")
    assert registry.espeak_data_path(long_ascii) == r"C:\STUDYF~1\STUDYF~1\ESPEAK~1"


def test_piper_hands_espeak_an_ascii_data_path(monkeypatch):
    phonemize_espeak = pytest.importorskip("piper.phonemize_espeak")
    import piper

    monkeypatch.setattr(phonemize_espeak, "ESPEAK_DATA_DIR", Path(ZOE_DATA))
    _windows_short_names(monkeypatch, ZOE_SHORT)
    seen: list[dict] = []
    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(lambda *a, **kw: seen.append(kw) or FakePiperVoice(a[0])))
    install_fake_voice("en_US-test-medium")
    engine = PiperEngine()
    assert engine.is_available() == (True, "")
    engine.synthesize("Hello")
    assert seen[0]["espeak_data_dir"] == ZOE_SHORT

    _windows_short_names(monkeypatch, None)
    engine = PiperEngine()
    ok, reason = engine.is_available()
    assert not ok and "plain name" in reason and ZOE_DATA in reason
    with pytest.raises(EngineUnavailable, match="plain name"):  # a friendly error, not espeak's exit(1)
        engine.synthesize("Hello")
    assert len(seen) == 1


def test_kokoro_hands_espeak_an_ascii_data_path(monkeypatch):
    import espeakng_loader
    import onnxruntime
    from kokoro_onnx import Kokoro
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    write_fake_kokoro(config.voices_dir() / "kokoro")
    monkeypatch.setattr(espeakng_loader, "get_data_path", lambda: ZOE_DATA)
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **kw: "session")

    def from_session(session, voices, espeak_config=None):
        EspeakWrapper.set_data_path(ZOE_DATA)  # what kokoro-onnx's tokenizer does
        return FakeKokoro()

    monkeypatch.setattr(Kokoro, "from_session", staticmethod(from_session))
    monkeypatch.setattr(EspeakWrapper, "_ESPEAK_DATA_PATH", None)
    monkeypatch.setenv("ESPEAK_DATA_PATH", "")  # restored (to unset) after the test
    monkeypatch.setenv("PHONEMIZER_ESPEAK_DATA_PATH", ZOE_DATA)
    _windows_short_names(monkeypatch, ZOE_SHORT)
    engine = KokoroEngine()
    assert engine.is_available() == (True, "")
    engine.synthesize("Hello")
    # phonemizer would turn a short folder it is given back into the long one, so espeak-ng gets no
    # folder from it and reads ESPEAK_DATA_PATH itself.
    assert os.environ["ESPEAK_DATA_PATH"] == ZOE_SHORT and EspeakWrapper._ESPEAK_DATA_PATH is None
    assert "PHONEMIZER_ESPEAK_DATA_PATH" not in os.environ

    _windows_short_names(monkeypatch, None)
    engine = KokoroEngine()
    ok, reason = engine.is_available()
    assert not ok and "plain name" in reason
    with pytest.raises(EngineUnavailable, match="plain name"):
        engine.synthesize("Hello")


def test_kokoro_damaged_model_is_friendly():
    write_fake_kokoro(config.voices_dir() / "kokoro")
    with pytest.raises(EngineUnavailable, match="damaged"):
        KokoroEngine().synthesize("Hello")


def test_physical_cores(monkeypatch):
    from destiny_bot.tts import kokoro_engine

    cores = kokoro_engine.physical_cores()
    assert 1 <= cores <= (os.cpu_count() or 1)
    monkeypatch.setattr(sys, "platform", "sunos5")
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert kokoro_engine.physical_cores() == 4  # unknown OS: assume hyper-threading
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert kokoro_engine.physical_cores() == 2


@pytest.mark.skipif(struct.calcsize("P") != 8, reason="the record layout below is 64-bit Windows'")
def test_physical_cores_windows_layout(monkeypatch):
    # GetLogicalProcessorInformation fills SYSTEM_LOGICAL_PROCESSOR_INFORMATION records, 32 bytes each
    # on 64-bit Windows: ProcessorMask (8), Relationship (4), padding (4), a 16-byte union.
    import ctypes

    from destiny_bot.tts import kokoro_engine

    relationships = [3, 0, 2, 2, 0, 2, 0, 1, 0, 2]  # package, 4 cores, caches, a NUMA node
    records = b"".join(struct.pack("<QI4xQQ", 0b11 << (2 * i), rel, 0x0040000801010000, 0) for i, rel in enumerate(relationships))

    def get_info(buffer, size_ref):
        size = size_ref._obj
        if buffer is None or size.value < len(records):
            size.value = len(records)
            return 0  # ERROR_INSUFFICIENT_BUFFER
        ctypes.memmove(buffer, records, len(records))
        return 1

    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(GetLogicalProcessorInformation=get_info)), raising=False)
    assert kokoro_engine._windows_physical_cores() == 4
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert kokoro_engine.physical_cores() == 4


def test_kokoro_real_synthesis(kokoro_model_dir, monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv(catalog.EXTRA_DIRS_ENV, str(kokoro_model_dir.parent))
    if kokoro_model_dir.name != "kokoro":
        engine = KokoroEngine(kokoro_model_dir)
    else:
        engine = KokoroEngine()
    assert engine.is_available() == (True, "")
    assert engine.default_voice() == "af_heart"
    assert {"af_heart", "am_michael", "bf_emma", "bm_george"} <= {v.id for v in engine.list_voices()}
    text = "The mitochondria release energy from food during cellular respiration."
    clip = engine.synthesize(text)
    pcm, rate, channels, width = audio.read_wav(clip.data)
    assert clip.mime == "audio/wav" and (clip.sample_rate, rate, channels, width) == (24000, 24000, 1, 2)
    samples = np.frombuffer(pcm, dtype="<i2").astype(float)
    normal = len(samples) / rate
    assert 2.0 < normal < 8.0
    assert 3000 < np.abs(samples).max() < 32767  # loud enough, never clipped
    assert np.mean(np.abs(samples) >= 32766) == 0
    # Most energy sits in the speech band, and there is no DC offset.
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / rate)
    assert spectrum[(freqs > 80) & (freqs < 4000)].sum() / spectrum.sum() > 0.8
    assert abs(samples.mean()) < 50
    durations = [wav_seconds(engine.synthesize(text, rate=r).data) for r in (0.5, 2.0, 3.0)]
    assert durations[0] > normal * 1.6 and durations[1] < normal * 0.7
    assert durations[2] == pytest.approx(durations[1] * 2 / 3, rel=0.15)  # 2.0 native, then stretched
    british = engine.synthesize("Hello from London.", "bm_george")
    assert wav_seconds(british.data) > 0.5


def test_kokoro_real_concurrent(kokoro_model_dir):
    engine = KokoroEngine(kokoro_model_dir)
    results: list[AudioClip] = []
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            results.append(engine.synthesize(f"This is sentence number {i}."))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 3 and all(wav_seconds(c.data) > 0.8 for c in results)


# --------------------------------------------------------------------------- system: Linux (real espeak-ng)


@pytest.mark.skipif(not HAS_ESPEAK or sys.platform in ("win32", "darwin"), reason="needs espeak-ng on Linux")
def test_system_espeak_real():
    engine = SystemEngine()
    assert engine.is_available() == (True, "")
    voices = engine.list_voices()
    ids = [v.id for v in voices]
    assert ids[:2] == ["en-us", "en-gb"]
    first_other = next(i for i, v in enumerate(ids) if not v.startswith("en"))
    assert all(not v.startswith("en") for v in ids[first_other:])
    assert not any(v.engine != "system" or v.online for v in voices)
    assert engine.default_voice() == "en-us"
    assert voices[0].language == "en-US"

    clip = engine.synthesize("--help -v me. Cells are the basic unit of life.")
    pcm, rate, channels, width = audio.read_wav(clip.data)
    assert clip.mime == "audio/wav" and clip.sample_rate == rate and (channels, width) == (1, 2)
    # The header was rewritten with real sizes (espeak writes a bogus length to stdout).
    with wave.open(io.BytesIO(clip.data)) as w:
        assert w.getnframes() == len(pcm) // 2
    normal = len(pcm) / (rate * 2)
    assert normal > 1.0
    fast = wav_seconds(engine.synthesize("--help -v me. Cells are the basic unit of life.", rate=2.0).data)
    assert fast < normal * 0.8
    fastest = wav_seconds(engine.synthesize("--help -v me. Cells are the basic unit of life.", rate=3.0).data)
    assert fastest < fast * 0.85
    assert wav_seconds(engine.synthesize("Hola, ¿qué tal?", "es").data) > 0.3
    assert wav_seconds(engine.synthesize("Hello", "en-us+f3").data) > 0.2  # voice + variant

    with pytest.raises(EngineUnavailable, match="isn't available"):
        engine.synthesize("Hello", "xx-notavoice")
    with pytest.raises(EngineUnavailable, match="isn't valid"):
        engine.synthesize("Hello", "--stdout")
    with pytest.raises(TTSError):
        engine.synthesize("  ")


def test_system_linux_without_espeak(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    engine = SystemEngine()
    ok, reason = engine.is_available()
    assert not ok and "espeak-ng" in reason
    assert engine.list_voices() == []
    with pytest.raises(EngineUnavailable):
        engine.synthesize("Hello")


# --------------------------------------------------------------------------- system: macOS (mocked)

SAY_VOICES = """\
Alex                en_US    # Most people recognize me by my voice.
Bad News            en_US    # The light you see at the end of the tunnel is the headlamp of a fast approaching train.
Daniel              en_GB    # Hello, my name is Daniel. I am a British-English voice.
Eddy (German (Germany)) de_DE    # Hallo! Ich heiße Eddy.
Samantha            en_US    # Hello, my name is Samantha. I am an American-English voice.
Thomas              fr_FR    # Bonjour, je m'appelle Thomas.
Zarvox              en_US    # That looks like a peaceful planet.
"""


class FakeRun:
    """Records subprocess.run calls; ``handler(cmd, kwargs)`` returns (returncode, stdout)."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        code, out = self.handler(list(cmd), kwargs)
        return subprocess.CompletedProcess(cmd, code, out, b"")


def test_system_macos_commands(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/say" if name == "say" else None)
    seen_text: list[str] = []

    def handler(cmd, kwargs):
        if cmd[1:] == ["-v", "?"]:
            return 0, SAY_VOICES.encode()
        out = Path(cmd[cmd.index("-o") + 1])
        seen_text.append(Path(cmd[cmd.index("-f") + 1]).read_text(encoding="utf-8"))
        out.write_bytes(sine_wav(0.4, 22050))
        return 0, b""

    fake = FakeRun(handler)
    monkeypatch.setattr(subprocess, "run", fake)
    engine = SystemEngine()
    assert engine.is_available() == (True, "")
    voices = engine.list_voices()
    assert [v.id for v in voices] == ["Alex", "Samantha", "Daniel", "Eddy (German (Germany))", "Thomas"]
    assert voices[0].name == "Alex (en-US)" and voices[2].language == "en-GB"
    assert voices[3].name == "Eddy (German (Germany))"
    assert engine.default_voice() == "Samantha"

    text = '"Quotes" & $(rm -rf ~) -- 5 > 3'
    clip = engine.synthesize(text, rate=2.0)
    cmd, kwargs = fake.calls[-1]
    tmp = Path(cmd[cmd.index("-o") + 1]).parent
    assert cmd == [
        "/usr/bin/say",
        "-v",
        "Samantha",
        "-r",
        "350",
        "-o",
        str(tmp / "speech.wav"),
        "--data-format=LEI16@22050",
        "-f",
        str(tmp / "text.txt"),
    ]
    assert text not in cmd and seen_text == [text]
    assert kwargs["timeout"] > 0 and "shell" not in kwargs
    assert not tmp.exists()  # temp files cleaned up
    assert clip.mime == "audio/wav" and clip.sample_rate == 22050
    assert wav_seconds(clip.data) == pytest.approx(0.4, abs=0.01)

    engine.synthesize("Hi", "Daniel")
    assert fake.calls[-1][0][1:5] == ["-v", "Daniel", "-r", "175"]
    with pytest.raises(EngineUnavailable):
        engine.synthesize("Hi", "Nobody")


def test_system_macos_failure_is_friendly(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/say" if name == "say" else None)
    monkeypatch.setattr(
        subprocess, "run", FakeRun(lambda cmd, kw: (0, SAY_VOICES.encode()) if cmd[1:] == ["-v", "?"] else (1, b""))
    )
    with pytest.raises(TTSError, match="couldn't read this text"):
        SystemEngine().synthesize("Hello")


# --------------------------------------------------------------------------- system: Windows (mocked)

PS_EXE = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def _decode_ps(cmd: list[str]) -> str:
    assert cmd[:4] == [PS_EXE, "-NoProfile", "-NonInteractive", "-EncodedCommand"] and len(cmd) == 5
    return base64.b64decode(cmd[4]).decode("utf-16-le")


def test_system_windows_powershell(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    _fake_comtypes(monkeypatch, None)  # comtypes (SAPI from Python) not installed
    monkeypatch.setattr(shutil, "which", lambda name: PS_EXE if name == "powershell" else None)
    seen: list[dict] = []

    def handler(cmd, kwargs):
        script = _decode_ps(cmd)
        if "GetInstalledVoices" in script:
            listing = "Microsoft Zira Desktop\ten-US\tFemale\r\nMicrosoft David Desktop\ten-US\tMale\r\nMicrosoft Hélène\tfr-FR\tFemale\r\n"
            return 0, listing.encode("utf-8")
        env = kwargs["env"]
        assert "SelectVoice($env:DESTINY_TTS_VOICE)" in script and "SetOutputToWaveFile" in script
        assert "ReadAllText($env:DESTINY_TTS_TEXT" in script
        seen.append({**{k: env[k] for k in env if k.startswith("DESTINY_TTS_")}, "text": Path(env["DESTINY_TTS_TEXT"]).read_text("utf-8")})
        Path(env["DESTINY_TTS_OUT"]).write_bytes(sine_wav(0.3, 22050))
        return 0, b""

    fake = FakeRun(handler)
    monkeypatch.setattr(subprocess, "run", fake)
    engine = SystemEngine()
    assert engine.is_available() == (True, "")
    voices = engine.list_voices()
    assert [v.id for v in voices] == ["Microsoft Zira Desktop", "Microsoft David Desktop", "Microsoft Hélène"]
    assert voices[0].gender == "female" and voices[2].language == "fr-FR"
    assert engine.default_voice() == "Microsoft Zira Desktop"

    text = "Line one; `whoami` & 'quotes' \"double\""
    clip = engine.synthesize(text, "Microsoft David Desktop", rate=2.0)
    cmd, kwargs = fake.calls[-1]
    assert text not in " ".join(cmd)
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    assert seen[-1]["text"] == text
    assert seen[-1]["DESTINY_TTS_VOICE"] == "Microsoft David Desktop"
    assert seen[-1]["DESTINY_TTS_RATE"] == "7"  # 350 wpm
    assert not Path(seen[-1]["DESTINY_TTS_TEXT"]).exists()  # temp files cleaned up
    assert clip.mime == "audio/wav" and wav_seconds(clip.data) == pytest.approx(0.3, abs=0.01)
    engine.synthesize("Hi", rate=1.0)
    assert seen[-1]["DESTINY_TTS_RATE"] == "1" and seen[-1]["DESTINY_TTS_VOICE"] == "Microsoft Zira Desktop"
    engine.synthesize("Hi", rate=0.25)
    assert seen[-1]["DESTINY_TTS_RATE"] == "-10"


class FakeToken:
    def __init__(self, token_id: str, description: str, language: str, gender: str | None):
        self.Id, self.description, self.attrs = token_id, description, {"Language": language, "Gender": gender}

    def GetDescription(self, locale=0):
        return self.description

    def GetAttribute(self, name):
        if self.attrs.get(name) is None:
            raise OSError("COMError: SPERR_NOT_FOUND")
        return self.attrs[name]


class FakeSapi:
    """What comtypes.client.CreateObject hands out for SAPI: SpVoice and SpFileStream."""

    def __init__(self, *, silent: bool = False, broken: bool = False, hang: threading.Event | None = None):
        self.silent, self.broken, self.hang = silent, broken, hang
        self.tokens = [
            FakeToken("TOKEN_ZIRA", "Microsoft Zira Desktop - English (United States)", "409", "Female"),
            FakeToken("TOKEN_DAVID", "Microsoft David Desktop - English (United States)", "409;9", "Male"),
            FakeToken("TOKEN_HEDDA", "Microsoft Hedda Desktop - German (Germany)", "407", None),
        ]
        self.threads: set[int] = set()
        self.created: list[str] = []
        self.spoken: list[dict] = []
        self.paths: list[str] = []

    def CreateObject(self, progid):
        self.threads.add(threading.get_ident())
        self.created.append(progid)
        if self.broken:
            raise OSError("COMError: class not registered")
        return FakeSpVoice(self) if progid == "SAPI.SpVoice" else FakeSpFileStream(self)


class FakeSpVoice:
    def __init__(self, sapi: FakeSapi):
        self.sapi, self.Voice, self.Rate, self.AudioOutputStream = sapi, sapi.tokens[0], 0, None

    def GetVoices(self):
        self.sapi.threads.add(threading.get_ident())
        return iter(self.sapi.tokens)

    def Speak(self, text, flags=0):
        self.sapi.threads.add(threading.get_ident())
        stream = self.AudioOutputStream
        self.sapi.spoken.append({"text": text, "flags": flags, "voice": self.Voice.Id, "rate": self.Rate, "format": stream.Format.Type})
        if self.sapi.hang is not None:
            self.sapi.hang.wait(10)
        if not self.sapi.silent:
            stream.pcm += audio.read_wav(sine_wav(0.25, 22050))[0]


class FakeSpFileStream:
    def __init__(self, sapi: FakeSapi):
        self.sapi, self.Format, self.path, self.pcm = sapi, SimpleNamespace(Type=None), None, b""

    def Open(self, path, mode, do_events=False):
        assert mode == 3  # SSFMCreateForWrite
        self.path = path
        self.sapi.paths.append(path)
        Path(path).write_bytes(audio.wav_bytes(b"", 22050))  # just the header until Close

    def Close(self):
        Path(self.path).write_bytes(audio.wav_bytes(self.pcm, 22050))


def _fake_comtypes(monkeypatch, sapi: FakeSapi | None) -> None:
    """Install a fake comtypes (or, with None, make it look uninstalled)."""
    if sapi is None:
        monkeypatch.setitem(sys.modules, "comtypes", None)
        return
    import importlib.machinery

    package = types.ModuleType("comtypes")
    package.__spec__ = importlib.machinery.ModuleSpec("comtypes", None)
    package.CoInitialize = lambda: None  # type: ignore[attr-defined]
    client = types.ModuleType("comtypes.client")
    client.CreateObject = sapi.CreateObject  # type: ignore[attr-defined]
    package.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comtypes", package)
    monkeypatch.setitem(sys.modules, "comtypes.client", client)
    monkeypatch.setitem(sys.modules, "pythoncom", None)


def _powershell_speaks(monkeypatch, seconds: float = 0.2) -> FakeRun:
    monkeypatch.setattr(shutil, "which", lambda name: PS_EXE if name == "powershell" else None)

    def handler(cmd, kwargs):
        if "GetInstalledVoices" in _decode_ps(cmd):
            return 0, "Microsoft Zira Desktop\ten-US\tFemale\r\n".encode()
        assert "Speak($text)" in _decode_ps(cmd)
        out = Path(kwargs["env"]["DESTINY_TTS_OUT"])
        out.write_bytes(sine_wav(seconds, 22050) if seconds else audio.wav_bytes(b"", 22050))
        return 0, b""

    fake = FakeRun(handler)
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def test_system_windows_sapi_worker(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    sapi = FakeSapi()
    _fake_comtypes(monkeypatch, sapi)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", FakeRun(lambda cmd, kw: pytest.fail("PowerShell should not be used")))
    engine = SystemEngine()
    assert engine.is_available() == (True, "")
    voices = engine.list_voices()
    assert [(v.id, v.language, v.gender) for v in voices] == [
        ("Microsoft Zira Desktop", "en-US", "female"),
        ("Microsoft David Desktop", "en-US", "male"),  # "409;9": several languages, the first counts
        ("Microsoft Hedda Desktop", "de-DE", None),  # no Gender attribute
    ]
    assert engine.default_voice() == "Microsoft Zira Desktop"

    results = []
    threads = [threading.Thread(target=lambda: results.append(engine.synthesize("<5 cells", "Microsoft David Desktop", 1.5))) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 3 and all(wav_seconds(c.data) == pytest.approx(0.25, abs=0.01) for c in results)
    assert sapi.created.count("SAPI.SpVoice") == 1  # one SpVoice, reused
    assert len(sapi.threads) == 1 and threading.get_ident() not in sapi.threads  # all COM calls on one worker thread
    assert sapi.spoken[-1] == {"text": "<5 cells", "flags": 16, "voice": "TOKEN_DAVID", "rate": 4, "format": 22}
    assert all(not Path(p).exists() for p in sapi.paths)  # temp files cleaned up
    engine.synthesize("Hi", rate=1.0)
    assert sapi.spoken[-1]["voice"] == "TOKEN_ZIRA" and sapi.spoken[-1]["rate"] == 1


def test_system_windows_header_only_wav_falls_back_to_powershell(monkeypatch):
    # Seen on windows-latest: pyttsx3 left a 46-byte WAV with no audio, which was played as silence.
    monkeypatch.setattr(sys, "platform", "win32")
    sapi = FakeSapi(silent=True)
    _fake_comtypes(monkeypatch, sapi)
    fake = _powershell_speaks(monkeypatch, 0.3)
    engine = SystemEngine()
    clip = engine.synthesize("Hello there", "Microsoft Zira Desktop")
    assert wav_seconds(clip.data) == pytest.approx(0.3, abs=0.01) and len(fake.calls) == 1
    engine.synthesize("Hello again", "Microsoft Zira Desktop")
    assert len(sapi.spoken) == 1 and len(fake.calls) == 2  # SAPI is broken here: PowerShell from now on


def test_system_windows_unspeakable_text_keeps_sapi(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    sapi = FakeSapi(silent=True)
    _fake_comtypes(monkeypatch, sapi)
    fake = _powershell_speaks(monkeypatch, 0)  # PowerShell has nothing to say either
    engine = SystemEngine()
    clip = engine.synthesize("...", "Microsoft Zira Desktop")
    assert wav_seconds(clip.data) == pytest.approx(0.2, abs=0.01)  # a short pause
    engine.synthesize("...", "Microsoft Zira Desktop")
    assert len(sapi.spoken) == 2 and len(fake.calls) == 2
    with pytest.raises(TTSError, match="couldn't read this text"):  # words never become silence
        engine.synthesize("Hello there", "Microsoft Zira Desktop")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("SystemRoot", "/nonexistent")
    with pytest.raises(TTSError, match="couldn't read this text"):  # no PowerShell to ask
        engine.synthesize("Hello there", "Microsoft Zira Desktop")


def test_system_windows_sapi_hang_falls_back(monkeypatch):
    from destiny_bot.tts import system_engine

    monkeypatch.setattr(sys, "platform", "win32")
    release = threading.Event()
    sapi = FakeSapi(hang=release)
    _fake_comtypes(monkeypatch, sapi)
    fake = _powershell_speaks(monkeypatch, 0.2)
    monkeypatch.setattr(system_engine, "_synth_timeout", lambda text: 0.3)
    try:
        clip = SystemEngine().synthesize("Hello", "Microsoft Zira Desktop")
    finally:
        release.set()
    assert wav_seconds(clip.data) == pytest.approx(0.2, abs=0.01) and len(fake.calls) == 1
    # The hung SAPI call may still hold its file open (Windows locks it): PowerShell writes elsewhere.
    assert fake.calls[0][1]["env"]["DESTINY_TTS_OUT"] != sapi.paths[0]


def test_system_windows_sapi_failure_falls_back(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    _fake_comtypes(monkeypatch, FakeSapi(broken=True))
    fake = _powershell_speaks(monkeypatch, 0.2)
    clip = SystemEngine().synthesize("Hello", "Microsoft Zira Desktop")
    assert wav_seconds(clip.data) == pytest.approx(0.2, abs=0.01)
    assert ["Speak($text)" in _decode_ps(cmd) for cmd, _ in fake.calls] == [False, True]  # voice list, then speech


def test_system_windows_language_codes():
    from destiny_bot.tts.system_engine import _lcid_language

    assert _lcid_language("409") == "en-US" and _lcid_language("809;9") == "en-GB"
    assert _lcid_language("c0a") == "es-ES" and _lcid_language("") is None and _lcid_language("zz") is None


def test_system_windows_nothing_available(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    _fake_comtypes(monkeypatch, None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    ok, reason = SystemEngine().is_available()
    assert not ok and "Settings → Voices" in reason


# --------------------------------------------------------------------------- edge (mocked)


def _mp3_bytes(seconds: float = 0.3) -> bytes:
    enc = audio.Mp3Encoder(24000, 1, 48)
    return enc.encode(audio.silence_pcm(int(seconds * 1000), 24000)) + enc.flush()


class FakeCommunicate:
    instances: list[FakeCommunicate] = []
    error: BaseException | None = None

    def __init__(self, text, voice="en-US-AvaNeural", *, rate="+0%", **kwargs):
        if not voice.endswith("Neural"):
            raise ValueError(f"Invalid voice '{voice}'.")
        self.text, self.voice, self.rate = text, voice, rate
        FakeCommunicate.instances.append(self)

    async def stream(self):
        data = _mp3_bytes()
        yield {"type": "WordBoundary", "offset": 0, "duration": 1, "text": "x"}
        if FakeCommunicate.error is not None:
            raise FakeCommunicate.error
        yield {"type": "audio", "data": data[:500]}
        yield {"type": "audio", "data": data[500:]}


@pytest.fixture
def fake_edge(monkeypatch):
    import edge_tts

    FakeCommunicate.instances = []
    FakeCommunicate.error = None
    monkeypatch.setattr(edge_tts, "Communicate", FakeCommunicate)
    return FakeCommunicate


def test_edge_voices_are_static(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network used"))
    engine = EdgeEngine()
    voices = engine.list_voices()
    assert len(voices) == 10 and voices[0].id == "en-US-AvaNeural"
    assert {v.id for v in voices} >= {"en-GB-SoniaNeural", "en-AU-NatashaNeural", "en-US-AndrewNeural"}
    assert all(v.online and v.engine == "edge" for v in voices)
    assert engine.default_voice() == "en-US-AvaNeural"
    assert engine.online is True and engine.is_available() == (True, "")


def test_edge_synthesize(fake_edge):
    engine = EdgeEngine()
    clip = engine.synthesize("Hello world", rate=1.25)
    call = fake_edge.instances[-1]
    assert (call.text, call.voice, call.rate) == ("Hello world", "en-US-AvaNeural", "+25%")
    assert clip.mime == "audio/mpeg" and clip.data == _mp3_bytes() and clip.sample_rate == 24000
    engine.synthesize("Hi", "en-GB-RyanNeural", 0.5)
    assert (fake_edge.instances[-1].voice, fake_edge.instances[-1].rate) == ("en-GB-RyanNeural", "-50%")
    engine.synthesize("Hi")
    assert fake_edge.instances[-1].rate == "+0%"
    # Microsoft honours 0.5x-2x only.
    engine.synthesize("Hi", rate=3.0)
    assert fake_edge.instances[-1].rate == "+100%"
    engine.synthesize("Hi", rate=0.25)
    assert fake_edge.instances[-1].rate == "-50%"
    with pytest.raises(EngineUnavailable):
        engine.synthesize("Hi", "not-a-voice")


def test_edge_network_errors(fake_edge):
    import aiohttp
    from edge_tts.exceptions import NoAudioReceived, WebSocketError

    engine = EdgeEngine()
    for error in (aiohttp.ClientConnectionError("down"), OSError("no route"), WebSocketError("closed"), asyncio.TimeoutError()):
        fake_edge.error = error
        with pytest.raises(TTSError) as info:
            engine.synthesize("Hello")
        assert str(info.value) == NETWORK_ERROR
    fake_edge.error = NoAudioReceived("none")
    with pytest.raises(TTSError, match="no audio"):
        engine.synthesize("Hello")


def test_edge_inside_running_event_loop(fake_edge):
    async def main():
        return EdgeEngine().synthesize("From async code")

    clip = asyncio.run(main())
    assert clip.mime == "audio/mpeg" and fake_edge.instances[-1].text == "From async code"


# --------------------------------------------------------------------------- browser


def test_browser_engine():
    engine = BrowserEngine()
    assert engine.is_available() == (True, "")
    assert engine.list_voices() == [] and engine.default_voice() is None
    with pytest.raises(EngineUnavailable, match="speaks inside the web reader"):
        engine.synthesize("Hello")


# --------------------------------------------------------------------------- catalog + downloader


def test_catalog_entries():
    ids = [v.id for v in catalog.CATALOG]
    assert len(ids) == len(set(ids)) >= 12
    assert all(catalog.is_valid_voice_id(i) for i in ids)
    lessac = catalog.catalog_voice("en_US-lessac-medium")
    assert lessac is not None and lessac.recommended and 55 < lessac.size_mb < 70
    assert [v.id for v in catalog.CATALOG if v.recommended] == ["kokoro-v1.0", "en_US-lessac-medium"]
    kokoro = catalog.catalog_voice("kokoro-v1.0")
    assert kokoro is not None and kokoro.engine == "kokoro" and 115 < kokoro.size_mb < 125
    assert all(v.engine == "piper" for v in catalog.CATALOG if v.id != "kokoro-v1.0")
    assert {"es_ES-davefx-medium", "es_MX-ald-medium", "en_GB-cori-high"} <= set(ids)
    assert all(v.description and v.language.count("-") == 1 for v in catalog.CATALOG if v.engine == "piper")


def test_voice_urls_match_piper_downloader():
    from piper.download_voices import URL_FORMAT

    model, cfg = catalog.voice_urls("en_GB-northern_english_male-medium")
    args = dict(lang_family="en", lang_code="en_GB", voice_name="northern_english_male", voice_quality="medium")
    assert model == URL_FORMAT.format(extension=".onnx", **args)
    assert cfg == URL_FORMAT.format(extension=".onnx.json", **args)
    model, _ = catalog.voice_urls("en_US-amy-medium", "http://127.0.0.1:9/base/")
    assert model == "http://127.0.0.1:9/base/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true"


def test_installed_piper_voices(tmp_path):
    (tmp_path / "a-b-medium.onnx").write_bytes(b"x")
    (tmp_path / "a-b-medium.onnx.json").write_text("{}")
    (tmp_path / "no-json-medium.onnx").write_bytes(b"x")
    (tmp_path / "empty-x-low.onnx").write_bytes(b"")
    (tmp_path / "empty-x-low.onnx.json").write_text("{}")
    assert catalog.installed_piper_voices(tmp_path) == [tmp_path / "a-b-medium.onnx"]


class VoiceServer:
    """A local HTTP server laid out like the Hugging Face voice repository."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: list[str] = []
        outer = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)

            def do_GET(self):
                outer.requests.append(self.path)
                super().do_GET()

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def add_voice(self, voice_id: str, model: bytes, cfg: bytes) -> None:
        m = catalog.VOICE_ID_RE.match(voice_id)
        assert m
        folder = self.root / m["family"] / f"{m['family']}_{m['region']}" / m["name"] / m["quality"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{voice_id}.onnx").write_bytes(model)
        (folder / f"{voice_id}.onnx.json").write_bytes(cfg)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def voice_server(tmp_path):
    server = VoiceServer(tmp_path / "hf")
    server.root.mkdir()
    yield server
    server.close()


def _leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_download_success(voice_server):
    model = os.urandom(1_300_000)
    cfg = json.dumps({"audio": {"sample_rate": 22050}, "language": {"code": "en_US"}}).encode()
    voice_server.add_voice("en_US-amy-medium", model, cfg)
    events: list[tuple[float, str]] = []
    path = catalog.download_piper_voice(
        "en_US-amy-medium", base_url=voice_server.url, progress=lambda f, m: events.append((f, m))
    )
    folder = config.voices_dir() / "piper"
    assert path == folder / "en_US-amy-medium.onnx"
    assert path.read_bytes() == model
    assert (folder / "en_US-amy-medium.onnx.json").read_bytes() == cfg
    assert _leftovers(folder) == ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"]
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and len(events) >= 6
    assert any("Downloading Amy (US English): " in m and "of 1.3 MB" in m for _, m in events)
    assert voice_server.requests == [
        "/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true",
        "/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true",
    ]
    # Now installed: Piper sees it, and downloading again does not touch the network.
    assert [v.id for v in PiperEngine().list_voices()] == ["en_US-amy-medium"]
    assert catalog.download_piper_voice("en_US-amy-medium", base_url=voice_server.url) == path
    assert len(voice_server.requests) == 2


def test_download_404(voice_server):
    with pytest.raises(catalog.VoiceDownloadError, match="404"):
        catalog.download_piper_voice("en_US-joe-medium", base_url=voice_server.url)
    assert _leftovers(config.voices_dir() / "piper") == []


def test_download_cancel(voice_server):
    voice_server.add_voice("en_US-joe-medium", os.urandom(2_000_000), b"{}")
    cancel = threading.Event()

    def progress(fraction: float, message: str) -> None:
        if fraction > 0.05:
            cancel.set()

    with pytest.raises(catalog.DownloadCancelled):
        catalog.download_piper_voice("en_US-joe-medium", base_url=voice_server.url, cancel=cancel, progress=progress)
    assert _leftovers(config.voices_dir() / "piper") == []


def test_download_rejects_bad_ids(voice_server):
    for bad in (
        "../../etc/passwd",
        "en_US-../x-medium",
        "en_US-lessac-medium/../../x",
        "en_US-les.sac-medium",
        "en_US-lessac-medium.onnx",
        "en_US-lessac-ultra",
        "EN_us-lessac-medium",
        "",
    ):
        with pytest.raises(ValueError):
            catalog.download_piper_voice(bad, base_url=voice_server.url)
    assert voice_server.requests == []


def test_download_damaged_config(voice_server):
    voice_server.add_voice("en_US-kristin-medium", b"model", b"<html>not json</html>")
    with pytest.raises(catalog.VoiceDownloadError, match="damaged"):
        catalog.download_piper_voice("en_US-kristin-medium", base_url=voice_server.url)
    assert _leftovers(config.voices_dir() / "piper") == []


def test_download_network_failure():
    import socket

    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(catalog.VoiceDownloadError) as info:
        catalog.download_piper_voice("en_US-amy-medium", base_url=f"http://127.0.0.1:{port}")
    assert str(info.value) == "Couldn't download the voice. Check your internet connection."


def _locked_for(monkeypatch, times: int) -> list[str]:
    """os.replace fails like Windows does while a virus scanner holds the new file."""
    real_replace, attempts = os.replace, []

    def replace(src, dst):
        attempts.append(Path(dst).name)
        if attempts.count(Path(dst).name) <= times:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(catalog.time, "sleep", lambda seconds: None)
    return attempts


def test_download_waits_for_the_virus_scanner(voice_server, monkeypatch):
    voice_server.add_voice("en_US-amy-medium", b"model bytes", b"{}")
    monkeypatch.setattr(catalog, "sys", SimpleNamespace(platform="win32"))
    attempts = _locked_for(monkeypatch, times=3)
    path = catalog.download_piper_voice("en_US-amy-medium", base_url=voice_server.url)
    assert path.read_bytes() == b"model bytes" and attempts.count("en_US-amy-medium.onnx") == 4
    assert _leftovers(path.parent) == ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"]


def test_download_gives_up_on_a_file_that_stays_locked(voice_server, monkeypatch):
    voice_server.add_voice("en_US-amy-medium", b"model bytes", b"{}")
    monkeypatch.setattr(catalog, "sys", SimpleNamespace(platform="win32"))
    _locked_for(monkeypatch, times=100)
    with pytest.raises(catalog.VoiceDownloadError, match="Couldn't save the voice"):
        catalog.download_piper_voice("en_US-amy-medium", base_url=voice_server.url)
    assert _leftovers(config.voices_dir() / "piper") == []


def test_download_tls_uses_certifi_when_python_has_no_certificates(monkeypatch):
    # python.org's macOS Python starts with an empty certificate store.
    import ssl

    certifi = pytest.importorskip("certifi")
    monkeypatch.setattr(ssl, "create_default_context", lambda: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    context = catalog._ssl_context()
    assert context.cert_store_stats()["x509_ca"] > 0 and context.verify_mode == ssl.CERT_REQUIRED
    assert Path(certifi.where()).is_file()


# --------------------------------------------------------------------------- bundled voice folders


def test_extra_voice_dirs_for_piper(tmp_path, monkeypatch):
    bundle_a, bundle_b = tmp_path / "bundle-a", tmp_path / "bundle-b"
    for bundle, vid in ((bundle_a, "en_US-amy-medium"), (bundle_b, "en_US-joe-medium"), (bundle_b, "en_US-lessac-medium")):
        folder = bundle / "piper"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{vid}.onnx").write_bytes(b"bundled")
        (folder / f"{vid}.onnx.json").write_text('{"language": {"code": "en_US"}}')
    monkeypatch.setenv(catalog.EXTRA_DIRS_ENV, os.pathsep.join([str(bundle_a), "", str(tmp_path / "missing"), str(bundle_b)]))
    assert catalog.extra_voice_dirs() == [bundle_a, tmp_path / "missing", bundle_b]
    downloaded = install_fake_voice("en_US-amy-medium")  # same id as a bundled voice: the download wins
    found = catalog.installed_piper_voices()
    assert found == [downloaded, bundle_b / "piper" / "en_US-joe-medium.onnx", bundle_b / "piper" / "en_US-lessac-medium.onnx"]
    assert catalog.installed_piper_voices(piper_dir()) == [downloaded]  # an explicit folder is searched alone
    engine = PiperEngine()
    assert engine.is_available() == (True, "") and engine.default_voice() == "en_US-lessac-medium"
    assert [v.id for v in engine.list_voices()] == ["en_US-lessac-medium", "en_US-amy-medium", "en_US-joe-medium"]
    assert catalog.installed_catalog_ids() == {"en_US-amy-medium", "en_US-joe-medium", "en_US-lessac-medium"}
    # Bundled voices count as installed: no download needed (and no network touched).
    assert catalog.download_piper_voice("en_US-joe-medium", base_url="http://127.0.0.1:9") == bundle_b / "piper" / "en_US-joe-medium.onnx"
    # The read-only bundle is never written to.
    assert sorted(p.name for p in (bundle_b / "piper").iterdir()) == [
        "en_US-joe-medium.onnx", "en_US-joe-medium.onnx.json", "en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"]  # fmt: skip


def test_extra_voice_dirs_for_kokoro(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundled_model, bundled_table = write_fake_kokoro(bundle / "kokoro")
    monkeypatch.setenv(catalog.EXTRA_DIRS_ENV, str(bundle))
    assert catalog.find_kokoro_model() == (bundled_model, bundled_table)
    assert "kokoro-v1.0" in catalog.installed_catalog_ids()
    assert KokoroEngine().is_available() == (True, "")
    # A downloaded model wins over the bundled one, whatever its variant.
    user_model, user_table = write_fake_kokoro(config.voices_dir() / "kokoro", variant="fp32")
    assert catalog.find_kokoro_model() == (user_model, user_table)
    # Inside one folder the default (int8) variant is preferred unless one is asked for.
    int8_model, _ = write_fake_kokoro(config.voices_dir() / "kokoro", variant="int8")
    assert catalog.find_kokoro_model() == (int8_model, user_table)
    assert catalog.find_kokoro_model(variant="fp32") == (user_model, user_table)
    # A model without its voice table doesn't count.
    user_table.unlink()
    assert catalog.find_kokoro_model() == (bundled_model, bundled_table)
    assert catalog.download_kokoro_model(base_url="http://127.0.0.1:9") == bundled_model  # already there


# --------------------------------------------------------------------------- kokoro downloader


@pytest.fixture
def kokoro_release(voice_server, monkeypatch):
    """Fake release files on the local server, with KOKORO_FILES pointing at their hashes."""
    contents = {"voices": os.urandom(300_000), "int8": os.urandom(700_000), "fp32": os.urandom(1_100_000)}
    files = {}
    for key, data in contents.items():
        name = catalog.KOKORO_FILES[key].name
        (voice_server.root / name).write_bytes(data)
        files[key] = catalog.KokoroFile(name, len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(catalog, "KOKORO_FILES", files)
    return contents


def test_kokoro_download_success(voice_server, kokoro_release):
    events: list[tuple[float, str]] = []
    model = catalog.download_kokoro_model(base_url=voice_server.url, progress=lambda f, m: events.append((f, m)))
    folder = config.voices_dir() / "kokoro"
    assert model == folder / "kokoro-v1.0.int8.onnx" and model.read_bytes() == kokoro_release["int8"]
    assert (folder / "voices-v1.0.bin").read_bytes() == kokoro_release["voices"]
    assert _leftovers(folder) == ["kokoro-v1.0.int8.onnx", "voices-v1.0.bin"]
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any(m.startswith("Downloading Kokoro voice pack: ") and m.endswith("of 1.0 MB") for _, m in events)
    assert voice_server.requests == ["/voices-v1.0.bin", "/kokoro-v1.0.int8.onnx"]
    assert catalog.find_kokoro_model() == (model, folder / "voices-v1.0.bin")
    # Installed now: nothing is fetched again.
    assert catalog.download_kokoro_model(base_url=voice_server.url) == model
    # Adding the fp32 model reuses the voice table.
    fp32 = catalog.download_kokoro_model(base_url=voice_server.url, variant="fp32")
    assert fp32.read_bytes() == kokoro_release["fp32"] and voice_server.requests[2:] == ["/kokoro-v1.0.onnx"]
    with pytest.raises(ValueError):
        catalog.download_kokoro_model(base_url=voice_server.url, variant="fp64")


def test_kokoro_download_through_catalog_flow(voice_server, kokoro_release):
    # The UI's catalog flow calls download_piper_voice with the catalog id.
    assert catalog.is_valid_voice_id("kokoro-v1.0")
    path = catalog.download_piper_voice("kokoro-v1.0", base_url=voice_server.url)
    assert path == config.voices_dir() / "kokoro" / "kokoro-v1.0.int8.onnx"
    assert "kokoro-v1.0" in catalog.installed_catalog_ids()
    assert catalog.download_voice("kokoro-v1.0", base_url=voice_server.url) == path
    # With an explicit Piper folder, Kokoro goes to the sibling "kokoro" folder.
    other = config.data_dir() / "elsewhere" / "piper"
    path2 = catalog.download_piper_voice("kokoro-v1.0", voices_dir=other, base_url=voice_server.url)
    assert path2 == other.parent / "kokoro" / "kokoro-v1.0.int8.onnx"


def test_kokoro_download_rejects_damaged_file(voice_server, kokoro_release, monkeypatch):
    bad = dict(catalog.KOKORO_FILES)
    bad["int8"] = catalog.KokoroFile(bad["int8"].name, bad["int8"].size, "0" * 64)
    monkeypatch.setattr(catalog, "KOKORO_FILES", bad)
    with pytest.raises(catalog.VoiceDownloadError, match="damaged"):
        catalog.download_kokoro_model(base_url=voice_server.url)
    assert _leftovers(config.voices_dir() / "kokoro") == []


def test_kokoro_download_404_and_cancel(voice_server, kokoro_release):
    (voice_server.root / "kokoro-v1.0.int8.onnx").unlink()
    with pytest.raises(catalog.VoiceDownloadError, match="404"):
        catalog.download_kokoro_model(base_url=voice_server.url)
    assert _leftovers(config.voices_dir() / "kokoro") == []
    cancel = threading.Event()

    def progress(fraction: float, message: str) -> None:
        if fraction > 0.1:
            cancel.set()

    with pytest.raises(catalog.DownloadCancelled):
        catalog.download_kokoro_model(base_url=voice_server.url, variant="fp32", cancel=cancel, progress=progress)
    assert _leftovers(config.voices_dir() / "kokoro") == []


# --------------------------------------------------------------------------- cache


def _clip(seconds: float = 0.2, freq: float = 440.0) -> AudioClip:
    return AudioClip(sine_wav(seconds, 16000, freq), "audio/wav", 16000)


def test_cache_hit_and_miss(tmp_path):
    cache = AudioCache(tmp_path / "cache")
    calls = []
    clip = _clip()

    def make():
        calls.append(1)
        return clip

    key = ("piper", "en_US-lessac-medium", "1.0", "Hello")
    first = cache.get_or_create(key, make)
    second = cache.get_or_create(key, make)
    assert first == clip and second.data == clip.data and second.sample_rate == 16000 and second.mime == "audio/wav"
    assert len(calls) == 1
    digest = hashlib.sha256("\x1f".join(key).encode()).hexdigest()
    assert [p.name for p in (tmp_path / "cache").iterdir()] == [f"{digest}.wav"]
    # A different key part is a different entry.
    cache.get_or_create(("piper", "en_US-lessac-medium", "1.0", "Hello!"), make)
    assert len(calls) == 2


def test_cache_mp3_roundtrip(tmp_path):
    tmp_path = tmp_path / "cache"
    cache = AudioCache(tmp_path)
    mp3 = AudioClip(_mp3_bytes(), "audio/mpeg", None)
    cache.get_or_create(("edge", "x"), lambda: mp3)
    hit = cache.get_or_create(("edge", "x"), lambda: pytest.fail("should be cached"))
    assert hit.mime == "audio/mpeg" and hit.data == mp3.data and hit.sample_rate == 24000
    assert next(tmp_path.iterdir()).suffix == ".mp3"


@pytest.mark.parametrize("damage", [b"", b"garbage that is not audio", b"RIFF\x00\x00\x00\x00WAVE"])
def test_cache_corrupt_files_are_misses(tmp_path, damage):
    tmp_path = tmp_path / "cache"
    cache = AudioCache(tmp_path)
    key = ("k",)
    cache.get_or_create(key, _clip)
    path = tmp_path / f"{AudioCache.key_for(key)}.wav"
    path.write_bytes(damage)
    fresh = _clip(0.3)
    assert cache.get_or_create(key, lambda: fresh).data == fresh.data
    assert path.read_bytes() == fresh.data


def test_cache_lru_eviction(tmp_path):
    tmp_path = tmp_path / "cache"
    clip_size = len(_clip().data)
    cache = AudioCache(tmp_path, max_bytes=int(clip_size * 3.5))
    now = time.time()
    for i in range(3):
        cache.get_or_create((f"k{i}",), lambda i=i: _clip(freq=300 + i))
        os.utime(tmp_path / f"{AudioCache.key_for((f'k{i}',))}.wav", (now - 100 + i, now - 100 + i))
    # Touch k0 (a hit refreshes its time), then add two more clips.
    cache.get_or_create(("k0",), lambda: pytest.fail("cached"))
    cache.get_or_create(("k3",), _clip)
    cache.get_or_create(("k4",), _clip)
    left = {p.name for p in tmp_path.iterdir()}
    names = {f"k{i}": f"{AudioCache.key_for((f'k{i}',))}.wav" for i in range(5)}
    assert names["k0"] in left and names["k4"] in left
    assert names["k1"] not in left  # least recently used goes first
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= cache.max_bytes


def test_cache_concurrent_same_key(tmp_path):
    cache = AudioCache(tmp_path / "cache")
    calls = []

    def make():
        calls.append(1)
        time.sleep(0.05)
        return _clip()

    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get_or_create(("same",), make))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(results) == 8 and len({r.data for r in results}) == 1
    assert cache._key_locks == {}


def test_cache_clear_and_errors(tmp_path):
    tmp_path = tmp_path / "cache"
    cache = AudioCache(tmp_path)
    cache.get_or_create(("a",), _clip)
    cache.get_or_create(("b",), lambda: AudioClip(_mp3_bytes(), "audio/mpeg"))
    (tmp_path / "keep-me.txt").write_text("not ours")
    cache.clear()
    assert [p.name for p in tmp_path.iterdir()] == ["keep-me.txt"]
    with pytest.raises(TTSError):
        cache.get_or_create(("c",), lambda: (_ for _ in ()).throw(TTSError("boom")))
    assert cache.get(("c",)) is None
    shutil.rmtree(tmp_path)  # folder wiped while the app runs: caching still works
    cache.get_or_create(("d",), _clip)
    assert cache.get(("d",)) is not None
