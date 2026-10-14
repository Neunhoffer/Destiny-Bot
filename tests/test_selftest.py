"""Tests for the self-test (destiny_bot.selftest)."""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import threading
import wave

import pytest

from destiny_bot import config, desktop, selftest
from destiny_bot.tts.base import AudioClip, EngineUnavailable, TTSEngine, Voice


@pytest.fixture(autouse=True)
def no_bundled_voices(monkeypatch):
    """Self-tests here run as from a source checkout without bundled voices."""
    monkeypatch.setenv("DESTINY_EXTRA_VOICE_DIRS", "")
    monkeypatch.setattr(desktop, "bundled_voices_dir", lambda: None)


def make_wav(seconds: float, *, amplitude: int = 8000, rate: int = 22050) -> bytes:
    frames = int(seconds * rate)
    samples = (int(amplitude * math.sin(2 * math.pi * 220 * i / rate)) for i in range(frames))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return buf.getvalue()


class FakeEngine(TTSEngine):
    label = "Fake voice"
    online = False

    def __init__(self, name: str = "piper", *, available: bool = True, audio: bytes | None = None,
                 error: Exception | None = None, online: bool = False, hang: threading.Event | None = None):  # fmt: skip
        self.name = name  # type: ignore[misc]
        self.online = online  # type: ignore[misc]
        self.available = available
        self.audio = audio if audio is not None else make_wav(1.5)
        self.error = error
        self.hang = hang
        self.spoken: list[str] = []

    def is_available(self):
        return (True, "") if self.available else (False, "Download a voice in Settings.")

    def list_voices(self):
        return [Voice(id="v1", name="Voice One", engine=self.name, language="en-US")]

    def synthesize(self, text, voice=None, rate=1.0):
        if self.hang is not None:
            self.hang.wait(5)
        if self.error is not None:
            raise self.error
        self.spoken.append(text)
        return AudioClip(self.audio, "audio/wav", 22050)


def run_checks(monkeypatch, engines, quick=False):
    from destiny_bot import tts

    monkeypatch.setattr(tts, "all_engines", lambda: engines)
    report = selftest.Report(io.StringIO())
    selftest._check_engines(report, quick)
    return report


# -- audio validation


def test_wav_stats_of_speech_like_audio():
    stats = selftest._wav_stats(AudioClip(make_wav(1.0), "audio/wav", 22050))
    assert stats["duration_s"] == pytest.approx(1.0, abs=0.01)
    assert stats["sample_rate"] == 22050 and stats["channels"] == 1
    assert stats["peak"] > 7000
    assert -20 < stats["rms_dbfs"] < -5
    assert selftest._valid_speech(stats) == ""


def test_silent_and_short_audio_is_rejected():
    silent = selftest._wav_stats(AudioClip(make_wav(1.0, amplitude=0), "audio/wav"))
    assert "silent" in selftest._valid_speech(silent)
    short = selftest._wav_stats(AudioClip(make_wav(0.1), "audio/wav"))
    assert "only 0.10 s" in selftest._valid_speech(short)


def test_mp3_is_only_checked_for_size():
    assert selftest._valid_speech(selftest._wav_stats(AudioClip(b"\xff\xfb" * 4000, "audio/mpeg"))) == ""


# -- engines


def test_benchmark_measures_cold_and_warm_runs():
    engine = FakeEngine()
    result = selftest._benchmark(engine, "v1", quick=False)
    assert "error" not in result
    assert result["cold_s"] >= 0 and result["audio"]["duration_s"] == pytest.approx(1.5, abs=0.01)
    assert result["cold_rtf"] == pytest.approx(result["cold_s"] / 1.5, abs=0.01)
    assert len(result["warm_runs"]) == 2 and "warm_rtf" in result
    assert engine.spoken == [selftest.TEST_SENTENCE, *selftest.WARM_SENTENCES]


def test_quick_benchmark_speaks_once():
    engine = FakeEngine()
    result = selftest._benchmark(engine, "v1", quick=True)
    assert "warm_s" not in result and len(engine.spoken) == 1


def test_benchmark_reports_errors_and_timeouts(monkeypatch):
    broken = selftest._benchmark(FakeEngine(error=EngineUnavailable("No voice installed.")), None, quick=True)
    assert broken["error"] == "EngineUnavailable: No voice installed."
    monkeypatch.setattr(selftest, "COLD_TIMEOUT_S", 0.2)
    release = threading.Event()
    hung = selftest._benchmark(FakeEngine(hang=release), None, quick=True)
    release.set()
    assert hung["error"].startswith("TimeoutError")


def test_missing_optional_engine_is_a_warning(monkeypatch):
    report = run_checks(monkeypatch, [FakeEngine("piper", available=False), FakeEngine("system")])
    statuses = {(c.section, c.name): c.status for c in report.checks}
    assert statuses[("engines", "Fake voice")] in (selftest.WARN, selftest.PASS)
    assert report.count(selftest.FAIL) == 0
    assert any(c.name == "system speaks" and c.status == selftest.PASS for c in report.checks)
    assert not any(c.name == "piper speaks" for c in report.checks)


def test_bundled_engine_must_speak(monkeypatch):
    monkeypatch.setattr(selftest, "_bundled_engines", lambda: {"piper"})
    report = run_checks(monkeypatch, [FakeEngine("piper", error=RuntimeError("onnxruntime DLL failed to load"))])
    failed = [c for c in report.checks if c.status == selftest.FAIL]
    assert [c.name for c in failed] == ["piper speaks"]
    assert "onnxruntime DLL failed to load" in failed[0].detail


def test_bundled_engine_that_is_unavailable_fails(monkeypatch):
    monkeypatch.setattr(selftest, "_bundled_engines", lambda: {"kokoro"})
    report = run_checks(monkeypatch, [FakeEngine("kokoro", available=False)])
    assert report.count(selftest.FAIL) == 1


def test_online_and_browser_engines_are_not_synthesized(monkeypatch):
    online = FakeEngine("edge", online=True, error=AssertionError("must not be called"))
    browser = FakeEngine("browser", error=AssertionError("must not be called"))
    report = run_checks(monkeypatch, [online, browser, FakeEngine("system")])
    assert report.count(selftest.FAIL) == 0
    assert any(c.name == "edge speaks" and c.status == selftest.SKIP for c in report.checks)
    assert not any(c.name == "browser speaks" for c in report.checks)


def test_no_working_offline_voice_is_a_warning(monkeypatch):
    report = run_checks(monkeypatch, [FakeEngine("system", audio=make_wav(1.0, amplitude=0))])
    names = {c.name: c.status for c in report.checks}
    assert names["system speaks"] == selftest.WARN
    assert names["an offline voice"] == selftest.WARN
    assert report.count(selftest.FAIL) == 0


def test_bundled_engines_detection(tmp_path, monkeypatch):
    (tmp_path / "piper").mkdir()
    (tmp_path / "piper" / "en_US-lessac-medium.onnx").write_bytes(b"x")
    (tmp_path / "kokoro").mkdir()
    monkeypatch.setattr(desktop, "bundled_voices_dir", lambda: tmp_path)
    assert selftest._bundled_engines() == {"piper"}


# -- whole runs


def _skip_slow_steps(monkeypatch):
    for name in (
        "_check_window",
        "_check_engines",
        "_check_windows_voices",
        "_check_ingest",
        "_check_server",
        "_check_ocr",
    ):
        monkeypatch.setattr(selftest, name, lambda *args, **kwargs: None)


def test_missing_core_module_fails_the_self_test(monkeypatch):
    _skip_slow_steps(monkeypatch)
    monkeypatch.setattr(selftest, "CORE_MODULES", ("destiny_bot.models", "destiny_bot.not_a_module"))
    out = io.StringIO()
    assert selftest.run_self_test(quick=True, out=out) == 1
    json_path, text_path = selftest.report_paths()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["ok"] is False and report["summary"]["fail"] == 1
    failed = [c for c in report["checks"] if c["status"] == "fail"]
    assert failed[0]["name"] == "destiny_bot.not_a_module"
    assert "[FAIL] destiny_bot.not_a_module" in out.getvalue()
    assert "problem(s) found" in text_path.read_text(encoding="utf-8")


def test_a_crashing_check_is_a_failure_not_a_crash(monkeypatch):
    _skip_slow_steps(monkeypatch)

    def explode(report):
        raise ValueError("boom")

    monkeypatch.setattr(selftest, "_check_ingest", explode)
    assert selftest.run_self_test(quick=True, out=io.StringIO()) == 1


def test_missing_optional_module_is_a_warning(monkeypatch):
    _skip_slow_steps(monkeypatch)
    monkeypatch.setattr(selftest, "CORE_MODULES", ("destiny_bot.models",))
    monkeypatch.setattr(
        selftest, "OPTIONAL_MODULES", {"not_installed_extra": "something nice", "destiny_bot.gone": "x"}
    )
    out = io.StringIO()
    assert selftest.run_self_test(out=out) == 0
    text = out.getvalue()
    assert "[WARN] not_installed_extra" in text and "needed for something nice" in text
    assert "[SKIP] destiny_bot.gone" in text


def test_output_survives_a_missing_console(monkeypatch):
    _skip_slow_steps(monkeypatch)
    monkeypatch.setattr(selftest, "CORE_MODULES", ("destiny_bot.models",))
    monkeypatch.setattr(sys, "stdout", None)
    assert selftest.run_self_test(quick=True) == 0
    assert selftest.report_paths()[0].is_file()


def test_real_quick_self_test():
    """The real thing: imports, a voice, PDF + Word files, the reader's server, OCR."""
    out = io.StringIO()
    code = selftest.run_self_test(quick=True, out=out)
    text = out.getvalue()
    assert code == 0, text
    report = json.loads(selftest.report_paths()[0].read_text(encoding="utf-8"))
    assert report["app"] == "Athena" and report["ok"] is True and report["quick"] is True
    assert report["environment"]["python"]["version"] == ".".join(map(str, sys.version_info[:3]))
    assert report["environment"]["data_dir"] == str(config.data_dir())
    names = {c["name"]: c["status"] for c in report["checks"]}
    assert names["PDF file"] == "pass" and names["Word file"] == "pass"
    assert names["GET /api/status"] == "pass"
    assert {e["name"] for e in report["engines"]} >= {"piper", "system", "edge", "browser"}
    assert report["server"]["resolved_engine"]
    assert text.startswith("Athena self-test, version")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows voice registry")
def test_windows_voice_registry():
    voices = selftest._registry_voices()
    assert set(voices) == {"sapi5", "sapi5_32bit", "sapi5_user", "onecore"}
    for entries in voices.values():
        for entry in entries:
            assert entry["token"]
