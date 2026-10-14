"""Audio helpers (WAV/MP3, joining, playback) and audiobook export."""

from __future__ import annotations

import io
import json
import math
import shutil
import struct
import subprocess
import sys
import threading
import types
import wave
from pathlib import Path

import pytest

from destiny_bot import audio, export
from destiny_bot.models import Block, BlockKind, Document, ReadingOptions, Section, Utterance
from destiny_bot.tts.base import AudioClip, EngineUnavailable, TTSEngine, TTSError, Voice

FFPROBE = shutil.which("ffprobe")
needs_ffprobe = pytest.mark.skipif(FFPROBE is None, reason="ffprobe not installed")


# --------------------------------------------------------------------------- helpers


def sine_pcm(seconds: float, rate: int, freq: float = 440.0, channels: int = 1, amp: int = 8000) -> bytes:
    out = bytearray()
    for i in range(int(seconds * rate)):
        out += struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / rate))) * channels
    return bytes(out)


def wav_clip(seconds: float = 0.2, rate: int = 22050, freq: float = 440.0, channels: int = 1) -> AudioClip:
    return AudioClip(audio.wav_bytes(sine_pcm(seconds, rate, freq, channels), rate, channels), "audio/wav", rate)


def mp3_clip(seconds: float = 0.3, rate: int = 24000, freq: float = 300.0) -> AudioClip:
    enc = audio.Mp3Encoder(rate, 1, 48)
    return AudioClip(enc.encode(sine_pcm(seconds, rate, freq)) + enc.flush(), "audio/mpeg", rate)


def seconds_of(clip: AudioClip) -> float:
    return audio.clip_duration(clip)


def zero_crossings(pcm16: bytes) -> int:
    samples = struct.unpack(f"<{len(pcm16) // 2}h", pcm16)
    return sum(1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0))


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    return json.loads(out)


def parse_id3(data: bytes) -> dict[str, str]:
    assert data[:5] == b"ID3\x03\x00"
    size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
    body, frames, pos = data[10 : 10 + size], {}, 0
    while pos + 10 <= len(body):
        fid, fsize = body[pos : pos + 4].decode(), struct.unpack(">I", body[pos + 4 : pos + 8])[0]
        payload = body[pos + 10 : pos + 10 + fsize]
        frames[fid] = payload[1:].decode("latin-1" if payload[0] == 0 else "utf-16")
        pos += 10 + fsize
    return frames


# --------------------------------------------------------------------------- WAV basics


def test_wav_roundtrip_and_silence():
    pcm = sine_pcm(0.1, 16000)
    data = audio.wav_bytes(pcm, 16000)
    assert audio.read_wav(data) == (pcm, 16000, 1, 2)
    assert audio.wav_format(data) == (16000, 1, 2)
    assert audio.silence_pcm(250, 16000) == b"\x00" * 8000
    assert audio.silence_pcm(10, 8000, channels=2, sampwidth=1) == b"\x80" * 160
    assert audio.silence_pcm(0, 22050) == b""
    assert audio.read_wav(audio.wav_bytes(pcm + b"\x01", 16000))[0] == pcm  # partial frame dropped


def test_read_wav_tolerates_streamed_header():
    pcm = sine_pcm(0.2, 22050)
    data = bytearray(audio.wav_bytes(pcm, 22050))
    data[4:8] = struct.pack("<I", 0x7FFFFFFF)  # what espeak writes to a pipe
    data[40:44] = struct.pack("<I", 0x7FFFFFFF)
    assert audio.read_wav(bytes(data)) == (pcm, 22050, 1, 2)


def test_read_wav_rejects_garbage_and_handles_extensible():
    for bad in (b"", b"hello", b"RIFF\x04\x00\x00\x00WAVE", b"ID3\x03\x00" + b"\x00" * 40):
        with pytest.raises(ValueError):
            audio.read_wav(bad)
    pcm = sine_pcm(0.05, 16000)
    fmt = struct.pack("<HHIIHHHHIH14s", 0xFFFE, 1, 16000, 32000, 2, 16, 22, 16, 4, 1, b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71")
    riff = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    data = b"RIFF" + struct.pack("<I", len(riff)) + riff
    assert audio._read_wav_fallback(data) == (pcm, 16000, 1, 2)
    assert audio.read_wav(data) == (pcm, 16000, 1, 2)


def test_sample_width_conversion():
    pcm16 = struct.pack("<4h", 0, 256, -256, 32767)
    pcm8 = bytes([128, 129, 127, 255])
    assert audio.to_int16(pcm8, 1) == struct.pack("<4h", 0, 256, -256, 32512)
    pcm24 = b"".join(b"\x55" + struct.pack("<h", v) for v in (0, 256, -256, 32767))
    assert audio.to_int16(pcm24, 3) == pcm16
    pcm32 = b"".join(b"\x11\x22" + struct.pack("<h", v) for v in (0, 256, -256, 32767))
    assert audio.to_int16(pcm32, 4) == pcm16


# --------------------------------------------------------------------------- joining WAV


def test_concat_same_format_is_exact():
    a, b = wav_clip(0.2), wav_clip(0.3, freq=880)
    joined = audio.concat_clips([(a, 500), (b, 0)])
    pcm, rate, channels, width = audio.read_wav(joined.data)
    assert (joined.mime, joined.sample_rate, rate, channels, width) == ("audio/wav", 22050, 22050, 1, 2)
    a_pcm, b_pcm = audio.read_wav(a.data)[0], audio.read_wav(b.data)[0]
    assert pcm == a_pcm + b"\x00" * (2 * 11025) + b_pcm


@pytest.mark.parametrize("use_numpy", [True, False])
def test_concat_resamples_and_downmixes(monkeypatch, use_numpy):
    if not use_numpy:
        monkeypatch.setattr(audio, "_np", lambda: None)
    low = wav_clip(0.5, 16000, 440)
    stereo = wav_clip(0.5, 22050, 440, channels=2)
    joined = audio.concat_clips([(low, 100), (stereo, 0)])
    pcm, rate, channels, width = audio.read_wav(joined.data)
    assert (rate, channels, width) == (22050, 1, 2)  # highest rate; mono because channel counts differ
    assert seconds_of(joined) == pytest.approx(1.1, abs=0.002)
    first = pcm[: 2 * 11025]
    # 440 Hz survives resampling: ~2 crossings per cycle over 0.5 s.
    assert zero_crossings(first) == pytest.approx(440, abs=6)
    second = pcm[-2 * 11025 :]
    assert zero_crossings(second) == pytest.approx(440, abs=6)
    assert max(abs(s) for s in struct.unpack(f"<{len(second) // 2}h", second)) > 7000


def test_resample_numpy_matches_pure_python(monkeypatch):
    pcm = sine_pcm(0.1, 16000, 300)
    fast = audio.resample(pcm, 1, 16000, 22050)
    monkeypatch.setattr(audio, "_np", lambda: None)
    slow = audio.resample(pcm, 1, 16000, 22050)
    assert len(fast) == len(slow)
    a = struct.unpack(f"<{len(fast) // 2}h", fast)
    b = struct.unpack(f"<{len(slow) // 2}h", slow)
    assert max(abs(x - y) for x, y in zip(a, b)) <= 1
    stereo = sine_pcm(0.05, 16000, 300, channels=2)
    assert audio.change_channels(stereo, 2, 1) == audio.change_channels(stereo, 2, 1)
    assert audio.change_channels(audio.change_channels(pcm, 1, 2), 2, 1) == pcm


def test_concat_rejects_empty():
    with pytest.raises(ValueError):
        audio.concat_clips([])


# --------------------------------------------------------------------------- MP3


def test_encode_mp3_frames_and_duration(tmp_path):
    wav = audio.wav_bytes(sine_pcm(2.0, 22050), 22050)
    mp3 = audio.encode_mp3(wav)
    assert mp3[0] == 0xFF and (mp3[1] & 0xE0) == 0xE0  # MPEG frame sync
    info = audio.mp3_info(mp3)
    assert (info.sample_rate, info.channels, info.bitrate_kbps) == (22050, 1, 64)
    assert info.duration == pytest.approx(2.0, abs=0.12)
    stereo = audio.encode_mp3(audio.wav_bytes(sine_pcm(0.5, 44100, channels=2), 44100, 2), bitrate_kbps=128)
    assert audio.mp3_info(stereo).channels == 2 and audio.mp3_info(stereo).sample_rate == 44100
    odd = audio.mp3_info(audio.encode_mp3(audio.wav_bytes(sine_pcm(0.5, 11000), 11000)))
    assert odd.sample_rate == 11025 and odd.duration == pytest.approx(0.5, abs=0.15)  # 52 ms frames + encoder delay
    eight_bit = audio.wav_bytes(bytes([128 + int(60 * math.sin(i / 5)) for i in range(8000)]), 8000, 1, 1)
    assert audio.mp3_info(audio.encode_mp3(eight_bit)).sample_rate == 8000
    if FFPROBE:
        path = tmp_path / "x.mp3"
        path.write_bytes(mp3)
        probe = ffprobe(path)
        assert probe["streams"][0]["codec_name"] == "mp3"
        assert float(probe["format"]["duration"]) == pytest.approx(2.0, abs=0.12)


def test_mp3_info_rejects_garbage():
    for bad in (b"", b"\xff\xff\xff\xff", b"not an mp3" * 100):
        with pytest.raises(ValueError):
            audio.mp3_info(bad)


def test_mp3_silence_keeps_rate():
    for rate in (16000, 22050, 24000, 44100):
        info = audio.mp3_info(audio.mp3_silence(800, rate, 1, 48))
        assert info.sample_rate == rate and info.duration == pytest.approx(0.8, abs=0.15)
    assert audio.mp3_silence(0, 24000) == b""


def test_concat_mp3_clips(tmp_path):
    a, b = mp3_clip(0.5), mp3_clip(0.4, freq=500)
    tagged = AudioClip(b"ID3\x03\x00\x00\x00\x00\x00\x05" + b"\x00" * 5 + b.data + b"TAG" + b"\x00" * 125, "audio/mpeg")
    joined = audio.concat_clips([(a, 300), (tagged, 200)])
    assert joined.mime == "audio/mpeg" and joined.sample_rate == 24000
    assert b"ID3" not in joined.data and b"TAG\x00" not in joined.data
    info = audio.mp3_info(joined.data)
    expected = seconds_of(a) + 0.3 + seconds_of(b) + 0.2
    assert info.sample_rate == 24000 and info.bitrate_kbps == 48
    assert info.duration == pytest.approx(expected, abs=0.15)
    if FFPROBE:
        path = tmp_path / "joined.mp3"
        path.write_bytes(joined.data)
        probe = ffprobe(path)
        assert int(probe["streams"][0]["sample_rate"]) == 24000
        assert float(probe["format"]["duration"]) == pytest.approx(info.duration, abs=0.1)


def test_concat_mixed_wav_and_mp3():
    joined = audio.concat_clips([(wav_clip(0.5, 16000), 250), (mp3_clip(0.5), 0)])
    info = audio.mp3_info(joined.data)
    assert joined.mime == "audio/mpeg" and info.sample_rate == 24000
    assert info.duration == pytest.approx(1.25 + 0.05, abs=0.2)


def test_joiner_streams_to_file(tmp_path):
    path = tmp_path / "out.wav"
    with open(path, "wb") as fh:
        joiner = audio.ClipJoiner(fh, "wav")
        joiner.add_silence(500)  # before the format is known
        joiner.add(wav_clip(0.25, 16000), 250)
        joiner.add(wav_clip(0.25, 22050), 0)
        assert joiner.close() == pytest.approx(1.25, abs=0.001)
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 20000
    with pytest.raises(ValueError):
        audio.ClipJoiner(io.BytesIO(), "wav").add(mp3_clip())


# --------------------------------------------------------------------------- playback


class FakePopen:
    instances: list["FakePopen"] = []
    interrupt = False

    def __init__(self, cmd, **kwargs):
        self.cmd, self.kwargs = cmd, kwargs
        self.file_existed = Path(cmd[-1]).is_file()
        self.terminated = False
        FakePopen.instances.append(self)

    def wait(self, timeout=None):
        if FakePopen.interrupt and not self.terminated:
            raise KeyboardInterrupt
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


@pytest.fixture
def popen(monkeypatch):
    FakePopen.instances = []
    FakePopen.interrupt = False
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return FakePopen


def _which(available: set[str]):
    return lambda name: f"/usr/bin/{name}" if name in available else None


def test_play_linux_wav_and_mp3(monkeypatch, popen):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", _which({"pw-play", "aplay", "mpg123"}))
    audio.play_clip(wav_clip())
    call = popen.instances[-1]
    assert call.cmd[0] == "/usr/bin/pw-play" and call.cmd[1].endswith(".wav") and call.file_existed
    assert not Path(call.cmd[1]).exists()  # temp file removed
    audio.play_clip(mp3_clip())
    assert popen.instances[-1].cmd[:2] == ["/usr/bin/mpg123", "-q"] and popen.instances[-1].cmd[2].endswith(".mp3")
    monkeypatch.setattr(shutil, "which", _which({"paplay", "ffplay"}))
    audio.play_clip(wav_clip())
    assert popen.instances[-1].cmd[0] == "/usr/bin/paplay"
    audio.play_clip(mp3_clip())
    assert popen.instances[-1].cmd[:4] == ["/usr/bin/ffplay", "-nodisp", "-autoexit", "-loglevel"]


def test_play_without_player(monkeypatch, popen):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", _which({"paplay"}))
    with pytest.raises(audio.AudioPlaybackError, match="mpg123"):
        audio.play_clip(mp3_clip())
    assert isinstance(audio.AudioPlaybackError("x"), TTSError)


def test_play_interrupt_stops_player(monkeypatch, popen):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", _which({"aplay"}))
    popen.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        audio.play_clip(wav_clip())
    call = popen.instances[-1]
    assert call.cmd[:2] == ["/usr/bin/aplay", "-q"] and call.terminated
    assert not Path(call.cmd[-1]).exists()


def test_play_macos(monkeypatch, popen):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", _which({"afplay"}))
    audio.play_clip(mp3_clip())
    assert popen.instances[-1].cmd[0] == "/usr/bin/afplay" and popen.instances[-1].cmd[1].endswith(".mp3")


def test_play_windows(monkeypatch, popen):
    monkeypatch.setattr(sys, "platform", "win32")
    played = []
    fake = types.SimpleNamespace(SND_MEMORY=4, SND_NODEFAULT=2, PlaySound=lambda data, flags: played.append((data, flags)))
    monkeypatch.setitem(sys.modules, "winsound", fake)
    clip = wav_clip()
    audio.play_clip(clip)
    assert played == [(clip.data, 6)]
    monkeypatch.setattr(shutil, "which", _which({"ffplay"}))
    audio.play_clip(mp3_clip())
    call = popen.instances[-1]
    assert call.cmd[:3] == ["/usr/bin/ffplay", "-nodisp", "-autoexit"]
    assert call.kwargs["creationflags"] == 0x08000000
    # No ffplay: MCI through ctypes plays the temp file.
    commands: list[str] = []
    modes = iter(["playing", "playing", "stopped"])

    def mci(command, buf, size, hwnd):
        commands.append(command)
        if command.startswith("status"):
            buf.value = next(modes)
        return 0

    import ctypes

    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(winmm=types.SimpleNamespace(mciSendStringW=mci)), raising=False)
    monkeypatch.setattr(audio.time, "sleep", lambda s: None)
    monkeypatch.setattr(shutil, "which", _which(set()))
    audio.play_clip(mp3_clip())
    assert commands[0].startswith('open "') and commands[0].endswith(' type mpegvideo alias ' + commands[1].split()[1])
    assert [c.split()[0] for c in commands] == ["open", "play", "status", "status", "status", "stop", "close"]


# --------------------------------------------------------------------------- export


class SineEngine(TTSEngine):
    """Fake voice: each utterance becomes a short sine tone."""

    name = "system"
    label = "Fake"

    def __init__(self, seconds: float = 0.2, rate: int = 22050, mp3: bool = False) -> None:
        self.seconds, self.rate, self.mp3 = seconds, rate, mp3
        self.calls: list[tuple[str, str | None, float]] = []
        self.on_call = None

    def is_available(self):
        return True, ""

    def list_voices(self):
        return [Voice(id="sine", name="Sine", engine="system")]

    def synthesize(self, text, voice=None, rate=1.0):
        self.calls.append((text, voice, rate))
        if self.on_call:
            self.on_call(len(self.calls))
        if self.mp3:
            return mp3_clip(self.seconds)
        return wav_clip(self.seconds, self.rate, 300 + 10 * len(self.calls))


def fake_plan(section: Section, options: ReadingOptions | None = None) -> list[Utterance]:
    utts = []
    for i, block in enumerate(section.blocks):
        if not block.text.strip():
            continue
        pause = 700 if block.kind == BlockKind.HEADING else 350
        utts.append(Utterance(index=len(utts), block=i, start=0, end=len(block.text), kind=block.kind, speech=block.text, pause_after_ms=pause))
    return utts


@pytest.fixture
def planner(monkeypatch):
    monkeypatch.setattr(export, "_plan", fake_plan)


def make_doc() -> Document:
    def sec(sid, title, *texts):
        blocks = [Block(kind=BlockKind.HEADING, text=title, level=1)] if title else []
        blocks += [Block(kind=BlockKind.PARAGRAPH, text=t) for t in texts]
        return Section(id=sid, title=title or "Untitled", blocks=blocks)

    return Document(
        id="abc123",
        title="Campbell Biology",
        author="Urry et al.",
        source_name="bio.pdf",
        source_type="pdf",
        sections=[
            sec("s0", "Chapter 1 Cells", "Cells are the unit of life.", "They divide."),
            Section(id="s1", title="Empty", blocks=[Block(kind=BlockKind.PARAGRAPH, text="   ")]),
            sec("s2", 'Chapter 2: Energy / "ATP"?', "ATP stores energy."),
        ],
    )


def test_export_mp3(planner, tmp_path):
    engine = SineEngine()
    events: list[tuple[float, str]] = []
    out = tmp_path / "out"
    files = export.export_audiobook(
        make_doc(), engine=engine, voice="sine", options=ReadingOptions(), out_dir=out, rate=1.5,
        progress=lambda f, m: events.append((f, m)),
    )  # fmt: skip
    names = [p.name for p in files]
    assert names == ["01 - Chapter 1 Cells.mp3", "02 - Chapter 2 Energy ATP.mp3", "playlist.m3u"]
    assert sorted(p.name for p in out.iterdir()) == sorted(names)  # no temp files left
    assert [c[1:] for c in engine.calls] == [("sine", 1.5)] * 5
    assert [c[0] for c in engine.calls][:3] == ["Chapter 1 Cells", "Cells are the unit of life.", "They divide."]

    tags = parse_id3(files[0].read_bytes())
    assert tags == {"TIT2": "Chapter 1 Cells", "TALB": "Campbell Biology", "TPE1": "Urry et al.", "TRCK": "1/2"}
    assert parse_id3(files[1].read_bytes())["TIT2"] == 'Chapter 2: Energy / "ATP"?'

    # 0.5 lead + 3 x 0.2 voice + 0.7 + 0.35 + 0.35 pauses + 0.5 trail
    expected = 0.5 + 0.6 + 1.4 + 0.5
    if FFPROBE:
        probe = ffprobe(files[0])
        assert probe["streams"][0]["codec_name"] == "mp3"
        assert float(probe["format"]["duration"]) == pytest.approx(expected, abs=0.15)
        assert probe["format"]["tags"]["album"] == "Campbell Biology"
        assert probe["format"]["tags"]["track"] == "1/2"
    assert audio.mp3_info(files[0].read_bytes()).duration == pytest.approx(expected, abs=0.15)

    playlist = (out / "playlist.m3u").read_text(encoding="utf-8").splitlines()
    assert playlist[0] == "#EXTM3U"
    assert playlist[1] == "#EXTINF:3,Chapter 1 Cells" and playlist[2] == "01 - Chapter 1 Cells.mp3"
    assert playlist[3].startswith("#EXTINF:2,Chapter 2: Energy") and playlist[4] == "02 - Chapter 2 Energy ATP.mp3"

    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    messages = [m for _, m in events]
    assert "Recording Chapter 1 Cells (1/3)" in messages and "Recording Chapter 1 Cells (3/3)" in messages
    assert 'Recording Chapter 2: Energy / "ATP"? (1/2)' in messages


def test_export_wav_and_subset(planner, tmp_path):
    files = export.export_audiobook(
        make_doc(), engine=SineEngine(rate=16000), voice=None, options=ReadingOptions(), out_dir=tmp_path,
        section_ids=["s2"], fmt="wav",
    )  # fmt: skip
    assert [p.name for p in files] == ["01 - Chapter 2 Energy ATP.wav", "playlist.m3u"]
    with wave.open(str(files[0])) as w:
        assert w.getframerate() == 16000
        assert w.getnframes() / 16000 == pytest.approx(0.5 + 0.2 + 0.7 + 0.2 + 0.35 + 0.5, abs=0.001)
    with pytest.raises(export.ExportError):
        export.export_audiobook(make_doc(), engine=SineEngine(), voice=None, options=ReadingOptions(), out_dir=tmp_path, section_ids=["s9"])
    with pytest.raises(export.ExportError, match="nothing to read"):
        export.export_audiobook(make_doc(), engine=SineEngine(), voice=None, options=ReadingOptions(), out_dir=tmp_path, section_ids=["s1"])


def test_export_mp3_engine_and_wav_request(planner, tmp_path):
    files = export.export_audiobook(
        make_doc(), engine=SineEngine(mp3=True), voice=None, options=ReadingOptions(), out_dir=tmp_path, fmt="wav"
    )
    assert [p.suffix for p in files] == [".mp3", ".mp3", ".m3u"]  # MP3-only voice: saved as MP3
    info = audio.mp3_info(files[0].read_bytes())
    assert info.sample_rate == 24000 and info.duration == pytest.approx(0.5 + 3 * 0.3 + 1.4 + 0.5, abs=0.3)
    assert parse_id3(files[0].read_bytes())["TRCK"] == "1/2"


def test_export_cancel_cleans_up(planner, tmp_path):
    engine = SineEngine()
    cancel = threading.Event()
    engine.on_call = lambda n: cancel.set() if n == 4 else None  # during the second track
    events = []
    with pytest.raises(export.ExportCancelled):
        export.export_audiobook(
            make_doc(), engine=engine, voice=None, options=ReadingOptions(), out_dir=tmp_path / "out",
            cancel=cancel, progress=lambda f, m: events.append(m),
        )  # fmt: skip
    assert list((tmp_path / "out").iterdir()) == []
    assert len(engine.calls) == 4 and events[-1] == "Export cancelled"


def test_export_engine_failure_cleans_up(planner, tmp_path):
    engine = SineEngine()

    def fail(n):
        if n == 5:
            raise EngineUnavailable("voice went away")

    engine.on_call = fail
    with pytest.raises(EngineUnavailable):
        export.export_audiobook(make_doc(), engine=engine, voice=None, options=ReadingOptions(), out_dir=tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["destiny-home"]


def test_export_retries_once(planner, tmp_path):
    engine = SineEngine()
    failures = []

    def flaky(n):
        if n == 2:
            failures.append(n)
            raise TTSError("hiccup")

    engine.on_call = flaky
    files = export.export_audiobook(make_doc(), engine=engine, voice=None, options=ReadingOptions(), out_dir=tmp_path / "o")
    assert failures == [2] and len(files) == 3 and len(engine.calls) == 6


def test_export_three_digit_tracks(planner, tmp_path):
    doc = Document(
        id="x", title="Big", source_name="b.txt", source_type="txt",
        sections=[Section(id=f"s{i}", title=f"Part {i}", blocks=[Block(kind=BlockKind.PARAGRAPH, text="Hi.")]) for i in range(100)],
    )  # fmt: skip
    files = export.export_audiobook(doc, engine=SineEngine(0.05), voice=None, options=ReadingOptions(), out_dir=tmp_path / "big")
    assert files[0].name == "001 - Part 0.mp3" and files[99].name == "100 - Part 99.mp3" and len(files) == 101
    assert parse_id3(files[99].read_bytes())["TRCK"] == "100/100"
    assert "TPE1" not in parse_id3(files[0].read_bytes())  # no author


def test_safe_filename():
    assert export.safe_filename('a<b>c:d"e/f\\g|h?i*j') == "a b c d e f g h i j"
    assert export.safe_filename("Tab\there\x00\x1f and​ more  ") == "Tab here and more"
    assert export.safe_filename("CON") == "CON_"
    assert export.safe_filename("lpt1.txt") == "lpt1.txt_"
    assert export.safe_filename("Ends with dots... ") == "Ends with dots"
    assert export.safe_filename("  ") == "Untitled"
    assert export.safe_filename("???") == "Untitled"
    assert export.safe_filename("Capítulo 3 – Células") == "Capítulo 3 – Células"
    long = export.safe_filename("word " * 40)
    assert len(long) <= 80 and not long.endswith(" ") and long.endswith("word")
    assert len(export.safe_filename("x" * 200)) == 80


def test_id3_unicode_title():
    tag = export.id3v2_tag(title="Capítulo – Células ✓", album="Libro", track="3/9")
    frames = parse_id3(tag)
    assert frames == {"TIT2": "Capítulo – Células ✓", "TALB": "Libro", "TRCK": "3/9"}


def test_export_real_piper_end_to_end(piper_voice_path, tmp_path, monkeypatch):
    if not FFPROBE:
        pytest.skip("ffprobe not installed")
    try:
        from destiny_bot.speech import plan_section  # noqa: F401
    except ImportError:
        monkeypatch.setattr(export, "_plan", fake_plan)
    from destiny_bot.tts.piper_engine import PiperEngine

    folder = Path(__import__("destiny_bot.config", fromlist=["voices_dir"]).voices_dir()) / "piper"
    folder.mkdir(parents=True, exist_ok=True)
    for src in (piper_voice_path, piper_voice_path.with_name(piper_voice_path.name + ".json")):
        try:
            (folder / src.name).symlink_to(src)
        except OSError:
            shutil.copyfile(src, folder / src.name)
    doc = Document(
        id="e2e", title="Tiny Textbook", author="A. Writer", source_name="tiny.txt", source_type="txt",
        sections=[
            Section(id="s0", title="Chapter 1: Water", blocks=[
                Block(kind=BlockKind.HEADING, text="Chapter 1: Water", level=1),
                Block(kind=BlockKind.PARAGRAPH, text="Water is a polar molecule. It forms hydrogen bonds."),
            ]),
            Section(id="s1", title="Chapter 2: Carbon", blocks=[
                Block(kind=BlockKind.HEADING, text="Chapter 2: Carbon", level=1),
                Block(kind=BlockKind.PARAGRAPH, text="Carbon forms four covalent bonds."),
            ]),
        ],
    )  # fmt: skip
    files = export.export_audiobook(doc, engine=PiperEngine(), voice=None, options=ReadingOptions(), out_dir=tmp_path / "book")
    assert [p.name for p in files] == ["01 - Chapter 1 Water.mp3", "02 - Chapter 2 Carbon.mp3", "playlist.m3u"]
    rate = json.loads(piper_voice_path.with_name(piper_voice_path.name + ".json").read_text())["audio"]["sample_rate"]
    for path, title in zip(files[:2], ("Chapter 1: Water", "Chapter 2: Carbon")):
        probe = ffprobe(path)
        stream = probe["streams"][0]
        assert stream["codec_name"] == "mp3" and int(stream["sample_rate"]) == rate and stream["channels"] == 1
        assert 2.0 < float(probe["format"]["duration"]) < 20.0
        assert probe["format"]["tags"]["title"] == title and probe["format"]["tags"]["artist"] == "A. Writer"
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, timeout=60
        )
        assert decoded.returncode == 0 and not decoded.stderr.strip()
    playlist = files[2].read_text(encoding="utf-8")
    assert playlist.startswith("#EXTM3U\n#EXTINF:") and "01 - Chapter 1 Water.mp3" in playlist
