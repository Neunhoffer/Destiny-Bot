"""Audio helpers: WAV/MP3 handling, joining clips with pauses, MP3 encoding, playback.

Everything here works on in-memory bytes. WAV is 16-bit PCM wherever we create
it; other PCM widths are accepted on input. MP3 is never decoded (that would
need ffmpeg): MP3 clips are joined frame-by-frame and pauses are encoded as
silent MP3 at the same sample rate.
"""

from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Literal

from .tts.base import AudioClip, TTSError

WAV_MIME = "audio/wav"
MP3_MIME = "audio/mpeg"
MP3_SAMPLE_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
_PLAYER_TIMEOUT_S = 6 * 3600


class AudioPlaybackError(TTSError):
    """Audio could not be played on the speakers; the message is safe to show."""


# --------------------------------------------------------------------------- WAV


def wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw little-endian PCM in a WAV container."""
    frame = channels * sampwidth
    if len(pcm) % frame:
        pcm = pcm[: len(pcm) - len(pcm) % frame]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm)
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[bytes, int, int, int]:
    """Return ``(pcm, sample_rate, channels, sampwidth)``. Raises ``ValueError`` if not PCM WAV.

    Tolerates the bogus length fields written by programs that stream WAV to a pipe.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, sampwidth, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            pcm = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError, struct.error) as exc:
        # Python < 3.12 rejects WAVE_FORMAT_EXTENSIBLE; parse those by hand.
        parsed = _read_wav_fallback(data)
        if parsed is None:
            raise ValueError(f"Not a valid WAV file ({exc})") from exc
        pcm, rate, channels, sampwidth = parsed
    if channels < 1 or sampwidth not in (1, 2, 3, 4) or rate <= 0:
        raise ValueError("Unsupported WAV format")
    frame = channels * sampwidth
    if len(pcm) % frame:
        pcm = pcm[: len(pcm) - len(pcm) % frame]
    return pcm, rate, channels, sampwidth


def _read_wav_fallback(data: bytes) -> tuple[bytes, int, int, int] | None:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            if tag != 1:
                return None
            fmt = (rate, channels, (bits + 7) // 8)
        elif cid == b"data" and fmt is not None:
            rate, channels, width = fmt
            return body, rate, channels, width
        pos += 8 + size + (size & 1)
    return None


def wav_format(data: bytes) -> tuple[int, int, int]:
    """``(sample_rate, channels, sampwidth)`` of a WAV clip without copying its audio."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            return wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
    except (wave.Error, EOFError, struct.error):
        _, rate, channels, width = read_wav(data)
        return rate, channels, width


def silence_pcm(ms: int, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """``ms`` milliseconds of digital silence (8-bit WAV is unsigned, so its zero is 0x80)."""
    frames = max(0, round(sample_rate * ms / 1000))
    fill = b"\x80" if sampwidth == 1 else b"\x00"
    return fill * (frames * channels * sampwidth)


def _np():
    try:
        import numpy

        return numpy
    except ImportError:  # numpy comes with piper-tts, but keep a pure-Python path
        return None


_FLIP_SIGN = bytes(b ^ 0x80 for b in range(256))


def to_int16(pcm: bytes, sampwidth: int) -> bytes:
    """Convert little-endian PCM of any width to 16-bit."""
    if sampwidth == 2:
        return pcm
    out = bytearray(len(pcm) // sampwidth * 2)
    if sampwidth == 1:  # unsigned 8-bit: high byte is the sample with its sign bit flipped
        out[1::2] = pcm.translate(_FLIP_SIGN)
    elif sampwidth in (3, 4):  # keep the two most significant bytes
        out[0::2] = pcm[sampwidth - 2 :: sampwidth]
        out[1::2] = pcm[sampwidth - 1 :: sampwidth]
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    return bytes(out)


def _samples(pcm16: bytes) -> array:
    arr = array("h")
    arr.frombytes(pcm16)
    if sys.byteorder == "big":
        arr.byteswap()
    return arr


def _pcm(arr: array) -> bytes:
    if sys.byteorder == "big":
        arr = array("h", arr)
        arr.byteswap()
    return arr.tobytes()


def change_channels(pcm16: bytes, channels: int, to_channels: int) -> bytes:
    """Mix 16-bit PCM down to mono, or copy mono out to ``to_channels``."""
    if channels == to_channels:
        return pcm16
    np = _np()
    if to_channels == 1:
        if np is not None:
            frames = np.frombuffer(pcm16, dtype="<i2").reshape(-1, channels)
            return frames.mean(axis=1).round().astype("<i2").tobytes()
        src = _samples(pcm16)
        mono = array("h", (sum(src[i : i + channels]) // channels for i in range(0, len(src), channels)))
        return _pcm(mono)
    if channels == 1:
        if np is not None:
            mono = np.frombuffer(pcm16, dtype="<i2")
            return np.repeat(mono, to_channels).astype("<i2").tobytes()
        src = _samples(pcm16)
        return _pcm(array("h", (s for s in src for _ in range(to_channels))))
    return change_channels(change_channels(pcm16, channels, 1), 1, to_channels)


def resample(pcm16: bytes, channels: int, rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resampling of 16-bit PCM (good enough for speech)."""
    if rate == to_rate or not pcm16:
        return pcm16
    n_in = len(pcm16) // (2 * channels)
    n_out = max(1, round(n_in * to_rate / rate))
    np = _np()
    if np is not None:
        frames = np.frombuffer(pcm16, dtype="<i2").reshape(-1, channels).astype(np.float64)
        pos = np.arange(n_out) * (rate / to_rate)
        idx = np.arange(n_in)
        out = np.empty((n_out, channels), dtype=np.float64)
        for c in range(channels):
            out[:, c] = np.interp(pos, idx, frames[:, c])
        return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()
    src = _samples(pcm16)
    out_arr = array("h", bytes(2 * n_out * channels))
    step = rate / to_rate
    last = n_in - 1
    for i in range(n_out):
        x = i * step
        j = min(int(x), last)
        k = min(j + 1, last)
        t = x - j
        for c in range(channels):
            a, b = src[j * channels + c], src[k * channels + c]
            out_arr[i * channels + c] = int(round(a + (b - a) * t))
    return _pcm(out_arr)


def convert_pcm(
    pcm: bytes, *, sample_rate: int, channels: int, sampwidth: int, to_rate: int, to_channels: int
) -> bytes:
    """Convert PCM to 16-bit at ``to_rate`` Hz with ``to_channels`` channels."""
    pcm16 = to_int16(pcm, sampwidth)
    if to_channels < channels:
        pcm16 = change_channels(pcm16, channels, to_channels)
        channels = to_channels
    pcm16 = resample(pcm16, channels, sample_rate, to_rate)
    return change_channels(pcm16, channels, to_channels)


# --------------------------------------------------------------------------- MP3

_MP3_BITRATES = {
    1: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),  # MPEG-1 layer III
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),  # MPEG-2/2.5 layer III
}
_MP3_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


@dataclass(frozen=True)
class Mp3Info:
    sample_rate: int
    channels: int
    bitrate_kbps: int  # of the first audio frame
    frames: int
    duration: float  # seconds
    start: int  # offset of the first audio frame
    end: int  # offset just past the last audio frame


def _id3v2_size(data: bytes) -> int:
    if len(data) >= 10 and data[:3] == b"ID3":
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
        return 10 + size + (10 if data[5] & 0x10 else 0)
    return 0


def _frame_header(data: bytes, pos: int) -> tuple[int, int, int, int, int] | None:
    """(frame length, sample rate, channels, bitrate kbps, samples) for a layer III header at pos."""
    if pos + 4 > len(data) or data[pos] != 0xFF or (data[pos + 1] & 0xE0) != 0xE0:
        return None
    b1, b2, b3 = data[pos + 1], data[pos + 2], data[pos + 3]
    version = (b1 >> 3) & 3  # 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5
    layer = (b1 >> 1) & 3  # 1 = layer III
    br_idx, sr_idx, padding = b2 >> 4, (b2 >> 2) & 3, (b2 >> 1) & 1
    if version == 1 or layer != 1 or br_idx in (0, 15) or sr_idx == 3:
        return None
    bitrate = _MP3_BITRATES[1 if version == 3 else 2][br_idx]
    rate = _MP3_RATES[version][sr_idx]
    samples = 1152 if version == 3 else 576
    length = (samples // 8) * bitrate * 1000 // rate + padding
    channels = 1 if (b3 >> 6) == 3 else 2
    return length, rate, channels, bitrate, samples


def _is_info_frame(data: bytes, pos: int) -> bool:
    """True for a Xing/Info/VBRI header frame (it describes a file, not audio)."""
    mpeg1 = (data[pos + 1] >> 3) & 3 == 3
    mono = data[pos + 3] >> 6 == 3
    side_info = (17 if mono else 32) if mpeg1 else (9 if mono else 17)
    tag = data[pos + 4 + side_info : pos + 8 + side_info]
    return tag in (b"Xing", b"Info") or data[pos + 36 : pos + 40] == b"VBRI"


def mp3_info(data: bytes) -> Mp3Info:
    """Scan MP3 frames. Raises ``ValueError`` if there is no MPEG layer III audio."""
    pos = _id3v2_size(data)
    # Find the first header that is followed by another valid header (avoids false syncs).
    first = None
    while True:
        pos = data.find(b"\xff", pos)
        if pos < 0:
            break
        hdr = _frame_header(data, pos)
        if hdr and (pos + hdr[0] >= len(data) or _frame_header(data, pos + hdr[0])):
            first = hdr
            break
        pos += 1
    if first is None:
        raise ValueError("Not a valid MP3 file")
    start, frames, duration = pos, 0, 0.0
    bitrate = first[3]
    if _is_info_frame(data, pos):
        start = pos = pos + first[0]
        nxt = _frame_header(data, pos)
        bitrate = nxt[3] if nxt else bitrate
    while True:
        hdr = _frame_header(data, pos)
        if hdr is None or pos + hdr[0] > len(data):
            break
        frames += 1
        duration += hdr[4] / hdr[1]
        pos += hdr[0]
    if frames == 0:
        raise ValueError("Not a valid MP3 file")
    return Mp3Info(first[1], first[2], bitrate, frames, duration, start, pos)


def mp3_audio_frames(data: bytes) -> bytes:
    """Just the audio frames: drops ID3 tags and Xing/Info headers so clips can be joined."""
    info = mp3_info(data)
    return data[info.start : info.end]


def _clamp_bitrate(bitrate_kbps: int, sample_rate: int) -> int:
    allowed = _MP3_BITRATES[1 if sample_rate >= 32000 else 2][1:]
    return min(allowed, key=lambda b: abs(b - bitrate_kbps))


def _mp3_rate(sample_rate: int) -> int:
    return min(MP3_SAMPLE_RATES, key=lambda r: (abs(r - sample_rate), -r))


class Mp3Encoder:
    """Thin wrapper around a lameenc encoder for 16-bit PCM."""

    def __init__(self, sample_rate: int, channels: int = 1, bitrate_kbps: int = 64) -> None:
        try:
            import lameenc
        except ImportError as exc:  # it is a core dependency, but fail politely
            raise TTSError("MP3 saving needs the 'lameenc' package. Reinstall Destiny Bot to fix this.") from exc
        if channels not in (1, 2):
            raise ValueError("MP3 supports mono or stereo only")
        self.sample_rate, self.channels = sample_rate, channels
        out_rate = _mp3_rate(sample_rate)
        self._enc = lameenc.Encoder()
        self._enc.set_bit_rate(_clamp_bitrate(bitrate_kbps, out_rate))
        self._enc.set_in_sample_rate(sample_rate)
        # Pin the output rate: LAME otherwise may pick a lower one for low bitrates,
        # and joined MP3 clips must all share one sample rate.
        self._enc.set_out_sample_rate(out_rate)
        self._enc.set_channels(channels)
        self._enc.set_quality(2)
        self._enc.silence()

    def encode(self, pcm16: bytes) -> bytes:
        return bytes(self._enc.encode(pcm16)) if pcm16 else b""

    def flush(self) -> bytes:
        return bytes(self._enc.flush())


def encode_mp3(wav: bytes, *, bitrate_kbps: int = 64) -> bytes:
    """Encode a WAV clip to MP3 (mono stays mono, stereo stays stereo)."""
    pcm, rate, channels, width = read_wav(wav)
    pcm16 = to_int16(pcm, width)
    if channels > 2:
        pcm16, channels = change_channels(pcm16, channels, 1), 1
    enc = Mp3Encoder(rate, channels, bitrate_kbps)
    return enc.encode(pcm16) + enc.flush()


def mp3_silence(ms: int, sample_rate: int, channels: int = 1, bitrate_kbps: int = 32) -> bytes:
    """``ms`` of silence as MP3 frames (no tags) at ``sample_rate``."""
    if ms <= 0:
        return b""
    enc = Mp3Encoder(sample_rate, channels, bitrate_kbps)
    return enc.encode(silence_pcm(ms, sample_rate, channels)) + enc.flush()


# --------------------------------------------------------------------------- joining


class ClipJoiner:
    """Streams clips and pauses into one WAV or MP3 written to ``sink``.

    The audio format is fixed by the first clip (or by ``sample_rate``/``channels``
    for WAV, or by ``mp3_like`` to append to existing MP3 audio). Later WAV clips are
    converted to match. MP3 clips are copied frame by frame, so MP3 output from MP3
    input never re-encodes. MP3 can't be turned into WAV (no decoder): ``add`` raises
    ``ValueError`` for that.
    """

    def __init__(
        self,
        sink: BinaryIO,
        fmt: Literal["wav", "mp3"],
        *,
        bitrate_kbps: int = 64,
        sample_rate: int | None = None,
        channels: int | None = None,
        mp3_like: Mp3Info | None = None,
    ) -> None:
        self._sink, self.fmt, self._bitrate = sink, fmt, bitrate_kbps
        self._want_rate, self._want_channels = sample_rate, channels
        self._mode: Literal["wav", "encode", "copy"] | None = None
        self.sample_rate: int | None = None
        self.channels = 1
        self._wav: wave.Wave_write | None = None
        self._encoder: Mp3Encoder | None = None
        self._frames = 0  # PCM frames written (wav / encode modes)
        self._mp3_seconds = 0.0  # copy mode
        self._pending_ms = 0
        self._closed = False
        if mp3_like is not None:
            if fmt != "mp3":
                raise ValueError("mp3_like needs fmt='mp3'")
            self._start_copy(mp3_like.sample_rate, mp3_like.channels, mp3_like.bitrate_kbps)

    @property
    def mime(self) -> str:
        return WAV_MIME if self.fmt == "wav" else MP3_MIME

    @property
    def duration(self) -> float:
        if self._mode == "copy":
            return self._mp3_seconds
        return self._frames / self.sample_rate if self.sample_rate else 0.0

    def add(self, clip: AudioClip, pause_after_ms: int = 0) -> None:
        if clip.mime == MP3_MIME:
            self._add_mp3(clip.data)
        else:
            self._add_wav(clip.data)
        self.add_silence(pause_after_ms)

    def add_silence(self, ms: int) -> None:
        if ms <= 0:
            return
        if self._mode is None:
            self._pending_ms += ms
            return
        if self._mode == "copy":
            self._write_mp3(mp3_silence(ms, self.sample_rate or 24000, self.channels, self._bitrate))
        else:
            self._write_pcm(silence_pcm(ms, self.sample_rate or 22050, self.channels))

    def close(self) -> float:
        """Finish the stream (flushes the encoder / WAV header). Returns the duration in seconds."""
        if self._closed:
            return self.duration
        self._closed = True
        if self._mode is None:  # nothing but silence (or nothing at all)
            self._start_pcm(self._want_rate or 22050, self._want_channels or 1)
        if self._encoder is not None:
            self._sink.write(self._encoder.flush())
        if self._wav is not None:
            self._wav.close()
        return self.duration

    # -- internals

    def _start_pcm(self, rate: int, channels: int) -> None:
        channels = 1 if channels > 2 else channels
        if self.fmt == "wav":
            self._mode = "wav"
            self._wav = wave.open(self._sink, "wb")  # type: ignore[arg-type]
            self._wav.setnchannels(channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(rate)
        else:
            self._mode = "encode"
            rate = _mp3_rate(rate)
            self._encoder = Mp3Encoder(rate, channels, self._bitrate)
        self.sample_rate, self.channels = rate, channels
        self._flush_pending()

    def _start_copy(self, rate: int, channels: int, bitrate: int) -> None:
        self._mode = "copy"
        self.sample_rate, self.channels, self._bitrate = rate, channels, bitrate
        self._flush_pending()

    def _flush_pending(self) -> None:
        pending, self._pending_ms = self._pending_ms, 0
        self.add_silence(pending)

    def _add_wav(self, data: bytes) -> None:
        pcm, rate, channels, width = read_wav(data)
        if self._mode is None:
            self._start_pcm(self._want_rate or rate, self._want_channels or channels)
        assert self.sample_rate is not None
        pcm16 = convert_pcm(
            pcm, sample_rate=rate, channels=channels, sampwidth=width, to_rate=self.sample_rate, to_channels=self.channels
        )
        if self._mode == "copy":
            enc = Mp3Encoder(self.sample_rate, self.channels, self._bitrate)
            self._write_mp3(enc.encode(pcm16) + enc.flush())
        else:
            self._write_pcm(pcm16)

    def _add_mp3(self, data: bytes) -> None:
        if self.fmt == "wav":
            raise ValueError("MP3 audio can't be saved as WAV; choose MP3 instead.")
        info = mp3_info(data)
        if self._mode is None:
            self._start_copy(info.sample_rate, info.channels, info.bitrate_kbps)
        elif self._mode == "encode":
            raise ValueError("Can't mix MP3 audio into audio that is being encoded from WAV.")
        if info.sample_rate != self.sample_rate:
            raise ValueError("Can't join MP3 clips with different sample rates.")
        self._sink.write(data[info.start : info.end])
        self._mp3_seconds += info.duration

    def _write_pcm(self, pcm16: bytes) -> None:
        if not pcm16:
            return
        self._frames += len(pcm16) // (2 * self.channels)
        if self._wav is not None:
            self._wav.writeframesraw(pcm16)
        elif self._encoder is not None:
            self._sink.write(self._encoder.encode(pcm16))

    def _write_mp3(self, frames: bytes) -> None:
        if frames:
            self._mp3_seconds += mp3_info(frames).duration
            self._sink.write(frames)


def concat_clips(clips: list[tuple[AudioClip, int]]) -> AudioClip:
    """Join ``(clip, pause_after_ms)`` pairs into one clip.

    All WAV gives WAV (converted to the highest sample rate, and to mono if the
    channel counts differ). Any MP3 gives MP3 at the first MP3 clip's sample rate;
    WAV parts are encoded to match.
    """
    if not clips:
        raise ValueError("There is no audio to join.")
    buf = io.BytesIO()
    mp3s = [clip for clip, _ in clips if clip.mime == MP3_MIME]
    if mp3s:
        joiner = ClipJoiner(buf, "mp3", mp3_like=mp3_info(mp3s[0].data))
    else:
        formats = [wav_format(clip.data) for clip, _ in clips]
        rate = max(f[0] for f in formats)
        channels = {min(f[1], 2) for f in formats}
        joiner = ClipJoiner(buf, "wav", sample_rate=rate, channels=channels.pop() if len(channels) == 1 else 1)
    for clip, pause in clips:
        joiner.add(clip, pause)
    joiner.close()
    return AudioClip(buf.getvalue(), joiner.mime, joiner.sample_rate)


def clip_duration(clip: AudioClip) -> float:
    """Length of a clip in seconds."""
    if clip.mime == MP3_MIME:
        return mp3_info(clip.data).duration
    pcm, rate, channels, width = read_wav(clip.data)
    return len(pcm) / (rate * channels * width)


# --------------------------------------------------------------------------- playback


def play_clip(clip: AudioClip) -> None:
    """Play a clip on the speakers and block until it finishes (Ctrl+C stops it)."""
    if not clip.data:
        return
    is_mp3 = clip.mime == MP3_MIME
    if sys.platform == "win32":
        if not is_mp3:
            _play_winsound(clip.data)
            return
        ffplay = shutil.which("ffplay")
        with _temp_audio(clip) as path:
            if ffplay:
                _run_player([ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])
            else:
                _play_mci(path)
        return
    if sys.platform == "darwin":
        player = shutil.which("afplay")
        if not player:
            raise AudioPlaybackError("Couldn't find the macOS audio player (afplay).")
        with _temp_audio(clip) as path:
            _run_player([player, str(path)])
        return
    cmd = _linux_player(is_mp3)
    if cmd is None:
        need = "ffmpeg (ffplay) or mpg123" if is_mp3 else "pulseaudio-utils (paplay), pipewire (pw-play) or alsa-utils (aplay)"
        raise AudioPlaybackError(f"No audio player was found. Install {need} to listen from the command line.")
    with _temp_audio(clip) as path:
        _run_player([*cmd, str(path)])


def _linux_player(is_mp3: bool) -> list[str] | None:
    candidates: list[list[str]] = []
    if not is_mp3:
        candidates += [["paplay"], ["pw-play"], ["aplay", "-q"]]
    candidates += [["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]]
    if is_mp3:
        candidates += [["mpg123", "-q"]]
    for cmd in candidates:
        exe = shutil.which(cmd[0])
        if exe:
            return [exe, *cmd[1:]]
    return None


@contextmanager
def _temp_audio(clip: AudioClip) -> Iterator[Path]:
    """The clip written to a temp file that is removed afterwards."""
    fd, name = tempfile.mkstemp(prefix="destiny-", suffix=clip.extension)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(clip.data)
        yield path
    finally:
        for _ in range(5):  # Windows may hold the file briefly after the player exits
            try:
                path.unlink()
                break
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(0.1)


def _run_player(cmd: list[str]) -> None:
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except OSError as exc:
        raise AudioPlaybackError(f"Couldn't start the audio player: {exc}") from exc
    try:
        code = proc.wait(timeout=_PLAYER_TIMEOUT_S)
    except BaseException:  # KeyboardInterrupt / timeout: stop the sound right away
        _stop(proc)
        raise
    if code != 0:
        raise AudioPlaybackError(f"The audio player stopped with an error (exit code {code}).")


def _stop(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _play_winsound(data: bytes) -> None:
    import winsound  # type: ignore[import-not-found]

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            winsound.PlaySound(data, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            errors.append(exc)

    thread = threading.Thread(target=worker, name="destiny-play", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(0.1)  # short joins keep Ctrl+C responsive
    except KeyboardInterrupt:
        winsound.PlaySound(None, 0)  # stops the sound playing in the worker thread
        raise
    if errors:
        raise AudioPlaybackError(f"Couldn't play the audio: {errors[0]}")


def _play_mci(path: Path) -> None:
    import ctypes

    mci = ctypes.windll.winmm.mciSendStringW  # type: ignore[attr-defined]
    alias = f"destiny{threading.get_ident()}"

    def send(command: str) -> str:
        buf = ctypes.create_unicode_buffer(256)
        err = mci(command, buf, 255, None)
        if err:
            raise AudioPlaybackError(f"Windows couldn't play the audio (MCI error {err}).")
        return buf.value

    send(f'open "{path}" type mpegvideo alias {alias}')
    try:
        send(f"play {alias}")
        while send(f"status {alias} mode").lower() == "playing":
            time.sleep(0.1)
    finally:
        try:
            send(f"stop {alias}")
        except AudioPlaybackError:
            pass
        try:
            send(f"close {alias}")
        except AudioPlaybackError:
            pass
