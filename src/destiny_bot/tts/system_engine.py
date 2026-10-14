"""The operating system's own voices: macOS ``say``, Linux espeak-ng, Windows SAPI.

Text is always handed over through a temp file or stdin, never on a command
line, and no shell is involved.
"""

from __future__ import annotations

import base64
import concurrent.futures
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .base import AudioClip, EngineUnavailable, TTSEngine, TTSError, Voice

BASE_WPM = 175
_VOICE_LIST_TTL_S = 300
_FAILED_MSG = "The system voice couldn't read this text. Try another voice."


def _create_no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def _run(
    cmd: list[str], *, timeout: float, input: bytes | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    kwargs: dict[str, Any] = {"capture_output": True, "timeout": timeout, "check": False}
    if input is not None:
        kwargs["input"] = input
    else:
        kwargs["stdin"] = subprocess.DEVNULL
    if env is not None:
        kwargs["env"] = env
    if sys.platform == "win32":
        kwargs["creationflags"] = _create_no_window()
    try:
        return subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise TTSError("The system voice took too long to answer. Please try again.") from exc
    except OSError as exc:
        raise EngineUnavailable(f"Couldn't start the system voice ({exc}).") from exc


def _synth_timeout(text: str) -> float:
    return 60 + len(text) * 0.05


def _stderr(proc: subprocess.CompletedProcess) -> str:
    raw = proc.stderr or b""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    return text.strip().splitlines()[0] if text.strip() else ""


def _wav_clip(data: bytes) -> AudioClip:
    """Re-wrap a tool's WAV output as plain 16-bit PCM WAV with correct sizes."""
    from ..audio import read_wav, to_int16, wav_bytes

    try:
        pcm, rate, channels, width = read_wav(data)
    except ValueError as exc:
        raise TTSError(_FAILED_MSG) from exc
    if not pcm:
        pcm, width = b"\x00\x00" * channels * (rate // 5), 2  # nothing speakable: a short pause
    return AudioClip(wav_bytes(to_int16(pcm, width), rate, channels, 2), "audio/wav", rate)


def _bcp47(code: str) -> str:
    parts = [p for p in code.replace("_", "-").split("-") if p]
    if not parts:
        return code
    parts[0] = parts[0].lower()
    if len(parts) > 1 and len(parts[1]) == 2:
        parts[1] = parts[1].upper()
    return "-".join(parts)


def _check_voice_arg(voice: str) -> None:
    if voice.startswith("-") or any(ord(ch) < 32 for ch in voice) or len(voice) > 200:
        raise EngineUnavailable("That system voice name isn't valid. Pick another voice in Settings.")


class _Backend:
    """One platform's way of listing voices and speaking."""

    def __init__(self) -> None:
        self._voices: list[Voice] | None = None
        self._voices_at = 0.0
        self._voices_lock = threading.Lock()

    def is_available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def _list_voices(self) -> list[Voice]:
        raise NotImplementedError

    def default_voice(self) -> str | None:
        voices = self.voices()
        return voices[0].id if voices else None

    def synthesize(self, text: str, voice: str | None, wpm: int, rate: float) -> AudioClip:
        raise NotImplementedError

    def voices(self) -> list[Voice]:
        with self._voices_lock:
            stale = time.monotonic() - self._voices_at > _VOICE_LIST_TTL_S
            if self._voices is None or stale:
                try:
                    self._voices = self._list_voices() if self.is_available()[0] else []
                except (TTSError, OSError, subprocess.SubprocessError, ValueError):
                    self._voices = []
                self._voices_at = time.monotonic()
            return list(self._voices)

    def knows_voice(self, voice: str) -> bool:
        voices = self.voices()
        return not voices or any(v.id == voice for v in voices)


# --------------------------------------------------------------------------- macOS


_MAC_VOICE_RE = re.compile(r"^(?P<name>.+?)\s+(?P<locale>[A-Za-z]{2,3}(?:[_-][A-Za-z0-9]+)*)\s+#")
# Sound-effect voices that are no good for reading a textbook.
_MAC_NOVELTY = {
    "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Deranged", "Good News",
    "Hysterical", "Jester", "Organ", "Pipe Organ", "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox",
}  # fmt: skip


class _MacBackend(_Backend):
    def _say(self) -> str | None:
        return shutil.which("say")

    def is_available(self) -> tuple[bool, str]:
        if self._say():
            return True, ""
        return False, "The macOS speech command ('say') wasn't found on this Mac."

    def _list_voices(self) -> list[Voice]:
        say = self._say()
        if not say:
            return []
        proc = _run([say, "-v", "?"], timeout=30)
        voices: list[Voice] = []
        seen: set[str] = set()
        for line in (proc.stdout or b"").decode("utf-8", "replace").splitlines():
            match = _MAC_VOICE_RE.match(line.strip())
            if not match:
                continue
            name, language = match.group("name").strip(), _bcp47(match.group("locale"))
            if name in seen or name in _MAC_NOVELTY:
                continue
            seen.add(name)
            label = name if "(" in name else f"{name} ({language})"
            voices.append(Voice(id=name, name=label, engine="system", language=language))
        voices.sort(key=lambda v: (not (v.language or "").startswith("en"), v.language != "en-US", v.name.lower()))
        return voices

    def default_voice(self) -> str | None:
        voices = self.voices()
        ids = [v.id for v in voices]
        for preferred in ("Samantha", "Alex"):
            if preferred in ids:
                return preferred
        return ids[0] if ids else None

    def synthesize(self, text: str, voice: str | None, wpm: int, rate: float) -> AudioClip:
        say = self._say()
        if not say:
            raise EngineUnavailable(self.is_available()[1])
        with tempfile.TemporaryDirectory(prefix="destiny-say-", ignore_cleanup_errors=True) as tmp:
            text_path = Path(tmp) / "text.txt"
            text_path.write_text(text, encoding="utf-8")
            out = Path(tmp) / "speech.wav"
            cmd = [say]
            if voice:
                cmd += ["-v", voice]
            cmd += ["-r", str(wpm), "-o", str(out), "--data-format=LEI16@22050", "-f", str(text_path)]
            proc = _run(cmd, timeout=_synth_timeout(text))
            if proc.returncode != 0 or not out.is_file():
                raise TTSError(_FAILED_MSG)
            return _wav_clip(out.read_bytes())


# --------------------------------------------------------------------------- Linux (espeak)


class _EspeakBackend(_Backend):
    def _exe(self) -> str | None:
        return shutil.which("espeak-ng") or shutil.which("espeak")

    def is_available(self) -> tuple[bool, str]:
        if self._exe():
            return True, ""
        return False, (
            "No system voice was found. Install espeak-ng (for example: sudo apt install espeak-ng), "
            "or download a natural voice in Settings → Voices."
        )

    def _list_voices(self) -> list[Voice]:
        exe = self._exe()
        if not exe:
            return []
        proc = _run([exe, "--voices"], timeout=30)
        voices: list[Voice] = []
        seen: set[str] = set()
        for line in (proc.stdout or b"").decode("utf-8", "replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) < 5 or parts[1] == "variant" or parts[4].startswith("mb/"):
                continue
            lang, age_gender, name = parts[1], parts[2], parts[3]
            if lang in seen:
                continue
            seen.add(lang)
            gender = {"M": "male", "F": "female"}.get(age_gender.rsplit("/", 1)[-1])
            label = name.replace("_", " ").strip()
            voices.append(Voice(id=lang, name=label, engine="system", language=_bcp47(lang), gender=gender))

        def order(v: Voice) -> tuple[int, str]:
            rank = {"en-us": 0, "en-gb": 1}.get(v.id, 2 if v.id.startswith("en") else 3)
            return rank, v.name.lower()

        voices.sort(key=order)
        return voices

    def default_voice(self) -> str | None:
        ids = [v.id for v in self.voices()]
        for preferred in ("en-us", "en"):
            if preferred in ids:
                return preferred
        return ids[0] if ids else "en"

    def knows_voice(self, voice: str) -> bool:
        return super().knows_voice(voice.split("+", 1)[0])  # "en-us+f3" = voice + variant

    def synthesize(self, text: str, voice: str | None, wpm: int, rate: float) -> AudioClip:
        exe = self._exe()
        if not exe:
            raise EngineUnavailable(self.is_available()[1])
        speed = min(max(wpm, 80), 450)
        cmd = [exe, "--stdout", "-b", "1", "-v", voice or "en", "-s", str(speed), "--stdin"]
        proc = _run(cmd, input=text.encode("utf-8"), timeout=_synth_timeout(text))
        if proc.returncode != 0 or not proc.stdout:
            raise TTSError(_FAILED_MSG)
        return _wav_clip(proc.stdout)


# --------------------------------------------------------------------------- Windows (SAPI)


def _sapi_rate(wpm: int) -> int:
    # Same words-per-minute -> SAPI rate mapping pyttsx3 uses, so both paths sound alike.
    return max(-10, min(10, int(math.log(wpm / 156.63, 1.11))))


def _sapi_name(description: str) -> str:
    """'Microsoft Zira Desktop - English (United States)' -> 'Microsoft Zira Desktop'."""
    return description.split(" - ", 1)[0].strip()


_PS_LIST = """
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  foreach ($v in $s.GetInstalledVoices()) {
    if ($v.Enabled) { $i = $v.VoiceInfo; "{0}`t{1}`t{2}" -f $i.Name, $i.Culture.Name, $i.Gender }
  }
} finally { $s.Dispose() }
"""

_PS_SPEAK = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  if ($env:DESTINY_TTS_VOICE) { $s.SelectVoice($env:DESTINY_TTS_VOICE) }
  $s.Rate = [int]$env:DESTINY_TTS_RATE
  $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
  $s.SetOutputToWaveFile($env:DESTINY_TTS_OUT, $fmt)
  $text = [System.IO.File]::ReadAllText($env:DESTINY_TTS_TEXT, [System.Text.Encoding]::UTF8)
  $s.Speak($text)
} finally { $s.Dispose() }
"""


def _encoded(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _co_initialize() -> None:
    """SAPI is COM: every thread that uses it must initialize COM first."""
    try:
        import pythoncom  # type: ignore[import-not-found]

        pythoncom.CoInitialize()
        return
    except ImportError:
        pass
    try:
        import comtypes  # type: ignore[import-not-found]

        comtypes.CoInitialize()
    except ImportError:
        pass


class _SapiWorker:
    """One thread that owns the pyttsx3 engine (SAPI objects must stay on their thread)."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[..., Any], tuple[Any, ...], concurrent.futures.Future]] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="destiny-sapi", daemon=True)
        self._thread.start()

    def call(self, fn: Callable[..., Any], *args: Any, timeout: float) -> Any:
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((fn, args, future))
        return future.result(timeout=timeout)

    def _loop(self) -> None:
        engine: Any = None
        init_error: BaseException | None = None
        try:
            _co_initialize()
            import pyttsx3

            engine = pyttsx3.init("sapi5")
        except BaseException as exc:  # noqa: BLE001 - reported to every caller
            init_error = exc
        while True:
            fn, args, future = self._jobs.get()
            if not future.set_running_or_notify_cancel():
                continue
            if init_error is not None:
                future.set_exception(init_error)
                continue
            try:
                future.set_result(fn(engine, *args))
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)


def _pyttsx3_voices(engine: Any) -> list[Voice]:
    voices: list[Voice] = []
    seen: set[str] = set()
    for v in engine.getProperty("voices"):
        name = _sapi_name(str(getattr(v, "name", "") or ""))
        if not name or name in seen:
            continue
        seen.add(name)
        langs = getattr(v, "languages", None) or []
        language = _bcp47(str(langs[0])) if langs and "Unknown" not in str(langs[0]) else None
        gender = (getattr(v, "gender", None) or "").lower() or None
        voices.append(Voice(id=name, name=name, engine="system", language=language, gender=gender))
    return voices


def _pyttsx3_save(engine: Any, text: str, voice: str | None, wpm: int, out_path: str) -> None:
    if voice:
        for v in engine.getProperty("voices"):
            if _sapi_name(str(getattr(v, "name", "") or "")) == voice:
                engine.setProperty("voice", v.id)
                break
        else:
            raise EngineUnavailable(f"The Windows voice '{voice}' isn't installed. Pick another voice in Settings.")
    engine.setProperty("rate", wpm)  # after the voice: changing voice resets the rate
    engine.save_to_file(text, out_path)
    engine.runAndWait()


class _WindowsBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self._worker: _SapiWorker | None = None
        self._pyttsx3_ok: bool | None = None
        self._lock = threading.Lock()

    def _powershell(self) -> str | None:
        exe = shutil.which("powershell") or shutil.which("powershell.exe")
        if exe:
            return exe
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        return str(candidate) if candidate.is_file() else None

    def _sapi(self) -> _SapiWorker | None:
        with self._lock:
            if self._pyttsx3_ok is None:
                try:
                    import pyttsx3  # noqa: F401

                    self._pyttsx3_ok = True
                except Exception:
                    self._pyttsx3_ok = False
            if not self._pyttsx3_ok:
                return None
            if self._worker is None:
                self._worker = _SapiWorker()
            return self._worker

    def _give_up_on_pyttsx3(self) -> None:
        # A hung or broken SAPI thread is abandoned (it is a daemon); PowerShell takes over.
        with self._lock:
            self._pyttsx3_ok = False
            self._worker = None

    def is_available(self) -> tuple[bool, str]:
        if self._sapi() is not None or self._powershell():
            return True, ""
        return False, "Windows speech isn't available on this computer. Download a natural voice in Settings → Voices."

    def _list_voices(self) -> list[Voice]:
        worker = self._sapi()
        if worker is not None:
            try:
                return worker.call(_pyttsx3_voices, timeout=30)
            except Exception:
                self._give_up_on_pyttsx3()
        exe = self._powershell()
        if not exe:
            return []
        proc = _run([exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", _encoded(_PS_LIST)], timeout=60)
        voices: list[Voice] = []
        for line in (proc.stdout or b"").decode("utf-8", "replace").splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 2 or not parts[0]:
                continue
            gender = parts[2].lower() if len(parts) > 2 and parts[2] in ("Male", "Female") else None
            voices.append(Voice(id=parts[0], name=parts[0], engine="system", language=parts[1] or None, gender=gender))
        return voices

    def default_voice(self) -> str | None:
        voices = self.voices()
        english = [v.id for v in voices if (v.language or "").startswith("en")]
        return (english or [v.id for v in voices] or [None])[0]

    def synthesize(self, text: str, voice: str | None, wpm: int, rate: float) -> AudioClip:
        with tempfile.TemporaryDirectory(prefix="destiny-sapi-", ignore_cleanup_errors=True) as tmp:
            out = Path(tmp) / "speech.wav"
            worker = self._sapi()
            if worker is not None:
                try:
                    worker.call(_pyttsx3_save, text, voice, wpm, str(out), timeout=_synth_timeout(text))
                    if out.is_file() and out.stat().st_size > 44:
                        return _wav_clip(out.read_bytes())
                except EngineUnavailable:
                    raise
                except Exception:
                    pass
                self._give_up_on_pyttsx3()
            return self._powershell_speak(text, voice, wpm, Path(tmp), out)

    def _powershell_speak(self, text: str, voice: str | None, wpm: int, tmp: Path, out: Path) -> AudioClip:
        exe = self._powershell()
        if not exe:
            raise EngineUnavailable(self.is_available()[1])
        text_path = tmp / "text.txt"
        text_path.write_text(text, encoding="utf-8")
        env = dict(os.environ)
        env.update(
            DESTINY_TTS_TEXT=str(text_path),
            DESTINY_TTS_OUT=str(out),
            DESTINY_TTS_VOICE=voice or "",
            DESTINY_TTS_RATE=str(_sapi_rate(wpm)),
        )
        cmd = [exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", _encoded(_PS_SPEAK)]
        proc = _run(cmd, env=env, timeout=_synth_timeout(text) + 30)
        if proc.returncode != 0 or not out.is_file():
            raise TTSError(_FAILED_MSG)
        return _wav_clip(out.read_bytes())


# --------------------------------------------------------------------------- engine


def _platform_key() -> str:
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform == "win32":
        return "win32"
    return "linux"


_BACKENDS: dict[str, type[_Backend]] = {"darwin": _MacBackend, "win32": _WindowsBackend, "linux": _EspeakBackend}


class SystemEngine(TTSEngine):
    name = "system"
    label = "System voice (offline)"
    online = False

    def __init__(self) -> None:
        self._backends: dict[str, _Backend] = {}
        self._lock = threading.Lock()

    def _backend(self) -> _Backend:
        key = _platform_key()
        with self._lock:
            backend = self._backends.get(key)
            if backend is None:
                backend = self._backends[key] = _BACKENDS[key]()
            return backend

    def is_available(self) -> tuple[bool, str]:
        return self._backend().is_available()

    def list_voices(self) -> list[Voice]:
        return self._backend().voices()

    def default_voice(self) -> str | None:
        return self._backend().default_voice()

    def synthesize(self, text: str, voice: str | None = None, rate: float = 1.0) -> AudioClip:
        if not text or not text.strip():
            raise TTSError("There's no text to read aloud.")
        backend = self._backend()
        ok, reason = backend.is_available()
        if not ok:
            raise EngineUnavailable(reason)
        if voice:
            _check_voice_arg(voice)
            if not backend.knows_voice(voice):
                raise EngineUnavailable(
                    f"The system voice '{voice}' isn't available on this computer. Pick another voice in Settings."
                )
        else:
            voice = backend.default_voice()
        rate = min(max(float(rate or 1.0), 0.25), 4.0)
        return backend.synthesize(text, voice, int(round(BASE_WPM * rate)), rate)
