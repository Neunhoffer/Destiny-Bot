"""Microsoft Edge online neural voices (edge-tts). Sends the text to Microsoft.

Only used when the user turns on online voices in Settings.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from . import clamp_rate
from .base import AudioClip, EngineUnavailable, TTSEngine, TTSError, Voice

DEFAULT_VOICE = "en-US-AvaNeural"
SAMPLE_RATE = 24000  # edge-tts default output: audio-24khz-48kbitrate-mono-mp3
NETWORK_ERROR = (
    "Couldn't reach Microsoft's online voices. Check your internet connection or switch to an offline voice."
)

# (id, display name, language, gender). Listing never touches the network.
_VOICES: list[tuple[str, str, str, str]] = [
    ("en-US-AvaNeural", "Ava (US English)", "en-US", "female"),
    ("en-US-AndrewNeural", "Andrew (US English)", "en-US", "male"),
    ("en-US-EmmaNeural", "Emma (US English)", "en-US", "female"),
    ("en-US-BrianNeural", "Brian (US English)", "en-US", "male"),
    ("en-US-AriaNeural", "Aria (US English)", "en-US", "female"),
    ("en-US-JennyNeural", "Jenny (US English)", "en-US", "female"),
    ("en-US-GuyNeural", "Guy (US English)", "en-US", "male"),
    ("en-GB-SoniaNeural", "Sonia (British English)", "en-GB", "female"),
    ("en-GB-RyanNeural", "Ryan (British English)", "en-GB", "male"),
    ("en-AU-NatashaNeural", "Natasha (Australian English)", "en-AU", "female"),
]


def _rate_string(rate: float) -> str:
    # Microsoft's voices only honour 0.5x-2x; ask for no more than that.
    percent = round((clamp_rate(rate, 0.5, 2.0) - 1.0) * 100)
    return f"{percent:+d}%"


class EdgeEngine(TTSEngine):
    name = "edge"
    label = "Microsoft online voices"
    online = True

    def is_available(self) -> tuple[bool, str]:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False, (
                "Microsoft online voices need the edge-tts package, which isn't installed. "
                "Reinstall Athena with online voices included, or use an offline voice."
            )
        return True, ""

    def list_voices(self) -> list[Voice]:
        return [
            Voice(id=vid, name=name, engine=self.name, language=lang, gender=gender, online=True)
            for vid, name, lang, gender in _VOICES
        ]

    def default_voice(self) -> str | None:
        return DEFAULT_VOICE

    def synthesize(self, text: str, voice: str | None = None, rate: float = 1.0) -> AudioClip:
        if not text or not text.strip():
            raise TTSError("There's no text to read aloud.")
        ok, reason = self.is_available()
        if not ok:
            raise EngineUnavailable(reason)
        data = _run_coroutine_blocking(lambda: _collect(text, voice or DEFAULT_VOICE, _rate_string(rate)))
        if not data:
            raise TTSError("Microsoft's online voice sent back no audio for this text.")
        return AudioClip(data, "audio/mpeg", SAMPLE_RATE)


async def _collect(text: str, voice: str, rate: str) -> bytes:
    import aiohttp
    import edge_tts
    from edge_tts import exceptions as edge_exc

    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
    except ValueError as exc:  # bad voice name / rate format
        raise EngineUnavailable(f"'{voice}' isn't a Microsoft online voice. Pick another voice in Settings.") from exc
    audio = bytearray()
    try:
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                audio += chunk["data"]
    except edge_exc.NoAudioReceived as exc:
        raise TTSError("Microsoft's online voice sent back no audio for this text.") from exc
    except (edge_exc.EdgeTTSException, aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
        raise TTSError(NETWORK_ERROR) from exc
    return bytes(audio)


def _run_coroutine_blocking(make_coro: Any) -> Any:
    """``asyncio.run`` here, or in a helper thread if this thread already runs a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())

    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["value"] = asyncio.run(make_coro())
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            result["error"] = exc

    thread = threading.Thread(target=worker, name="athena-edge-tts", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
