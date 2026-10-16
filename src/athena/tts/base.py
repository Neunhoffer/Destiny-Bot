"""The interface every text-to-speech engine implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel


class TTSError(RuntimeError):
    """Synthesis failed; the message is safe to show to the user."""


class EngineUnavailable(TTSError):
    """The engine (or the requested voice) is not installed / not usable right now."""


class Voice(BaseModel):
    id: str  # engine-specific identifier passed back to synthesize()
    name: str  # human friendly, e.g. "Lessac (US English, medium quality)"
    engine: str
    language: str | None = None  # BCP-47-ish, e.g. "en-US"
    gender: str | None = None
    online: bool = False  # True if using this voice sends text over the internet
    installed: bool = True  # False for downloadable-but-not-downloaded voices
    size_mb: float | None = None


@dataclass(frozen=True)
class AudioClip:
    data: bytes
    mime: str  # "audio/wav" or "audio/mpeg"
    sample_rate: int | None = None  # known for WAV output

    @property
    def extension(self) -> str:
        return ".mp3" if self.mime == "audio/mpeg" else ".wav"


class TTSEngine(ABC):
    name: ClassVar[str]  # "piper" | "system" | "edge" | "browser"
    label: ClassVar[str]  # shown in the UI
    online: ClassVar[bool] = False  # True if it sends text over the internet

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(usable, reason). ``reason`` explains how to fix it when not usable."""

    @abstractmethod
    def list_voices(self) -> list[Voice]:
        """Installed voices for this engine (may be empty)."""

    def default_voice(self) -> str | None:
        voices = self.list_voices()
        return voices[0].id if voices else None

    @abstractmethod
    def synthesize(self, text: str, voice: str | None = None, rate: float = 1.0) -> AudioClip:
        """Speak ``text`` (one utterance, plain text) with ``voice`` (None = default).

        ``rate`` is a speed multiplier (1.0 = normal). The web reader always asks
        for 1.0 and changes speed in the browser; the CLI and exporter pass other
        values. Raises :class:`EngineUnavailable` or :class:`TTSError`.
        """
