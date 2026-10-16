"""The browser's built-in Web Speech voices.

The web reader speaks with ``speechSynthesis`` itself, so this engine only
exists so the UI can list and choose it; it can't produce audio files.
"""

from __future__ import annotations

from .base import AudioClip, EngineUnavailable, TTSEngine, Voice


class BrowserEngine(TTSEngine):
    name = "browser"
    label = "Browser voice"
    online = False

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def list_voices(self) -> list[Voice]:
        return []  # the browser lists its own voices

    def synthesize(self, text: str, voice: str | None = None, rate: float = 1.0) -> AudioClip:
        raise EngineUnavailable("The browser voice speaks inside the web reader and can't be used for this.")
