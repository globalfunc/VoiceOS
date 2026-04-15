"""
pyttsx3 TTS — system fallback (no external model required).

pyttsx3 is synchronous and blocks until speech is done.
We initialize it once and reuse the engine to avoid per-call overhead.
"""
from __future__ import annotations

import logging

from voice_os.core.tts.base import TTSService

logger = logging.getLogger(__name__)


class Pyttsx3TTS(TTSService):
    """Wraps pyttsx3 as a synchronous TTS fallback."""

    def __init__(self, rate: int = 175, volume: float = 1.0) -> None:
        self._engine = None
        self._available = False
        self._load(rate, volume)

    # ------------------------------------------------------------------
    # TTSService interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def speak(self, text: str) -> None:
        if not self._available or not text.strip():
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            logger.error("Pyttsx3TTS.speak failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, rate: int, volume: float) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            self._engine = engine
            self._available = True
            logger.info("pyttsx3 TTS initialized (rate=%d, vol=%.1f).", rate, volume)
        except Exception as exc:
            logger.warning("pyttsx3 TTS unavailable: %s", exc)
            self._available = False


def build_tts_service():
    """
    Factory: try Kokoro first, fall back to pyttsx3.
    Import here to avoid circular imports at module level.
    """
    from voice_os.core.tts.kokoro_tts import KokoroTTS

    kokoro = KokoroTTS()
    if kokoro.is_available():
        return kokoro

    logger.warning("Falling back to pyttsx3 TTS.")
    fallback = Pyttsx3TTS()
    if fallback.is_available():
        return fallback

    raise RuntimeError("No TTS backend available. Install kokoro-onnx or pyttsx3.")
