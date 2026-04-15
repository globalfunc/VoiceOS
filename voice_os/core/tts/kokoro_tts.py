"""
Kokoro TTS — primary, near-human quality.

Uses kokoro-onnx (~330 MB) loaded once at startup.
Model files are downloaded automatically to ~/.voice_os/models/ on first run.
Audio is played back via sounddevice so we get the same device routing as the mic.

Kokoro API: kokoro.create(text, voice, speed, lang) → (samples_float32, sample_rate)
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import sounddevice as sd

from voice_os.core.tts.base import TTSService

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "af_heart"   # warm female voice; configurable later
_DEFAULT_SPEED = 1.0
_DEFAULT_LANG  = "en-us"

_MODELS_DIR = Path.home() / ".voice_os" / "models"
_ONNX_FILE  = _MODELS_DIR / "kokoro-v1.0.onnx"
_VOICES_FILE = _MODELS_DIR / "voices-v1.0.bin"

# Official release URLs from the kokoro-onnx project
_ONNX_URL   = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _download_if_missing() -> None:
    """Download Kokoro model files to ~/.voice_os/models/ if not already present."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in [(_ONNX_FILE, _ONNX_URL), (_VOICES_FILE, _VOICES_URL)]:
        if path.exists():
            continue
        logger.info("Downloading Kokoro model file: %s → %s", url, path)
        try:
            urllib.request.urlretrieve(url, path, reporthook=_log_progress(path.name))
            logger.info("Downloaded %s (%.1f MB).", path.name, path.stat().st_size / 1e6)
        except Exception as exc:
            logger.error("Failed to download %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            raise


def _log_progress(name: str):
    last = [0]
    def hook(count, block_size, total):
        downloaded = count * block_size
        pct = downloaded / total * 100 if total > 0 else 0
        if pct - last[0] >= 10:
            logger.info("  %s: %.0f%%", name, pct)
            last[0] = pct
    return hook


class KokoroTTS(TTSService):
    """Wraps kokoro-onnx for synchronous speech synthesis + playback."""

    def __init__(
        self,
        voice: str = _DEFAULT_VOICE,
        speed: float = _DEFAULT_SPEED,
        lang: str = _DEFAULT_LANG,
    ) -> None:
        self._voice = voice
        self._speed = speed
        self._lang  = lang
        self._kokoro = None
        self._available = False
        self._load()

    # ------------------------------------------------------------------
    # TTSService interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def speak(self, text: str) -> None:
        if not self._available or not text.strip():
            return
        try:
            samples, sr = self._kokoro.create(
                text,
                voice=self._voice,
                speed=self._speed,
                lang=self._lang,
            )
            # samples is float32; play it and wait until finished
            sd.play(samples, samplerate=sr)
            sd.wait()
        except Exception as exc:
            logger.error("KokoroTTS.speak failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from kokoro_onnx import Kokoro
            _download_if_missing()
            self._kokoro = Kokoro(str(_ONNX_FILE), str(_VOICES_FILE))
            self._available = True
            logger.info("Kokoro TTS loaded.")
        except Exception as exc:
            logger.warning("Kokoro TTS unavailable (%s); will fall back to pyttsx3.", exc)
            self._available = False
