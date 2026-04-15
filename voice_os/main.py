"""
VoiceOS — Phase 1 entry point.

State machine:
    IDLE  →  ACTIVE  →  PROCESSING  →  SPEAKING  →  IDLE
               ↑                                       │
               └──────────── idle timer fires ─────────┘

Phase 1 goal: wake word → VAD captures command → Whisper transcribes →
              TTS echoes text back.  No LLM involved yet.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
from enum import Enum, auto

from voice_os.config.settings import settings
from voice_os.core.mic_manager import MicManager
from voice_os.core.wake_word import WakeWordListener
from voice_os.core.vad import VADRecorder
from voice_os.core.speech_to_text import WhisperTranscriber
from voice_os.core.tts.pyttsx3_tts import build_tts_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice_os.main")

# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE       = auto()
    ACTIVE     = auto()   # capturing utterance
    PROCESSING = auto()   # transcribing
    SPEAKING   = auto()   # TTS playing


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------

IDLE_TIMEOUT_SECS = 60   # 1 minute of inactivity → session clear + speak notice


class VoiceAssistant:
    """Orchestrates the 4-state voice loop for Phase 1."""

    def __init__(self) -> None:
        self._state = State.IDLE
        self._state_lock = threading.RLock()
        self._idle_timer: threading.Timer | None = None

        # Shared mic manager so all layers use the same device config
        self._mic = MicManager()

        logger.info("Loading TTS…")
        self._tts = build_tts_service()

        logger.info("Loading Whisper STT…")
        self._stt = WhisperTranscriber()

        logger.info("Loading VAD recorder…")
        self._vad = VADRecorder(mic_manager=self._mic)

        logger.info("Loading wake-word listener…")
        self._ww = WakeWordListener(
            on_detected=self._on_wake_word,
            mic_manager=self._mic,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the assistant and block until shutdown (Ctrl-C / SIGTERM)."""
        logger.info("VoiceOS Phase 1 starting…")
        self._tts.speak(f"Voice OS is ready. Say '{settings.wake_phrase}' to begin.")
        self._reset_idle_timer()
        self._ww.start()

        # Block the main thread until a shutdown signal arrives
        shutdown = threading.Event()

        def _sig_handler(sig, frame):
            logger.info("Shutdown signal received.")
            shutdown.set()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        logger.info("Listening for wake word: '%s'", settings.wake_phrase)
        shutdown.wait()
        self._shutdown()

    def _shutdown(self) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
        self._ww.stop()
        self._tts.speak("Goodbye.")
        logger.info("VoiceOS stopped.")

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _on_wake_word(self) -> None:
        """Called by WakeWordListener when the wake phrase is detected."""
        with self._state_lock:
            if self._state != State.IDLE:
                logger.info("Wake word ignored — current state is %s, not IDLE.", self._state.name)
                self._ww.resume()
                return
            self._set_state(State.ACTIVE)

        self._reset_idle_timer()
        threading.Thread(target=self._capture_and_respond_safe, daemon=True).start()

    def _capture_and_respond_safe(self) -> None:
        try:
            self._capture_and_respond()
        except Exception:
            logger.exception("Unhandled error in capture/respond pipeline — returning to IDLE.")
            self._go_idle()

    def _capture_and_respond(self) -> None:
        """ACTIVE → PROCESSING → SPEAKING → IDLE."""
        # --- ACTIVE: capture utterance ---
        logger.info("[ACTIVE] Capturing utterance via VAD…")
        self._tts.speak("Listening.")
        audio = self._vad.record_until_silence()
        self._reset_idle_timer()

        # --- PROCESSING: transcribe ---
        self._set_state(State.PROCESSING)
        logger.info("[PROCESSING] Transcribing…")
        text = self._stt.transcribe(audio)
        self._reset_idle_timer()

        if not text:
            logger.warning("Transcription returned empty — returning to IDLE.")
            self._tts.speak("Sorry, I didn't catch that.")
            self._go_idle()
            return

        logger.info("[PROCESSING] Transcribed: '%s'", text)

        # --- SPEAKING: echo back (Phase 1 — no LLM) ---
        self._set_state(State.SPEAKING)
        logger.info("[SPEAKING] Echoing transcription.")
        self._tts.speak(f"You said: {text}")
        self._reset_idle_timer()

        # --- back to IDLE ---
        self._go_idle()

    def _go_idle(self) -> None:
        self._set_state(State.IDLE)
        logger.info("[IDLE] Waiting for wake word.")
        self._ww.resume()

    def _on_idle_timeout(self) -> None:
        """Called after IDLE_TIMEOUT_SECS of no activity."""
        with self._state_lock:
            current = self._state
        logger.info("Idle timeout fired (state=%s).", current.name)
        if current == State.IDLE:
            self._tts.speak("Going to idle state.")
            # In Phase 2+ this will also clear session memory
            logger.info("Session memory cleared (Phase 2).")
        # Reset the timer so we keep firing periodically while idle
        # (keeps the "going to idle" message from repeating — Phase 1 only notifies once)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_state(self, new_state: State) -> None:
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            logger.debug("State: %s → %s", old.name, new_state.name)

    def _reset_idle_timer(self) -> None:
        """Cancel any running idle timer and start a fresh one."""
        if self._idle_timer:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(IDLE_TIMEOUT_SECS, self._on_idle_timeout)
        self._idle_timer.daemon = True
        self._idle_timer.start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
