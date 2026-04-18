"""
VoiceOS — Phase 2 entry point.

State machine:
    IDLE  →  ACTIVE  →  PROCESSING  →  SPEAKING  →  IDLE
               ↑                                       │
               └──────────── idle timer fires ─────────┘

Phase 2 adds:
  - LangChain ReAct agent (Ollama) with volume, open-app, and system-control tools
  - In-session memory (last 5 turns), cleared on idle timeout
  - WW listener muted during all TTS playback (fixes own-voice false triggers)
  - Wake phrase stripped from Whisper transcription
"""
from __future__ import annotations

import logging
import pathlib
import re
import signal
import threading
from enum import Enum, auto

import numpy as np

from voice_os.config.settings import settings
from voice_os.core.mic_manager import MicManager
from voice_os.core.wake_word import WakeWordListener
from voice_os.core.vad import VADRecorder
from voice_os.core.speech_to_text import WhisperTranscriber
from voice_os.core.tts.pyttsx3_tts import build_tts_service
from voice_os.agent.executor import AgentRunner

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

# Third-party namespaces that spam at DEBUG/INFO even when we don't need them.
_NOISY_LOGGERS = (
    "httpcore",
    "httpx",
    "urllib3",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langgraph",
    "openai",
    "asyncio",
    "ollama",
)


def _configure_logging() -> None:
    """
    Set up console logging.

    minimal_debug_logs=True  (default): root at WARNING, voice_os.* at INFO,
                                        all known noisy third-party loggers
                                        silenced to WARNING.
    minimal_debug_logs=False           : root at DEBUG — full firehose.
    """
    if settings.minimal_debug_logs:
        logging.basicConfig(
            level=logging.WARNING,
            format=_LOG_FORMAT,
            datefmt=_LOG_DATEFMT,
        )
        logging.getLogger("voice_os").setLevel(logging.INFO)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format=_LOG_FORMAT,
            datefmt=_LOG_DATEFMT,
        )


_configure_logging()
logger = logging.getLogger("voice_os.main")

# Dedicated conversation log — one line per user turn, tool execution, and response.
_CONV_LOG_PATH = pathlib.Path.home() / ".voice_os" / "conversation.log"
_CONV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_conv_file_handler = logging.FileHandler(_CONV_LOG_PATH, encoding="utf-8")
_conv_file_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
_conv_logger = logging.getLogger("voice_os.conversation")
_conv_logger.setLevel(logging.INFO)
_conv_logger.addHandler(_conv_file_handler)
_conv_logger.propagate = False   # keep conversation events out of the debug log

# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE       = auto()
    ACTIVE     = auto()   # capturing utterance
    PROCESSING = auto()   # transcribing + agent
    SPEAKING   = auto()   # TTS playing


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------

IDLE_TIMEOUT_SECS = 60   # 1 minute of inactivity → session clear + speak notice

# Words to strip from the front of the Whisper transcription.
# Covers the Phase 1 placeholder wake word ("alexa") plus the configured phrase.
_WAKE_WORD_TOKENS = re.compile(
    r"^(?:alexa|hey\s+alexa|os\s+assistant|hey\s+os|assistant)[,.\s]*",
    re.IGNORECASE,
)

# Phrases that dismiss the assistant back to idle and clear the session.
_DISMISS_PATTERN = re.compile(
    r"^(?:bye|goodbye|see\s+you|that'?s?\s+all|dismiss)[\s,.]*(alexa)?[\s.]*$",
    re.IGNORECASE,
)


class VoiceAssistant:
    """Orchestrates the 4-state voice loop."""

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

        logger.info("Building agent runner…")
        self._agent = AgentRunner(tts=self._tts, stt=self._stt, vad=self._vad)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the assistant and block until shutdown (Ctrl-C / SIGTERM)."""
        logger.info("VoiceOS Phase 2 starting…")
        # Speak startup message *before* WW listener starts — no need to mute WW.
        self._tts.speak(f"Voice OS is ready. Say '{settings.wake_phrase}' to begin.")
        self._reset_idle_timer()
        self._ww.start()

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
        # WW is stopped — no need to mute it; speak directly.
        self._tts.speak("Goodbye.")
        logger.info("VoiceOS stopped.")

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _on_wake_word(self, trailing_audio: np.ndarray) -> None:
        """Called by WakeWordListener when the wake phrase is detected."""
        with self._state_lock:
            if self._state != State.IDLE:
                logger.info(
                    "Wake word ignored — current state is %s, not IDLE.",
                    self._state.name,
                )
                self._ww.resume()
                return
            self._set_state(State.ACTIVE)

        self._reset_idle_timer()
        threading.Thread(
            target=self._capture_and_respond_safe,
            args=(trailing_audio,),
            daemon=True,
        ).start()

    def _capture_and_respond_safe(self, trailing_audio: np.ndarray) -> None:
        try:
            self._capture_and_respond(trailing_audio)
        except Exception:
            logger.exception(
                "Unhandled error in capture/respond pipeline — returning to IDLE."
            )
            self._go_idle()

    def _capture_and_respond(self, trailing_audio: np.ndarray) -> None:
        """ACTIVE → PROCESSING → SPEAKING → IDLE."""

        # --- ACTIVE: capture utterance ---
        # Start VAD immediately after wake-word detection — no blocking TTS prompt.
        # Speaking "Listening." delays VAD by ~500 ms during which the user's
        # command is already being spoken and gets lost.  VAD will record until
        # the user stops speaking; trailing_audio (buffered during WW detection)
        # is prepended so nothing is dropped at the seam.
        logger.info(
            "[ACTIVE] Capturing utterance via VAD (%d trailing samples from WW)…",
            len(trailing_audio),
        )
        audio = self._vad.record_until_silence(
            prefix_audio=trailing_audio if len(trailing_audio) > 0 else None
        )
        self._reset_idle_timer()

        # --- PROCESSING: transcribe ---
        self._set_state(State.PROCESSING)
        logger.info("[PROCESSING] Transcribing…")
        raw_text = self._stt.transcribe(audio)
        self._reset_idle_timer()

        if not raw_text:
            logger.warning("Transcription returned empty — returning to IDLE silently.")
            self._go_idle()
            return

        # Strip wake-phrase prefix that Whisper may include in the transcript.
        text = _WAKE_WORD_TOKENS.sub("", raw_text).strip()
        if not text:
            logger.warning("Nothing left after stripping wake phrase — returning to IDLE silently.")
            self._go_idle()
            return

        logger.info("[PROCESSING] Transcribed: '%s' (raw: '%s')", text, raw_text)

        # --- PROCESSING: check for dismiss command before hitting the agent ---
        if _DISMISS_PATTERN.match(text):
            logger.info("[PROCESSING] Dismiss command detected — going idle.")
            self._agent.clear_session()
            self._speak("Goodbye.")
            self._go_idle()
            return

        # --- PROCESSING: run agent ---
        logger.info("[PROCESSING] Running agent…")
        response = self._agent.handle(text)
        self._reset_idle_timer()

        # --- SPEAKING: TTS ---
        self._set_state(State.SPEAKING)
        logger.info("[SPEAKING] Speaking response.")
        self._speak(response)
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
            # Mute WW while speaking so TTS doesn't trigger itself.
            self._ww.resume()          # resume — still idle after speaking
            self._agent.clear_session()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _speak(self, text: str) -> None:
        """
        Speak ``text`` while keeping the wake-word listener muted.

        Always calls pause() before speaking.  Callers that want to resume
        WW afterwards must do so explicitly (via _go_idle or _ww.resume()).

        Rationale: the TTS audio leaks into the microphone and can falsely
        trigger the wake-word detector (the so-called "own voice" problem).
        Muting WW for the entire duration of TTS playback prevents this.
        """
        self._ww.pause()
        self._tts.speak(text)

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
