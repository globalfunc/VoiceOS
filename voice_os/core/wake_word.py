"""
Wake-word listener using openWakeWord.

Runs the detection loop in a background thread.  When the configured wake
phrase is detected above the threshold, it calls the provided ``on_detected``
callback and pauses itself until ``resume()`` is called.

Usage:
    from voice_os.core.wake_word import WakeWordListener

    def handle():
        print("Wake word!")

    listener = WakeWordListener(on_detected=handle)
    listener.start()
    ...
    listener.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from voice_os.config.settings import settings
from voice_os.core.mic_manager import MicManager, SAMPLE_RATE, BLOCKSIZE

logger = logging.getLogger(__name__)

# openWakeWord scores are 0.0–1.0; above this we consider it a detection.
DETECTION_THRESHOLD = 0.5

# openWakeWord expects 80 ms chunks at 16 kHz → 1280 samples.
_OWW_CHUNK = 1280


class WakeWordListener:
    """
    Wraps openWakeWord in a background thread with start/pause/resume/stop.

    openWakeWord ships a handful of pre-trained models (hey_jarvis, alexa, etc.)
    as well as a generic model that can be fine-tuned.  For Phase 1 we use the
    built-in "hey_mycroft" / "hey_jarvis" model as a placeholder until the user
    trains a custom "OS Assistant" model via the UI.

    The model name comes from config.wake_phrase but openWakeWord uses a *model
    file*, not raw text.  Phase 1 loads the pre-bundled model; Phase 6 adds the
    retraining flow.
    """

    def __init__(
        self,
        on_detected: Callable[[], None],
        mic_manager: Optional[MicManager] = None,
    ) -> None:
        self._on_detected = on_detected
        self._mic = mic_manager or MicManager()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # not paused initially
        self._thread: Optional[threading.Thread] = None
        self._oww_model = None
        self._audio_buffer: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load the model and begin the detection loop in a daemon thread."""
        self._load_model()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ww-listener")
        self._thread.start()
        logger.info(
            "Wake-word listener started (placeholder: 'alexa'; target phrase: '%s')",
            settings.wake_phrase,
        )

    def stop(self) -> None:
        """Signal the loop to exit and join the thread."""
        self._stop_event.set()
        self._paused.set()  # unblock if paused
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Wake-word listener stopped.")

    def pause(self) -> None:
        """Pause detection while the assistant is active/speaking."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume detection (called when the assistant returns to IDLE)."""
        self._paused.set()
        logger.debug("Wake-word listener resumed.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model

            # Download pre-trained models on first run (hey_mycroft, alexa, etc.)
            logger.info("Downloading openWakeWord pre-trained models (first run only)…")
            openwakeword.utils.download_models()

            # Use the pre-trained "alexa" wakeword as Phase 1 placeholder —
            # it is reliably included in the downloaded set.
            # Phase 6 will add custom "OS Assistant" model training via the UI.
            self._oww_model = Model(wakeword_models=["alexa"], inference_framework="onnx")
            logger.info("openWakeWord model loaded (alexa placeholder).")
        except Exception as exc:
            logger.error("Failed to load openWakeWord model: %s", exc)
            raise

    def _run(self) -> None:
        """Main detection loop — runs in the background thread."""
        stream = self._mic.open_input_stream(callback=self._audio_callback)
        stream.start()
        chunks_processed = 0
        try:
            while not self._stop_event.is_set():
                self._paused.wait()  # blocks while paused
                if self._stop_event.is_set():
                    break

                chunk = self._drain_buffer()
                if chunk is None:
                    time.sleep(0.01)
                    continue

                chunks_processed += 1

                # openWakeWord expects int16 numpy array
                prediction = self._oww_model.predict(chunk)

                # Log scores every ~5 seconds (≈100 chunks @ 1280 samples/16kHz ≈ 8ms each)
                # so we can confirm audio is flowing and see score magnitudes.
                if chunks_processed % 100 == 0:
                    score_str = "  ".join(
                        f"{k}={v:.3f}" for k, v in prediction.items()
                    )
                    logger.debug("WW scores [chunk %d]: %s", chunks_processed, score_str)

                for model_name, score in prediction.items():
                    if score >= DETECTION_THRESHOLD:
                        logger.info(
                            "Wake word detected! model=%s score=%.3f", model_name, score
                        )
                        self.pause()
                        try:
                            self._on_detected()
                        except Exception:
                            logger.exception("Exception in on_detected callback.")
                        break
        except Exception:
            logger.exception("Wake word run loop crashed.")
        finally:
            stream.stop()
            stream.close()

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status) -> None:
        """Called by sounddevice on each audio block; feeds the buffer."""
        self._audio_buffer.append(indata[:, 0].astype(np.int16))

    def _drain_buffer(self) -> Optional[np.ndarray]:
        """Return a chunk of _OWW_CHUNK samples if enough data is buffered."""
        if not self._audio_buffer:
            return None
        combined = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        if len(combined) < _OWW_CHUNK:
            # Not enough data yet; put it back
            self._audio_buffer.append(combined)
            return None
        # Return exactly one chunk; keep the remainder
        chunk = combined[:_OWW_CHUNK]
        remainder = combined[_OWW_CHUNK:]
        if len(remainder):
            self._audio_buffer.append(remainder)
        return chunk
