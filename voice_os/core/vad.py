"""
Voice Activity Detection using Silero VAD.

Records audio after wake-word activation, uses Silero to detect when the user
has finished speaking (1.5 s of silence after speech begins), then returns the
captured audio buffer as a numpy int16 array ready for Whisper.

Usage:
    from voice_os.core.vad import VADRecorder
    recorder = VADRecorder()
    audio = recorder.record_until_silence()  # blocks; returns np.ndarray int16
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
import torch

from voice_os.core.mic_manager import MicManager, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Silero VAD parameters
_VAD_THRESHOLD = 0.5          # probability threshold for "speech"
_SILENCE_DURATION = 1.5       # seconds of continuous silence → end of utterance
_MAX_RECORD_SECS = 30         # safety cap — prevent endless recording
_SILERO_CHUNK = 512           # samples per VAD inference (must be 256 or 512 at 16 kHz)


class VADRecorder:
    """
    Records one utterance from the mic, stopping on 1.5 s of silence.

    The Silero VAD model is loaded once at construction time (~2 MB).
    """

    def __init__(self, mic_manager: Optional[MicManager] = None) -> None:
        self._mic = mic_manager or MicManager()
        self._model, self._utils = self._load_silero()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_until_silence(self, prefix_audio: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Block until the user finishes speaking.

        Args:
            prefix_audio: Optional audio to prepend, e.g. trailing audio captured
                during wake-word detection (contains the start of an inline command).

        Returns:
            np.ndarray: int16 audio samples at SAMPLE_RATE Hz.
        """
        audio_frames: list[np.ndarray] = []
        vad_chunk_buffer: list[np.ndarray] = []
        speech_started = False
        silence_start: Optional[float] = None
        recording_done = threading.Event()
        result_frames: list[np.ndarray] = []

        # Pre-process any prefix audio (trailing WW buffer) through the VAD so
        # that speech_started and silence_start are already set correctly when
        # the live stream opens.  This captures the portion of an inline command
        # that was spoken concurrently with or just after the wake word.
        if prefix_audio is not None and len(prefix_audio) > 0:
            audio_frames.append(prefix_audio)
            buf = prefix_audio.copy()
            while len(buf) >= _SILERO_CHUNK:
                window = buf[:_SILERO_CHUNK]
                buf = buf[_SILERO_CHUNK:]
                prob = self._vad_probability(window)
                if prob >= _VAD_THRESHOLD:
                    speech_started = True
                    silence_start = None
                elif speech_started and silence_start is None:
                    silence_start = time.monotonic()
            # Seed the live buffer with any sub-chunk remainder
            if len(buf) > 0:
                vad_chunk_buffer.append(buf)

        def callback(indata: np.ndarray, frames, time_info, status):
            nonlocal speech_started, silence_start

            if recording_done.is_set():
                return

            chunk = indata[:, 0].astype(np.int16)
            audio_frames.append(chunk)
            vad_chunk_buffer.append(chunk)

            # Run VAD when we have enough samples
            buffered = np.concatenate(vad_chunk_buffer)
            while len(buffered) >= _SILERO_CHUNK:
                window = buffered[:_SILERO_CHUNK]
                buffered = buffered[_SILERO_CHUNK:]
                prob = self._vad_probability(window)

                if prob >= _VAD_THRESHOLD:
                    speech_started = True
                    silence_start = None
                elif speech_started:
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= _SILENCE_DURATION:
                        result_frames.extend(audio_frames)
                        recording_done.set()
                        return

            vad_chunk_buffer.clear()
            if len(buffered):
                vad_chunk_buffer.append(buffered)

        stream = self._mic.open_input_stream(callback=callback)
        stream.start()

        deadline = time.monotonic() + _MAX_RECORD_SECS
        while not recording_done.is_set():
            if time.monotonic() > deadline:
                logger.warning("VAD: max recording duration reached without silence; stopping.")
                result_frames.extend(audio_frames)
                break
            time.sleep(0.05)

        stream.stop()
        stream.close()

        if not result_frames:
            logger.warning("VAD: no audio captured.")
            return np.array([], dtype=np.int16)

        audio = np.concatenate(result_frames)
        logger.info(
            "VAD: recorded %.2f s of audio (%d samples).",
            len(audio) / SAMPLE_RATE,
            len(audio),
        )
        return audio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_silero():
        """Download (first time) and load the Silero VAD model from torch hub."""
        logger.info("Loading Silero VAD model…")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        model.eval()
        logger.info("Silero VAD model loaded.")
        return model, utils

    def _vad_probability(self, chunk: np.ndarray) -> float:
        """Run one Silero inference; return speech probability [0.0, 1.0]."""
        tensor = torch.from_numpy(chunk).float() / 32768.0  # int16 → float [-1, 1]
        with torch.no_grad():
            prob = self._model(tensor, SAMPLE_RATE).item()
        return prob
