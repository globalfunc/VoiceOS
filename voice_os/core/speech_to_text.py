"""
Speech-to-text using faster-whisper (CTranslate2 backend).

Model: small.en (English-only, ~244 MB, very fast on CPU).
The WhisperTranscriber loads once at construction; subsequent calls are cheap.

Usage:
    from voice_os.core.speech_to_text import WhisperTranscriber
    stt = WhisperTranscriber()
    text = stt.transcribe(audio_np)   # audio_np: np.ndarray int16 @ 16 kHz
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "small.en"
_COMPUTE_TYPE = "int8"     # fast on CPU; use "float16" if CUDA available


class WhisperTranscriber:
    """Wraps faster-whisper for single-shot transcription."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str = "auto") -> None:
        self._model = self._load(model_name, device)
        self._initial_prompt: Optional[str] = None   # built lazily on first transcribe()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe int16 audio sampled at 16 kHz.

        Args:
            audio: 1-D numpy int16 array.

        Returns:
            Stripped transcription string, or empty string on failure.
        """
        if audio is None or len(audio) == 0:
            logger.warning("STT: received empty audio buffer.")
            return ""

        # faster-whisper expects float32 in [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        if self._initial_prompt is None:
            self._initial_prompt = self._build_initial_prompt()
            logger.debug("Whisper initial_prompt (%d chars): %s…", len(self._initial_prompt), self._initial_prompt[:120])

        try:
            segments, info = self._model.transcribe(
                audio_f32,
                language="en",
                beam_size=5,
                vad_filter=True,           # internal VAD for trailing silence
                vad_parameters=dict(min_silence_duration_ms=300),
                initial_prompt=self._initial_prompt,
            )
            text = " ".join(seg.text for seg in segments).strip()
            logger.info("STT: '%s' (lang=%s, prob=%.2f)", text, info.language, info.language_probability)
            return text
        except Exception as exc:
            logger.error("STT transcription failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_initial_prompt(self) -> str:
        """
        Build a Whisper initial_prompt from the installed app catalog so the
        STT model recognises proper nouns (HeidiSQL, OBS Studio, DBeaver, …)
        instead of phonetically mangling them.

        Called once on the first transcribe() and cached.  Falls back to a
        minimal static string if the OS handler is unavailable.
        """
        try:
            from voice_os.os_handlers import get_os_handler
            names = get_os_handler().list_app_names(top_n=200)
            # Exclude very short names (≤3 chars: "sh", "ls", "vi").
            candidates = [n for n in names if len(n) > 3]
            # Sort: proper nouns (mixed-case, e.g. "HeidiSQL", "OBS Studio") first
            # so the most voice-recognition-valuable names fit within the budget.
            # All-lowercase names ("dbeaver-ce") come second.
            def _sort_key(n: str) -> int:
                return 0 if not n.islower() else 1
            candidates.sort(key=_sort_key)
            # Whisper's initial_prompt budget is ~375 tokens (~1 500 chars).
            joined = ", ".join(candidates)[:1400]
            prompt = f"Voice OS commands: open, close, volume, shutdown. Apps: {joined}."
            logger.info("Built Whisper initial_prompt from %d app names.", len(candidates))
            return prompt
        except Exception as exc:
            logger.warning("Could not build dynamic Whisper prompt: %s", exc)
            return (
                "Voice OS assistant commands: open, close, volume, shutdown, restart, sleep."
            )

    @staticmethod
    def _load(model_name: str, device: str):
        """Load the faster-whisper model (downloads on first run, ~244 MB)."""
        from faster_whisper import WhisperModel
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try compute types from best to most compatible.
        # Pascal GPUs (GTX 10xx) only support int8 on CUDA; newer cards support int8_float16.
        cuda_candidates = ["int8_float16", "int8", "float32"]
        cpu_candidates  = ["int8", "float32"]
        candidates = cuda_candidates if device == "cuda" else cpu_candidates

        for compute_type in candidates:
            try:
                logger.info(
                    "Loading Whisper model '%s' on %s (compute_type=%s)…",
                    model_name, device, compute_type,
                )
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
                logger.info("Whisper model loaded (compute_type=%s).", compute_type)
                return model
            except ValueError as exc:
                logger.warning("compute_type=%s not supported: %s — trying next.", compute_type, exc)

        raise RuntimeError(
            f"Could not load Whisper model '{model_name}' on {device}: "
            "no supported compute type found."
        )
