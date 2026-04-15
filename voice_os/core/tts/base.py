"""Abstract TTS interface. All engines implement TTSService."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TTSService(ABC):
    """Minimal contract every TTS backend must satisfy."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesize and play ``text`` synchronously (blocks until done)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can currently produce audio."""
