"""
In-session conversation history.

Keeps the last N (user, assistant) turn pairs and formats them as a
context string for the LLM prompt.  Cleared when the idle timer fires.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Tuple


class SessionMemory:
    """Ring-buffer of the last ``maxlen`` conversation turns."""

    def __init__(self, maxlen: int = 5) -> None:
        self._history: Deque[Tuple[str, str]] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, user: str, assistant: str) -> None:
        """Append one (user, assistant) turn."""
        self._history.append((user, assistant))

    def get_context(self) -> str:
        """Return history formatted for insertion into the LLM prompt.

        Returns an empty string when there is no history yet.
        """
        if not self._history:
            return ""
        lines: list[str] = []
        for user, assistant in self._history:
            lines.append(f"User: {user}")
            lines.append(f"Assistant: {assistant}")
        return "\n".join(lines)

    def get_turns(self) -> list[tuple[str, str]]:
        """Return all stored turns as a list of (user, assistant) pairs."""
        return list(self._history)

    def clear(self) -> None:
        """Wipe all stored turns (called on idle timeout)."""
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)

    def __bool__(self) -> bool:
        return bool(self._history)
