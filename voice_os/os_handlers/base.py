"""Abstract OS handler interface. Linux and Windows implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class OSHandler(ABC):
    """Abstracts OS-level operations so tools stay platform-agnostic."""

    # ------------------------------------------------------------------
    # File / app operations
    # ------------------------------------------------------------------

    @abstractmethod
    def open_file(self, path: str, app: Optional[str] = None) -> bool:
        """Open ``path`` with ``app`` (or the system default). Returns True on success."""

    @abstractmethod
    def find_app(self, name: str) -> Optional[str]:
        """
        Search PATH (and OS app registries) for an executable named ``name``.
        Returns the full path or None if not found.
        """

    @abstractmethod
    def list_apps(self) -> List[str]:
        """List installed / known application names (best-effort)."""

    def find_app_candidates(
        self, query: str, top_n: int = 3, min_score: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """
        Fuzzy-match ``query`` against installed app names.

        Returns up to ``top_n`` tuples of (display_name, launch_path, score)
        sorted by descending score.  Only results with score >= ``min_score``
        are included.  Default implementation returns an empty list; platform
        handlers override for richer results.
        """
        return []

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def find_processes(self, name: str) -> List[Tuple[int, str]]:
        """
        Find running processes whose command line contains ``name`` (substring,
        case-insensitive).  Returns [(pid, process_name), ...].
        Default: empty list.  Platform handlers override.
        """
        return []

    def find_processes_fuzzy(
        self, query: str, min_score: float = 0.5
    ) -> List[Tuple[str, List[int], float]]:
        """
        Fuzzy-match ``query`` against running process names.
        Returns [(process_name, [pids], score), ...] sorted descending by score.
        Default: empty list.  Platform handlers override.
        """
        return []

    def close_processes(self, pids: List[int]) -> bool:
        """
        Gracefully terminate the given PIDs (SIGTERM / equivalent).
        Returns True if all signals were sent without error.
        Default: False.  Platform handlers override.
        """
        return False

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    @abstractmethod
    def get_volume(self) -> int:
        """Return current default output volume as 0–100."""

    @abstractmethod
    def set_volume(self, level: int) -> None:
        """Set default output volume to ``level`` (0–100, clamped)."""

    # ------------------------------------------------------------------
    # Power management
    # ------------------------------------------------------------------

    @abstractmethod
    def sleep(self) -> None:
        """Suspend / sleep the system."""

    @abstractmethod
    def shutdown(self) -> None:
        """Power off the system."""

    @abstractmethod
    def restart(self) -> None:
        """Reboot the system."""
