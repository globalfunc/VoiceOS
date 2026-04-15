"""Abstract OS handler interface. Linux and Windows implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


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
