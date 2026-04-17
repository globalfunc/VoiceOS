"""
LinuxAppCatalog — single source of truth for user-facing installed apps.

Reads .desktop files from all standard locations and returns a filtered,
deduplicated list of dicts.  Only apps that should appear in an app launcher
are included:
  - NoDisplay=true  → excluded (background daemons, portals, helpers)
  - Categories containing "Settings" → excluded (control-panel utilities)

Entry keys:
    display_name  (str) — human-readable name, e.g. "HeidiSQL"
    launch_path   (str) — resolved binary path or .desktop path for wrapper launchers
    generic_name  (str) — e.g. "Web Browser", "Universal Database Manager"
    categories    (str) — space-separated, e.g. "Development Database"

The catalog is built once per process and cached at the class level.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from typing import List, Optional

logger = logging.getLogger(__name__)

_WRAPPERS = frozenset({"flatpak", "snap", "env", "bash", "sh", "python", "python3"})

_DESKTOP_DIRS = [
    "/usr/share/applications",
    "/var/lib/snapd/desktop/applications",
    "/var/lib/flatpak/exports/share/applications",
    os.path.expanduser("~/.local/share/applications"),
    os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
]


class LinuxAppCatalog:
    """Builds and caches the installed user-app catalog from .desktop files."""

    _cache: Optional[List[dict]] = None

    @classmethod
    def build(cls) -> List[dict]:
        """Return the cached catalog, building it on first call."""
        if cls._cache is not None:
            return cls._cache
        cls._cache = cls._build()
        return cls._cache

    @classmethod
    def invalidate(cls) -> None:
        """Force a rebuild on the next call to build()."""
        cls._cache = None

    @classmethod
    def _build(cls) -> List[dict]:
        import glob as _glob

        catalog: List[dict] = []
        seen_exec: set = set()

        for desktop_dir in _DESKTOP_DIRS:
            try:
                for fpath in _glob.glob(os.path.join(desktop_dir, "*.desktop")):
                    entry = cls._parse_desktop(fpath)
                    if entry is None:
                        continue
                    exec_base = os.path.basename(entry["_exec"]).lower()
                    if exec_base in seen_exec:
                        continue
                    if exec_base in _WRAPPERS:
                        launch = fpath
                    else:
                        launch = shutil.which(entry["_exec"]) or fpath
                    catalog.append({
                        "display_name": entry["name"],
                        "launch_path":  launch,
                        "generic_name": entry["generic_name"],
                        "categories":   entry["categories"],
                    })
                    seen_exec.add(exec_base)
            except Exception as exc:
                logger.warning("LinuxAppCatalog (%s): %s", desktop_dir, exc)

        logger.debug("LinuxAppCatalog: %d user apps loaded.", len(catalog))
        return catalog

    @staticmethod
    def _parse_desktop(fpath: str) -> Optional[dict]:
        """
        Parse a single .desktop file.  Returns None if the entry should be
        excluded (NoDisplay, Settings category, missing Name/Exec).
        """
        name: Optional[str] = None
        exec_raw: Optional[str] = None
        generic_name = ""
        categories = ""
        no_display = ""

        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                in_entry = False
                for line in f:
                    line = line.strip()
                    if line == "[Desktop Entry]":
                        in_entry = True
                        continue
                    if line.startswith("[") and in_entry:
                        break  # left [Desktop Entry] section
                    if not in_entry:
                        continue
                    if line.startswith("Name=") and name is None:
                        name = line[5:].strip()
                    elif line.startswith("Exec=") and exec_raw is None:
                        tokens = line[5:].strip().split()
                        if tokens:
                            cmd = re.sub(r"%\w", "", tokens[0]).strip()
                            exec_raw = cmd or None
                    elif line.startswith("GenericName="):
                        generic_name = line[12:].strip()
                    elif line.startswith("Categories="):
                        categories = line[11:].strip().replace(";", " ").strip()
                    elif line.startswith("NoDisplay="):
                        no_display = line[10:].strip()
        except Exception:
            return None

        if not name or not exec_raw:
            return None
        if no_display.lower() == "true":
            return None
        if "settings" in categories.lower():
            return None

        return {
            "name":         name,
            "_exec":        exec_raw,
            "generic_name": generic_name,
            "categories":   categories,
        }
