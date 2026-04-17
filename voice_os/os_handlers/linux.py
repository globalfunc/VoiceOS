"""
Linux OS handler.

Volume: pactl (PulseAudio / PipeWire-PulseAudio compat)
File open: xdg-open
Power: systemctl (user session suspend/poweroff/reboot)
App lookup: which + /usr/share/applications
"""
from __future__ import annotations

import difflib
import logging
import os
import re
import shutil
import subprocess
from typing import List, Optional, Tuple

from voice_os.os_handlers.base import OSHandler

logger = logging.getLogger(__name__)


class LinuxHandler(OSHandler):

    # ------------------------------------------------------------------
    # File / app
    # ------------------------------------------------------------------

    def open_file(self, path: str, app: Optional[str] = None) -> bool:
        try:
            if app:
                subprocess.Popen([app, path])
            else:
                subprocess.Popen(["xdg-open", path])
            logger.info("Opened '%s' with %s.", path, app or "xdg-open")
            return True
        except Exception as exc:
            logger.error("open_file failed: %s", exc)
            return False

    def find_app(self, name: str) -> Optional[str]:
        # 1. Fast exact binary lookup in PATH.
        path = shutil.which(name)
        if path:
            return path
        # 2. Exact case-insensitive match against the catalog — reuses the
        #    already-built LinuxAppCatalog instead of grepping /usr/share/applications.
        name_lower = name.lower().strip()
        for entry in self._catalog():
            if entry["display_name"].lower() == name_lower:
                return entry["launch_path"]
            exec_base = os.path.basename(entry["launch_path"]).replace(".desktop", "").lower()
            if exec_base == name_lower:
                return entry["launch_path"]
        return None

    def list_apps(self) -> List[str]:
        """Return executable names from the catalog."""
        return sorted({
            os.path.basename(e["launch_path"]).replace(".desktop", "")
            for e in self._catalog()
        })

    def _catalog(self) -> List[dict]:
        """Return the shared user-app catalog (built once, cached)."""
        from voice_os.os_handlers.app_catalog import LinuxAppCatalog
        return LinuxAppCatalog.build()

    def search_apps(
        self, query: str, top_n: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Search installed apps by name, generic name, and categories.
        Returns [(display_name, launch_path, score), ...] sorted descending.
        """
        def _norm(s: str) -> str:
            return re.sub(r"[\s\-_/;,]", "", s).lower()

        q_norm = _norm(query)
        q_words = [w for w in query.lower().split() if len(w) > 2]

        scored: List[Tuple[str, str, float]] = []
        for entry in self._catalog():
            display_name: str = entry["display_name"]
            launch_path: str  = entry["launch_path"]

            base  = difflib.SequenceMatcher(None, q_norm, _norm(display_name)).ratio()
            generic = entry.get("generic_name", "").lower()
            cats    = entry.get("categories",   "").lower()

            boost = 0.0
            for word in q_words:
                if word in generic or word in cats:
                    boost = max(boost, 0.5)
                elif word in display_name.lower():
                    boost = max(boost, 0.15)

            score = min(1.0, base + boost)
            if score >= 0.35:
                scored.append((display_name, launch_path, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]

    def list_app_names(self, top_n: int = 200) -> List[str]:
        """Return sorted display names for Whisper vocabulary building."""
        names = sorted({e["display_name"] for e in self._catalog()})
        return names[:top_n]

    def find_app_candidates(
        self, query: str, top_n: int = 3, min_score: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """
        Fuzzy-match ``query`` against installed app display names and executable names.
        Returns up to ``top_n`` results with score >= ``min_score``, sorted descending.
        """
        def _norm(s: str) -> str:
            return re.sub(r"[\s\-_]", "", s).lower()

        q = _norm(query)
        scored: List[Tuple[str, str, float]] = []
        for entry in self._catalog():
            display_name: str = entry["display_name"]
            launch_path: str  = entry["launch_path"]
            dn = _norm(display_name)
            en = _norm(os.path.basename(launch_path).replace(".desktop", ""))
            score = max(
                difflib.SequenceMatcher(None, q, dn).ratio(),
                difflib.SequenceMatcher(None, q, en).ratio(),
            )
            if score >= min_score:
                scored.append((display_name, launch_path, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def find_processes(self, name: str, fullcmd: bool = True) -> List[Tuple[int, str]]:
        """
        Find running processes matching ``name``.
        fullcmd=True  — pgrep -af: search the full command line (default).
        fullcmd=False — pgrep -a: match only the process comm/binary name.
                        Use when you have an exact exec name from the catalog
                        to avoid false positives from cmdline paths/args
                        (e.g. "obs" matching ".../models/blobs/sha256-...").
        Returns [(pid, comm), ...].  The current process is excluded.
        """
        my_pid = os.getpid()
        flags = ["-af"] if fullcmd else ["-a"]
        try:
            result = subprocess.run(
                ["pgrep"] + flags + [name],
                capture_output=True, text=True, timeout=5,
            )
        except Exception as exc:
            logger.warning("find_processes(%r): %s", name, exc)
            return []

        matches: List[Tuple[int, str]] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid == my_pid:
                continue
            # Resolve the short process name (comm) for display / grouping.
            try:
                comm_out = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "comm="],
                    capture_output=True, text=True, timeout=3,
                )
                comm = comm_out.stdout.strip() or (parts[1].split()[0] if len(parts) > 1 else str(pid))
            except Exception:
                comm = parts[1].split()[0] if len(parts) > 1 else str(pid)
            matches.append((pid, comm))
        return matches

    def find_processes_fuzzy(
        self, query: str, min_score: float = 0.5
    ) -> List[Tuple[str, List[int], float]]:
        """
        Fuzzy-match ``query`` against running process names (comm).

        Used when find_processes() returns nothing — e.g. user said
        "haydysql" for a process actually named "heidisql".

        Returns [(process_name, [pids], score), ...] sorted descending.
        """
        my_pid = os.getpid()
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,comm="],
                capture_output=True, text=True, timeout=5,
            )
        except Exception as exc:
            logger.warning("find_processes_fuzzy: %s", exc)
            return []

        # Build {comm → [pids]} map, skipping ourselves.
        name_to_pids: dict = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid == my_pid:
                continue
            comm = parts[1].strip()
            name_to_pids.setdefault(comm, []).append(pid)

        def _norm(s: str) -> str:
            return re.sub(r"[\s\-_]", "", s).lower()

        q = _norm(query)
        scored: List[Tuple[str, List[int], float]] = []
        for comm, pids in name_to_pids.items():
            cn = _norm(comm)
            score = difflib.SequenceMatcher(None, q, cn).ratio()
            # Prefix bonus: "obs" is a prefix of "obsstudio" — strong signal
            # that this short process name IS the app the user meant.
            if q.startswith(cn) or cn.startswith(q):
                score = max(score, 0.75)
            if score >= min_score:
                scored.append((comm, pids, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored

    def close_processes(self, pids: List[int]) -> bool:
        """Send SIGTERM to each PID.  Returns True if all signals were delivered."""
        import signal as _signal

        success = True
        for pid in pids:
            try:
                os.kill(pid, _signal.SIGTERM)
                logger.info("SIGTERM → PID %d.", pid)
            except ProcessLookupError:
                logger.info("PID %d already gone.", pid)
            except Exception as exc:
                logger.error("close_processes(%d): %s", pid, exc)
                success = False
        return success

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def get_volume(self) -> int:
        try:
            out = subprocess.check_output(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                text=True, timeout=3,
            )
            # "Volume: front-left: 49152 /  75% / ..."
            match = re.search(r"(\d+)%", out)
            if match:
                return int(match.group(1))
        except Exception as exc:
            logger.warning("get_volume: %s", exc)
        return 50  # safe fallback

    def set_volume(self, level: int) -> None:
        level = max(0, min(100, level))
        try:
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                check=True, timeout=3,
            )
            logger.info("Volume set to %d%%.", level)
        except Exception as exc:
            logger.error("set_volume: %s", exc)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def sleep(self) -> None:
        logger.info("Suspending system…")
        subprocess.Popen(["systemctl", "suspend"])

    def shutdown(self) -> None:
        logger.info("Shutting down system…")
        subprocess.Popen(["systemctl", "poweroff"])

    def restart(self) -> None:
        logger.info("Rebooting system…")
        subprocess.Popen(["systemctl", "reboot"])
