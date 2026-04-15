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
        path = shutil.which(name)
        if path:
            return path
        # Fuzzy search in .desktop files
        try:
            result = subprocess.run(
                ["grep", "-ril", name, "/usr/share/applications"],
                capture_output=True, text=True, timeout=3,
            )
            if result.stdout.strip():
                return result.stdout.strip().splitlines()[0]
        except Exception:
            pass
        return None

    def list_apps(self) -> List[str]:
        """Return executable names from /usr/share/applications *.desktop files."""
        apps: List[str] = []
        try:
            result = subprocess.run(
                ["grep", "-rh", "^Exec=", "/usr/share/applications"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                # Exec=firefox %u → firefox
                exe = line.removeprefix("Exec=").split()[0]
                exe = re.sub(r"[^a-zA-Z0-9_\-./]", "", exe)
                if exe:
                    apps.append(exe)
        except Exception as exc:
            logger.warning("list_apps: %s", exc)
        return sorted(set(apps))

    def _build_app_catalog(self) -> List[Tuple[str, str]]:
        """
        Build a catalog of installed applications from all known sources.

        Sources (in order):
        1. .desktop files — /usr/share/applications, snap, flatpak, and user dirs.
           Gives nice display names (e.g. "dbeaver-ce" for the snap).
        2. PATH executables — every executable found in $PATH that wasn't already
           covered by a .desktop entry.  Catches AppImages, manual installs, etc.

        Returns [(display_name, launch_path), ...].
        """
        import glob as _glob

        catalog: List[Tuple[str, str]] = []
        seen_exec: set = set()   # lower-cased binary base-names already in catalog

        desktop_dirs = [
            "/usr/share/applications",
            "/var/lib/snapd/desktop/applications",                           # snap
            "/var/lib/flatpak/exports/share/applications",                  # flatpak (system)
            os.path.expanduser("~/.local/share/applications"),
            os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),  # flatpak (user)
        ]
        for desktop_dir in desktop_dirs:
            try:
                for fpath in _glob.glob(os.path.join(desktop_dir, "*.desktop")):
                    name: Optional[str] = None
                    exec_bin: Optional[str] = None
                    try:
                        with open(fpath, encoding="utf-8", errors="replace") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("Name=") and name is None:
                                    name = line[5:].strip()
                                elif line.startswith("Exec=") and exec_bin is None:
                                    tokens = line[5:].strip().split()
                                    if tokens:
                                        cmd = re.sub(r"%\w", "", tokens[0]).strip()
                                        exec_bin = cmd or None
                    except Exception:
                        continue
                    if not name or not exec_bin:
                        continue
                    launch = shutil.which(exec_bin) or fpath
                    catalog.append((name, launch))
                    seen_exec.add(os.path.basename(exec_bin).lower())
            except Exception as exc:
                logger.warning("_build_app_catalog (%s): %s", desktop_dir, exc)

        # Supplement with PATH executables not already covered by a .desktop file.
        for dir_path in os.environ.get("PATH", "").split(os.pathsep):
            try:
                if not os.path.isdir(dir_path):
                    continue
                for entry in os.scandir(dir_path):
                    if entry.is_file() and os.access(entry.path, os.X_OK):
                        if entry.name.lower() not in seen_exec:
                            catalog.append((entry.name, entry.path))
                            seen_exec.add(entry.name.lower())
            except Exception:
                continue

        return catalog

    def find_app_candidates(
        self, query: str, top_n: int = 3, min_score: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """
        Fuzzy-match ``query`` against installed desktop apps.

        Normalises both sides (lowercase, strip spaces/dashes/underscores) then
        computes difflib SequenceMatcher ratio against both the display name
        and the executable name.  Returns up to ``top_n`` results with score
        >= ``min_score``, sorted descending.

        Examples that work:
          "db beaver"   → DBeaver  (score ~0.93 after normalising "dbbeaver"/"dbeaver")
          "haydysql"    → HeidiSQL (score ~0.63 for "haydysql"/"heidisql")
        """
        catalog = self._build_app_catalog()

        def _norm(s: str) -> str:
            return re.sub(r"[\s\-_]", "", s).lower()

        q = _norm(query)
        scored: List[Tuple[str, str, float]] = []
        for display_name, launch_path in catalog:
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

    def find_processes(self, name: str) -> List[Tuple[int, str]]:
        """
        Find running processes whose full command line contains ``name``
        (case-insensitive substring via pgrep -af).

        Returns [(pid, comm), ...].  The current process is excluded so
        VoiceOS can never accidentally close itself.
        """
        my_pid = os.getpid()
        try:
            result = subprocess.run(
                ["pgrep", "-af", name],
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
