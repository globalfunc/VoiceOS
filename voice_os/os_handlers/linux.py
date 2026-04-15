"""
Linux OS handler.

Volume: pactl (PulseAudio / PipeWire-PulseAudio compat)
File open: xdg-open
Power: systemctl (user session suspend/poweroff/reboot)
App lookup: which + /usr/share/applications
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import List, Optional

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
