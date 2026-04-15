"""
Windows OS handler.

Volume: pycaw (Windows Core Audio API)
File open: os.startfile / ShellExecute
Power: shutdown.exe
App lookup: winreg App Paths
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Optional

from voice_os.os_handlers.base import OSHandler

logger = logging.getLogger(__name__)


class WindowsHandler(OSHandler):

    # ------------------------------------------------------------------
    # File / app
    # ------------------------------------------------------------------

    def open_file(self, path: str, app: Optional[str] = None) -> bool:
        try:
            if app:
                os.startfile(path, "open")  # let Windows use app association
            else:
                os.startfile(path)
            logger.info("Opened '%s'.", path)
            return True
        except Exception as exc:
            logger.error("open_file: %s", exc)
            return False

    def find_app(self, name: str) -> Optional[str]:
        import shutil
        # Try PATH first
        path = shutil.which(name) or shutil.which(f"{name}.exe")
        if path:
            return path
        # Try App Paths registry
        try:
            import winreg
            key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{name}.exe"
            for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(root, key_path) as key:
                        value, _ = winreg.QueryValueEx(key, "")
                        return value
                except OSError:
                    pass
        except Exception:
            pass
        return None

    def list_apps(self) -> List[str]:
        """Best-effort: list exe names from App Paths registry."""
        apps: List[str] = []
        try:
            import winreg
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                i = 0
                while True:
                    try:
                        name = winreg.EnumKey(key, i)
                        apps.append(name.removesuffix(".exe"))
                        i += 1
                    except OSError:
                        break
        except Exception as exc:
            logger.warning("list_apps: %s", exc)
        return sorted(set(apps))

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def get_volume(self) -> int:
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from comtypes import CLSCTX_ALL
            import ctypes

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
            scalar = volume.GetMasterVolumeLevelScalar()
            return round(scalar * 100)
        except Exception as exc:
            logger.warning("get_volume: %s", exc)
            return 50

    def set_volume(self, level: int) -> None:
        level = max(0, min(100, level))
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from comtypes import CLSCTX_ALL
            import ctypes

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
            logger.info("Volume set to %d%%.", level)
        except Exception as exc:
            logger.error("set_volume: %s", exc)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def sleep(self) -> None:
        logger.info("Suspending system…")
        subprocess.Popen(
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"]
        )

    def shutdown(self) -> None:
        logger.info("Shutting down system…")
        subprocess.Popen(["shutdown", "/s", "/t", "0"])

    def restart(self) -> None:
        logger.info("Rebooting system…")
        subprocess.Popen(["shutdown", "/r", "/t", "0"])
