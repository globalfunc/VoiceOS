"""Platform detection — import get_os_handler() to get the right implementation."""
import platform

from voice_os.os_handlers.base import OSHandler


def get_os_handler() -> OSHandler:
    system = platform.system()
    if system == "Linux":
        from voice_os.os_handlers.linux import LinuxHandler
        return LinuxHandler()
    elif system == "Windows":
        from voice_os.os_handlers.windows import WindowsHandler
        return WindowsHandler()
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


__all__ = ["OSHandler", "get_os_handler"]
