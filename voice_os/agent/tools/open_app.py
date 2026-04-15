"""
LangChain tool: open_app

Launches applications by name, or opens files with a specified app or the
system default.  Uses the platform OSHandler for portability.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Optional, Type

from langchain_classic.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _OpenAppInput(BaseModel):
    app_name: str = Field(
        description=(
            "Name of the application to open, e.g. 'firefox', 'vlc', 'terminal', "
            "'nautilus', 'code'. Use an empty string if you only want to open a file "
            "with its default app."
        )
    )
    file_path: Optional[str] = Field(
        default=None,
        description=(
            "Absolute path of a file to open. "
            "If provided with app_name, the file is opened with that app. "
            "If provided without app_name, the system default app is used."
        ),
    )


class OpenAppTool(BaseTool):
    """Launches an application or opens a file."""

    name: str = "open_app"
    description: str = (
        "Open an application by name, or open a file with a specific or default app. "
        "Examples: open Firefox, open terminal, open /home/user/report.pdf with evince."
    )
    args_schema: Type[BaseModel] = _OpenAppInput

    def _run(self, app_name: str, file_path: Optional[str] = None) -> str:  # type: ignore[override]
        from voice_os.os_handlers import get_os_handler

        handler = get_os_handler()
        app_name = (app_name or "").strip()

        # ── Case 1: open a file (with optional app override) ───────────────
        if file_path:
            file_path = file_path.strip()
            if app_name:
                # Use the named app to open the file
                success = handler.open_file(file_path, app=app_name)
            else:
                # System default
                success = handler.open_file(file_path)
            if success:
                app_label = app_name if app_name else "the default app"
                return f"Opening {file_path} with {app_label}."
            return f"Failed to open {file_path}."

        # ── Case 2: launch an application by name ─────────────────────────
        if not app_name:
            return "Please specify an app name or a file path."

        app_path = handler.find_app(app_name)
        if not app_path:
            return (
                f"Could not find '{app_name}'. "
                "It may not be installed, or try a different name."
            )

        # If find_app returned a .desktop file rather than an executable,
        # fall back to xdg-open on Linux (handled gracefully).
        try:
            if app_path.endswith(".desktop"):
                subprocess.Popen(["xdg-open", app_path])
            else:
                subprocess.Popen([app_path])
            logger.info("Launched '%s' (%s).", app_name, app_path)
            return f"Opening {app_name}."
        except Exception as exc:
            logger.error("Failed to launch '%s': %s", app_name, exc)
            return f"Failed to open {app_name}: {exc}"

    async def _arun(self, app_name: str, file_path: Optional[str] = None) -> str:  # type: ignore[override]
        return self._run(app_name, file_path)
