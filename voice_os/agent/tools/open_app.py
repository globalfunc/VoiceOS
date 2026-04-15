"""
LangChain tool: open_app

Launches applications by name, or opens files with a specified app or the
system default.  Uses the platform OSHandler for portability.

When an exact match is not found, fuzzy candidates are retrieved from the
OS handler and the user is asked to confirm via voice before launching.
This mitigates STT mis-transcriptions such as "haydysql" → HeidiSQL or
"db beaver" → DBeaver.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Callable, Optional, Type

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

    # Injected by AgentRunner so the tool can speak and listen during
    # the fuzzy-match confirmation flow.
    speak: Optional[Callable[[str], None]] = None
    listen_for_response: Optional[Callable[[], str]] = None

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Tool entry point
    # ------------------------------------------------------------------

    def _run(self, app_name: str, file_path: Optional[str] = None) -> str:  # type: ignore[override]
        from voice_os.os_handlers import get_os_handler

        handler = get_os_handler()
        app_name = (app_name or "").strip()

        # ── Case 1: open a file (with optional app override) ──────────
        if file_path:
            file_path = file_path.strip()
            if app_name:
                success = handler.open_file(file_path, app=app_name)
            else:
                success = handler.open_file(file_path)
            if success:
                app_label = app_name if app_name else "the default app"
                return f"Opening {file_path} with {app_label}."
            return f"Failed to open {file_path}."

        # ── Case 2: launch an application by name ─────────────────────
        if not app_name:
            return "Please specify an app name or a file path."

        # Try exact / case-insensitive binary / desktop-file match first.
        app_path = handler.find_app(app_name)
        if app_path:
            return self._launch(app_name, app_path)

        # ── Case 3: exact match failed — try fuzzy candidates ─────────
        candidates = handler.find_app_candidates(app_name, top_n=1)
        if not candidates:
            return (
                f"Could not find '{app_name}'. "
                "It may not be installed, or try a different name."
            )

        display_name, launch_path, score = candidates[0]
        logger.info(
            "open_app: no exact match for %r; best fuzzy candidate: %r (score=%.2f)",
            app_name, display_name, score,
        )

        # If we have voice callbacks, ask the user to confirm.
        if self.speak and self.listen_for_response:
            self.speak(
                f"I couldn't find {app_name}. "
                f"Did you mean {display_name}? "
                "Say yes to open it or no to cancel."
            )
            response = self.listen_for_response()
            logger.info("Confirmation response for '%s': %r", display_name, response)
            if "yes" in response.lower():
                return self._launch(display_name, launch_path)
            return "Cancelled."

        # No voice callbacks — suggest the best match as text so the LLM
        # can relay it to the user.
        return (
            f"Could not find '{app_name}'. "
            f"The closest match is '{display_name}'. "
            "Please try again with that name."
        )

    async def _arun(self, app_name: str, file_path: Optional[str] = None) -> str:  # type: ignore[override]
        return self._run(app_name, file_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch(self, label: str, path: str) -> str:
        """Launch ``path`` and return a spoken confirmation string."""
        try:
            if path.endswith(".desktop"):
                subprocess.Popen(["xdg-open", path])
            else:
                subprocess.Popen([path])
            logger.info("Launched '%s' (%s).", label, path)
            return f"Opening {label}."
        except Exception as exc:
            logger.error("Failed to launch '%s': %s", label, exc)
            return f"Failed to open {label}: {exc}"
