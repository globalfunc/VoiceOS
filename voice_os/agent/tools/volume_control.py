"""
LangChain tool: volume_control

Controls the system audio volume via the platform OS handler.
Works on Linux (pactl) and Windows (pycaw) through the OSHandler abstraction.
"""
from __future__ import annotations

import logging
from typing import Optional, Type

from langchain_classic.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _VolumeInput(BaseModel):
    action: str = Field(
        description=(
            "What to do with the volume. "
            "One of: set, increase, decrease, mute, unmute."
        )
    )
    value: Optional[int] = Field(
        default=None,
        description=(
            "Volume level or change amount as a whole number 0-100. "
            "Required for 'set'; optional for 'increase'/'decrease' (defaults to 10). "
            "Not used for 'mute'/'unmute'."
        ),
    )


class VolumeControlTool(BaseTool):
    """Adjusts system volume: set, increase, decrease, mute, unmute."""

    name: str = "volume_control"
    description: str = (
        "Control system audio volume. "
        "Use action='set' with value=N to set volume to N percent (0-100). "
        "Use action='increase' or action='decrease' with optional value=N to change by N percent (default 10). "
        "Use action='mute' or action='unmute' with no value."
    )
    args_schema: Type[BaseModel] = _VolumeInput

    def _run(self, action: str, value: Optional[int] = None) -> str:  # type: ignore[override]
        from voice_os.os_handlers import get_os_handler

        handler = get_os_handler()
        action = action.lower().strip()

        if action == "set":
            level = max(0, min(100, value if value is not None else 50))
            handler.set_volume(level)
            return f"Volume set to {level} percent."

        if action in ("increase", "up", "raise"):
            delta = max(1, value if value is not None else 10)
            current = handler.get_volume()
            level = min(100, current + delta)
            handler.set_volume(level)
            return f"Volume increased to {level} percent."

        if action in ("decrease", "down", "lower"):
            delta = max(1, value if value is not None else 10)
            current = handler.get_volume()
            level = max(0, current - delta)
            handler.set_volume(level)
            return f"Volume decreased to {level} percent."

        if action == "mute":
            handler.set_volume(0)
            return "Volume muted."

        if action == "unmute":
            # Restore to a reasonable level — get_volume returns 0 when muted,
            # so we jump to 50% as a safe unmute default.
            handler.set_volume(50)
            return "Volume unmuted and set to 50 percent."

        if action in ("get", "check", "status"):
            level = handler.get_volume()
            return f"Current volume is {level} percent."

        return (
            f"Unknown volume action: '{action}'. "
            "Use set, increase, decrease, mute, or unmute."
        )

    # BaseTool requires _arun for async; we just delegate to _run.
    async def _arun(self, action: str, value: Optional[int] = None) -> str:  # type: ignore[override]
        return self._run(action, value)
