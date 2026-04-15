"""
LangChain tool: system_control

Executes system power actions (sleep, restart, shutdown) with a mandatory
10-second spoken countdown during which the user can say "cancel" to abort.

The tool requires two callbacks injected at construction time:
    speak(text)              — TTS speak function
    listen_for_cancel()      — records up to 10 s and returns True if user said "cancel"

These are provided by AgentRunner so the tool can interact with audio I/O.
"""
from __future__ import annotations

import logging
from typing import Callable, Type

from langchain_classic.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"sleep", "suspend", "restart", "reboot", "shutdown", "poweroff"}


class _SystemControlInput(BaseModel):
    action: str = Field(
        description=(
            "Power action to perform. "
            "One of: sleep, restart, shutdown."
        )
    )


class SystemControlTool(BaseTool):
    """
    Executes system power actions with a 10-second safety countdown.

    Speak and listen_for_cancel callbacks must be injected by the caller
    (AgentRunner) so the tool can interact with TTS and STT.
    """

    name: str = "system_control"
    description: str = (
        "Control system power state. "
        "Use action='sleep' to suspend. "
        "Use action='restart' to reboot. "
        "Use action='shutdown' to power off. "
        "Always issues a 10-second warning before executing — user can say 'cancel' to abort."
    )
    args_schema: Type[BaseModel] = _SystemControlInput

    # Injected callbacks — declared as fields so Pydantic v2 accepts them.
    speak: Callable[[str], None]
    listen_for_cancel: Callable[[], bool]

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, action: str) -> str:  # type: ignore[override]
        from voice_os.os_handlers import get_os_handler

        action = action.lower().strip()

        # Normalise synonyms
        if action in ("suspend",):
            action = "sleep"
        elif action in ("reboot",):
            action = "restart"
        elif action in ("poweroff", "power off", "power-off"):
            action = "shutdown"

        if action not in ("sleep", "restart", "shutdown"):
            return (
                f"Unknown action: '{action}'. "
                "Use sleep, restart, or shutdown."
            )

        # ── Safety countdown ───────────────────────────────────────────────
        action_label = {
            "sleep":    "going to sleep",
            "restart":  "restarting",
            "shutdown": "shutting down",
        }[action]

        warning = (
            f"I am {action_label} in 10 seconds. "
            "Say cancel to abort."
        )
        logger.info("system_control: %s — starting countdown.", action)
        self.speak(warning)

        cancelled = self.listen_for_cancel()
        if cancelled:
            logger.info("system_control: cancelled by user.")
            return f"{action.capitalize()} cancelled."

        # ── Execute ────────────────────────────────────────────────────────
        handler = get_os_handler()
        logger.info("system_control: executing %s.", action)

        if action == "sleep":
            self.speak("Good night.")
            handler.sleep()
            return "System is going to sleep."

        if action == "restart":
            self.speak("Restarting now.")
            handler.restart()
            return "System is restarting."

        if action == "shutdown":
            self.speak("Shutting down. Goodbye.")
            handler.shutdown()
            return "System is shutting down."

        return "Done."  # unreachable but satisfies type checker

    async def _arun(self, action: str) -> str:  # type: ignore[override]
        return self._run(action)
