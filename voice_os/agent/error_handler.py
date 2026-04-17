"""
AgentInvokeErrorHandler — classifies exceptions raised by agent.invoke()
into structured AgentInvokeError instances with a spoken response,
a logging level, and a flag indicating whether session history should be cleared.

Usage:
    handler = AgentInvokeErrorHandler()
    err = handler.handle(exc)
    logger.log(err.level, "Agent invoke failed: %s", exc, exc_info=err.level == logging.ERROR)
    if err.clear_session:
        session.clear()
    return err.response
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field


@dataclass
class AgentInvokeError:
    """Structured result of a classified agent invocation error."""
    response: str
    level: int          # logging.WARNING or logging.ERROR
    clear_session: bool = field(default=False)


class AgentInvokeErrorHandler:
    """
    Classifies exceptions raised by agent.invoke() and returns an AgentInvokeError.

    Classification is based on exception type name and message content.
    Falls through to a generic error if no specific pattern matches.
    """

    def handle(self, exc: Exception) -> AgentInvokeError:
        exc_type = type(exc).__name__
        exc_str = str(exc).lower()

        # --- Recursion limit (chain too long) ---
        if exc_type == "GraphRecursionError" or "recursion" in exc_str:
            return AgentInvokeError(
                response=(
                    "That request was too complex for me to complete in one go. "
                    "Try breaking it into smaller steps."
                ),
                level=logging.WARNING,
                clear_session=False,
            )

        # --- Ollama not reachable ---
        if any(k in exc_str for k in ("connection", "connect", "refused")):
            return AgentInvokeError(
                response="I can't reach the AI model. Please make sure Ollama is running.",
                level=logging.ERROR,
                clear_session=False,
            )

        # --- Model response timeout ---
        if any(k in exc_str for k in ("timeout", "timed out")):
            return AgentInvokeError(
                response="The AI model took too long to respond. Please try again.",
                level=logging.WARNING,
                clear_session=False,
            )

        # --- Model not pulled in Ollama ---
        if any(k in exc_str for k in ("not found", "no such", "404")):
            return AgentInvokeError(
                response=(
                    "The AI model isn't available. "
                    "Run 'ollama pull' to download the configured model."
                ),
                level=logging.ERROR,
                clear_session=False,
            )

        # --- Context window exceeded ---
        if "context" in exc_str and any(k in exc_str for k in ("length", "limit", "window")):
            return AgentInvokeError(
                response=(
                    "The conversation got too long. "
                    "I've cleared the session history — please repeat your request."
                ),
                level=logging.WARNING,
                clear_session=True,
            )

        # --- Unexpected / unclassified error ---
        return AgentInvokeError(
            response="Sorry, I couldn't process that. Please try again.",
            level=logging.ERROR,
            clear_session=False,
        )
