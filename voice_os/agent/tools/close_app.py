"""
LangChain tool: close_app

Terminates a running application by name.

Flow:
  1. Exact/substring match (pgrep -af) → close immediately.
  2. Nothing found → fuzzy match against all running process names.
     Best match offered to user for voice confirmation.
  3. Multiple PIDs for the same process → user asked whether to close
     all instances or just one.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Type

from langchain_classic.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _CloseAppInput(BaseModel):
    app_name: str = Field(
        description=(
            "Name of the application to close, e.g. 'firefox', 'dbeaver', 'terminal'. "
            "Use the spoken name — fuzzy matching handles minor STT errors."
        )
    )


class CloseAppTool(BaseTool):
    """Close / quit a running application by name."""

    name: str = "close_app"
    description: str = (
        "Close or quit a running application by name. "
        "Examples: close Firefox, quit DBeaver, exit terminal."
    )
    args_schema: Type[BaseModel] = _CloseAppInput

    # Injected by AgentRunner for interactive confirmation.
    speak: Optional[Callable[[str], None]] = None
    listen_for_response: Optional[Callable[[], str]] = None

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Tool entry point
    # ------------------------------------------------------------------

    def _run(self, app_name: str) -> str:  # type: ignore[override]
        import os as _os
        from voice_os.os_handlers import get_os_handler

        handler = get_os_handler()
        app_name = (app_name or "").strip()
        if not app_name:
            return "Please specify an application name to close."

        # ── Step 1: direct command-line substring match ────────────────
        # pgrep -af searches the full cmdline, so "dbeaver" matches "dbeaver-ce".
        raw_matches = handler.find_processes(app_name)
        confirmed_display = app_name

        # ── Step 1.5: catalog bridge ───────────────────────────────────
        # "OBS Studio" → catalog → exec "obs" → pgrep "obs".
        # This mirrors how open_app resolves multi-word product names to their
        # real binary name, and avoids needing "studio" to appear in the cmdline.
        if not raw_matches:
            app_candidates = handler.find_app_candidates(app_name, top_n=1)
            if app_candidates:
                display_name, launch_path, app_score = app_candidates[0]
                exec_name = _os.path.basename(launch_path)
                # Skip if catalog only returned a .desktop path (binary not in PATH).
                if exec_name and not launch_path.endswith(".desktop"):
                    bridged = handler.find_processes(exec_name)
                    if bridged:
                        raw_matches = bridged
                        confirmed_display = display_name
                        logger.info(
                            "close_app: bridged %r → exec %r → %d process(es) "
                            "(catalog score=%.2f)",
                            app_name, exec_name, len(bridged), app_score,
                        )
                        # Only ask for confirmation when the catalog match is
                        # uncertain (score < 0.8).  High-confidence matches
                        # (e.g. "obs studio" → "OBS Studio", score=1.0) close
                        # directly without interrupting the user.
                        if app_score < 0.8 and self.speak and self.listen_for_response:
                            self.speak(
                                f"I couldn't find {app_name}. "
                                f"Did you mean {display_name}? "
                                "Say yes to close it or no to cancel."
                            )
                            response = self.listen_for_response()
                            if "yes" not in response.lower():
                                return "Cancelled."

        # ── Step 2: fuzzy match against running process names ──────────
        # Last resort — handles STT mis-transcriptions like "haydysql" → heidisql.
        if not raw_matches:
            fuzzy = handler.find_processes_fuzzy(app_name, min_score=0.5)
            if not fuzzy:
                return f"No running process found matching '{app_name}'."

            proc_name, pids, score = fuzzy[0]
            logger.info(
                "close_app: no exact match for %r; best fuzzy: %r (score=%.2f)",
                app_name, proc_name, score,
            )

            if self.speak and self.listen_for_response:
                self.speak(
                    f"I couldn't find {app_name}. "
                    f"Did you mean {proc_name}? "
                    "Say yes to close it or no to cancel."
                )
                response = self.listen_for_response()
                logger.info("Fuzzy-close confirmation for %r: %r", proc_name, response)
                if "yes" not in response.lower():
                    return "Cancelled."
            else:
                return (
                    f"No process named '{app_name}' found. "
                    f"The closest running process is '{proc_name}'. "
                    "Please try again with that name."
                )

            raw_matches = [(pid, proc_name) for pid in pids]

        # ── Step 3: group by process name → handle multiple instances ─
        groups: Dict[str, List[int]] = defaultdict(list)
        for pid, proc_name in raw_matches:
            groups[proc_name].append(pid)

        parts: List[str] = []
        for proc_name, pids in groups.items():
            # Deduplicate PIDs (find_processes may return duplicates for
            # multi-thread processes sharing a comm name).
            pids = list(dict.fromkeys(pids))

            if len(pids) > 1 and self.speak and self.listen_for_response:
                self.speak(
                    f"There are {len(pids)} instances of {proc_name} running. "
                    "Close all of them? Say yes or no."
                )
                response = self.listen_for_response()
                logger.info("Multi-close confirmation for %r: %r", proc_name, response)
                if "yes" not in response.lower():
                    parts.append(f"Skipped {proc_name}.")
                    continue

            handler.close_processes(pids)
            if len(pids) == 1:
                parts.append(f"Closed {proc_name}.")
            else:
                parts.append(f"Closed {len(pids)} instances of {proc_name}.")

        return " ".join(parts) if parts else f"Nothing to close for '{app_name}'."

    async def _arun(self, app_name: str) -> str:  # type: ignore[override]
        return self._run(app_name)
