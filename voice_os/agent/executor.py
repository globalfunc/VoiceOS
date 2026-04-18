"""
AgentRunner — orchestrates the LangChain agent for Phase 2.

Responsibilities:
  - Build the agent using create_agent (LangGraph, native tool calling via bind_tools)
  - Inject TTS/STT callbacks into tools that need them (open_app, close_app, system_control)
  - Maintain in-session conversation history (last 5 turns)
  - Log all conversation events to voice_os.conversation logger
  - Handle Ollama connectivity errors gracefully

Usage (called from main.py):
    runner = AgentRunner(tts=tts_service, stt=stt_service, vad=vad_recorder)
    response_text = runner.handle("turn up the volume")
    runner.clear_session()   # called on idle timeout
"""
from __future__ import annotations

import datetime
import logging
import pathlib
import threading
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from voice_os.core.tts.base import TTSService
    from voice_os.core.speech_to_text import WhisperTranscriber
    from voice_os.core.vad import VADRecorder

logger = logging.getLogger(__name__)

# Dedicated conversation logger — file handler configured by main.py.
_conv_logger = logging.getLogger("voice_os.conversation")

# Fallback response when something goes wrong before/inside the agent
_FALLBACK = "Sorry, I couldn't process that. Please try again."


class AgentRunner:
    """
    Wraps a LangGraph create_agent with session memory and audio callbacks.

    The agent uses ChatOllama with native tool calling (bind_tools) — no
    text-based ReAct parsing, no FORMAT_INSTRUCTIONS, no agent_scratchpad.
    The agent is built lazily on the first handle() call.
    """

    def __init__(
        self,
        tts: "TTSService",
        stt: "WhisperTranscriber",
        vad: "VADRecorder",
    ) -> None:
        self._tts = tts
        self._stt = stt
        self._vad = vad

        from voice_os.memory.session import SessionMemory
        from voice_os.agent.error_handler import AgentInvokeErrorHandler
        self._session = SessionMemory(maxlen=5)
        self._agent = None          # built lazily
        self._build_lock = threading.Lock()
        self._error_handler = AgentInvokeErrorHandler()
        self._debug_fh: IO | None = None   # per-session dump file

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(self, text: str) -> str:
        """
        Process one user utterance and return the assistant's spoken response.

        Args:
            text: The (already wake-phrase-stripped) transcribed command.

        Returns:
            A plain-English string suitable for TTS.
        """
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        try:
            agent = self._get_agent()
        except Exception as exc:
            logger.error("AgentRunner._get_agent raised: %s", exc, exc_info=True)
            agent = None

        if agent is None:
            return (
                "The AI agent is unavailable. "
                "Make sure Ollama is running and try again."
            )

        from voice_os.config.settings import settings

        # Open debug file lazily (once per session).
        if self._debug_fh is None:
            self._debug_fh = self._open_debug_file()

        # Build message list: optionally inject session history.
        messages = []
        if not settings.stateless_commands:
            for user_text, ai_text in self._session.get_turns():
                messages.append(HumanMessage(content=user_text))
                messages.append(AIMessage(content=ai_text))
        messages.append(HumanMessage(content=text))

        _conv_logger.info('User said: "%s"', text)
        logger.debug(
            "Agent input — user: %r  history_turns: %d",
            text,
            len(self._session),
        )

        try:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 12},
            )
            all_msgs = result.get("messages", [])
            # Only examine messages produced in this turn, not injected history.
            new_msgs = all_msgs[len(messages):]

            # Log every tool execution and track whether any tool was called.
            tool_was_called = False
            for msg in new_msgs:
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    tool_was_called = True
                    for tc in msg.tool_calls:
                        _conv_logger.info(
                            'AI Assistant Executed: "%s(%s)"',
                            tc.get("name", "unknown"),
                            tc.get("args", {}),
                        )

            # Prefer the last synthesized AIMessage with text content 
            # AI models almost always create as synth summary message after tool chain executions.
            response = ""
            for msg in reversed(new_msgs):
                if isinstance(msg, AIMessage):
                    content = getattr(msg, "content", "") or ""
                    if content.strip():
                        response = content
                        break

            # Tool-only turn (no synthesis): surface errors, accept silence for success.
            _TOOL_ERROR_MARKERS = ("not found", "no running", "failed", "could not", "error", "cancelled")
            if not response.strip() and tool_was_called:
                for msg in reversed(new_msgs):
                    if isinstance(msg, ToolMessage):
                        content = getattr(msg, "content", "") or ""
                        if content.strip() and any(m in content.lower() for m in _TOOL_ERROR_MARKERS):
                            response = content
                            break
                # No errors found — tool executed silently, that is acceptable.

            # Agent produced no tool calls and no text — something went wrong.
            if not response.strip() and not tool_was_called:
                response = _FALLBACK

            # Guard: detect when the model describes an action without calling a tool.
            _ACTION_VERBS = (
                "open", "launch", "start", "close", "quit", "exit", "kill",
                "shutdown", "shut down", "restart", "reboot", "sleep",
                "volume", "louder", "quieter", "mute",
            )
            _REFUSAL_SIGNALS = (
                "can't", "cannot", "couldn't", "not found", "not recognized",
                "not installed", "sorry", "don't", "unable", "no app", "unknown",
                "i can't help", "i cannot",
            )
            text_lower = text.lower()
            response_lower = response.lower().strip()
            if not tool_was_called and any(v in text_lower for v in _ACTION_VERBS):
                is_refusal = any(s in response_lower for s in _REFUSAL_SIGNALS)
                # Clarifying questions (e.g. "Did you mean HeidiSQL?") are
                # legitimate — they are not hallucinated action confirmations.
                is_question = response_lower.rstrip(".… ").endswith("?")
                if not is_refusal and not is_question:
                    # Non-refusal, non-question response to an action command
                    # with no tool call → hallucinated confirmation, discard it.
                    logger.warning(
                        "Model faked a successful action for %r without calling a tool — "
                        "discarding response %r.",
                        text, response,
                    )
                    response = (
                        "Sorry, I couldn't complete that action. Please try again."
                    )

        except Exception as exc:
            err = self._error_handler.handle(exc)
            logger.log(
                err.level,
                "Agent invoke failed: %s",
                exc,
                exc_info=err.level == logging.ERROR,
            )
            if err.clear_session:
                self._session.clear()
            self._session.add(text, "")
            return err.response

        self._dump_turn(messages, new_msgs, response)
        _conv_logger.info('AI Assistant responded: "%s"', response)
        logger.debug("Agent output: %r", response)
        self._session.add(text, response)
        return response

    def clear_session(self) -> None:
        """Clear conversation history (called on idle timeout)."""
        self._session.clear()
        if self._debug_fh is not None:
            try:
                self._debug_fh.write("\n=== SESSION CLEARED ===\n")
                self._debug_fh.flush()
                self._debug_fh.close()
            except Exception:
                pass
            self._debug_fh = None
        logger.info("Session memory cleared.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_debug_file(self) -> IO | None:
        """
        Open (once per session) a timestamped dump file under ~/.voice_os/debug/.
        Returns the file handle, or None if the setting is off or opening fails.
        """
        from voice_os.config.settings import settings
        if not settings.debug_session_dump:
            return None
        try:
            debug_dir = pathlib.Path.home() / ".voice_os" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = debug_dir / f"session_{ts}.log"
            fh = path.open("w", encoding="utf-8")
            fh.write(f"VoiceOS session dump — {ts}\n{'=' * 60}\n\n")
            fh.flush()
            logger.info("Debug session dump: %s", path)
            return fh
        except Exception as exc:
            logger.warning("Could not open debug session dump file: %s", exc)
            return None

    def _dump_turn(
        self,
        messages: list,
        new_msgs: list,
        response: str,
    ) -> None:
        """Append one agent turn to the open debug file."""
        if self._debug_fh is None:
            return
        from langchain_core.messages import AIMessage, ToolMessage
        try:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            lines = [f"--- Turn @ {ts} ---"]
            lines.append("[Messages sent to agent]")
            for m in messages:
                role = type(m).__name__.replace("Message", "")
                content = getattr(m, "content", "") or ""
                lines.append(f"  {role}: {content!r}")
            lines.append("[New messages from agent]")
            for m in new_msgs:
                role = type(m).__name__.replace("Message", "")
                content = getattr(m, "content", "") or ""
                tool_calls = getattr(m, "tool_calls", None)
                if tool_calls:
                    lines.append(f"  {role} tool_calls: {tool_calls}")
                if content.strip():
                    lines.append(f"  {role}: {content!r}")
            lines.append(f"[Final response] {response!r}\n")
            self._debug_fh.write("\n".join(lines) + "\n")
            self._debug_fh.flush()
        except Exception as exc:
            logger.warning("Failed to write debug turn: %s", exc)

    def _get_agent(self):
        """Return cached agent, building it if necessary."""
        if self._agent is not None:
            return self._agent
        with self._build_lock:
            if self._agent is None:   # double-checked
                self._agent = self._build_agent()
        return self._agent

    def _build_agent(self):
        """
        Build and return a LangGraph agent using create_agent + ChatOllama.

        create_agent uses the model's native tool-calling API (bind_tools),
        which sends structured JSON tool calls — no text-based ReAct parsing,
        no FORMAT_INSTRUCTIONS, no agent_scratchpad template variable.
        """
        try:
            from langchain.agents import create_agent
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            logger.error(
                "LangChain/Ollama not installed — Phase 2 agent unavailable. "
                "Run: pip install langchain langchain-ollama. "
                "Error: %s", exc
            )
            return None

        from voice_os.config.settings import settings
        from voice_os.agent.prompts import SYSTEM_PROMPT
        from voice_os.agent.tools.volume_control import VolumeControlTool
        from voice_os.agent.tools.open_app import OpenAppTool
        from voice_os.agent.tools.close_app import CloseAppTool
        from voice_os.agent.tools.system_control import SystemControlTool
        from voice_os.agent.tools.search_apps import SearchAppsTool
        from voice_os.agent.tools.search_files import SearchFilesTool
        from voice_os.agent.tools.play_media import PlayMediaTool

        # --- Build tools -----------------------------------------------
        try:
            tools = [
                VolumeControlTool(),
                OpenAppTool(
                    speak=self._tts.speak,
                    listen_for_response=self._listen_for_response,
                ),
                CloseAppTool(
                    speak=self._tts.speak,
                    listen_for_response=self._listen_for_response,
                ),
                SystemControlTool(
                    speak=self._tts.speak,
                    listen_for_cancel=self._listen_for_cancel,
                ),
                SearchAppsTool(),
                SearchFilesTool(),
                PlayMediaTool(),
            ]
            logger.debug("AgentRunner: tools built: %s", [t.name for t in tools])
        except Exception as exc:
            logger.error("Failed to instantiate agent tools: %s", exc, exc_info=True)
            return None

        # --- Build ChatOllama LLM --------------------------------------
        try:
            llm = ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )
            logger.info(
                "AgentRunner: ChatOllama model '%s' at %s.",
                settings.llm_model,
                settings.ollama_base_url,
            )
        except Exception as exc:
            logger.error("Failed to initialise ChatOllama: %s", exc)
            return None

        # --- Build agent -----------------------------------------------
        # create_agent returns a CompiledStateGraph.  Invoked with
        # {"messages": [HumanMessage, ...]} and returns {"messages": [...]}.
        # No prompt template needed — system_prompt is injected directly.
        try:
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
            )
            logger.info(
                "AgentRunner: create_agent ready (ChatOllama native tool calling)."
            )
            return agent
        except Exception as exc:
            logger.error("Failed to build agent: %s", exc)
            return None

    def _listen_for_response(self) -> str:
        """
        Record up to 10 seconds and return the transcribed text.

        Used by open_app / close_app during fuzzy-match confirmation.
        Returns an empty string on timeout or transcription failure.
        """
        result: list[str] = [""]
        done = threading.Event()

        def _record_and_transcribe() -> None:
            try:
                audio = self._vad.record_until_silence()
                if audio is not None and len(audio) > 0:
                    text = self._stt.transcribe(audio) or ""
                    logger.info("Confirmation response transcribed: '%s'", text)
                    result[0] = text
            except Exception as exc:
                logger.error("_listen_for_response error: %s", exc)
            finally:
                done.set()

        t = threading.Thread(target=_record_and_transcribe, daemon=True)
        t.start()
        done.wait(timeout=10.0)
        return result[0]

    def _listen_for_cancel(self) -> bool:
        """
        Record up to 10 seconds and return True if the user said "cancel".

        Used by the system_control tool during the safety countdown.
        """
        result: list[bool] = [False]
        done = threading.Event()

        def _record_and_transcribe() -> None:
            try:
                audio = self._vad.record_until_silence()
                if audio is not None and len(audio) > 0:
                    text = self._stt.transcribe(audio) or ""
                    logger.info("Cancel-listen transcribed: '%s'", text)
                    if "cancel" in text.lower():
                        result[0] = True
            except Exception as exc:
                logger.error("_listen_for_cancel error: %s", exc)
            finally:
                done.set()

        t = threading.Thread(target=_record_and_transcribe, daemon=True)
        t.start()
        done.wait(timeout=10.0)
        return result[0]
