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

import logging
import threading
from typing import TYPE_CHECKING

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
        self._session = SessionMemory(maxlen=5)
        self._agent = None          # built lazily
        self._build_lock = threading.Lock()

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

        # Build message list: session history + current input
        messages = []
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
                config={"recursion_limit": 5},
            )
            all_msgs = result.get("messages", [])

            # Log every tool execution
            for msg in all_msgs:
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        _conv_logger.info(
                            'AI Assistant Executed: "%s(%s)"',
                            tc.get("name", "unknown"),
                            tc.get("args", {}),
                        )

            # Extract the final spoken response from the last AIMessage with content.
            response = ""
            for msg in reversed(all_msgs):
                if isinstance(msg, AIMessage):
                    content = getattr(msg, "content", "") or ""
                    if content.strip():
                        response = content
                        break

            # If the model produced no final text (tool-only turn), use the
            # last ToolMessage as the spoken confirmation.
            if not response.strip():
                for msg in reversed(all_msgs):
                    if isinstance(msg, ToolMessage):
                        content = getattr(msg, "content", "") or ""
                        if content.strip():
                            response = content
                            logger.info(
                                "Agent: using tool result as spoken response: %r",
                                response,
                            )
                            break

            if not response.strip():
                response = _FALLBACK

        except Exception as exc:
            logger.error("AgentRunner.handle error: %s", exc)
            response = _FALLBACK

        _conv_logger.info('AI Assistant responded: "%s"', response)
        logger.debug("Agent output: %r", response)
        self._session.add(text, response)
        return response

    def clear_session(self) -> None:
        """Clear conversation history (called on idle timeout)."""
        self._session.clear()
        logger.info("Session memory cleared.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
