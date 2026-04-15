"""
AgentRunner — orchestrates the LangChain ReAct agent for Phase 2.

Responsibilities:
  - Build the LangChain agent (lazy, on first handle() call)
  - Inject TTS/STT callbacks into tools that need them (system_control)
  - Maintain in-session conversation history (last 5 turns)
  - Handle Ollama connectivity errors gracefully

Usage (called from main.py):
    runner = AgentRunner(tts=tts_service, stt=stt_service, vad=vad_recorder)
    response_text = runner.handle("turn up the volume")
    runner.clear_session()   # called on idle timeout
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from voice_os.memory.session import SessionMemory

if TYPE_CHECKING:
    from voice_os.core.tts.base import TTSService
    from voice_os.core.speech_to_text import WhisperTranscriber
    from voice_os.core.vad import VADRecorder

logger = logging.getLogger(__name__)


def _make_debug_callback():
    """
    Build a LangChain BaseCallbackHandler that logs rendered chat prompts at
    DEBUG level.  Constructed lazily inside handle() so the import only happens
    when langchain_core is already loaded.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _Cb(BaseCallbackHandler):
        def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
            for i, msg_batch in enumerate(messages):
                parts = []
                for msg in msg_batch:
                    role = getattr(msg, "type", msg.__class__.__name__)
                    content = getattr(msg, "content", str(msg))
                    parts.append(f"[{role}] {content}")
                logger.debug(
                    "── Chat prompt [batch %d] ────────────────────\n%s\n"
                    "────────────────────────────────────────────",
                    i, "\n".join(parts),
                )

    return _Cb()

# Fallback response when something goes wrong before/inside the agent
_FALLBACK = "Sorry, I couldn't process that. Please try again."


class AgentRunner:
    """
    Wraps LangChain AgentExecutor with session memory and audio callbacks.

    The agent is built lazily on the first handle() call so that import
    errors (e.g. langchain not installed) surface at runtime with a clear
    error message rather than at startup.
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
        self._session = SessionMemory(maxlen=5)
        self._executor = None          # built lazily
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
        try:
            executor = self._get_executor()
        except Exception as exc:
            logger.error("AgentRunner._get_executor raised: %s", exc, exc_info=True)
            executor = None

        if executor is None:
            return (
                "The AI agent is unavailable. "
                "Make sure Ollama is running and try again."
            )

        chat_history = self._session.get_context()

        logger.debug(
            "Agent input — user: %r  chat_history: %r",
            text,
            chat_history or "<empty>",
        )

        try:
            result = executor.invoke(
                {"input": text, "chat_history": chat_history},
                config={"callbacks": [_make_debug_callback()]},
            )
            response: str = result.get("output", _FALLBACK)

            # Mistral often calls the tool correctly but then fails to wrap
            # its Final Answer in the required JSON blob.  When we detect a
            # generic/stuck output, fall back to the first *successful* tool
            # observation — tools already return well-formed spoken sentences
            # like "Volume set to 50 percent."
            _stuck = (
                "stopped due to iteration limit" in response
                or response.strip() in ("Done.", "Done")
            )
            if _stuck:
                steps = result.get("intermediate_steps", [])
                _error_markers = ("unknown", "failed", "error", "could not")
                successful = [
                    str(obs)
                    for _, obs in steps
                    if not any(m in str(obs).lower() for m in _error_markers)
                ]
                if successful:
                    response = successful[0]
                    logger.info(
                        "Agent loop resolved — using first successful tool observation: %r",
                        response,
                    )
                elif steps:
                    response = str(steps[0][1])   # first obs even if imperfect
                else:
                    response = _FALLBACK

        except Exception as exc:
            logger.error("AgentRunner.handle error: %s", exc)
            response = _FALLBACK

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

    def _get_executor(self):
        """Return cached executor, building it if necessary."""
        if self._executor is not None:
            return self._executor
        with self._build_lock:
            if self._executor is None:   # double-checked
                self._executor = self._build_executor()
        return self._executor

    def _build_executor(self):
        """
        Build and return a LangChain AgentExecutor using ChatOllama +
        create_tool_calling_agent.

        This approach uses Mistral's native function-calling API instead of
        text-based ReAct parsing, which eliminates the "action='set' value=50"
        mis-formatting that occurred with the old ReAct text agent.
        """
        try:
            from langchain_ollama import ChatOllama
            from langchain_classic.agents import create_structured_chat_agent, AgentExecutor
            from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            from langchain_classic.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS
        except ImportError as exc:
            logger.error(
                "LangChain/Ollama not installed — Phase 2 agent unavailable. "
                "Run: pip install langchain langchain-classic langchain-ollama. "
                "Error: %s", exc
            )
            return None

        from voice_os.config.settings import settings
        from voice_os.agent.prompts import SYSTEM_PROMPT
        from voice_os.agent.tools.volume_control import VolumeControlTool
        from voice_os.agent.tools.open_app import OpenAppTool
        from voice_os.agent.tools.system_control import SystemControlTool

        # --- Build tools -----------------------------------------------
        try:
            volume_tool = VolumeControlTool()
            open_app_tool = OpenAppTool()
            system_tool = SystemControlTool(
                speak=self._tts.speak,
                listen_for_cancel=self._listen_for_cancel,
            )
            tools = [volume_tool, open_app_tool, system_tool]
            logger.debug("AgentRunner: tools built: %s", [t.name for t in tools])
        except Exception as exc:
            logger.error("Failed to instantiate agent tools: %s", exc, exc_info=True)
            return None

        # --- Build ChatOllama LLM --------------------------------------
        # ChatOllama uses Ollama's /api/chat endpoint which supports native
        # tool/function calling — no text parsing required.
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

        # --- Build chat prompt -----------------------------------------
        # structured_chat_agent requires {tools}, {tool_names} (injected
        # automatically), {chat_history} (our session string), {input}, and
        # {agent_scratchpad} (the running scratchpad).
        #
        # FORMAT_INSTRUCTIONS tells Mistral to emit a JSON blob for every
        # action — much more reliable than the plain-ReAct text format, and
        # avoids the "action='set' value=50" mis-formatting problem.
        system_template = (
            SYSTEM_PROMPT
            + "\n\n{tools}\n\n"
            + FORMAT_INSTRUCTIONS
            + "\n\nPrevious conversation:\n{chat_history}"
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}"),
        ])

        logger.debug(
            "Agent prompt template messages: %s",
            [m.__class__.__name__ for m in prompt.messages],
        )

        # --- Build agent + executor ------------------------------------
        try:
            agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
            # When Mistral writes free text after a tool call instead of a
            # proper JSON blob, the parser fails.  Rather than feeding
            # "Invalid or incomplete response" back (which causes looping),
            # we inject a Final Answer that stops the chain immediately.
            def _force_final_answer(error: Exception) -> str:
                return '{"action": "Final Answer", "action_input": "Done."}'

            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=1,
                handle_parsing_errors=_force_final_answer,
                # Must be True so handle() can fall back to the first
                # successful tool observation if needed.
                return_intermediate_steps=True,
            )
            logger.info("AgentRunner: structured-chat agent ready (ChatOllama).")
            return executor
        except Exception as exc:
            logger.error("Failed to build AgentExecutor: %s", exc)
            return None

    def _listen_for_cancel(self) -> bool:
        """
        Record up to 10 seconds and return True if the user said "cancel".

        Used by the system_control tool during the safety countdown.
        The VAD recorder handles end-of-speech detection naturally;
        we impose a 10-second overall cap via a daemon thread + Event.
        """
        import numpy as np

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

        # Wait at most 10 s — if user doesn't speak, assume no cancel.
        done.wait(timeout=10.0)
        return result[0]
