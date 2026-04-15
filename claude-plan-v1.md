# VoiceOS — Implementation Plan v1

## Context

Build a voice-controlled local AI assistant agent with OS-level capabilities for Linux and Windows.
The assistant listens for a custom wake phrase, transcribes the user's command, routes it through
a LangChain tool-calling agent backed by a locally-running Ollama LLM, executes OS-level actions,
and speaks the result back using Kokoro TTS. A FastAPI + React config UI runs on localhost.

This document reflects all architectural decisions made during planning Q&A.

---

## Architectural Decisions (locked)

| Concern | Decision |
|---|---|
| Target OS | Linux + Windows (abstracted via os_handlers) |
| Wake word engine | openWakeWord (free, custom phrase, local) |
| STT | faster-whisper (local, no API key) |
| VAD | Silero VAD (detects end-of-speech) |
| LLM backend | Ollama (local), model configurable via UI |
| LLM approach | Prompt engineering only — no training/fine-tuning |
| Agent framework | LangChain LCEL + ReAct tool-calling agent |
| TTS | Kokoro TTS (primary), pyttsx3 (fallback), abstracted interface |
| Web search | duckduckgo-search Python library (free, no key) |
| File search | Filename scan + ChromaDB semantic search |
| Safety model | Whitelist-only: assistant can ONLY operate in user-configured dirs |
| System control safety | 10s countdown + spoken cancel window |
| Config UI | FastAPI backend + React frontend (localhost) |
| Memory | In-session only (last 5 turns), cleared on 1-min idle |
| Idle timeout | 1 minute of TRUE inactivity (resets on any event) |
| App launching | xdg-open (Linux) / start (Windows) + per-type app overrides |
| Distribution | pip install + setup script → systemd service (Linux) / Task Scheduler (Windows) |
| Language | English only |

---

## Technology Stack

### Core Dependencies
```
faster-whisper          # STT (Whisper via CTranslate2, fast + local)
openwakeword            # Wake word detection (custom phrase)
silero-vad              # Voice activity detection (end-of-speech)
sounddevice             # Microphone capture
numpy                   # Audio buffer handling

langchain               # Agent framework
langchain-ollama        # Ollama LLM integration
langchain-community     # Tool utilities
ollama                  # Ollama Python client

kokoro-onnx             # TTS (primary, near-human quality)
pyttsx3                 # TTS fallback (no extra model needed)

duckduckgo-search       # Web search (free, no API key)
chromadb                # Vector store for RAG
sentence-transformers   # Embeddings for ChromaDB

fastapi                 # Config UI backend
uvicorn                 # ASGI server for FastAPI

psutil                  # System control (sleep, shutdown)
pyautogui               # UI automation (fallback for volume)
pathlib                 # File navigation
python-dotenv           # Config loading
```

### System Requirements
- Python 3.10+
- Ollama installed and running (`ollama serve`)
- Default model suggestion in UI: Mistral 7B (`ollama pull mistral`)
- GPU: Optional (CUDA for faster-whisper and Kokoro), CPU mode supported
- RAM: ~4GB minimum (Whisper small + Ollama Q4 + ChromaDB)

---

## Project Structure

```
voice_os/
├── main.py                         # Entry point: starts all services
├── setup.py                        # Install systemd/startup service
├── requirements.txt
├── config/
│   ├── settings.py                 # Config load/save logic
│   └── config.json                 # User config (auto-generated)
├── core/
│   ├── wake_word.py                # openWakeWord listener loop
│   ├── speech_to_text.py           # faster-whisper transcription
│   ├── vad.py                      # Silero VAD: detect end of speech
│   ├── mic_manager.py              # Enumerate + select mic device
│   └── tts/
│       ├── base.py                 # Abstract TTSService interface
│       ├── kokoro_tts.py           # Kokoro TTS implementation
│       └── pyttsx3_tts.py          # pyttsx3 fallback implementation
├── agent/
│   ├── intent_parser.py            # LangChain ReAct agent setup
│   ├── executor.py                 # Orchestrates agent + tools + TTS
│   ├── prompts.py                  # System prompt definition
│   └── tools/
│       ├── open_app.py             # Tool: open application or file
│       ├── play_media.py           # Tool: play media with configured player
│       ├── search_web.py           # Tool: DuckDuckGo + LLM summarize
│       ├── search_files.py         # Tool: filename + ChromaDB search
│       ├── system_control.py       # Tool: sleep/restart/shutdown + countdown
│       └── volume_control.py       # Tool: get/set system volume
├── os_handlers/
│   ├── base.py                     # Abstract OSHandler interface
│   ├── linux.py                    # Linux implementation
│   └── windows.py                  # Windows implementation
├── memory/
│   ├── session.py                  # In-session conversation history (last 5 turns)
│   ├── vector_store.py             # ChromaDB wrapper
│   └── file_indexer.py             # Index whitelisted dirs into ChromaDB
├── ui/
│   ├── server.py                   # FastAPI app
│   ├── routes/
│   │   ├── config.py               # GET/POST config settings
│   │   ├── indexer.py              # POST trigger file indexing
│   │   └── status.py               # GET assistant status
│   └── static/                     # React frontend (built)
│       ├── index.html
│       └── assets/
└── tests/
    ├── test_intent_parser.py
    ├── test_tools.py
    └── test_os_handlers.py
```

---

## Layer-by-Layer Design

### Layer 1: Microphone Manager (`core/mic_manager.py`)
- On startup: call `sounddevice.query_devices()` to list input devices
- Read `config.json` for `mic_device_id`; if missing, use system default
- Expose `get_input_stream(callback)` used by wake word loop
- Config UI shows device dropdown; on save → writes `mic_device_id` to config.json

### Layer 2: Wake Word Detection (`core/wake_word.py`)
- Use `openwakeword` with a custom wake phrase model
- Custom model: trained offline using text-to-speech audio augmentation
  (openWakeWord provides `train.py` — generates ~5000 synthetic audio clips from TTS)
- Default wake phrase: `"OS Assistant"` (configurable up to 25 chars)
- If user changes phrase → retrain takes ~5 minutes on CPU, runs in background
- Wake word loop runs continuously in Idle state (very low CPU)
- On detection → emit signal → transition to Active Listening state

### Layer 3: VAD + STT (`core/vad.py`, `core/speech_to_text.py`)
- On wake word trigger: begin recording via sounddevice
- Silero VAD monitors audio stream in real-time
- Recording stops when 1.5s of silence detected after speech starts
- Pass audio buffer to `faster-whisper` (model: `small.en` for English-only)
- Returns: transcribed text string

### Layer 4: LLM Agent (`agent/intent_parser.py`)
- Create Ollama LLM client with model from config (default: `mistral`)
- Build LangChain ReAct agent with 6 tools registered
- System prompt (see below) defines all intents, response format, safety rules
- Agent receives transcribed text + session history (last 5 turns)
- Returns: tool call(s) + final response text

**System Prompt Design (`agent/prompts.py`):**
```
You are OS Assistant, a voice-controlled AI assistant running locally on the user's computer.
You help with: opening apps and files, playing media, searching the web, finding files,
controlling system state (volume, sleep, shutdown), and searching indexed documents.

SAFETY RULES:
- You can ONLY access files and directories that are in the user's whitelist.
- NEVER suggest deleting files. NEVER execute file deletion commands.
- For shutdown/restart/sleep: always use the system_control tool which has a built-in safety countdown.
- If the user asks you to do something outside the whitelist, say: "That path is not in your
  whitelist. Please add it in settings first."

RESPONSE STYLE:
- Be concise. This response will be spoken aloud.
- Avoid markdown, bullet points, or formatting — speak naturally.
- Confirm what you're doing: "Playing The Godfather using VLC."
- If you can't find something, offer alternatives if available.

SESSION CONTEXT:
You remember the last 5 interactions in this session. Use them for follow-up commands
(e.g., "play it louder" refers to the last media action).
```

### Layer 5: Tools (`agent/tools/`)

**`open_app.py`:**
- Input: `app_name` (str), optional `file_path` (str)
- If `file_path` given: check whitelist → open with `xdg-open` (Linux) / `os.startfile` (Windows), or configured override app
- If `app_name` only: search PATH for executable; if not found → suggest alternatives from config
- Returns: success message or error with alternatives

**`play_media.py`:**
- Input: `title` (str), optional `application` (str)
- Search ChromaDB + filesystem for media files matching title
- If found: launch with configured media player (or `xdg-open`)
- If not found: say "Could not find [title]. Did you mean [closest match]?"
- If app not found: "VLC not found. I can try [alternative from config]."

**`search_web.py`:**
- Input: `query` (str), optional `chain_prompts` (list of strings)
- Step 1: `DDGS().text(query, max_results=10)` → raw results
- If `chain_prompts`: build LangChain LCEL chain:
  `search_results | format_prompt | ollama_llm | parse_output`
  Each chain_prompt becomes a sequential step
- Returns: summarized result string (to be spoken by TTS)

**`search_files.py`:**
- Input: `query` (str), optional `search_type` ("name" | "content" | "auto")
- "name" search: `os.walk()` over whitelisted dirs, `fnmatch` for filename patterns
- "content" search: ChromaDB similarity search over indexed documents
- "auto": try filename first; if <3 results, fall back to content search
- Returns: list of matching file paths (spoken as "I found 3 files: ...")

**`system_control.py`:**
- Input: `action` ("sleep" | "restart" | "shutdown" | "hibernate")
- Speak: "Shutting down in 10 seconds. Say cancel to abort."
- Start 10-second countdown, listen for "cancel" via STT
- If no cancel: execute via `psutil` / `subprocess`
- Linux: `systemctl suspend/reboot/poweroff`
- Windows: `shutdown /s`, `shutdown /r`, `rundll32.exe powrprof.dll,SetSuspendState`

**`volume_control.py`:**
- Input: `action` ("set" | "increase" | "decrease" | "mute" | "unmute"), `value` (int, %)
- Linux: `pactl set-sink-volume @DEFAULT_SINK@ [value]%` via subprocess
- Windows: `pycaw` library (Windows Core Audio API)
- "increase by 10%" → get current level, clamp to 100% max
- Returns: "Volume set to 70%"

### Layer 6: OS Handlers (`os_handlers/`)

**`base.py` — Abstract interface:**
```python
class OSHandler(ABC):
    def open_file(self, path: str, app: str = None) -> bool: ...
    def get_volume(self) -> int: ...
    def set_volume(self, level: int) -> None: ...
    def sleep(self) -> None: ...
    def shutdown(self) -> None: ...
    def restart(self) -> None: ...
    def list_apps(self) -> list[str]: ...
    def find_app(self, name: str) -> str | None: ...
```

**`linux.py`:** Uses `pactl`, `xdg-open`, `systemctl`, `/proc`, `which`
**`windows.py`:** Uses `pycaw`, `os.startfile`, `ctypes`, `winreg`, `shutdown /s`

Loaded at startup via `platform.system()` detection.

### Layer 7: TTS (`core/tts/`)

**`base.py` — Abstract interface:**
```python
class TTSService(ABC):
    def speak(self, text: str) -> None: ...
    def is_available(self) -> bool: ...
```

**`kokoro_tts.py`:** Kokoro ONNX model, ~330MB, near-human quality. Loaded once at startup.
**`pyttsx3_tts.py`:** System TTS, instant, robotic. Used as fallback if Kokoro fails to load.

`executor.py` instantiates Kokoro first; falls back to pyttsx3 if not available.

---

## Memory Design

### In-Session History (`memory/session.py`)
```python
class SessionMemory:
    history: deque[tuple[str, str]]  # (user_text, assistant_response), maxlen=5
    
    def add(user: str, assistant: str)
    def get_context() -> str  # formatted for LLM prompt
    def clear()
```
- Cleared when idle timer fires (1 min of true inactivity)
- Passed to LangChain agent as `chat_history` in each call

### ChromaDB File Index (`memory/vector_store.py`, `memory/file_indexer.py`)
- Collection: `voice_os_files`
- Each document: file path, filename, first 500 chars of content (for text files), metadata
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (small, fast, ~80MB)
- Persist path: `~/.voice_os/chroma_db/`
- Indexing: triggered via UI button or on first launch for configured dirs
- Supported file types for content extraction: `.txt`, `.md`, `.pdf`, `.docx`, `.py`, `.json`, `.csv`

---

## State Machine (`main.py`)

```
                    ┌─────────────────┐
                    │   IDLE STATE    │◄──────────────────────────┐
                    │ openWakeWord    │                           │
                    │ listening loop  │                           │
                    └────────┬────────┘                          │
                             │ wake word detected                 │
                    ┌────────▼────────┐                          │
                    │ ACTIVE LISTEN   │                          │
                    │ sounddevice +   │                          │
                    │ Silero VAD      │                          │
                    └────────┬────────┘                          │
                             │ speech ended (1.5s silence)       │
                    ┌────────▼────────┐                          │
                    │ PROCESSING      │                          │
                    │ faster-whisper  │                          │
                    │ LangChain agent │                          │
                    │ tool execution  │                          │
                    └────────┬────────┘                          │
                             │ result ready                       │
                    ┌────────▼────────┐                          │
                    │ SPEAKING        │                          │
                    │ Kokoro TTS      │──► 1-min idle timer starts│
                    └─────────────────┘                          │
                                          timer fires ───────────┘
                                          OR user says wake word 
                                          (resets timer, goes active)
```

### Idle Timeout
- Timer is a `threading.Timer(60, go_idle)` reset on any event:
  - Wake word detected
  - Command received
  - Tool executed
  - TTS finished speaking
- On fire: speak "Going to idle state." → clear session memory → restart wake word loop

---

## UI Configuration Panel

### FastAPI Backend (`ui/server.py`)
Runs on `http://localhost:7860` (separate thread from main assistant loop)

Endpoints:
```
GET  /api/config          → return current config.json
POST /api/config          → save config.json, hot-reload settings
GET  /api/devices/mic     → list sounddevice input devices
GET  /api/models/ollama   → list installed Ollama models (ollama list)
GET  /api/status          → assistant state (idle/active/processing)
POST /api/index           → trigger file indexer for whitelisted dirs
GET  /api/index/status    → indexing progress (SSE stream)
```

### React Frontend (`ui/static/`)
Config sections:
1. **General**: Wake phrase (text input, max 25 chars), default mic (dropdown)
2. **LLM**: Ollama model selector (fetched from GET /api/models/ollama)
3. **Default Apps**: per-type overrides (browser, video, music, PDF, images, archives, docs)
4. **Whitelisted Directories**: add/remove dir paths; "Index Now" button per dir
5. **Status Panel**: current state indicator, last command/response, TTS test button

---

## Safety & Security Model

### Whitelist Enforcement
- `config.json` stores `whitelisted_dirs: ["/home/user/docs", "/home/user/media"]`
- Every tool that touches the filesystem calls `is_whitelisted(path)` before proceeding
- `is_whitelisted()` resolves symlinks and checks if `path.startswith(whitelist_entry)`
- Operations outside whitelist → tool returns error string → TTS speaks error
- Whitelist includes all subdirectories recursively (as specced)

### System File Blocking
Even within the whitelist, these are always blocked (defense-in-depth):
- Linux: `/`, `/etc`, `/boot`, `/sys`, `/proc`, `/dev`, `/usr`, `/lib`, `/bin`, `/sbin`
- Windows: `C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)`, `%SystemRoot%`

### File Deletion
- NO delete tool is registered with the agent
- LLM system prompt explicitly prohibits suggesting deletion
- Even if user asks "delete file X", the agent has no tool to do so

---

## Configuration File (`config/config.json`)

```json
{
  "mic_device_id": null,
  "wake_phrase": "OS Assistant",
  "llm_model": "mistral",
  "tts_engine": "kokoro",
  "whitelisted_dirs": [],
  "default_apps": {
    "browser": null,
    "video": null,
    "audio": null,
    "pdf": null,
    "docx": null,
    "xls": null,
    "image": null,
    "archive": null
  },
  "ollama_base_url": "http://localhost:11434",
  "ui_port": 7860
}
```

`settings.py` loads this with validation (pydantic BaseSettings) and exposes a singleton.

---

## Service Architecture

### Linux (`setup.py install-service`)
Generates and installs:
```ini
[Unit]
Description=VoiceOS AI Assistant
After=network.target sound.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/voice_os/main.py
Restart=on-failure
User=%i

[Install]
WantedBy=default.target
```
`systemctl --user enable voice_os && systemctl --user start voice_os`

### Windows (`setup.py install-service`)
Uses `schtasks` to register a startup task:
```
schtasks /create /tn "VoiceOS" /tr "python C:\voice_os\main.py" /sc ONLOGON
```

---

## Answering Spec's Open Questions

**Q: Suggest API-based approach for LLM**
Ollama exposes a local REST API (`http://localhost:11434`). LangChain's `OllamaLLM` calls it
via HTTP. This IS an API-based approach — cleanly decoupled from the assistant process.
The LLM can be restarted/swapped without restarting the assistant.

**Q: On what data to "train" the LLM?**
We use prompt engineering, not training. The system prompt IS the "training". To improve it:
- Add more few-shot examples per intent to `prompts.py`
- Log misunderstood commands → add them as examples
- Create a `prompts/examples/` directory with per-intent example files loaded at startup
- The UI could have a "teach" mode where user corrects misunderstandings → appended to examples

**Q: Suggest ways to swap the LLM model**
UI config panel → "LLM" section → dropdown populated from `ollama list`. On save → config.json
updated → `intent_parser.py` hot-reloads LLM client without restart.

**Q: Does switching models require retraining?**
No — prompt engineering is model-agnostic. The same system prompt works across Mistral, LLaMA 3,
Phi-3, Gemma, etc. Minor prompt adjustments may be needed (instruction format varies slightly)
but no data or compute is required.

---

## Implementation Phases

### Phase 1 — Core Loop (MVP)
1. `config/settings.py` — config loading/saving
2. `core/mic_manager.py` — device enumeration
3. `core/wake_word.py` — openWakeWord listener
4. `core/vad.py` — Silero VAD end-of-speech detection
5. `core/speech_to_text.py` — faster-whisper transcription
6. `core/tts/` — Kokoro + pyttsx3 implementations
7. `os_handlers/base.py`, `linux.py`, `windows.py` — OS abstraction
8. `main.py` — state machine (Idle → Active → Processing → Speaking → Idle)
**Goal**: Say wake word → command transcribed → spoken back (no LLM yet)

### Phase 2 — Agent + Basic Tools
1. `agent/prompts.py` — system prompt
2. `agent/intent_parser.py` — LangChain ReAct agent with Ollama
3. `agent/tools/volume_control.py`
4. `agent/tools/open_app.py`
5. `agent/tools/system_control.py` (with countdown)
6. `memory/session.py` — in-session history
**Goal**: Voice-controlled volume, app launching, safe shutdown

### Phase 3 — File System + Media
1. `memory/file_indexer.py` + `memory/vector_store.py` (ChromaDB)
2. `agent/tools/search_files.py` (filename + semantic)
3. `agent/tools/play_media.py` (with fuzzy match + alternatives)
**Goal**: "Find my notes about budget" works; "Play Godfather" with fallbacks

### Phase 4 — Web Search + Chaining
1. `agent/tools/search_web.py` with DDGS + LangChain LCEL chain
2. Multi-step chain_prompts support
**Goal**: "Search YouTube for RAG tutorials and read me the top 5"

### Phase 5 — Configuration UI
1. `ui/server.py` FastAPI app
2. `ui/routes/` — config, indexer, status, models endpoints
3. React frontend with all config sections
4. Indexing progress via SSE
**Goal**: Full settings panel on localhost:7860

### Phase 6 — Polish + Service
1. `setup.py` — systemd (Linux) + Task Scheduler (Windows) integration
2. Error handling: Ollama not running, mic unavailable, model not installed
3. Logging (`~/.voice_os/logs/voice_os.log`)
4. Wake phrase retraining UI flow
5. End-to-end testing across all tools

---

## Verification Plan

1. **Unit tests** (`tests/`): mock OS calls, test intent → tool routing, test whitelist logic
2. **Integration**: `pytest tests/` with a running Ollama instance
3. **Manual end-to-end checklist:**
   - [ ] Wake word triggers in quiet and noisy environments
   - [ ] "Raise volume by 10%" → volume increases by 10%
   - [ ] "Open Firefox" → browser opens
   - [ ] "Play [file outside whitelist]" → refused with explanation
   - [ ] "Shutdown" → 10s countdown → cancel works
   - [ ] "Find my notes about [topic]" → returns relevant file
   - [ ] "Search YouTube for RAG tutorials and read me the top 5" → speaks 5 results
   - [ ] 1-minute idle → "Going to idle state" spoken → session memory cleared
   - [ ] Config UI: change model → agent uses new model without restart
   - [ ] Index files button → progress shown → search finds indexed content

---

## Open Items for Next Plan Iteration

- [ ] openWakeWord custom model training UX (how does user train from UI?)
- [ ] Exact React UI component breakdown and mockup
- [ ] Error recovery: what does assistant say when Ollama is unreachable?
- [ ] PDF/DOCX text extraction libraries for file indexer
- [ ] Windows volume control via pycaw — verify API details
- [ ] Fuzzy matching library for "did you mean" suggestions (rapidfuzz?)
