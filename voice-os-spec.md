# Overview: 
Build a voice-controlled local AI assistant agent with OS-level capabilities — a mix of accessibility tool + autonomous assistant.
The assistant should be able to: 
    - change volume,
    - open local files (docs, images, audio/video media)
    - browse the web and offer/open suggested results 
    - control system state: sleep/shutdown,restart

The assistant should be safeguarded against:
    - File and folder deletion (of system files and folders - os specific)
    - File modification (renaming, moving, copying of files inside system protected directories - os specific)


# High-Level Layers (6 core layers)

1. Voice Input (Speech → Text)
Capture microphone input and transcribe it, use Whisper (local via faster-whisper)
2. Intent Recognition (Text → Action)
Map user command → structured intent, use Ollama (run models locally like Mistral, LLaMA)
3. Command Engine (Action → OS execution)
Execute system-level commands safely, use LangChain or LlamaIndex
- Tool calling
- Prompt routing, chain prompts (if specified in the initial prompt)
- Agent behavior

4. OS Interaction Layer
Python (best choice for OS control)

Use libraries:

os, subprocess → run commands
psutil → system control
pyautogui → UI automation
pathlib → file navigation

5. Context + Memory (Optional but powerful)
Track user preferences, history, file index

6. Voice Output (Text → Speech)
Respond back to the user (use lightweight tts like pyttsx3, should be abstracted into interface so the TTS service layer can be swapped)

# High level architecture

I want interface-based services and LLM agent models. 
Use optimal (not bloated or very GPU heavy) LLM model for prompt execution, task chaining and RAG.


Suggested architecture structure:
core/
 ├── main.py
 ├── wake_word.py
 ├── speech_to_text.py
 ├── intent_parser.py
 ├── executor.py
 ├── tts.py
 ├── mic_manager.py
 └── os_handlers/
      ├── windows.py
      └── linux.py
      

# How the Assistant Runs (Runtime Model)

The app will run a background service with 3 states:

1. 💤 Idle (always running, low CPU)
Only listens for wake word
Uses lightweight model (not full Whisper)
2. 🎤 Active Listening
Triggered by: “OS Assistant”
Records full command
Sends to STT + LLM
3. ⚙️ Execution + Response
Runs command
Speaks result
Returns to idle
🔊 Wake Word Detection (CRITICAL)

Use a dedicated wake-word engine (not Whisper):

Best options:
Porcupine (Picovoice)

👉 Example wake phrase:

“OS Assistant”

🎤 Microphone Handling

Use:

pyaudio or sounddevice
On startup:
List all input devices
Let user choose (or auto-pick default)
import sounddevice as sd
print(sd.query_devices())

Store selected device in config:

{
  "mic_device_id": 2
}
🖥 OS Detection + Bootstrapping

`
import platform

os_type = platform.system()
Then load OS-specific handlers:
if os_type == "Windows":
    from commands.windows import *
elif os_type == "Linux":
    from commands.linux import *
`

🔁 Main Background Loop
while True:
    if detect_wake_word():
        speak("Listening...")
        
        audio = record_command()
        text = transcribe(audio)
        
        intent = parse_with_llm(text)
        result = execute(intent)
        
        speak(result)


# Process flow loop
    idle state -> listen for wake up word, example: "OS Assistant"
    audio -> listen()
    text -> transcribe(audio)
    intent -> parse(text)
    result -> execute(intent)
    speak(result)
    -> after 1 minute speak ('going to idle state') -> idle state (needs reactivation)

# UI/Configuration
    - Use gradient or other python UI to persist user preferences and expose to localhost for configuration
    - Select default apps for (browser, video media player, music media player, pdf documents, docx documents, xls documents, other documents, images, archives (.zip, .rar))
    - Choose wakeup word/phrase (up to 25 characters)
    - Choose default mic
    - Whitelist directories (choose specific directories the AI assistant can use and index), each directory provided will include all subdirectories and files (recursively)
    - Add action button to index the files in the directories into vector database (Chroma) for RAG purposes.


# Suggestions, and further considerations (TBD)

Sugggest API based approach for LLM prompt execution if applicapble to the app context. 
The OS assistant should be relatively lightweight and be able to run locally on GPUs with 8GB or less. 
Suggest ways to train the LLM model on diverse prompt use cases.
Suggest ways to swap the LLM model (offer alternatives using the UI)
Question: On what data should I train the LLM to perform as OS Assistant (be creative and give me ideas for polishing the LLM processing and prompt understanding).
Question: If the user starts with one LLM and then switches to another model (do the training has to be repeated on the new model)?

# User prompts and actions (Introduce LLM for Intent Parsing)
1. Register commands and associate executable actions
{
  "commands": [
    {
      "name": "open_app",
      "keywords": ["open", "start", "launch", "run"],
      "action": "open_app",
      "target": "string",
      "application": "string"
    },
    {
      "name": "play_media",
      "keywords": ["play", "watch", "listen", "stream"],
      "action": "play_media",
      "target": "string",
      "application": "string"
    },
    {
      "name": "search_web",
      "keywords": ["search", "find", "lookup", "google", "check", "web", "browse"],
      "action": "search_web",
      "target": "string",
      "chain_prompts": ["list"]
    },
    {
      "name": "search_files",
      "keywords": ["find file", "search documents", "locate"],
      "action": "search_files",
      "target": "string"
    },
    {
      "name": "system_control",
      "keywords": ["shutdown", "sleep", "restart", "reboot", "hibernate"],
      "action": "system_control",
      "target": "state"
    },
    {
      "name": "volume_control",
      "keywords": ["volume", "mute", "unmute", "louder", "quieter"],
      "action": "volume_control",
      "target": "level"
    }
  ]
}

2. Agent tool mappings:
tools = [
    {
        "name": "open_app",
        "function": open_app  # Reference to your local OS function
    },
    {
        "name": "play_media",
        "function": play_media
    },
    {
        "name": "search_files",
        "function": search_files
    },
    {
        "name": "search_web",
        "function": search_web
    },
    {
        "name": "system_control",
        "function": handle_system_state # Handles shutdown, sleep, etc.
    },
    {
        "name": "volume_control",
        "function": set_volume
    }
]

3. Example user prompts:

Examples:
User prompt 1:
"Play The Godfather part 1 using VLC"

Output:
{
  "intent": "play_media",
  "target": "The Godfather part 1",
  "application": "VLC"
}

Expected result:
- If found: 
I found The godfather part 1 movie on your machine and I am proceeding with playing it.
(Opens the found movie file using VLC media player)

- If movie not found:
Could not find godfather part 1 movie.
If similar movie found like `Godfather 2` -> would you like me to play `Godfather 2`

- If application not found: 
Could not find VLC media player on your machine, would you like me to use (found media player alternatives list)


User prompt 2:
"Search the web for Youtube videos on the topic of `AI RAG tutorials` and read me a list of the top 5 recommended video titles with a short brief of each video description"

Output:
{
  "intent": "search_web",
  "target": "Youtube videos on the topic of `AI RAG tutorials`",
  "chain_prompts": ["List top 5 videos", "Read each video title and give short description of its content and target audience"]
}

Expected results to be read to the user using TTS:
1. "RAG Fundamentals and Advanced Techniques – Full Course" (FreeCodeCamp/Vincibits)
Content: A comprehensive, multi-hour guide covering the basics of RAG, setting up vector databases, document chunking, and advanced techniques to improve retrieval accuracy.
Target Audience: Beginners to Intermediate developers wanting a complete, structured understanding from scratch.
2. "Complete RAG Tutorial 2025: Build AI Apps" (Harish Neel | AI)
Content: A 17-lesson series (approx. 4 hours) focusing on building practical AI applications that combine LLMs with external knowledge bases.
Target Audience: Developers aiming to build production-level applications in 2025.
3. "Building RAG Applications With LangChain in 2026!" (Pavan Belagatti/SingleStore)
Content: A hands-on tutorial that covers loading documents, splitting text, creating embeddings, storing them in a vector database (SingleStore), and building a chat interface with Streamlit.
Target Audience: Software engineers and developers looking for a "production-style" project tutorial.
4. "LangChain & LlamaIndex Tutorials: From RAG to Multi-Agent" (Pradip Nichite)
Content: This playlist covers advanced RAG techniques, focusing on using both LangChain and LlamaIndex for building sophisticated, agentic retrieval systems.
Target Audience: Intermediate/Advanced users wanting to move beyond basic RAG into multi-agent systems.
5. "RAG Explained in 12 Minutes: Architecture, Myths & 10 Patterns" (Aishwarya Srinivasan)
Content: A concise, high-level overview of RAG architecture, evaluating techniques, and exploring 10 key patterns (Self-RAG, Corrective RAG, Graph RAG) to build enterprise AI applications.
Target Audience: AI architects and Data Scientists needing to understand RAG design patterns quickly.

User prompt 3:

"Raise the volume with 10%"
Output:
{
  "intent": "volume_control",
  "target": "10%",
  "action_type": "increase"
}
Expected result:
 Raises the system speakers volume with 10% or to maximum if the ,[max volume] - [current volume] is less than 10%.