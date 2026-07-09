# cli-agent

A terminal-based AI coding agent in Python. It streams responses from an LLM via OpenRouter, manages conversation context, and provides a tool framework for file operations and future agent capabilities.

Inspired by a tutorial: [YouTube](https://www.youtube.com/watch?v=3GjE_YAs03s)

## Features

- **Streaming responses** — async LLM client with retry handling and live terminal output via Rich
- **Conversation context** — session history with token counting for context window tracking
- **System prompt** — identity, security, operational, and AGENTS.md guidelines assembled at runtime
- **Tool framework** — abstract `Tool` base class with Pydantic schemas and OpenAI function-calling format
- **Built-in tools** — `read_file` for reading text files with line numbers, offset/limit, and truncation

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Configure your OpenRouter API key in `client/llm_client.py` before running.

## Run

Pass a prompt as an argument to run a single request:

```bash
python main.py "Explain what this project does"
```

The agent streams its response to the terminal and exits when complete.

## Project layout

```
agent/
  agent.py          # Agent orchestration and LLM loop
  events.py         # Agent lifecycle and streaming events
client/
  llm_client.py     # Async OpenAI client (OpenRouter)
  response.py       # StreamEvent, TokenUsage, etc.
context/
  manager.py        # Conversation history and message assembly
prompts/
  system.py         # System prompt generation
tools/
  base.py           # Tool base class, schemas, and result types
  builtin/
    read_file.py    # Read text files with line numbers
ui/
  tui.py            # Rich terminal UI for streaming output
utils/
  paths.py          # Path resolution and binary file detection
  text.py           # Token counting and text truncation
main.py             # CLI entry point
```
