# TALK — Terminal Agent Linux Kit

TALK is a single-file terminal agent for FastFlowLM and other OpenAI-compatible Chat Completions APIs. It provides a browser UI with a shared xterm.js terminal on the left and an LLM conversation/debug panel on the right.

The agent can read from and write to a local PTY through native tool/function calls. It is designed for local experimentation with terminal-capable language models, especially models that support `tool_calls` and optional thinking/reasoning streams.

## Features

- OpenAI-compatible Chat Completions client via `openai.AsyncOpenAI`.
- FastFlowLM defaults with `reasoning_effort=high`.
- Native terminal tools: `terminal_read`, `terminal_send_text`, `terminal_send_keys`, `terminal_resize`, `sleep`, and `finish_task`.
- Streaming LLM output, including optional thinking/reasoning fields when the backend exposes them.
- Fallback parser for models that emit fake `tool_calls` inside normal text content.
- Browser UI built into the Python file using FastAPI, WebSockets, and xterm.js.
- Shared human/AI terminal: the user can type directly in the terminal while the agent receives terminal snapshots.
- Conversation reset button that clears the UI and backend model context without resetting the terminal.
- JSONL logs for debugging and LLM conversation traces.

## Requirements

- Linux, macOS, or WSL. Native Windows terminals are not supported because TALK uses PTY APIs.
- Python 3.10+ recommended.
- A running FastFlowLM server or any OpenAI-compatible Chat Completions backend.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Or install them directly:

```bash
pip install fastapi uvicorn openai
```

## Quick start

Start a compatible model server:

```bash
flm serve gemma4-it:e4b
# or:
flm serve qwen3.5:9b
```

Run TALK:

```bash
python talk.py --model gemma4-it:e4b --prompt "Check the working directory"
```

Open the web UI:

```text
http://127.0.0.1:8000
```

## Configuration

TALK can be configured with CLI flags or environment variables.

| Option | Default | Description |
| --- | --- | --- |
| `--base-url` | `FASTFLOWLM_BASE_URL` or `http://127.0.0.1:52625/v1` | OpenAI-compatible API base URL. |
| `--api-key` | `FASTFLOWLM_API_KEY` or `flm` | API key sent to the backend. |
| `--model` | `FASTFLOWLM_MODEL` or `qwen3.5:9b` | Model name. |
| `--host` | `127.0.0.1` | Web server host. |
| `--port` | `8000` | Web server port. |
| `--shell` | `$SHELL` or `/bin/bash` | Shell launched in the PTY. |
| `--cwd` | current working directory | Terminal working directory. |
| `--prompt` | none | Startup prompt; if omitted, enter it in the UI. |
| `--temperature` | `0.0` | LLM sampling temperature. |
| `--max-steps` | `40` | Maximum agent loop steps per run. |
| `--tool-choice` | `FASTFLOWLM_TOOL_CHOICE` or `required` | Tool-choice strategy: `required`, `auto`, `none`, or `omit`. |
| `--empty-retries` | `2` | Retries when the model returns no content and no tool calls. |
| `--terminal-context-chars` | `6000` | Trailing terminal characters attached to each LLM request. |
| `--reasoning-effort` | `FASTFLOWLM_REASONING_EFFORT` or `high` | Thinking/reasoning level: `none`, `low`, `medium`, `high`. |
| `--no-think` | off | Shortcut for `--reasoning-effort none`. |
| `--log-dir` | `logs` | Directory for debug and conversation logs. |

## Safety warning

TALK gives an LLM access to a real local terminal. Run it only in environments where command execution is acceptable. Prefer a temporary test directory, a container, a disposable VM, or a restricted user account. Keep the web server bound to `127.0.0.1` unless you fully understand the risks.

See [SECURITY.md](SECURITY.md) for more details.

## Documentation

- [Usage guide](docs/USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Prompt and tool contract](docs/PROMPTS.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Project status

This project is a compact experimental agent runtime. Expect backend compatibility differences across OpenAI-compatible providers, especially around streaming tool calls and reasoning/thinking fields.
