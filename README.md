# TALK — Terminal Agent Linux Kit

TALK is a single-file terminal agent runtime with a browser UI, a shared xterm.js terminal, native tool/function calling, session memory, and streaming LLM output.

## Features in v1.2

- OpenAI-compatible Chat Completions mode through `openai.AsyncOpenAI`.
- Native Ollama `/api/chat` mode with streaming, thinking, and tools.
- FastFlowLM defaults with `reasoning_effort=high`.
- Native terminal tools:
  - `terminal_read`
  - `terminal_send_text`
  - `terminal_send_keys`
  - `terminal_resize`
  - `terminal_create_session`
  - `terminal_list_sessions`
  - `terminal_switch_session`
  - `terminal_close_session`
- File tools:
  - `file_search`
  - `file_read`
  - `file_write`
- Session memory tools:
  - `memory_save`
  - `memory_search`
  - `memory_list`
  - `memory_forget`
- Task coordination tools:
  - `task_declare`
  - `task_mark_done`
  - `task_status`
- Multiple PTY terminal sessions with `session_id` targeting.
- Automatic new terminal session creation when the current session is busy/Working and a new command is sent without `session_id`.
- Session state tracking: `Working` / idle, active app, active command, current working directory, prompt state, and PID.
- Automatic terminal output capture after terminal-writing tools.
- Stronger `finish_task` discipline: the agent must verify terminal output before claiming completion.
- Fallback parser for fake or inline tool calls emitted inside normal text.
- FastFlowLM inline tool mode for backends that stream reasoning but hide native OpenAI tool calls.
- Context compaction and oversized tool-result trimming to reduce max-context failures.
- Browser UI with shared human/AI terminal, LLM debug panel, and session state updates.
- JSONL logs for debugging, LLM conversation traces, and session memory.

## Requirements

- Linux, macOS, or WSL. Native Windows terminals are not supported because TALK uses PTY APIs.
- Python 3.10+ recommended.
- One of:
  - FastFlowLM or another OpenAI-compatible Chat Completions backend.
  - Native Ollama server.

Install Python dependencies:

```bash
pip install fastapi uvicorn openai httpx
```

`httpx` is required for native Ollama streaming.

## Quick start

### FastFlowLM / OpenAI-compatible backend

Start a compatible model server:

```bash
flm serve gemma4-it:e4b
# or:
flm serve qwen3.5:9b
```

Run TALK:

```bash
python talk.py \
  --api-provider openai \
  --model gemma4-it:e4b \
  --prompt "Check the working directory"
```

Open the web UI:

```text
http://127.0.0.1:8000
```

### Native Ollama backend

Start Ollama and pull a model:

```bash
ollama serve
ollama pull qwen3.5:9b
```

Run TALK with native Ollama API:

```bash
python talk.py \
  --api-provider ollama \
  --model qwen3.5:9b \
  --prompt "Check the working directory"
```

Open the web UI:

```text
http://127.0.0.1:8000
```

## Configuration

TALK can be configured with CLI flags or environment variables.

| Option | Default | Description |
| --- | --- | --- |
| `--api-provider` | `TALK_API_PROVIDER`, `LLM_API_PROVIDER`, or `openai` | Backend mode: `openai` for OpenAI-compatible `/v1` Chat Completions, `ollama` for native Ollama `/api/chat`. |
| `--base-url` | Provider-dependent | For `openai`: `FASTFLOWLM_BASE_URL` or `http://127.0.0.1:52625/v1`. For `ollama`: `OLLAMA_HOST` or `http://127.0.0.1:11434`. |
| `--api-key` | Provider-dependent | For `openai`: `FASTFLOWLM_API_KEY` or `flm`. For `ollama`: `OLLAMA_API_KEY` or empty string. |
| `--model` | Provider-dependent | For `openai`: `FASTFLOWLM_MODEL` or `qwen3.5:9b`. For `ollama`: `OLLAMA_MODEL`, `FASTFLOWLM_MODEL`, or `qwen3.5:9b`. |
| `--host` | `127.0.0.1` | Web server host. |
| `--port` | `8000` | Web server port. |
| `--shell` | `$SHELL` or `/bin/bash` | Shell launched in the PTY. |
| `--cwd` | current working directory | Terminal working directory. |
| `--prompt` | none | Startup prompt; if omitted, enter it in the UI. |
| `--temperature` | `0.0` | LLM sampling temperature. |
| `--max-steps` | `40` | Maximum agent loop steps per run. |
| `--tool-choice` | `FASTFLOWLM_TOOL_CHOICE` or `required` | Tool-choice strategy for OpenAI-compatible mode: `required`, `auto`, `none`, or `omit`. |
| `--flm-tools-mode` | `TALK_FLM_TOOLS_MODE` or `auto` | FastFlowLM tool mode: `auto`, `native`, or `inline`. `inline` lets TALK parse `<|tool_call>` blocks itself. |
| `--empty-retries` | `2` | Retries when the model returns no content and no tool calls; also used for thinking-only retry budget. |
| `--llm-error-retries` | `5` | Retries for transient LLM/backend transport errors such as read timeouts. |
| `--terminal-context-chars` | `6000` | Trailing terminal characters attached to each LLM request. |
| `--terminal-tool-delay-ms` | `500` | Delay before attaching fresh terminal output after terminal-changing tools. |
| `--memory-enabled` / `--no-memory-enabled` | enabled | Enable or disable small session RAG memory. |
| `--memory-max-items` | `200` | Maximum number of facts kept in session memory. |
| `--memory-context-items` | `5` | Relevant memory items automatically attached to each LLM request. |
| `--max-request-chars` | `60000` | Approximate request character budget before conversation compaction. |
| `--compact-keep-recent-messages` | `30` | Number of newest messages kept verbatim during compaction. |
| `--compact-tool-result-chars` | `4000` | Maximum size of large strings inside tool results in the request copy. |
| `--reasoning-effort` | `TALK_REASONING_EFFORT`, `FASTFLOWLM_REASONING_EFFORT`, or `high` | Thinking/reasoning level: `none`, `low`, `medium`, `high`. |
| `--no-think` | off | Shortcut for `--reasoning-effort none`. |
| `--think` | off | Compatibility flag; thinking is already high by default. |
| `--log-dir` | `logs` | Directory for debug, conversation, and memory logs. |

## Changelog

### v1.2

v1.2 is the current version and a substantial runtime upgrade over v1.1.

#### Added

- **Native Ollama provider**
  - Added `--api-provider openai|ollama`.
  - Added native Ollama `/api/chat` streaming through `httpx`.
  - Added conversion between OpenAI-style tool calls and Ollama tool calls.
  - Added Ollama-aware thinking support through the `think` request field.
  - Added provider-specific defaults for base URL, API key, and model.

- **Multi-terminal PTY sessions**
  - Added `TerminalSession` management.
  - Added `terminal_create_session`.
  - Added `terminal_list_sessions`.
  - Added `terminal_switch_session`.
  - Added `terminal_close_session`.
  - Added `session_id` support to terminal tools.
  - Added `session_id` support to file tools so file operations can resolve paths relative to a chosen terminal session.
  - Added automatic session creation when the current session is `Working` and a new command is submitted without a target `session_id`.

- **Terminal session state tracking**
  - Tracks idle/Working state.
  - Tracks active app and active command.
  - Tracks prompt visibility.
  - Tracks terminal current working directory.
  - Broadcasts session state to the web UI.
  - Adds a background session status monitor.

- **Task coordination**
  - Added `task_declare`.
  - Added `task_mark_done`.
  - Added `task_status`.
  - Tasks can declare dependencies so the runtime can avoid starting dependent work too early.

- **FastFlowLM inline tool mode**
  - Added `--flm-tools-mode auto|native|inline`.
  - Added `<|tool_call>call:tool_name{...}<tool_call|>` parser.
  - Useful for FastFlowLM/Gemma-style backends that stream reasoning text but do not expose OpenAI-native tool calls reliably.

- **Improved LLM error handling**
  - Added `--llm-error-retries`.
  - Retries transient transport/backend errors with backoff.
  - Keeps the task unfinished instead of marking completion after backend failure.

- **Stronger completion verification**
  - The agent is instructed to inspect terminal results before `finish_task`.
  - `finish_task` is blocked when terminal commands were sent but their results were not verified.
  - Starting a command is no longer treated as successful completion.

#### Changed

- `terminal_read`, `terminal_send_text`, `terminal_send_keys`, and `terminal_resize` now accept an optional `session_id`.
- `file_search`, `file_read`, and `file_write` now accept an optional `session_id`.
- Terminal output attached after terminal-writing tools now comes from the targeted session.
- The system prompt now requires `terminal_list_sessions` before terminal operations.
- Tool-use discipline is stricter: the model is nudged to call tools instead of only describing the next command.
- The web UI now receives terminal session metadata, not only raw terminal output.
- Runtime defaults are provider-aware instead of assuming only FastFlowLM/OpenAI-compatible APIs.

#### Fixed

- Fixed a workflow issue where a UI step could force `max_steps=1` and stop the agent too early.
- Improved recovery from malformed fake tool-call JSON.
- Added repair for bare `_request_...` marker keys injected into streamed JSON.
- Improved parsing of loose JSON/Python-style tool arguments.
- Reduced risk of false completion after terminal actions.
- Improved behavior when a session is busy with an interactive app, REPL, editor, or long-running command.
- Improved resilience when backends reject `tool_choice=required`.

#### Dependency changes

- Added `httpx` for native Ollama streaming.

### v1.1

v1.1 is the baseline version.

#### Included

- OpenAI-compatible Chat Completions client through `openai.AsyncOpenAI`.
- FastFlowLM-oriented defaults.
- Single PTY terminal session.
- Native terminal tools:
  - `terminal_read`
  - `terminal_send_text`
  - `terminal_send_keys`
  - `terminal_resize`
  - `sleep`
  - `finish_task`
- File tools:
  - `file_search`
  - `file_read`
  - `file_write`
- Session memory tools:
  - `memory_save`
  - `memory_search`
  - `memory_list`
  - `memory_forget`
- Streaming LLM output with optional thinking/reasoning fields.
- Fallback parser for fake tool calls inside normal text.
- Browser UI with xterm.js and right-side LLM conversation/debug panel.
- Conversation reset button.
- JSONL logs for debug and conversation traces.
- Context compaction and large tool-result trimming.

## Safety warning

TALK gives an LLM access to a real local terminal. Run it only in environments where command execution is acceptable.

Recommended safety practices:

- Use a temporary test directory, container, disposable VM, or restricted user account.
- Keep the web server bound to `127.0.0.1` unless you fully understand the risks.
- Do not run destructive commands without explicit review.
- Treat all generated terminal actions as untrusted until verified.

## Project status

TALK is a compact experimental agent runtime. Backend compatibility can vary across OpenAI-compatible providers, especially around streaming tool calls, reasoning/thinking fields, and tool-call formatting. v1.2 improves compatibility by adding native Ollama support, provider-specific request handling, inline FastFlowLM tool mode, and stronger tool-call recovery.
