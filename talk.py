#!/usr/bin/env python3
"""
TALK — Terminal Agent Linux Kit
v1.1

A single-file terminal agent for FastFlowLM / OpenAI-compatible APIs.

Features:
- FastFlowLM API compatible with OpenAI Chat Completions
- streams LLM responses to the conversation panel, including thinking/reasoning fields when the backend exposes them
- streams tool execution start, parsed arguments, delayed terminal capture, results, and errors to the GUI
- maximum thinking mode by default: reasoning_effort=high
- native tool/function calling: terminal_read, terminal_send_text, terminal_send_keys,
  terminal_resize, sleep, file_search, file_read, file_write, finish_task
- after terminal-writing tools, automatically waits 500 ms and forwards fresh terminal output
  to the next LLM request in the tool result
- small session RAG memory: the model can save key facts with memory_save and
  receives relevant memories automatically before each LLM request
- context safety: old conversation history is compacted and oversized tool results are
  trimmed in request copies to avoid model max-context errors
- fallback parser: if the model returns fake tool_calls in regular content, TALK recovers and executes them
- web UI: xterm.js + shared human/AI terminal
- right panel: LLM conversation/debug log with its own scrollbar
- the text box on the right always sends messages to the LLM
- the human can type directly in the terminal; the LLM receives a terminal snapshot
- tolerant tool argument parsing and ignoring of extra fields
- “Clear conversation” button clears the UI and the model context on the backend
- debug and LLM conversation logs written to files

Requirements:
    pip install fastapi uvicorn openai

Start a model:
    flm serve gemma4-it:e4b
    # or: flm serve qwen3.5:9b

Start TALK:
    python talk.py --model gemma4-it:e4b --prompt "Check the working directory"

Security notes:
- The agent has access to a local terminal and can execute commands.
- Run it locally, preferably on 127.0.0.1 and inside a test directory.
- PTY works on Linux/macOS/WSL.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import pty
import re
import select
import shlex
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires Linux/macOS/WSL; on Windows, use WSL.") from exc

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from openai import AsyncOpenAI
import uvicorn


APP_NAME = "TALK"
APP_FULL_NAME = "TALK — Terminal Agent Linux Kit"


SYSTEM_PROMPT = """You are TALK — Terminal Agent Linux Kit. You share a Linux terminal with a human.

Working rules:
1. Complete the task by using NATIVE terminal tool calls.
2. If the task requires the terminal, use a tool. Do not merely describe what you would do.
3. Always use terminal_read first if you do not know the current terminal state.
4. After sending text/keys to the terminal, the runtime automatically waits 500 ms and attaches the fresh terminal output to the tool result. Use explicit sleep/terminal_read only when the command needs more time or more output.
5. Do not assume a long-running command has finished until you read the terminal.
6. Do not run destructive commands such as rm -rf, disk formatting, repository deletion, or global installations without an explicit user request.
7. When the task is finished, call finish_task with a short summary.
8. If the human types in the terminal, treat the visible terminal snapshot as shared work context.
9. If you need the human, write a short text message and wait for the next message from the conversation panel.
10. Be concise. The terminal is the source of truth.
11. Never stop after producing only private thinking/reasoning. If your analysis is not enough to finish, immediately choose and call the next useful native tool such as terminal_read, file_search, file_read, memory_search, sleep, or finish_task.

File tools protocol:
- Use file_read when you need to inspect a text file directly. Prefer it over cat/sed/awk for ordinary file reading because it is structured and can return selected line ranges.
- file_read requires path and accepts optional 1-based inclusive start_line and end_line. Use a narrow range for large files, then read more ranges if needed.
- file_read returns both text (raw file content without line-number prefixes) and numbered_text (the same returned content with line numbers). Use numbered_text to choose precise edit ranges, but never copy line-number prefixes into replacement content.
- Use file_search when you need to locate files or references before reading/editing. Prefer it over grep/find for ordinary searches because it returns structured path matches and content-line matches.
- file_search requires query and searches both file names/relative paths and file contents by default under the terminal's current working directory. It returns match_type="path" for filename/path matches and match_type="content" for content matches, with file path, line number, and matching line text for content matches.
- Use file_search results to choose the exact file and line range, then call file_read on a narrow range before making surgical file_write edits.
- file_search, file_read, and file_write resolve relative paths from the terminal's CURRENT working directory, equivalent to pwd in the shared shell, not from TALK's backend process directory or original startup directory. Use absolute paths when you need to avoid ambiguity.
- Use file_write to create or replace a whole text file when you know the complete desired content. It writes UTF-8 text. Use append=true only when intentionally appending.
- Use file_write with start_line and optional end_line to surgically replace a 1-based inclusive line range in an existing file. If start_line is provided without end_line, replace exactly that one line. If end_line is provided without start_line, replace lines 1..end_line.
- For surgical code edits, first call file_read on a focused range, inspect numbered_text, decide the exact start_line/end_line yourself, then call file_write with replacement content that does NOT include line numbers. Include the correct trailing newline in content when replacing whole lines unless you intentionally want to join with the following line.
- Do not use file_write for destructive edits, overwrites of important files, or broad generated changes unless the human requested or the plan clearly requires it. Prefer surgical line replacement for small code changes.

Planning and goal management:
- At the beginning of a task, form a short working plan before acting. The plan should identify the goal, constraints, likely files/commands to inspect, and the next verifiable step.
- Keep following the current plan across steps. Do not restart planning from scratch unless new evidence or a new human instruction requires it.
- Update the plan when tool results contradict assumptions, when a blocker appears, or when a safer/faster path is found.
- You may change sub-goals, priorities, and execution order when required to satisfy the human's real objective, preserve safety, or handle newly discovered constraints. Note the reason briefly.
- Prefer small verifiable steps: run a command, read the result, update memory/plan if needed, then continue.
- Do not over-plan. Keep plans compact and actionable.

Session memory protocol:
- Treat session memory as durable working memory for the current task/session. Use it to avoid rediscovering facts and to keep continuity when the conversation gets long or compacted.
- First read the automatically provided [auto_session_memory] block. If it contains enough information, use it directly without calling memory_search.
- Call memory_search when you need a fact that may have been saved earlier but is not visible in the current memory block, before asking the human again or repeating terminal exploration.
- Call memory_list only when you need a broad view of recent saved facts or when memory_search is too narrow.
- Save important facts with memory_save as soon as they become reliable and useful for the goal. Good memories include: repository root, relevant file paths, commands that worked, commands that failed in an informative way, chosen implementation decisions, user preferences, environment constraints, discovered APIs, ports, process IDs, test results, and unresolved blockers.
- Store memories as concise standalone facts. Include enough context to be useful later, for example: "Project root is /home/user/app; tests are run with pytest tests/unit".
- Use tags to make retrieval easier, such as repo, path, command, decision, error, test, user-preference, blocker, api, environment.
- Use importance 5 for facts that are essential to the current goal, 4 for strong decisions or recurring constraints, 3 for useful supporting facts, and 1-2 for low-priority notes.
- Do not save secrets, passwords, tokens, API keys, private keys, raw credentials, or transient terminal noise. If a secret appears, remember only a safe redacted fact such as "API token is configured in the environment".
- Do not save duplicate memories. If the same fact is already present, use it rather than saving it again.
- If a saved memory becomes wrong or obsolete and you know its id, call memory_forget, then save the corrected fact if still useful.

Non-repetition and context discipline:
- Treat prior assistant messages, tool results, terminal snapshots, and session memory as already known. Do not restate them unless the human explicitly asks for a recap.
- In thinking/reasoning, write only the delta: new observation, changed assumption, next action, plan update, memory decision, or blocker. Do not repeat earlier plans, explanations, command output, or conclusions.
- Keep thinking/reasoning compact: normally at most 5 short bullets or 1200 characters per assistant turn. Prefer fewer.
- Never copy more than one short sentence from your previous assistant response. Refer to earlier context with phrases like "as above", "from the last tool result", "same plan, next step", or "using saved memory".
- Do not paste terminal output into thinking unless a specific line is needed to decide the next action. Summarize repeated or long output.
- Final responses should contain only the current outcome, important changed files/commands, unresolved blockers, and the next useful action. Avoid re-explaining the full history.
- When continuing a multi-step task, start from the newest tool result, newest memory, or newest user instruction, not from a full recap.

Tool call contract:
- Use only the API-native tool_calls mechanism, not Markdown, XML, or textual JSON blocks.
- Function arguments must be a single JSON object.
- Do not use double braces such as {{...}}.
- Do not use Python dict syntax.
- Keys must be quoted.
- Booleans must be JSON: true or false.
- Do not add fields that are not in the schema.

Correct examples:
- terminal_read: {"max_chars":6000}
- terminal_send_text: {"text":"ls -la /home/user","newline":true}
- terminal_send_keys: {"keys":["CTRL_C"]}
- terminal_resize: {"cols":120,"rows":30}
- sleep: {"seconds":1}
- file_search: {"query":"TODO","path":".","search_filenames":true,"search_contents":true,"max_results":50}
- file_read: {"path":"src/app.py","start_line":1,"end_line":120}
- file_write whole file: {"path":"notes.txt","content":"hello\n","append":false}
- file_write replace lines: {"path":"src/app.py","start_line":42,"end_line":47,"content":"def fixed():\n    return True\n"}
- memory_save: {"text":"The repository root is /home/user/project.","tags":["repo","path"],"importance":4}
- memory_search: {"query":"repository root","max_items":5}
- memory_list: {"max_items":10}
- memory_forget: {"memory_id":"mem_0001"}
- finish_task: {"summary":"I completed the task."}
"""


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "terminal_read",
            "description": "Read recent visible/output text from the active shared terminal. Correct arguments example: {\"max_chars\":6000}",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum number of trailing characters to return.",
                        "default": 6000,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_send_text",
            "description": "Send exact text to the active terminal. Use newline=true to press Enter. Correct arguments example: {\"text\":\"ls -la /home/user\",\"newline\":true}",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "additionalProperties": False,
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Exact text to type into the terminal. Do not omit this field.",
                    },
                    "newline": {
                        "type": "boolean",
                        "description": "Append Enter/Return after the text.",
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_send_keys",
            "description": "Send special keys to the active terminal. Correct arguments example: {\"keys\":[\"ENTER\"]}. Valid keys: ENTER, CTRL_C, UP, DOWN, LEFT, RIGHT, TAB, ESC, BACKSPACE, CTRL_D.",
            "parameters": {
                "type": "object",
                "required": ["keys"],
                "additionalProperties": False,
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of special key names to send in order.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_resize",
            "description": "Resize the active terminal PTY. Correct arguments example: {\"cols\":120,\"rows\":30}",
            "parameters": {
                "type": "object",
                "required": ["cols", "rows"],
                "additionalProperties": False,
                "properties": {
                    "cols": {"type": "integer", "minimum": 20, "maximum": 300},
                    "rows": {"type": "integer", "minimum": 5, "maximum": 120},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sleep",
            "description": "Wait briefly, useful after sending commands before reading terminal output. Correct arguments example: {\"seconds\":1}",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "seconds": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 10,
                        "default": 1,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a UTF-8 text file from disk, optionally restricted to a 1-based inclusive line range. Returns raw text plus numbered_text with line-number prefixes for choosing surgical edit ranges. Relative paths are resolved from the terminal's current working directory. Correct arguments examples: {\"path\":\"src/app.py\"} or {\"path\":\"src/app.py\",\"start_line\":10,\"end_line\":40}",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "description": "File path to read. Relative paths are resolved from the terminal's current working directory, equivalent to pwd in the shared shell."},
                    "start_line": {"type": "integer", "minimum": 1, "description": "Optional 1-based first line to include."},
                    "end_line": {"type": "integer", "minimum": 1, "description": "Optional 1-based last line to include, inclusive."},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 200000, "default": 50000, "description": "Maximum characters of file text to return after line filtering."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write UTF-8 text to a file on disk. Relative paths are resolved from the terminal's current working directory. Replaces the whole file by default, appends with append=true, or surgically replaces a 1-based inclusive line range when start_line/end_line are provided. Correct arguments examples: {\"path\":\"notes.txt\",\"content\":\"hello\\n\",\"append\":false} or {\"path\":\"src/app.py\",\"start_line\":42,\"end_line\":47,\"content\":\"def fixed():\\n    return True\\n\"}",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "description": "File path to write. Relative paths are resolved from the terminal's current working directory, equivalent to pwd in the shared shell."},
                    "content": {"type": "string", "description": "Complete UTF-8 text content to write, text to append when append=true, or replacement content for the selected line range. Do not include line-number prefixes from numbered_text."},
                    "start_line": {"type": "integer", "minimum": 1, "description": "Optional 1-based first line to replace. When provided without end_line, only this line is replaced."},
                    "end_line": {"type": "integer", "minimum": 1, "description": "Optional 1-based last line to replace, inclusive. When provided without start_line, lines 1..end_line are replaced."},
                    "append": {"type": "boolean", "default": False, "description": "Append to the file instead of replacing it. Cannot be combined with start_line/end_line."},
                    "create_dirs": {"type": "boolean", "default": False, "description": "Create missing parent directories before writing whole-file or append content."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Search for a query in file names/relative paths and file contents under a directory resolved from the terminal's current working directory. Returns structured path matches and content matches with line numbers and matching line text. Prefer this over grep/find for ordinary codebase searches. Correct arguments example: {\"query\":\"TODO\",\"path\":\".\",\"search_filenames\":true,\"search_contents\":true,\"max_results\":50}",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "description": "Plain text or regex pattern to search for in file names/relative paths and/or file contents."},
                    "path": {"type": "string", "default": ".", "description": "Directory or file to search. Relative paths are resolved from the terminal's current working directory."},
                    "regex": {"type": "boolean", "default": False, "description": "Treat query as a Python regular expression instead of literal text."},
                    "case_sensitive": {"type": "boolean", "default": False, "description": "Whether matching is case-sensitive."},
                    "search_filenames": {"type": "boolean", "default": True, "description": "Search file names and relative paths."},
                    "search_contents": {"type": "boolean", "default": True, "description": "Search inside text file contents and return matching line text."},
                    "include_hidden": {"type": "boolean", "default": False, "description": "Include hidden files and directories. Common heavy directories such as .git and node_modules are still skipped by default unless named explicitly as the search path."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100, "description": "Maximum total matches to return."},
                    "max_files": {"type": "integer", "minimum": 1, "maximum": 20000, "default": 2000, "description": "Maximum files to inspect."},
                    "max_file_bytes": {"type": "integer", "minimum": 1, "maximum": 10000000, "default": 1000000, "description": "Skip content search for files larger than this many bytes."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "Save a durable, task-relevant fact to the session RAG memory. Do not store secrets, tokens, passwords, private keys, or transient terminal noise. Correct arguments example: {\"text\":\"The repository root is /home/user/project.\",\"tags\":[\"repo\",\"path\"],\"importance\":4}",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string", "description": "The concise fact to remember for this session."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional short tags such as repo, path, decision, error."},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search session RAG memory for relevant saved facts. Correct arguments example: {\"query\":\"repository root\",\"max_items\":5}",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "description": "What to search for in session memory."},
                    "max_items": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": "List recent saved facts from the session memory. Correct arguments example: {\"max_items\":10}",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_items": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_forget",
            "description": "Remove one saved session memory item by id. Correct arguments example: {\"memory_id\":\"mem_0001\"}",
            "parameters": {
                "type": "object",
                "required": ["memory_id"],
                "additionalProperties": False,
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory id to remove, such as mem_0001."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Mark the current task as finished and provide a concise summary for the human. Correct arguments example: {\"summary\":\"I completed the task.\"}",
            "parameters": {
                "type": "object",
                "required": ["summary"],
                "additionalProperties": False,
                "properties": {"summary": {"type": "string"}},
            },
        },
    },
]

TOOL_ALLOWED_ARGS: Dict[str, Set[str]] = {
    "terminal_read": {"max_chars"},
    "terminal_send_text": {"text", "newline"},
    "terminal_send_keys": {"keys"},
    "terminal_resize": {"cols", "rows"},
    "sleep": {"seconds"},
    "file_search": {"query", "path", "regex", "case_sensitive", "search_filenames", "search_contents", "include_hidden", "max_results", "max_files", "max_file_bytes"},
    "file_read": {"path", "start_line", "end_line", "max_chars"},
    "file_write": {"path", "content", "start_line", "end_line", "append", "create_dirs"},
    "memory_save": {"text", "tags", "importance"},
    "memory_search": {"query", "max_items"},
    "memory_list": {"max_items"},
    "memory_forget": {"memory_id"},
    "finish_task": {"summary"},
}

KEYS: Dict[str, bytes] = {
    "ENTER": b"\r",
    "RETURN": b"\r",
    "TAB": b"\t",
    "ESC": b"\x1b",
    "ESCAPE": b"\x1b",
    "BACKSPACE": b"\x7f",
    "DELETE": b"\x1b[3~",
    "UP": b"\x1b[A",
    "DOWN": b"\x1b[B",
    "RIGHT": b"\x1b[C",
    "LEFT": b"\x1b[D",
    "HOME": b"\x1b[H",
    "END": b"\x1b[F",
    "PAGE_UP": b"\x1b[5~",
    "PAGE_DOWN": b"\x1b[6~",
    "CTRL_C": b"\x03",
    "CTRL_D": b"\x04",
    "CTRL_Z": b"\x1a",
    "CTRL_L": b"\x0c",
}


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(exclude_none=True))
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    try:
        return json.loads(value.json())
    except Exception:
        return str(value)


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, data: Any) -> None:
        item = {"ts": time.time(), "event": event, "data": to_jsonable(data)}
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


class WebHub:
    def __init__(self, history_limit: int = 1000) -> None:
        self.clients: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.llm_history: Deque[Dict[str, Any]] = deque(maxlen=history_limit)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            self.clients.discard(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        async with self.lock:
            clients = list(self.clients)
        dead: List[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self.lock:
                for ws in dead:
                    self.clients.discard(ws)

    def broadcast_threadsafe(self, message: Dict[str, Any]) -> None:
        if not self.loop:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    async def emit_llm(self, event: str, data: Any) -> Dict[str, Any]:
        item = {"ts": time.time(), "event": event, "data": to_jsonable(data)}
        self.llm_history.append(item)
        await self.broadcast({"type": "llm_event", "item": item})
        return item

    async def clear_llm_history(self) -> None:
        self.llm_history.clear()
        await self.broadcast({"type": "llm_history", "events": []})


class PtyTerminal:
    def __init__(self, shell: str, cwd: Optional[str], rows: int = 28, cols: int = 100, scrollback_chars: int = 300_000) -> None:
        if os.name == "nt":
            raise RuntimeError("PTY w tym skrypcie wymaga Linux/macOS/WSL.")
        self.shell = shell
        self.cwd = cwd
        self.rows = rows
        self.cols = cols
        self.master_fd: Optional[int] = None
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.buffer: Deque[str] = deque(maxlen=scrollback_chars)
        self.buffer_start_abs = 0
        self.buffer_end_abs = 0
        self.buffer_lock = threading.Lock()
        self.hub: Optional[WebHub] = None

    def start(self, hub: WebHub) -> None:
        self.hub = hub
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd
        self._set_winsize(slave_fd, self.rows, self.cols)
        env = os.environ.copy()
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("COLORTERM", "truecolor")
        self.proc = subprocess.Popen(
            [self.shell],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=self.cwd,
            env=env,
            preexec_fn=os.setsid,
            close_fds=True,
        )
        os.close(slave_fd)
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self.reader_thread = threading.Thread(target=self._reader_loop, name="pty-reader", daemon=True)
        self.reader_thread.start()
        logging.info("Terminal started: shell=%s cwd=%s pid=%s", self.shell, self.cwd, self.proc.pid)

    def _reader_loop(self) -> None:
        assert self.master_fd is not None
        while not self.stop_event.is_set():
            try:
                readable, _, _ = select.select([self.master_fd], [], [], 0.05)
                if not readable:
                    continue
                data = os.read(self.master_fd, 8192)
                if not data:
                    continue
                text = data.decode("utf-8", errors="replace")
                with self.buffer_lock:
                    self.buffer.extend(text)
                    self.buffer_end_abs += len(text)
                    self.buffer_start_abs = self.buffer_end_abs - len(self.buffer)
                if self.hub:
                    self.hub.broadcast_threadsafe({"type": "terminal", "data": text})
            except OSError:
                break
            except Exception:
                logging.exception("Terminal reader error")
                break

    def _set_winsize(self, fd: int, rows: int, cols: int) -> None:
        packed = struct.pack("HHHH", int(rows), int(cols), 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)

    def resize(self, rows: int, cols: int) -> Dict[str, Any]:
        self.rows = max(5, min(int(rows), 120))
        self.cols = max(20, min(int(cols), 300))
        if self.master_fd is not None:
            self._set_winsize(self.master_fd, self.rows, self.cols)
        if self.proc and self.proc.pid:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGWINCH)
            except Exception:
                pass
        return {"ok": True, "rows": self.rows, "cols": self.cols}

    def write(self, data: bytes) -> Dict[str, Any]:
        if self.master_fd is None:
            return {"ok": False, "error": "terminal not started"}
        written = os.write(self.master_fd, data)
        return {"ok": True, "bytes": written}

    def send_text(self, text: str, newline: bool = False) -> Dict[str, Any]:
        data = text.encode("utf-8") + (b"\r" if newline else b"")
        return self.write(data)

    def send_keys(self, keys: Iterable[str]) -> Dict[str, Any]:
        sent: List[str] = []
        unknown: List[str] = []
        for key in keys:
            normalized = str(key).upper().replace("-", "_").replace("+", "_")
            data = KEYS.get(normalized)
            if data is None:
                unknown.append(str(key))
                continue
            self.write(data)
            sent.append(normalized)
        return {"ok": not unknown, "sent": sent, "unknown": unknown}

    def read(self, max_chars: int = 6000) -> str:
        max_chars = max(1, min(int(max_chars), 50_000))
        with self.buffer_lock:
            text = "".join(self.buffer)
        return text[-max_chars:]

    def cursor(self) -> int:
        """Return a monotonic character offset for the terminal output stream."""
        with self.buffer_lock:
            return self.buffer_end_abs

    def current_working_directory(self) -> Path:
        """Return the current working directory of the shared shell when available.

        On Linux/WSL this uses /proc/<shell-pid>/cwd, so it follows cd commands
        entered in the terminal. On platforms without /proc, fall back to the
        startup cwd and then the backend process cwd.
        """
        if self.proc and self.proc.pid:
            proc_cwd = Path(f"/proc/{self.proc.pid}/cwd")
            try:
                target = Path(os.readlink(proc_cwd))
                if target.exists() and target.is_dir():
                    return target.resolve(strict=False)
            except Exception:
                pass
        base = Path(self.cwd or os.getcwd()).expanduser()
        return base.resolve(strict=False)

    def read_since(self, cursor: int, max_chars: int = 6000) -> str:
        """Read terminal output produced after a previously captured cursor offset."""
        max_chars = max(1, min(int(max_chars), 50_000))
        with self.buffer_lock:
            text = "".join(self.buffer)
            start_abs = self.buffer_start_abs
            end_abs = self.buffer_end_abs
        cursor = max(start_abs, min(int(cursor), end_abs))
        rel_start = max(0, cursor - start_abs)
        return text[rel_start:][-max_chars:]

    def stop(self) -> None:
        self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGHUP)
            except Exception:
                self.proc.terminate()
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None


class SessionMemory:
    """Small in-process RAG memory for one TALK session.

    It intentionally avoids external dependencies. Retrieval uses a compact lexical
    score with recency and importance boosts, while save/search/list are exposed as
    LLM API tool calls.
    """

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
        "i", "you", "we", "they", "he", "she", "was", "were", "will", "would", "can", "could", "should", "do", "does", "did",
        "jest", "są", "sa", "to", "ten", "ta", "te", "oraz", "i", "w", "we", "z", "ze", "na", "do", "dla", "że", "ze", "jak", "czy", "się", "sie",
        "nie", "tak", "po", "od", "pod", "nad", "przy", "aby", "żeby", "zeby", "model", "agent", "terminal",
    }

    SECRET_PATTERNS = (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
        re.compile(r"\b(?:api[_-]?key|token|secret|password|passwd|hasło|haslo)\b\s*[:=]", re.IGNORECASE),
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"),
    )

    def __init__(self, max_items: int = 200, log_path: Optional[Path] = None) -> None:
        self.max_items = max(1, int(max_items))
        self.log_path = log_path
        self.items: List[Dict[str, Any]] = []
        self.next_id = 1
        self.lock = threading.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        item = {"ts": time.time(), "event": event, "data": to_jsonable(payload)}
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logging.exception("Could not write session memory log")

    @classmethod
    def _tokenize(cls, text: str) -> Set[str]:
        tokens = {t.lower() for t in re.findall(r"[\w./:@+-]{2,}", text or "", flags=re.UNICODE)}
        return {t for t in tokens if t not in cls.STOPWORDS}

    @classmethod
    def _looks_secret(cls, text: str) -> bool:
        return any(p.search(text or "") for p in cls.SECRET_PATTERNS)

    @staticmethod
    def _clean_text(text: str, limit: int = 1200) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text[:limit].rstrip()

    @staticmethod
    def _clean_tags(tags: Any) -> List[str]:
        if tags is None:
            return []
        if isinstance(tags, str):
            raw = [p.strip() for p in re.split(r"[,;]", tags) if p.strip()]
        elif isinstance(tags, (list, tuple, set)):
            raw = [str(p).strip() for p in tags if str(p).strip()]
        else:
            raw = [str(tags).strip()] if str(tags).strip() else []
        seen: Set[str] = set()
        out: List[str] = []
        for tag in raw:
            tag = re.sub(r"[^\w./:-]+", "_", tag.lower(), flags=re.UNICODE).strip("_")[:40]
            if tag and tag not in seen:
                seen.add(tag)
                out.append(tag)
        return out[:10]

    def save(self, text: str, tags: Any = None, importance: int = 3, source: str = "agent_tool") -> Dict[str, Any]:
        text = self._clean_text(text)
        if not text:
            return {"ok": False, "error": "memory text is empty"}
        if self._looks_secret(text):
            return {"ok": False, "error": "refused to store text that looks like a secret/token/password"}
        try:
            importance_i = int(importance)
        except Exception:
            importance_i = 3
        importance_i = max(1, min(importance_i, 5))
        tag_list = self._clean_tags(tags)
        now = time.time()
        text_key = text.lower()
        with self.lock:
            for item in self.items:
                if str(item.get("text", "")).lower() == text_key:
                    item["updated_at"] = now
                    item["importance"] = max(int(item.get("importance", 3)), importance_i)
                    item["tags"] = sorted(set(item.get("tags", [])) | set(tag_list))
                    result = {"ok": True, "deduped": True, "item": dict(item), "count": len(self.items)}
                    self._log("memory_update", result)
                    return result
            memory_id = f"mem_{self.next_id:04d}"
            self.next_id += 1
            item = {
                "id": memory_id,
                "text": text,
                "tags": tag_list,
                "importance": importance_i,
                "source": source,
                "created_at": now,
                "updated_at": now,
            }
            self.items.append(item)
            while len(self.items) > self.max_items:
                self.items.pop(0)
            result = {"ok": True, "deduped": False, "item": dict(item), "count": len(self.items)}
            self._log("memory_save", result)
            return result

    def _score(self, item: Dict[str, Any], query_tokens: Set[str], query_text: str, now: float) -> float:
        text = str(item.get("text", ""))
        tags = " ".join(str(t) for t in item.get("tags", []))
        item_tokens = self._tokenize(text + " " + tags)
        overlap = len(query_tokens & item_tokens)
        if query_tokens:
            coverage = overlap / max(1, len(query_tokens))
        else:
            coverage = 0.0
        phrase_bonus = 0.0
        q = (query_text or "").lower().strip()
        if q and len(q) >= 4 and q in text.lower():
            phrase_bonus = 2.0
        semantic_score = float(overlap) + coverage + phrase_bonus
        if query_tokens and semantic_score <= 0.0:
            return 0.0
        importance_bonus = 0.15 * int(item.get("importance", 3))
        age_hours = max(0.0, (now - float(item.get("updated_at", item.get("created_at", now)))) / 3600.0)
        recency_bonus = 0.25 / (1.0 + age_hours)
        return semantic_score + importance_bonus + recency_bonus

    def search(self, query: str, max_items: int = 5) -> Dict[str, Any]:
        query = self._clean_text(query, limit=2000)
        max_items = max(1, min(int(max_items or 5), 50))
        now = time.time()
        query_tokens = self._tokenize(query)
        with self.lock:
            candidates = [dict(item) for item in self.items]
        if not candidates:
            return {"ok": True, "query": query, "items": [], "count": 0}
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in candidates:
            score = self._score(item, query_tokens, query, now)
            if score > 0.0 or not query_tokens:
                item["score"] = round(score, 4)
                scored.append((score, item))
        scored.sort(key=lambda x: (x[0], float(x[1].get("updated_at", 0))), reverse=True)
        items = [item for _, item in scored[:max_items]]
        return {"ok": True, "query": query, "items": items, "count": len(items), "total_memories": len(candidates)}

    def list_recent(self, max_items: int = 10) -> Dict[str, Any]:
        max_items = max(1, min(int(max_items or 10), 100))
        with self.lock:
            items = [dict(item) for item in self.items[-max_items:]]
            total = len(self.items)
        items.reverse()
        return {"ok": True, "items": items, "count": len(items), "total_memories": total}

    def forget(self, memory_id: str) -> Dict[str, Any]:
        memory_id = str(memory_id or "").strip()
        if not memory_id:
            return {"ok": False, "error": "missing memory_id"}
        with self.lock:
            for idx, item in enumerate(self.items):
                if item.get("id") == memory_id:
                    removed = self.items.pop(idx)
                    result = {"ok": True, "removed": removed, "count": len(self.items)}
                    self._log("memory_forget", result)
                    return result
        return {"ok": False, "error": f"memory not found: {memory_id}"}

    def context_block(self, query: str, max_items: int = 5) -> str:
        result = self.search(query, max_items=max_items)
        items = result.get("items", [])
        if not items:
            return ""
        lines = [
            "[session_memory]",
            "Relevant facts saved during this TALK session. Treat them as hints; verify with terminal when needed.",
        ]
        for item in items:
            tags = ",".join(item.get("tags", [])) or "-"
            lines.append(f"- {item.get('id')} | importance={item.get('importance')} | tags={tags} | {item.get('text')}")
        lines.append("[/session_memory]")
        return "\n".join(lines)



class ToolArgParser:
    TEXT_ALIASES = ("text", "cmd", "command", "input", "data", "content")
    NEWLINE_ALIASES = ("newline", "enter", "press_enter", "submit", "return")

    @classmethod
    def parse_and_normalize(cls, tool_name: str, raw_args: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        meta: Dict[str, Any] = {"raw_arguments": raw_args, "parse_notes": [], "ignored_args": {}}
        parsed = cls._parse_any(raw_args, meta)
        if not isinstance(parsed, dict):
            meta["parse_notes"].append(f"arguments were {type(parsed).__name__}, coerced to empty object")
            parsed = {}
        normalized = cls._normalize(tool_name, parsed, meta)
        allowed = TOOL_ALLOWED_ARGS.get(tool_name)
        if allowed is not None:
            consumed_aliases = set(meta.get("consumed_aliases", {}).keys())
            ignored = {k: v for k, v in parsed.items() if k not in allowed and k not in consumed_aliases}
            if ignored:
                meta["ignored_args"] = ignored
            normalized = {k: v for k, v in normalized.items() if k in allowed}
        return normalized, meta

    @classmethod
    def _parse_any(cls, raw_args: Any, meta: Dict[str, Any]) -> Any:
        if raw_args is None or raw_args == "":
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, (list, tuple)):
            return {"value": raw_args}
        s = str(raw_args).strip()
        candidates = cls._candidate_strings(s)
        last_error: Optional[BaseException] = None
        for cand in candidates:
            for repaired in cls._json_candidates(cand):
                try:
                    value = json.loads(repaired)
                    if isinstance(value, str):
                        meta["parse_notes"].append("decoded double-encoded JSON string")
                        return cls._parse_any(value, meta)
                    if repaired != s:
                        meta["parse_notes"].append("repaired JSON-like arguments")
                    return value
                except Exception as exc:
                    last_error = exc
        for cand in candidates:
            for repaired in cls._python_candidates(cand):
                try:
                    value = ast.literal_eval(repaired)
                    if repaired != s:
                        meta["parse_notes"].append("repaired Python-like arguments")
                    else:
                        meta["parse_notes"].append("parsed Python-like arguments")
                    return value
                except Exception as exc:
                    last_error = exc
        meta["parse_notes"].append(f"argument parse failed: {last_error!r}")
        return {}

    @classmethod
    def _candidate_strings(cls, s: str) -> List[str]:
        out: List[str] = []
        def add(x: str) -> None:
            x = str(x).strip()
            if x and x not in out:
                out.append(x)
        add(s)
        if s.startswith("```"):
            stripped = s.strip("`").strip()
            if "\n" in stripped:
                first, rest = stripped.split("\n", 1)
                if first.strip().lower() in {"json", "javascript", "js", "python"}:
                    stripped = rest
            add(stripped)
        if s.startswith("{{") and s.endswith("}}"):
            add(s[1:-1])
        extracted = cls._extract_first_object(s)
        if extracted:
            add(extracted)
            if extracted.startswith("{{") and extracted.endswith("}}"):
                add(extracted[1:-1])
        for marker in ("arguments=", "args=", "parameters=", "input="):
            if marker in s:
                add(s.split(marker, 1)[1].strip())
        match = re.search(r"\w+\s*\((.*)\)\s*$", s, flags=re.DOTALL)
        if match:
            add(match.group(1))
        return out

    @staticmethod
    def _extract_first_object(s: str) -> Optional[str]:
        start = min([i for i in (s.find("{"), s.find("[")) if i >= 0], default=-1)
        if start < 0:
            return None
        stack: List[str] = []
        quote: Optional[str] = None
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue
            if ch in {'"', "'"}:
                quote = ch
            elif ch in "[{":
                stack.append("}" if ch == "{" else "]")
            elif ch in "]}":
                if not stack or ch != stack[-1]:
                    return None
                stack.pop()
                if not stack:
                    return s[start:i+1]
        return None

    @classmethod
    def _json_candidates(cls, s: str) -> List[str]:
        out = [s]
        repaired = cls._replace_words_outside_strings(s, {"True": "true", "False": "false", "None": "null"})
        repaired = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', lambda m: f'{m.group(1)}"{m.group(2)}":', repaired)
        if repaired not in out:
            out.append(repaired)
        repaired2 = re.sub(
            r':\s*([^\"\'\{\}\[\],][^,\}]*)',
            lambda m: cls._quote_loose_value(m.group(1)),
            repaired,
        )
        if repaired2 not in out:
            out.append(repaired2)
        return out

    @classmethod
    def _python_candidates(cls, s: str) -> List[str]:
        repaired = cls._replace_words_outside_strings(s, {"true": "True", "false": "False", "null": "None"})
        return [s, repaired] if repaired != s else [s]

    @staticmethod
    def _quote_loose_value(value: str) -> str:
        raw = value.strip()
        low = raw.lower()
        if low in {"true", "false", "null"}:
            return ":" + low
        if re.fullmatch(r"-?\d+(\.\d+)?", raw):
            return ":" + raw
        return ":" + json.dumps(raw, ensure_ascii=False)

    @staticmethod
    def _replace_words_outside_strings(s: str, mapping: Dict[str, str]) -> str:
        out: List[str] = []
        i = 0
        quote: Optional[str] = None
        escape = False
        while i < len(s):
            ch = s[i]
            if quote:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                i += 1
                continue
            if ch in {'"', "'"}:
                quote = ch
                out.append(ch)
                i += 1
                continue
            matched = False
            for src, dst in mapping.items():
                end = i + len(src)
                if s[i:end] == src:
                    before = s[i - 1] if i > 0 else ""
                    after = s[end] if end < len(s) else ""
                    if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                        out.append(dst)
                        i = end
                        matched = True
                        break
            if not matched:
                out.append(ch)
                i += 1
        return "".join(out)

    @classmethod
    def _normalize(cls, tool_name: str, args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "terminal_read":
            value = cls._first_present(args, ("max_chars", "chars", "limit", "max"), meta)
            return {"max_chars": cls._to_int(value, default=6000, minimum=1, maximum=50_000)}
        if tool_name == "terminal_send_text":
            text = cls._first_present(args, cls.TEXT_ALIASES, meta)
            newline = cls._first_present(args, cls.NEWLINE_ALIASES, meta)
            out: Dict[str, Any] = {"newline": cls._to_bool(newline, default=False)}
            if text is not None:
                out["text"] = str(text)
            return out
        if tool_name == "terminal_send_keys":
            keys = cls._first_present(args, ("keys", "key", "sequence"), meta)
            if keys is None:
                return {}
            if isinstance(keys, list):
                return {"keys": [str(k) for k in keys]}
            if isinstance(keys, str) and "," in keys:
                return {"keys": [p.strip() for p in keys.split(",") if p.strip()]}
            return {"keys": [str(keys)]}
        if tool_name == "terminal_resize":
            rows = cls._first_present(args, ("rows", "height"), meta)
            cols = cls._first_present(args, ("cols", "columns", "width"), meta)
            return {"rows": cls._to_int(rows, 28, 5, 120), "cols": cls._to_int(cols, 100, 20, 300)}
        if tool_name == "sleep":
            seconds = cls._first_present(args, ("seconds", "second", "secs", "duration"), meta)
            return {"seconds": cls._to_float(seconds, 1.0, 0.1, 10.0)}
        if tool_name == "file_search":
            query = cls._first_present(args, ("query", "q", "text", "pattern", "search", "needle"), meta)
            path = cls._first_present(args, ("path", "dir", "directory", "root", "folder", "base"), meta)
            regex = cls._first_present(args, ("regex", "regexp", "use_regex", "pattern_mode"), meta)
            case_sensitive = cls._first_present(args, ("case_sensitive", "case", "match_case"), meta)
            search_filenames = cls._first_present(args, ("search_filenames", "filenames", "names", "paths", "search_paths"), meta)
            search_contents = cls._first_present(args, ("search_contents", "contents", "content", "inside", "lines"), meta)
            include_hidden = cls._first_present(args, ("include_hidden", "hidden", "dotfiles"), meta)
            max_results = cls._first_present(args, ("max_results", "limit", "max", "count"), meta)
            max_files = cls._first_present(args, ("max_files", "file_limit", "files"), meta)
            max_file_bytes = cls._first_present(args, ("max_file_bytes", "max_bytes", "size_limit"), meta)
            out: Dict[str, Any] = {
                "path": str(path or "."),
                "regex": cls._to_bool(regex, default=False),
                "case_sensitive": cls._to_bool(case_sensitive, default=False),
                "search_filenames": cls._to_bool(search_filenames, default=True),
                "search_contents": cls._to_bool(search_contents, default=True),
                "include_hidden": cls._to_bool(include_hidden, default=False),
                "max_results": cls._to_int(max_results, 100, 1, 1000),
                "max_files": cls._to_int(max_files, 2000, 1, 20_000),
                "max_file_bytes": cls._to_int(max_file_bytes, 1_000_000, 1, 10_000_000),
            }
            if query is not None:
                out["query"] = str(query)
            return out
        if tool_name == "file_read":
            path = cls._first_present(args, ("path", "file", "filename", "filepath"), meta)
            start_line = cls._first_present(args, ("start_line", "start", "from_line", "line_start", "first_line"), meta)
            end_line = cls._first_present(args, ("end_line", "end", "to_line", "line_end", "last_line"), meta)
            max_chars = cls._first_present(args, ("max_chars", "chars", "limit", "max"), meta)
            out: Dict[str, Any] = {"max_chars": cls._to_int(max_chars, 50_000, 1, 200_000)}
            if path is not None:
                out["path"] = str(path)
            if start_line is not None:
                out["start_line"] = cls._to_int(start_line, 1, 1, 10_000_000)
            if end_line is not None:
                out["end_line"] = cls._to_int(end_line, 1, 1, 10_000_000)
            return out
        if tool_name == "file_write":
            path = cls._first_present(args, ("path", "file", "filename", "filepath"), meta)
            content = cls._first_present(args, ("content", "text", "data", "body", "replacement", "replacement_content"), meta)
            start_line = cls._first_present(args, ("start_line", "start", "from_line", "line_start", "first_line"), meta)
            end_line = cls._first_present(args, ("end_line", "end", "to_line", "line_end", "last_line"), meta)
            append = cls._first_present(args, ("append", "append_mode", "mode_append"), meta)
            create_dirs = cls._first_present(args, ("create_dirs", "mkdirs", "parents", "create_parent_dirs"), meta)
            out = {
                "append": cls._to_bool(append, default=False),
                "create_dirs": cls._to_bool(create_dirs, default=False),
            }
            if path is not None:
                out["path"] = str(path)
            if content is not None:
                out["content"] = str(content)
            if start_line is not None:
                out["start_line"] = cls._to_int(start_line, 1, 1, 10_000_000)
            if end_line is not None:
                out["end_line"] = cls._to_int(end_line, 1, 1, 10_000_000)
            return out
        if tool_name == "memory_save":
            text = cls._first_present(args, ("text", "fact", "memory", "content", "summary"), meta)
            tags = cls._first_present(args, ("tags", "tag", "labels", "label"), meta)
            importance = cls._first_present(args, ("importance", "priority", "score"), meta)
            out = {"importance": cls._to_int(importance, 3, 1, 5)}
            if text is not None:
                out["text"] = str(text)
            if tags is not None:
                if isinstance(tags, list):
                    out["tags"] = [str(t) for t in tags]
                elif isinstance(tags, str):
                    out["tags"] = [p.strip() for p in re.split(r"[,;]", tags) if p.strip()]
                else:
                    out["tags"] = [str(tags)]
            return out
        if tool_name == "memory_search":
            query = cls._first_present(args, ("query", "text", "q", "search"), meta)
            max_items = cls._first_present(args, ("max_items", "limit", "n", "count"), meta)
            return {"query": str(query or ""), "max_items": cls._to_int(max_items, 5, 1, 20)}
        if tool_name == "memory_list":
            max_items = cls._first_present(args, ("max_items", "limit", "n", "count"), meta)
            return {"max_items": cls._to_int(max_items, 10, 1, 50)}
        if tool_name == "memory_forget":
            memory_id = cls._first_present(args, ("memory_id", "id", "item_id"), meta)
            return {"memory_id": str(memory_id or "")}
        if tool_name == "finish_task":
            summary = cls._first_present(args, ("summary", "message", "text", "result"), meta)
            return {"summary": str(summary or "Task finished.")}
        return dict(args)

    @staticmethod
    def _first_present(args: Dict[str, Any], names: Iterable[str], meta: Dict[str, Any]) -> Any:
        names = tuple(names)
        canonical = names[0]
        for name in names:
            if name in args:
                if name != canonical:
                    meta.setdefault("consumed_aliases", {})[name] = canonical
                    meta.setdefault("parse_notes", []).append(f"used alias {name}->{canonical}")
                return args[name]
        return None

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in {"1", "true", "yes", "y", "on", "enter", "return"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _to_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            out = int(value)
        except Exception:
            out = default
        return max(minimum, min(out, maximum))

    @staticmethod
    def _to_float(value: Any, default: float, minimum: float, maximum: float) -> float:
        try:
            out = float(value)
        except Exception:
            out = default
        return max(minimum, min(out, maximum))


@dataclass
class AgentConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_steps: int
    tool_choice: str
    empty_retries: int
    terminal_context_chars: int
    terminal_tool_delay_ms: int
    memory_enabled: bool
    memory_max_items: int
    memory_context_items: int
    max_request_chars: int
    compact_keep_recent_messages: int
    compact_tool_result_chars: int
    reasoning_effort: str


class TerminalAgent:
    def __init__(self, cfg: AgentConfig, terminal: PtyTerminal, hub: WebHub, convo_log: JsonlLogger, memory: Optional[SessionMemory] = None) -> None:
        self.cfg = cfg
        self.terminal = terminal
        self.hub = hub
        self.convo_log = convo_log
        self.memory = memory
        self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.lock = asyncio.Lock()
        self.running = False
        self.finished = False
        self.generation = 0

    async def emit_agent(self, level: str, data: str, payload: Optional[Any] = None) -> None:
        message: Dict[str, Any] = {"type": "agent", "level": level, "data": data}
        if payload is not None:
            message["payload"] = to_jsonable(payload)
        await self.hub.broadcast(message)

    async def emit_llm(self, event: str, data: Any) -> None:
        self.convo_log.write(event, data)
        await self.hub.emit_llm(event, data)

    async def reset_conversation(self) -> None:
        self.generation += 1
        self.finished = False
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        memory_count = self.memory.list_recent(1).get("total_memories", 0) if self.memory else 0
        payload = {"generation": self.generation, "message": "The LLM context has been cleared. The terminal and session memory remain unchanged.", "session_memory_items": memory_count}
        self.convo_log.write("conversation_cleared", payload)
        await self.hub.clear_llm_history()
        await self.hub.emit_llm("conversation_cleared", payload)

    def _terminal_context_block(self, max_chars: Optional[int] = None) -> str:
        limit = self.cfg.terminal_context_chars if max_chars is None else max(100, int(max_chars))
        text = self.terminal.read(limit)
        if not text:
            return "[terminal_snapshot]\nThe terminal does not have visible output yet.\n[/terminal_snapshot]"
        return f"[terminal_snapshot: last {min(len(text), limit)} characters of the visible terminal]\n{text}\n[/terminal_snapshot]"

    def _terminal_context_message(self, emergency: bool = False) -> Dict[str, str]:
        # In emergency mode keep only a small terminal tail; large terminal snapshots are
        # a common cause of context-length failures after verbose commands.
        max_chars = min(self.cfg.terminal_context_chars, 1500) if emergency else self.cfg.terminal_context_chars
        return {
            "role": "user",
            "content": (
                "[auto_terminal_snapshot]\n"
                "This is not a new command from the human; it is the current shared terminal context. "
                "You can also see manual keystrokes/commands entered by the human and their output. "
                "If the state may be incomplete or the task concerns the terminal, use terminal_read. If you need to locate files or references, use file_search. If you need file contents, use file_read with an optional line range.\n"
                f"{self._terminal_context_block(max_chars=max_chars)}\n"
                "[/auto_terminal_snapshot]"
            ),
        }

    def _memory_query_text(self) -> str:
        parts: List[str] = []
        for msg in reversed(self.messages[-10:]):
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant", "tool"} and isinstance(content, str) and content.strip():
                parts.append(content.strip())
            if sum(len(p) for p in parts) > 3000:
                break
        terminal_tail = self.terminal.read(min(1000, self.cfg.terminal_context_chars))
        if terminal_tail:
            parts.append(terminal_tail)
        return "\n".join(reversed(parts))[-4000:]

    def _memory_context_message(self) -> Optional[Dict[str, str]]:
        if not self.cfg.memory_enabled or not self.memory:
            return None
        block = self.memory.context_block(self._memory_query_text(), max_items=self.cfg.memory_context_items)
        if not block:
            return None
        return {
            "role": "user",
            "content": (
                "[auto_session_memory]\n"
                "This is not a new command from the human; it is retrieved session memory. "
                "Use it when relevant, but verify with the terminal when correctness matters.\n"
                f"{block}\n"
                "[/auto_session_memory]"
            ),
        }

    def _build_request_messages(self, emergency: bool = False) -> List[Dict[str, Any]]:
        out = self._compacted_history_for_request(emergency=emergency)
        memory_message = self._memory_context_message()
        if memory_message:
            out.append(memory_message)
        out.append(self._terminal_context_message(emergency=emergency))
        return self._fit_request_messages(out, emergency=emergency)

    @staticmethod
    def _message_chars(message: Dict[str, Any]) -> int:
        try:
            return len(json.dumps(message, ensure_ascii=False, default=str))
        except Exception:
            return len(str(message))

    @classmethod
    def _messages_chars(cls, messages: List[Dict[str, Any]]) -> int:
        return sum(cls._message_chars(m) for m in messages)

    @staticmethod
    def _trim_text_middle(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        max_chars = max(200, int(max_chars))
        head = max_chars // 3
        tail = max_chars - head - 80
        omitted = len(text) - head - max(0, tail)
        return text[:head] + f"\n...[truncated {omitted} chars for request budget]...\n" + text[-max(0, tail):]

    def _trim_large_strings(self, value: Any, max_chars: int) -> Tuple[Any, int]:
        if isinstance(value, str):
            if len(value) > max_chars:
                return self._trim_text_middle(value, max_chars), 1
            return value, 0
        if isinstance(value, list):
            changed = 0
            out = []
            for item in value:
                trimmed, n = self._trim_large_strings(item, max_chars)
                changed += n
                out.append(trimmed)
            return out, changed
        if isinstance(value, dict):
            changed = 0
            out: Dict[str, Any] = {}
            for key, item in value.items():
                # Tool outputs and terminal snapshots are useful mostly near the tail,
                # but keep a small head too for command/error context.
                per_key_limit = max_chars
                if str(key) in {"text", "terminal_output_after_delay", "terminal_snapshot_after_delay", "stdout", "stderr"}:
                    per_key_limit = max(800, max_chars)
                trimmed, n = self._trim_large_strings(item, per_key_limit)
                changed += n
                out[key] = trimmed
            if changed:
                out.setdefault("_request_truncated", True)
                out.setdefault("_request_truncated_note", "Large strings were shortened only in the copy sent to the model; full logs remain on disk/UI.")
            return out, changed
        return value, 0

    def _copy_message_for_request(self, message: Dict[str, Any], emergency: bool = False) -> Dict[str, Any]:
        out = dict(message)
        content = out.get("content")
        if not isinstance(content, str):
            return out
        role = str(out.get("role") or "")
        tool_limit = min(self.cfg.compact_tool_result_chars, 1000) if emergency else self.cfg.compact_tool_result_chars
        generic_limit = 2500 if emergency else max(4000, min(12000, self.cfg.max_request_chars // 6))
        if role == "tool":
            try:
                parsed = json.loads(content)
                trimmed, changed = self._trim_large_strings(parsed, max(500, tool_limit))
                if changed:
                    out["content"] = json.dumps(trimmed, ensure_ascii=False)
                elif len(content) > tool_limit * 2:
                    out["content"] = json.dumps({
                        "ok": True,
                        "_request_truncated": True,
                        "text_tail": self._trim_text_middle(content, tool_limit),
                    }, ensure_ascii=False)
            except Exception:
                if len(content) > tool_limit:
                    out["content"] = self._trim_text_middle(content, tool_limit)
            return out
        if role != "system" and len(content) > generic_limit:
            out["content"] = self._trim_text_middle(content, generic_limit)
        return out

    def _safe_recent_suffix(self, history: List[Dict[str, Any]], keep_count: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if keep_count <= 0 or len(history) <= keep_count:
            return [], history
        start = max(0, len(history) - keep_count)
        # Do not start a retained suffix with orphaned tool results.
        while start < len(history) and history[start].get("role") == "tool":
            start += 1
        return history[:start], history[start:]

    def _compaction_summary_message(self, omitted: List[Dict[str, Any]], emergency: bool = False) -> Optional[Dict[str, str]]:
        if not omitted:
            return None
        role_counts: Dict[str, int] = {}
        last_user: List[str] = []
        tools: List[str] = []
        for msg in omitted:
            role = str(msg.get("role") or "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            content = msg.get("content")
            if role == "user" and isinstance(content, str) and content.strip():
                last_user.append(content.strip())
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    if name:
                        tools.append(str(name))
        recent_user_lines = []
        for text in last_user[-3:]:
            recent_user_lines.append("- " + self._trim_text_middle(text, 500).replace("\n", " "))
        tool_lines = ", ".join(tools[-12:]) if tools else "none recorded"
        max_summary = 1800 if emergency else 3500
        content = (
            "[conversation_compacted]\n"
            "Older conversation messages were omitted to stay within the model context window. "
            "Do not assume omitted terminal output or file contents are still visible; use terminal_read, file_search, file_read, or memory_search if needed.\n"
            f"Omitted messages: {len(omitted)}; role counts: {role_counts}.\n"
            f"Recent omitted user requests:\n" + ("\n".join(recent_user_lines) if recent_user_lines else "- none") + "\n"
            f"Recent omitted tool calls: {tool_lines}.\n"
            "[/conversation_compacted]"
        )
        return {"role": "user", "content": self._trim_text_middle(content, max_summary)}

    def _compacted_history_for_request(self, emergency: bool = False) -> List[Dict[str, Any]]:
        if not self.messages:
            return []
        system = self._copy_message_for_request(self.messages[0], emergency=emergency)
        history = self.messages[1:]
        keep = 8 if emergency else max(4, self.cfg.compact_keep_recent_messages)
        base_chars = self._messages_chars(self.messages)
        should_compact = emergency or len(history) > keep or base_chars > int(self.cfg.max_request_chars * 0.75)
        if not should_compact:
            return [self._copy_message_for_request(m, emergency=emergency) for m in self.messages]
        omitted, recent = self._safe_recent_suffix(history, keep)
        out: List[Dict[str, Any]] = [system]
        summary = self._compaction_summary_message(omitted, emergency=emergency)
        if summary:
            out.append(summary)
        out.extend(self._copy_message_for_request(m, emergency=emergency) for m in recent)
        return out

    def _fit_request_messages(self, messages: List[Dict[str, Any]], emergency: bool = False) -> List[Dict[str, Any]]:
        budget = max(8000, self.cfg.max_request_chars // 2) if emergency else max(8000, self.cfg.max_request_chars)
        if self._messages_chars(messages) <= budget:
            return messages
        # Second-pass shrinking: this is still only the outgoing request copy.
        out: List[Dict[str, Any]] = []
        per_message = 1200 if emergency else 2400
        for msg in messages:
            m = dict(msg)
            content = m.get("content")
            if isinstance(content, str) and m.get("role") != "system" and len(content) > per_message:
                m["content"] = self._trim_text_middle(content, per_message)
            out.append(m)
        if self._messages_chars(out) <= budget:
            return out
        # Last resort: keep system, compaction summary, and the newest non-auto messages.
        system = out[0:1]
        autos: List[Dict[str, Any]] = []
        non_auto: List[Dict[str, Any]] = []
        for m in out[1:]:
            content = m.get("content")
            if isinstance(content, str) and content.startswith("[auto_"):
                autos.append(m)
            else:
                non_auto.append(m)
        recent = non_auto[-6:]
        while recent and recent[0].get("role") == "tool":
            recent = recent[1:]
        kept: List[Dict[str, Any]] = system + recent + autos[-2:]
        while len(kept) > 2 and self._messages_chars(kept) > budget:
            del kept[1]
            while len(kept) > 1 and kept[1].get("role") == "tool":
                del kept[1]
        return kept

    @staticmethod
    def _is_context_length_error(exc: BaseException) -> bool:
        text = repr(exc).lower()
        return any(marker in text for marker in (
            "max length reached",
            "maximum context",
            "context length",
            "maximum length",
            "too many tokens",
        ))

    async def _emit_request_log(self, step: int, source: str, request_messages: List[Dict[str, Any]], compacted_retry: bool = False, emergency: bool = False) -> None:
        await self.emit_llm("request", {
            "step": step,
            "source": source,
            "model": self.cfg.model,
            "stream": True,
            "tool_choice": self.cfg.tool_choice,
            "reasoning_effort": self.cfg.reasoning_effort,
            "extra_body": self._extra_body(),
            "request_chars_estimate": self._messages_chars(request_messages),
            "request_message_count": len(request_messages),
            "context_budget_chars": self.cfg.max_request_chars,
            "compacted_retry": compacted_retry,
            "emergency_compaction": emergency,
            "messages_tail": request_messages[-20:],
        })

    def _extra_body(self) -> Dict[str, Any]:
        enable_thinking = self.cfg.reasoning_effort != "none"
        return {
            "reasoning_effort": self.cfg.reasoning_effort,
            "think": enable_thinking,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }

    async def add_user_message(self, content: str, source: str = "user", step_limit: Optional[int] = None) -> None:
        display_content = content.strip()
        if not display_content:
            return
        snapshot = self._terminal_context_block()
        self.messages.append({"role": "user", "content": display_content})
        await self.emit_llm("user_message", {"source": source, "content": display_content, "terminal_snapshot_at_receive": snapshot})
        asyncio.create_task(self.run_loop(max_steps=step_limit, source=source))

    async def step_once(self) -> None:
        await self.run_loop(max_steps=1, source="manual_step")

    async def run_loop(self, max_steps: Optional[int] = None, source: str = "auto") -> None:
        if self.lock.locked():
            await self.emit_agent("info", "The agent is already running; the message was added to the conversation.")
            return
        async with self.lock:
            run_generation = self.generation
            self.running = True
            self.finished = False
            steps_limit = max_steps or self.cfg.max_steps
            empty_retries_left = max(0, self.cfg.empty_retries)
            thinking_only_retries_left = max(1, self.cfg.empty_retries)
            await self.hub.broadcast({"type": "status", "agent_running": True})
            try:
                for step in range(1, steps_limit + 1):
                    if run_generation != self.generation:
                        await self.emit_agent("info", "The previous agent run was interrupted because the conversation context was cleared.")
                        break
                    logging.info("LLM step %s/%s source=%s", step, steps_limit, source)
                    request_messages = self._build_request_messages()
                    await self._emit_request_log(step, source, request_messages)

                    try:
                        assistant_dict, stream_payload = await self._stream_chat_completion(request_messages, step, source)
                    except Exception as exc:
                        if not self._is_context_length_error(exc):
                            raise
                        await self.emit_llm("context_compaction_retry", {
                            "step": step,
                            "source": source,
                            "error": repr(exc),
                            "action": "Retrying once with emergency context compaction.",
                        })
                        request_messages = self._build_request_messages(emergency=True)
                        await self._emit_request_log(step, source, request_messages, compacted_retry=True, emergency=True)
                        assistant_dict, stream_payload = await self._stream_chat_completion(request_messages, step, source)
                    if run_generation != self.generation:
                        await self.emit_agent("info", "Discarding a response from the old context after the conversation was cleared.")
                        break

                    native_tool_calls = assistant_dict.get("tool_calls") or []
                    content = str(assistant_dict.get("content") or "")
                    thinking = str(stream_payload.get("thinking") or "")
                    tool_calls = native_tool_calls

                    if not tool_calls:
                        recovered_tool_calls, recovery_meta = self._extract_tool_calls_from_content(content)
                        if recovered_tool_calls:
                            tool_calls = recovered_tool_calls
                            assistant_dict["tool_calls"] = recovered_tool_calls
                            stream_payload["recovered_tool_calls"] = recovered_tool_calls
                            stream_payload["recovered_tool_calls_count"] = len(recovered_tool_calls)
                            stream_payload["tool_calls_count"] = len(recovered_tool_calls)
                            stream_payload["native_tool_calls_count"] = 0
                            stream_payload["tool_call_recovery"] = recovery_meta
                            warning_payload = {
                                "step": step,
                                "warning": "Model returned fake tool_calls in content, recovered via fallback parser.",
                                "native_tool_calls_count": 0,
                                "recovered_tool_calls_count": len(recovered_tool_calls),
                                "recovered_tool_calls": recovered_tool_calls,
                                "recovery_meta": recovery_meta,
                            }
                            await self.emit_llm("fake_tool_calls_recovered", warning_payload)

                    history_msg = {k: v for k, v in assistant_dict.items() if k != "thinking"}
                    if not tool_calls and history_msg.get("content") is None:
                        history_msg["content"] = ""
                    self.messages.append(history_msg)
                    await self.emit_llm("assistant", stream_payload)

                    if not tool_calls:
                        no_visible_content = not content.strip()
                        has_private_thinking = bool(thinking.strip())

                        if no_visible_content and has_private_thinking and thinking_only_retries_left > 0:
                            thinking_only_retries_left -= 1
                            nudge = (
                                "Your previous response contained private thinking/reasoning but no visible answer and no tool_calls. "
                                "Do not stop after analysis. Inspect the latest terminal/tool state and convert your analysis into the next native tool call. "
                                "Call terminal_read if the terminal state may be incomplete, file_search if you need to locate files/references, file_read if file contents are needed, file_write with start_line/end_line for precise code edits, memory_search if saved context is needed, sleep if a command may still be running, or finish_task if the task is complete."
                            )
                            self.messages.append({"role": "user", "content": nudge})
                            await self.emit_llm("thinking_only_retry", {
                                "step": step,
                                "remaining": thinking_only_retries_left,
                                "thinking_chars": len(thinking),
                                "nudge": nudge,
                            })
                            continue

                        if no_visible_content and not has_private_thinking and empty_retries_left > 0:
                            empty_retries_left -= 1
                            nudge = (
                                "Your previous response was empty and did not contain tool_calls. "
                                "You must use native tool calling. If you do not know what to do, call terminal_read; if you need to locate files/references, call file_search; if you need file contents, call file_read; if you are editing code, use file_write with start_line/end_line for precise replacements when possible. "
                                "If the task is finished, call finish_task."
                            )
                            self.messages.append({"role": "user", "content": nudge})
                            await self.emit_llm("empty_retry", {"step": step, "remaining": empty_retries_left, "nudge": nudge})
                            continue
                        logging.info("LLM produced no tool calls; stopping loop")
                        break

                    for tc in tool_calls:
                        await self._execute_tool_call(tc, step)
                        if self.finished or run_generation != self.generation:
                            break
                    if self.finished or run_generation != self.generation:
                        break
                else:
                    await self.emit_llm("max_steps", {"max_steps": steps_limit})
            except Exception as exc:
                logging.exception("Agent loop crashed")
                err = {"error": repr(exc), "traceback": traceback.format_exc()}
                self.convo_log.write("error", err)
                await self.hub.emit_llm("error", err)
            finally:
                self.running = False
                await self.hub.broadcast({"type": "status", "agent_running": False})

    async def _stream_chat_completion(self, request_messages: List[Dict[str, Any]], step: int, source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        stream_id = f"s{int(time.time() * 1000)}_{step}"
        start_payload = {"stream_id": stream_id, "step": step, "source": source, "model": self.cfg.model, "reasoning_effort": self.cfg.reasoning_effort}
        self.convo_log.write("stream_start", start_payload)
        await self.hub.broadcast({"type": "llm_stream_start", "data": start_payload})

        content_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_acc: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        chunks_seen = 0
        usage: Any = None

        try:
            stream = await self._open_stream(request_messages)
        except Exception as exc:
            error_payload = {
                "stream_id": stream_id,
                "step": step,
                "finish_reason": "error",
                "chunks_seen": chunks_seen,
                "content": "",
                "thinking": "",
                "tool_calls": [],
                "tool_calls_count": 0,
                "native_tool_calls_count": 0,
                "usage": usage,
                "error": repr(exc),
            }
            self.convo_log.write("stream_error", error_payload)
            await self.hub.broadcast({"type": "llm_stream_delta", "stream_id": stream_id, "kind": "content", "text": f"[stream error before first chunk] {exc!r}"})
            await self.hub.broadcast({"type": "llm_stream_done", "stream_id": stream_id, "data": error_payload})
            raise
        async for chunk in stream:
            chunks_seen += 1
            chunk_json = to_jsonable(chunk)
            self.convo_log.write("stream_chunk", {"stream_id": stream_id, "step": step, "chunk": chunk_json})
            if chunk_json.get("usage"):
                usage = chunk_json.get("usage")
            for choice in chunk_json.get("choices", []) or []:
                if choice.get("finish_reason"):
                    finish_reason = choice.get("finish_reason")
                delta = choice.get("delta") or {}
                content_delta = self._delta_text(delta.get("content"))
                thinking_delta = self._extract_thinking_delta(delta)
                if thinking_delta:
                    thinking_parts.append(thinking_delta)
                    await self.hub.broadcast({"type": "llm_stream_delta", "stream_id": stream_id, "kind": "thinking", "text": thinking_delta})
                if content_delta:
                    content_parts.append(content_delta)
                    await self.hub.broadcast({"type": "llm_stream_delta", "stream_id": stream_id, "kind": "content", "text": content_delta})
                for tcd in delta.get("tool_calls") or []:
                    self._accumulate_tool_delta(tool_acc, tcd)
                    # Do not render raw assistant tool-call deltas in the panel.
                    # The following tool execution stream renders the same call once, together with its result.

        content = "".join(content_parts)
        thinking = "".join(thinking_parts)
        tool_calls = self._finalize_tool_calls(tool_acc)
        assistant_dict: Dict[str, Any] = {"role": "assistant", "content": content if content else None}
        if tool_calls:
            assistant_dict["tool_calls"] = tool_calls
        done_payload = {
            "stream_id": stream_id,
            "step": step,
            "finish_reason": finish_reason,
            "chunks_seen": chunks_seen,
            "content": content,
            "thinking": thinking,
            "tool_calls": tool_calls,
            "tool_calls_count": len(tool_calls),
            "native_tool_calls_count": len(tool_calls),
            "usage": usage,
        }
        self.convo_log.write("stream_done", done_payload)
        await self.hub.broadcast({"type": "llm_stream_done", "stream_id": stream_id, "data": done_payload})
        return assistant_dict, done_payload

    async def _open_stream(self, request_messages: List[Dict[str, Any]]) -> Any:
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": request_messages,
            "tools": TOOLS,
            "temperature": self.cfg.temperature,
            "stream": True,
            "extra_body": self._extra_body(),
        }
        if self.cfg.tool_choice != "omit":
            kwargs["tool_choice"] = self.cfg.tool_choice
        try:
            return await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._is_context_length_error(exc):
                raise
            if kwargs.get("tool_choice") == "required":
                await self.emit_llm("tool_choice_retry", {"reason": "backend rejected tool_choice=required", "error": repr(exc)})
                kwargs["tool_choice"] = "auto"
                return await self.client.chat.completions.create(**kwargs)
            raise

    @staticmethod
    def _delta_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _extract_thinking_delta(cls, delta: Dict[str, Any]) -> str:
        for key in ("reasoning_content", "reasoning", "thinking", "thought", "analysis"):
            if key in delta and delta[key] is not None:
                return cls._delta_text(delta[key])
        for key in ("reasoning_details", "thinking_details"):
            if key in delta and delta[key]:
                return cls._delta_text(delta[key])
        return ""

    @staticmethod
    def _accumulate_tool_delta(acc: Dict[int, Dict[str, Any]], tcd: Dict[str, Any]) -> None:
        idx = int(tcd.get("index") or 0)
        item = acc.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}, "index": idx})
        if tcd.get("id"):
            item["id"] = tcd["id"]
        if tcd.get("type"):
            item["type"] = tcd["type"]
        fn = tcd.get("function") or {}
        if fn.get("name"):
            item["function"]["name"] += str(fn["name"])
        if fn.get("arguments"):
            item["function"]["arguments"] += str(fn["arguments"])

    @staticmethod
    def _tool_delta_preview(tcd: Dict[str, Any]) -> str:
        fn = tcd.get("function") or {}
        if fn.get("name"):
            return str(fn["name"])
        if fn.get("arguments"):
            return str(fn["arguments"])
        return ""

    @staticmethod
    def _finalize_tool_calls(acc: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx in sorted(acc):
            item = acc[idx]
            if not item.get("id"):
                item["id"] = f"call_{int(time.time())}_{idx}"
            item.pop("index", None)
            fn = item.get("function") or {}
            if fn.get("name"):
                out.append(item)
        return out

    @classmethod
    def _extract_tool_calls_from_content(cls, content: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        meta: Dict[str, Any] = {"source": "assistant.content", "notes": [], "content_preview": (content or "")[:1000]}
        text = (content or "").strip()
        if not text:
            meta["notes"].append("empty content")
            return [], meta
        candidates: List[Tuple[Any, str]] = []
        seen: Set[str] = set()
        def add_candidate(value: Any, note: str) -> None:
            key = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, default=str) if not isinstance(value, str) else value
            if key not in seen:
                seen.add(key)
                candidates.append((value, note))
        parse_meta: Dict[str, Any] = {"parse_notes": [], "ignored_args": {}}
        parsed = ToolArgParser._parse_any(text, parse_meta)
        if parsed:
            add_candidate(parsed, "parsed whole content")
        meta["notes"].extend(parse_meta.get("parse_notes", []))
        for block in re.findall(r"```(?:json|javascript|js|python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
            pm: Dict[str, Any] = {"parse_notes": [], "ignored_args": {}}
            val = ToolArgParser._parse_any(block, pm)
            if val:
                add_candidate(val, "parsed fenced block")
                meta["notes"].extend(pm.get("parse_notes", []))
        for obj_text in cls._extract_json_objects_from_text(text, limit=25):
            pm = {"parse_notes": [], "ignored_args": {}}
            val = ToolArgParser._parse_any(obj_text, pm)
            if val:
                add_candidate(val, "parsed embedded object")
                meta["notes"].extend(pm.get("parse_notes", []))

        recovered: List[Dict[str, Any]] = []
        for candidate, note in candidates:
            calls = cls._candidate_to_tool_call_items(candidate)
            if not calls:
                continue
            for call in calls:
                normalized = cls._normalize_recovered_tool_call(call, len(recovered))
                if normalized:
                    recovered.append(normalized)
                    meta["notes"].append(f"{note}: recovered {normalized['function']['name']}")
        unique: List[Dict[str, Any]] = []
        signatures: Set[str] = set()
        for call in recovered:
            fn = call.get("function") or {}
            sig = json.dumps({"name": fn.get("name"), "arguments": fn.get("arguments")}, ensure_ascii=False, sort_keys=True, default=str)
            if sig not in signatures:
                signatures.add(sig)
                unique.append(call)
        if not unique:
            meta["notes"].append("no recoverable fake tool_calls found")
        meta["recovered_count"] = len(unique)
        return unique, meta

    @staticmethod
    def _extract_json_objects_from_text(text: str, limit: int = 20) -> List[str]:
        objects: List[str] = []
        for start, ch0 in enumerate(text):
            if ch0 not in "[{":
                continue
            stack: List[str] = []
            quote: Optional[str] = None
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if quote:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == quote:
                        quote = None
                    continue
                if ch in {'"', "'"}:
                    quote = ch
                    continue
                if ch in "[{":
                    stack.append("}" if ch == "{" else "]")
                    continue
                if ch in "}]":
                    if not stack or ch != stack[-1]:
                        break
                    stack.pop()
                    if not stack:
                        objects.append(text[start:i+1])
                        break
            if len(objects) >= limit:
                break
        return sorted(set(objects), key=len, reverse=True)[:limit]

    @staticmethod
    def _candidate_to_tool_call_items(candidate: Any) -> List[Any]:
        if isinstance(candidate, dict):
            for key in ("tool_calls", "toolCalls", "tools", "function_calls", "functionCalls"):
                value = candidate.get(key)
                if isinstance(value, list):
                    return value
                if isinstance(value, dict):
                    return [value]
            if any(k in candidate for k in ("function", "name", "tool", "tool_name", "function_name")):
                return [candidate]
        if isinstance(candidate, list):
            return candidate
        return []

    @staticmethod
    def _normalize_recovered_tool_call(item: Any, idx: int) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = fn.get("name") or item.get("name") or item.get("tool") or item.get("tool_name") or item.get("function_name")
        if not name:
            return None
        name = str(name).strip()
        if name not in TOOL_ALLOWED_ARGS:
            return None
        args = fn.get("arguments") if "arguments" in fn else item.get("arguments", item.get("args", item.get("parameters", item.get("input", {}))))
        if args is None:
            args = {}
        raw_arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return {"id": str(item.get("id") or f"recovered_call_{int(time.time())}_{idx}"), "type": "function", "function": {"name": name, "arguments": raw_arguments}}

    async def _execute_tool_call(self, tc: Dict[str, Any], step: int) -> None:
        fn = tc.get("function") or {}
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments") or "{}"
        args, parse_meta = ToolArgParser.parse_and_normalize(name, raw_args)
        tool_stream_id = await self._start_tool_execution_stream(step, tc.get("id"), name)
        capture_terminal_after = self._should_auto_capture_terminal_after_tool(name, args)
        terminal_cursor_before = self.terminal.cursor() if capture_terminal_after else None
        tool_call_payload = {
            "step": step,
            "id": tc.get("id"),
            "name": name,
            "raw_arguments": raw_args,
            "parsed_arguments": args,
            "parse_meta": parse_meta,
            # The live GUI already renders this through the tool execution stream.
            # Keep the structured event in JSONL/history for debugging, but hide the duplicate card in the panel.
            "ui_hidden": True,
        }
        await self._emit_stream_delta(tool_stream_id, "tool", self._format_tool_call_for_stream(tool_call_payload))
        await self.emit_llm("tool_call", tool_call_payload)
        result: Dict[str, Any] = {}
        try:
            result = await self._call_tool(name, args)
            if capture_terminal_after:
                delay_ms = max(0, min(int(self.cfg.terminal_tool_delay_ms), 10_000))
                if delay_ms:
                    await self._emit_stream_delta(tool_stream_id, "tool", f"\nwaiting {delay_ms} ms for terminal output...\n")
                await self._attach_delayed_terminal_output(result, name, terminal_cursor_before)
                terminal_after = result.get("terminal_output_after_delay") or {}
                new_output = str(terminal_after.get("new_output") or "")
                if new_output:
                    await self._emit_stream_delta(tool_stream_id, "result", "\n[terminal output after delay]\n")
                    await self._emit_stream_text(tool_stream_id, "result", new_output)
            if parse_meta.get("ignored_args"):
                result["ignored_args"] = parse_meta["ignored_args"]
            if parse_meta.get("parse_notes"):
                result["arg_parse_notes"] = parse_meta["parse_notes"]
            result_payload = {
                "step": step,
                "tool_call_id": tc.get("id"),
                "name": name,
                "args": args,
                "result": result,
                # The live GUI already renders this through the tool execution stream.
                # Keep the structured event in JSONL/history for debugging, but hide the duplicate card in the panel.
                "ui_hidden": True,
            }
            await self._emit_stream_delta(tool_stream_id, "result", "\n[tool result JSON]\n")
            await self._emit_stream_json(tool_stream_id, "result", result_payload)
            await self.emit_llm("tool_result", result_payload)
            self.messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result, ensure_ascii=False)})
            await self._finish_tool_execution_stream(tool_stream_id, step, name, ok=bool(result.get("ok", True)), result=result)
        except Exception as exc:
            logging.exception("Tool execution stream failed for %s", name)
            result = {"ok": False, "error": repr(exc)}
            result_payload = {
                "step": step,
                "tool_call_id": tc.get("id"),
                "name": name,
                "args": args,
                "result": result,
                # The live GUI already renders this through the tool execution stream.
                # Keep the structured event in JSONL/history for debugging, but hide the duplicate card in the panel.
                "ui_hidden": True,
            }
            await self._emit_stream_delta(tool_stream_id, "result", "\n[tool execution error]\n")
            await self._emit_stream_json(tool_stream_id, "result", result_payload)
            await self.emit_llm("tool_result", result_payload)
            self.messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result, ensure_ascii=False)})
            await self._finish_tool_execution_stream(tool_stream_id, step, name, ok=False, result=result)
        if name == "finish_task":
            self.finished = True

    async def _start_tool_execution_stream(self, step: int, tool_call_id: Any, name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name or "tool")[:40]
        stream_id = f"tool_{int(time.time() * 1000)}_{step}_{safe_name}"
        payload = {
            "stream_id": stream_id,
            "stream_type": "tool_execution",
            "step": step,
            "tool_call_id": tool_call_id,
            "tool_name": name,
        }
        self.convo_log.write("tool_stream_start", payload)
        await self.hub.broadcast({"type": "llm_stream_start", "data": payload})
        return stream_id

    async def _finish_tool_execution_stream(self, stream_id: str, step: int, name: str, ok: bool, result: Dict[str, Any]) -> None:
        payload = {
            "stream_id": stream_id,
            "stream_type": "tool_execution",
            "step": step,
            "tool_name": name,
            "finish_reason": "tool_result",
            "ok": ok,
            "result_preview": self._compact_json(result, max_chars=4000),
        }
        self.convo_log.write("tool_stream_done", payload)
        await self.hub.broadcast({"type": "llm_stream_done", "stream_id": stream_id, "data": payload})

    async def _emit_stream_delta(self, stream_id: str, kind: str, text: str) -> None:
        if text:
            await self.hub.broadcast({"type": "llm_stream_delta", "stream_id": stream_id, "kind": kind, "text": text})

    async def _emit_stream_text(self, stream_id: str, kind: str, text: str, chunk_size: int = 4000) -> None:
        if not text:
            return
        for start in range(0, len(text), chunk_size):
            await self._emit_stream_delta(stream_id, kind, text[start:start + chunk_size])

    async def _emit_stream_json(self, stream_id: str, kind: str, value: Any, chunk_size: int = 4000) -> None:
        await self._emit_stream_text(stream_id, kind, json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, default=str), chunk_size=chunk_size)

    @staticmethod
    def _compact_json(value: Any, max_chars: int = 1200) -> str:
        text = json.dumps(to_jsonable(value), ensure_ascii=False, default=str)
        if len(text) > max_chars:
            return text[:max_chars] + "…"
        return text

    @staticmethod
    def _format_tool_call_for_stream(payload: Dict[str, Any]) -> str:
        name = payload.get("name") or "unknown_tool"
        raw_args = payload.get("raw_arguments") or "{}"
        parsed = payload.get("parsed_arguments") or {}
        notes = (payload.get("parse_meta") or {}).get("parse_notes") or []
        out = [f"CALL {name}", "", "raw arguments:", str(raw_args), "", "parsed arguments:", json.dumps(to_jsonable(parsed), ensure_ascii=False, indent=2, default=str)]
        if notes:
            out.extend(["", "argument parser notes:", json.dumps(to_jsonable(notes), ensure_ascii=False, indent=2, default=str)])
        return "\n".join(out) + "\n"

    def _should_auto_capture_terminal_after_tool(self, name: str, args: Dict[str, Any]) -> bool:
        # These tools can change or advance the terminal state. terminal_read already returns
        # terminal text, and finish_task should not delay completion.
        if name in {"terminal_send_text", "terminal_send_keys", "terminal_resize", "sleep"}:
            return True
        return False

    async def _attach_delayed_terminal_output(self, result: Dict[str, Any], tool_name: str, terminal_cursor_before: Optional[int]) -> None:
        delay_ms = max(0, min(int(self.cfg.terminal_tool_delay_ms), 10_000))
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000.0)
        cursor = terminal_cursor_before if terminal_cursor_before is not None else self.terminal.cursor()
        new_output = self.terminal.read_since(cursor, self.cfg.terminal_context_chars)
        snapshot = self.terminal.read(self.cfg.terminal_context_chars)
        result["terminal_output_after_delay"] = {
            "ok": True,
            "delay_ms": delay_ms,
            "tool_name": tool_name,
            "new_output": new_output,
            "snapshot": snapshot,
        }

    def _resolve_tool_file_path(self, path_value: Any) -> Tuple[Optional[Path], Dict[str, Any]]:
        raw_path = str(path_value or "").strip()
        if not raw_path:
            return None, {"ok": False, "error": "missing required argument: path"}
        base_cwd = self.terminal.current_working_directory()
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base_cwd / path
        resolved = path.resolve(strict=False)
        return resolved, {
            "raw_path": raw_path,
            "base_cwd": str(base_cwd),
            "path": str(resolved),
        }

    @staticmethod
    def _number_text_lines_for_tool(text: str, start_line: int = 1) -> str:
        """Return text with stable line-number prefixes for model inspection.

        This is intentionally a separate field from raw text so models can use the
        numbers to choose edit ranges without accidentally writing the prefixes back
        into source files.
        """
        if not text:
            return ""
        lines = text.splitlines(keepends=True)
        if not lines:
            return ""
        last_line = start_line + len(lines) - 1
        width = max(4, len(str(last_line)))
        return "".join(f"{line_no:>{width}} | {line}" for line_no, line in enumerate(lines, start=start_line))

    @staticmethod
    def _tool_search_is_hidden_path(path: Path) -> bool:
        return any(part.startswith(".") for part in path.parts if part not in {".", ".."})

    @staticmethod
    def _tool_search_is_binary(data: bytes) -> bool:
        if not data:
            return False
        if b"\x00" in data[:4096]:
            return True
        return False

    @staticmethod
    def _tool_search_compile_matcher(query: str, regex: bool, case_sensitive: bool) -> Tuple[Optional[Any], Optional[str]]:
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(query, flags)
            except re.error as exc:
                return None, f"invalid regex: {exc}"

            def match_regex(haystack: str) -> Optional[str]:
                m = pattern.search(haystack)
                if not m:
                    return None
                try:
                    return m.group(0)
                except Exception:
                    return haystack[m.start():m.end()]

            return match_regex, None

        needle = query if case_sensitive else query.casefold()

        def match_literal(haystack: str) -> Optional[str]:
            target = haystack if case_sensitive else haystack.casefold()
            idx = target.find(needle)
            if idx < 0:
                return None
            return haystack[idx:idx + len(query)]

        return match_literal, None

    @staticmethod
    def _tool_search_short_line(line: str, max_chars: int = 2000) -> str:
        clean = line.rstrip("\r\n")
        if len(clean) <= max_chars:
            return clean
        head = max_chars // 2
        tail = max_chars - head - 32
        return clean[:head] + " ...[line truncated]... " + clean[-tail:]

    def _search_files_for_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        raw_query = args.get("query")
        query = str(raw_query or "")
        if not query:
            return {"ok": False, "error": "missing required argument: query", "received_args": args}

        path_value = args.get("path", ".")
        root, meta = self._resolve_tool_file_path(path_value)
        if root is None:
            return {"ok": False, **meta, "received_args": args}
        if not root.exists():
            return {"ok": False, **meta, "error": "search path does not exist"}
        if not (root.is_dir() or root.is_file()):
            return {"ok": False, **meta, "error": "search path is neither a directory nor a regular file"}

        regex = bool(args.get("regex", False))
        case_sensitive = bool(args.get("case_sensitive", False))
        search_filenames = bool(args.get("search_filenames", True))
        search_contents = bool(args.get("search_contents", True))
        include_hidden = bool(args.get("include_hidden", False))
        max_results = max(1, min(int(args.get("max_results", 100)), 1000))
        max_files = max(1, min(int(args.get("max_files", 2000)), 20_000))
        max_file_bytes = max(1, min(int(args.get("max_file_bytes", 1_000_000)), 10_000_000))

        if not search_filenames and not search_contents:
            return {"ok": False, **meta, "error": "at least one of search_filenames or search_contents must be true"}

        matcher, matcher_error = self._tool_search_compile_matcher(query, regex=regex, case_sensitive=case_sensitive)
        if matcher_error:
            return {"ok": False, **meta, "error": matcher_error, "query": query, "regex": regex}
        assert matcher is not None

        skip_dirs = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "env", "dist", "build", "target"}
        results: List[Dict[str, Any]] = []
        searched_files = 0
        path_matches = 0
        content_matches = 0
        skipped_hidden = 0
        skipped_dirs_count = 0
        skipped_large = 0
        skipped_binary = 0
        skipped_errors = 0
        truncated = False

        search_root = root if root.is_dir() else root.parent

        def rel_for(file_path: Path) -> str:
            try:
                return file_path.relative_to(search_root).as_posix()
            except Exception:
                return file_path.as_posix()

        def add_result(item: Dict[str, Any]) -> bool:
            nonlocal truncated
            if len(results) >= max_results:
                truncated = True
                return False
            results.append(item)
            return len(results) < max_results

        def iter_files() -> Iterable[Path]:
            nonlocal skipped_hidden, skipped_dirs_count
            if root.is_file():
                yield root
                return
            for dirpath, dirnames, filenames in os.walk(root):
                current = Path(dirpath)
                kept_dirs = []
                for dirname in dirnames:
                    child = current / dirname
                    if dirname in skip_dirs:
                        skipped_dirs_count += 1
                        continue
                    try:
                        rel_child = child.relative_to(search_root)
                    except Exception:
                        rel_child = Path(dirname)
                    if not include_hidden and self._tool_search_is_hidden_path(rel_child):
                        skipped_hidden += 1
                        continue
                    kept_dirs.append(dirname)
                dirnames[:] = kept_dirs
                for filename in filenames:
                    file_path = current / filename
                    try:
                        rel_path = file_path.relative_to(search_root)
                    except Exception:
                        rel_path = Path(filename)
                    if not include_hidden and self._tool_search_is_hidden_path(rel_path):
                        skipped_hidden += 1
                        continue
                    yield file_path

        for file_path in iter_files():
            if searched_files >= max_files or len(results) >= max_results:
                truncated = True
                break
            searched_files += 1
            rel_path = rel_for(file_path)

            if search_filenames:
                matched = matcher(rel_path)
                if matched is not None:
                    path_matches += 1
                    if not add_result({
                        "match_type": "path",
                        "path": str(file_path),
                        "relative_path": rel_path,
                        "name": file_path.name,
                        "matched_text": matched,
                        "line_number": None,
                        "line": None,
                    }):
                        break

            if not search_contents:
                continue
            try:
                stat = file_path.stat()
                if stat.st_size > max_file_bytes:
                    skipped_large += 1
                    continue
                data = file_path.read_bytes()
                if self._tool_search_is_binary(data):
                    skipped_binary += 1
                    continue
                text = data.decode("utf-8", errors="replace")
            except Exception:
                skipped_errors += 1
                continue

            for line_number, line in enumerate(text.splitlines(keepends=False), start=1):
                matched = matcher(line)
                if matched is None:
                    continue
                content_matches += 1
                if not add_result({
                    "match_type": "content",
                    "path": str(file_path),
                    "relative_path": rel_path,
                    "name": file_path.name,
                    "line_number": line_number,
                    "line": self._tool_search_short_line(line),
                    "matched_text": matched,
                }):
                    break

        return {
            "ok": True,
            **meta,
            "query": query,
            "regex": regex,
            "case_sensitive": case_sensitive,
            "search_filenames": search_filenames,
            "search_contents": search_contents,
            "include_hidden": include_hidden,
            "max_results": max_results,
            "max_files": max_files,
            "max_file_bytes": max_file_bytes,
            "searched_files": searched_files,
            "path_matches": path_matches,
            "content_matches": content_matches,
            "matches_returned": len(results),
            "truncated": truncated,
            "skipped_hidden": skipped_hidden,
            "skipped_dirs": skipped_dirs_count,
            "skipped_large": skipped_large,
            "skipped_binary": skipped_binary,
            "skipped_errors": skipped_errors,
            "results": results,
            "result_note": "For match_type='content', use relative_path plus line_number to choose a focused file_read range before file_write edits. For match_type='path', the match is in the file path/name.",
        }

    def _read_text_file_for_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path, meta = self._resolve_tool_file_path(args.get("path"))
        if path is None:
            return {"ok": False, **meta, "received_args": args}
        if not path.exists():
            return {"ok": False, **meta, "error": "file does not exist"}
        if not path.is_file():
            return {"ok": False, **meta, "error": "path is not a regular file"}
        max_chars = max(1, min(int(args.get("max_chars", 50_000)), 200_000))
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if start_line is not None:
            start_line = max(1, int(start_line))
        if end_line is not None:
            end_line = max(1, int(end_line))
        if start_line is not None and end_line is not None and start_line > end_line:
            return {"ok": False, **meta, "error": "start_line must be <= end_line"}

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        if start_line is not None or end_line is not None:
            first = start_line or 1
            last = end_line or total_lines
            start_index = max(0, first - 1)
            end_index = min(total_lines, last)
            selected = "".join(lines[start_index:end_index])
            returned_start_line = first
            returned_end_line = min(last, total_lines)
            if total_lines == 0 or start_index >= total_lines:
                returned_end_line = 0
        else:
            selected = text
            returned_start_line = 1 if total_lines else 0
            returned_end_line = total_lines

        truncated = len(selected) > max_chars
        if truncated:
            selected = selected[:max_chars]
        numbered_start = returned_start_line if returned_start_line > 0 else 1
        numbered_line_count = len(selected.splitlines(keepends=True)) if selected else 0
        numbered_end = numbered_start + numbered_line_count - 1 if numbered_line_count else 0
        numbered_text = self._number_text_lines_for_tool(selected, start_line=numbered_start)
        return {
            "ok": True,
            **meta,
            "text": selected,
            "numbered_text": numbered_text,
            "numbered_text_note": "Use numbered_text to choose start_line/end_line. Do not include the line-number prefixes in file_write content.",
            "numbered_start_line": numbered_start if numbered_line_count else 0,
            "numbered_end_line": numbered_end,
            "encoding": "utf-8",
            "total_lines": total_lines,
            "returned_start_line": returned_start_line,
            "returned_end_line": returned_end_line,
            "chars_returned": len(selected),
            "truncated": truncated,
            "max_chars": max_chars,
        }

    def _write_text_file_for_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path, meta = self._resolve_tool_file_path(args.get("path"))
        if path is None:
            return {"ok": False, **meta, "received_args": args}
        if "content" not in args or args.get("content") is None:
            return {"ok": False, **meta, "error": "missing required argument: content", "received_args": args}
        content = str(args.get("content"))
        append = bool(args.get("append", False))
        create_dirs = bool(args.get("create_dirs", False))
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        line_range_mode = start_line is not None or end_line is not None

        if append and line_range_mode:
            return {"ok": False, **meta, "error": "append=true cannot be combined with start_line/end_line"}

        parent = path.parent
        if not parent.exists():
            if create_dirs and not line_range_mode:
                parent.mkdir(parents=True, exist_ok=True)
            else:
                return {"ok": False, **meta, "error": "parent directory does not exist", "parent": str(parent)}
        if path.exists() and path.is_dir():
            return {"ok": False, **meta, "error": "path is a directory"}

        if line_range_mode:
            if not path.exists():
                return {"ok": False, **meta, "error": "cannot replace lines because file does not exist"}
            if not path.is_file():
                return {"ok": False, **meta, "error": "path is not a regular file"}
            existing = path.read_text(encoding="utf-8", errors="replace")
            lines = existing.splitlines(keepends=True)
            total_lines_before = len(lines)
            first = max(1, int(start_line)) if start_line is not None else 1
            last = max(1, int(end_line)) if end_line is not None else first
            if first > last:
                return {"ok": False, **meta, "error": "start_line must be <= end_line", "start_line": first, "end_line": last}
            if total_lines_before == 0:
                return {"ok": False, **meta, "error": "cannot replace lines in an empty file; omit start_line/end_line to write whole-file content"}
            if first > total_lines_before:
                return {"ok": False, **meta, "error": "start_line exceeds total_lines", "start_line": first, "total_lines": total_lines_before}
            if last > total_lines_before:
                return {"ok": False, **meta, "error": "end_line exceeds total_lines", "end_line": last, "total_lines": total_lines_before}

            start_index = first - 1
            end_index = last
            old_segment = "".join(lines[start_index:end_index])
            new_text = "".join(lines[:start_index]) + content + "".join(lines[end_index:])
            with path.open("w", encoding="utf-8", newline="") as f:
                written = f.write(new_text)
            total_lines_after = len(new_text.splitlines(keepends=True))
            replacement_line_count = len(content.splitlines(keepends=True))
            warning = None
            if content and not content.endswith(("\n", "\r")) and last < total_lines_before:
                warning = "replacement content does not end with a newline and may join with the following line"
            return {
                "ok": True,
                **meta,
                "operation": "replace_lines",
                "start_line": first,
                "end_line": last,
                "replaced_line_count": last - first + 1,
                "replacement_line_count": replacement_line_count,
                "total_lines_before": total_lines_before,
                "total_lines_after": total_lines_after,
                "old_chars_replaced": len(old_segment),
                "new_chars_inserted": len(content),
                "chars_written": written,
                "bytes_written_utf8": len(new_text.encode("utf-8")),
                "warning": warning,
            }

        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8", newline="") as f:
            written = f.write(content)
        return {
            "ok": True,
            **meta,
            "operation": "append" if append else "write_file",
            "append": append,
            "create_dirs": create_dirs,
            "chars_written": written,
            "bytes_written_utf8": len(content.encode("utf-8")),
        }

    async def _call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "terminal_read":
                text = self.terminal.read(args.get("max_chars", 6000))
                return {"ok": True, "text": text}
            if name == "terminal_send_text":
                if "text" not in args or args.get("text") is None or str(args.get("text")) == "":
                    return {"ok": False, "error": "missing required argument: text", "received_args": args}
                return self.terminal.send_text(str(args["text"]), bool(args.get("newline", False)))
            if name == "terminal_send_keys":
                keys = args.get("keys", [])
                if not isinstance(keys, list):
                    keys = [str(keys)]
                return self.terminal.send_keys([str(k) for k in keys])
            if name == "terminal_resize":
                return self.terminal.resize(int(args.get("rows", 28)), int(args.get("cols", 100)))
            if name == "sleep":
                seconds = max(0.1, min(float(args.get("seconds", 1)), 10.0))
                await asyncio.sleep(seconds)
                return {"ok": True, "slept": seconds}
            if name == "file_search":
                return self._search_files_for_tool(args)
            if name == "file_read":
                return self._read_text_file_for_tool(args)
            if name == "file_write":
                return self._write_text_file_for_tool(args)
            if name in {"memory_save", "memory_search", "memory_list", "memory_forget"}:
                if not self.cfg.memory_enabled or not self.memory:
                    return {"ok": False, "error": "session memory is disabled"}
                if name == "memory_save":
                    if "text" not in args or not str(args.get("text", "")).strip():
                        return {"ok": False, "error": "missing required argument: text", "received_args": args}
                    return self.memory.save(str(args["text"]), tags=args.get("tags", []), importance=int(args.get("importance", 3)))
                if name == "memory_search":
                    return self.memory.search(str(args.get("query", "")), max_items=int(args.get("max_items", 5)))
                if name == "memory_list":
                    return self.memory.list_recent(max_items=int(args.get("max_items", 10)))
                if name == "memory_forget":
                    return self.memory.forget(str(args.get("memory_id", "")))
            if name == "finish_task":
                summary = str(args.get("summary", "Task finished."))
                await self.emit_agent("done", summary)
                return {"ok": True, "summary": summary}
            return {"ok": False, "error": f"unknown tool: {name}"}
        except Exception as exc:
            logging.exception("Tool %s failed", name)
            return {"ok": False, "error": repr(exc)}


HTML = r"""
<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TALK — Terminal Agent Linux Kit</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css" />
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body { margin: 0; height: 100dvh; background: #111827; color: #e5e7eb; font-family: system-ui, sans-serif; overflow: hidden; }
    header { height: 54px; padding: 12px 16px; border-bottom: 1px solid #374151; display:flex; gap:12px; align-items:center; flex: 0 0 auto; }
    header strong { font-size: 16px; letter-spacing: .04em; }
    .pill { padding: 2px 8px; border-radius: 999px; background:#374151; font-size:12px; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 480px; height: calc(100dvh - 54px); min-height: 0; overflow: hidden; }
    .left { min-width: 0; border-right: 1px solid #374151; display:flex; flex-direction:column; min-height:0; overflow: hidden; }
    #terminal { flex: 1 1 auto; min-height: 0; padding: 8px; overflow: hidden; }
    .right { display:flex; flex-direction:column; min-width: 0; min-height: 0; overflow: hidden; background:#0f172a; }
    .panel { padding: 10px; border-bottom: 1px solid #374151; flex: 0 0 auto; }
    textarea, input { width: 100%; background:#0b1220; color:#e5e7eb; border:1px solid #4b5563; border-radius:8px; padding:9px; outline: none; }
    textarea:focus, input:focus { border-color:#60a5fa; }
    textarea { height: 68px; min-height: 68px; max-height: 120px; resize: vertical; }
    button { background:#2563eb; color:white; border:0; border-radius:8px; padding:9px 12px; cursor:pointer; white-space:nowrap; }
    button:disabled { opacity: 0.6; cursor:not-allowed; }
    #llmLog { flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden; padding: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.45; scroll-behavior: smooth; }
    .entry { padding: 7px 8px; margin-bottom: 7px; border-radius: 8px; background:#1f2937; white-space: pre-wrap; word-break: break-word; border-left: 3px solid #64748b; }
    .entry .meta { color:#9ca3af; font-size:11px; margin-bottom: 3px; display:flex; justify-content:space-between; gap:8px; }
    .summary { white-space: pre-wrap; }
    .event-user_message { border-left-color: #a78bfa; }
    .event-request { border-left-color: #60a5fa; }
    .event-assistant, .stream-entry { border-left-color: #22c55e; }
    .event-empty_retry, .event-assistant_empty, .event-fake_tool_calls_recovered { border-left-color: #f97316; color:#fed7aa; }
    .event-tool_call { border-left-color: #eab308; color:#fde68a; }
    .event-tool_result { border-left-color: #84cc16; color:#d9f99d; }
    .event-error { border-left-color: #ef4444; color:#fecaca; }
    .event-conversation_cleared { border-left-color: #38bdf8; color:#bae6fd; }
    .thinking-block { margin: 6px 0; padding: 6px; border: 1px dashed #475569; border-radius: 6px; color: #c4b5fd; background:#111827; max-height: 220px; overflow:auto; }
    .content-block { margin: 6px 0; color:#dcfce7; }
    .tool-stream { margin: 6px 0; color:#fde68a; }
    .result-block { margin: 6px 0; color:#d9f99d; white-space: pre-wrap; }
    .label { display:block; color:#94a3b8; font-size:11px; margin-bottom: 2px; }
    details { margin-top:6px; }
    summary { cursor:pointer; color:#93c5fd; }
    pre { white-space: pre-wrap; word-break: break-word; max-height: 280px; overflow:auto; background:#020617; padding:8px; border-radius:8px; border:1px solid #1f2937; }
    small { color:#9ca3af; }
    .hint { padding: 0 10px 8px 10px; color:#9ca3af; font-size:12px; flex: 0 0 auto; }
    #chatForm { display:flex; gap:8px; padding: 10px; border-top:1px solid #374151; flex: 0 0 auto; background:#0f172a; }
    #chatInput { flex: 1; min-width:0; }
  </style>
</head>
<body>
  <header>
    <strong>TALK</strong>
    <span class="pill">Terminal Agent Linux Kit</span>
    <span class="pill" id="wsStatus">connecting...</span>
    <span class="pill" id="agentStatus">agent idle</span>
  </header>
  <main>
    <section class="left"><div id="terminal"></div></section>
    <aside class="right">
      <div class="panel">
        <small>Startup prompt for the agent</small>
        <textarea id="prompt" placeholder="Example: Find files with the SUID bit set."></textarea>
        <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
          <button id="startBtn">Start agent</button>
          <button id="stepBtn" style="background:#7c3aed;">Step</button>
          <button id="clearBtn" style="background:#4b5563;">Clear conversation</button>
        </div>
      </div>
      <div id="llmLog" aria-label="LLM conversation"></div>
      <div class="hint">This field always writes to the LLM. Use the terminal directly in the window on the left.</div>
      <form id="chatForm">
        <input id="chatInput" autocomplete="off" placeholder="Message to the model, no slash command..." />
        <button type="submit">Send to LLM</button>
      </form>
    </aside>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.js"></script>
  <script>
    const term = new Terminal({ cursorBlink: true, scrollback: 10000, fontSize: 14, convertEol: true });
    const fitAddon = new FitAddon.FitAddon();
    term.loadAddon(fitAddon);
    term.open(document.getElementById('terminal'));
    fitAddon.fit();

    const wsStatus = document.getElementById('wsStatus');
    const agentStatus = document.getElementById('agentStatus');
    const llmLog = document.getElementById('llmLog');
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    const streams = new Map();
    let autoScroll = true;

    llmLog.addEventListener('scroll', () => {
      autoScroll = (llmLog.scrollTop + llmLog.clientHeight) >= (llmLog.scrollHeight - 40);
    });
    function scrollBottom(force=false) { if (force || autoScroll) llmLog.scrollTop = llmLog.scrollHeight; }
    function safeSend(obj) { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }
    function sendResize() { safeSend({type:'resize', cols: term.cols, rows: term.rows}); }
    function asTime(ts) { return new Date((ts || Date.now()/1000) * 1000).toLocaleTimeString(); }
    function escapeHtml(s) { return String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }

    function summarize(item) {
      const ev = item.event;
      const d = item.data || {};
      if (ev === 'user_message') return `USER [${d.source || 'user'}]: ${d.content || ''}`;
      if (ev === 'request') return `REQUEST step=${d.step} model=${d.model} stream=${d.stream} reasoning=${d.reasoning_effort} tool_choice=${d.tool_choice}`;
      if (ev === 'assistant') {
        const content = (d.content || '').trim();
        const thinking = (d.thinking || '').trim();
        const text = content ? content : (thinking ? '(the response contained only thinking)' : '(empty content)');
        return `ASSISTANT step=${d.step} finish=${d.finish_reason || '?'} tool_calls=${d.tool_calls_count || 0}: ${text}`;
      }
      if (ev === 'tool_call') return `TOOL CALL step=${d.step}: ${d.name}(${d.raw_arguments || '{}'})`;
      if (ev === 'tool_result') return `TOOL RESULT step=${d.step}: ${d.name} → ${JSON.stringify(d.result || {})}`;
      if (ev === 'empty_retry') return `EMPTY RETRY step=${d.step}: the model returned an empty response without tool_calls`;
      if (ev === 'thinking_only_retry') return `THINKING-ONLY RETRY step=${d.step}: the model reasoned privately but returned no tool_calls`;
      if (ev === 'fake_tool_calls_recovered') return `WARNING step=${d.step}: Model returned fake tool_calls in content, recovered via fallback parser (${d.recovered_tool_calls_count || 0})`;
      if (ev === 'tool_choice_retry') return `TOOL_CHOICE RETRY: backend rejected required, retrying with auto`;
      if (ev === 'conversation_cleared') return d.message || 'LLM context cleared.';
      if (ev === 'max_steps') return `MAX STEPS: ${d.max_steps}`;
      if (ev === 'error') return `ERROR: ${d.error || ''}`;
      if (ev === 'agent_log') return d.text || '';
      return `${ev}: ${JSON.stringify(d).slice(0, 500)}`;
    }

    function addLlmEvent(item) {
      // Tool calls/results are already shown as live tool execution streams.
      // Keep those events in backend logs/history, but do not render duplicate cards in the panel.
      if (item && item.data && item.data.ui_hidden === true) return;
      const div = document.createElement('div');
      div.className = `entry event-${item.event}`;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.innerHTML = `<span>${escapeHtml(item.event)}</span><span>${asTime(item.ts)}</span>`;
      const summary = document.createElement('div');
      summary.className = 'summary';
      summary.textContent = summarize(item);
      const details = document.createElement('details');
      const detSummary = document.createElement('summary');
      detSummary.textContent = 'raw JSON';
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(item.data, null, 2);
      details.appendChild(detSummary);
      details.appendChild(pre);
      div.appendChild(meta);
      div.appendChild(summary);
      div.appendChild(details);
      llmLog.appendChild(div);
      scrollBottom(true);
    }

    function addAgentLog(level, text) { addLlmEvent({ts: Date.now()/1000, event: 'agent_log', data: {level, text}}); }

    function streamStart(data) {
      const id = data.stream_id;
      const isToolExecution = data.stream_type === 'tool_execution';
      const div = document.createElement('div');
      div.className = 'entry stream-entry';
      div.dataset.streamId = id;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.innerHTML = `<span>${isToolExecution ? 'tool execution stream' : 'assistant stream'}</span><span>${asTime()}</span>`;
      const header = document.createElement('div');
      header.className = 'summary';
      header.textContent = isToolExecution
        ? `TOOL STREAM step=${data.step} ${data.tool_name || 'tool'}`
        : `STREAM step=${data.step} model=${data.model} reasoning=${data.reasoning_effort}`;
      const thinking = document.createElement('div');
      thinking.className = 'thinking-block';
      thinking.style.display = 'none';
      thinking.innerHTML = '<span class="label">thinking / reasoning</span><span class="thinking-text"></span>';
      const content = document.createElement('div');
      content.className = 'content-block';
      content.style.display = 'none';
      content.innerHTML = '<span class="label">content</span><span class="content-text"></span>';
      const tool = document.createElement('div');
      tool.className = 'tool-stream';
      tool.style.display = 'none';
      tool.innerHTML = '<span class="label">tool call / execution</span><span class="tool-text"></span>';
      const result = document.createElement('div');
      result.className = 'result-block';
      result.style.display = 'none';
      result.innerHTML = '<span class="label">tool result / output</span><span class="result-text"></span>';
      div.appendChild(meta); div.appendChild(header); div.appendChild(thinking); div.appendChild(content); div.appendChild(tool); div.appendChild(result);
      llmLog.appendChild(div);
      streams.set(id, {div, meta, header, thinking, content, tool, result, isToolExecution,
        thinkingText: thinking.querySelector('.thinking-text'),
        contentText: content.querySelector('.content-text'),
        toolText: tool.querySelector('.tool-text'),
        resultText: result.querySelector('.result-text')});
      scrollBottom(true);
    }

    function streamDelta(id, kind, text) {
      const s = streams.get(id);
      if (!s || !text) return;
      if (kind === 'thinking') { s.thinking.style.display = ''; s.thinkingText.textContent += text; }
      else if (kind === 'tool') { s.tool.style.display = ''; s.toolText.textContent += text; }
      else if (kind === 'result') { s.result.style.display = ''; s.resultText.textContent += text; }
      else { s.content.style.display = ''; s.contentText.textContent += text; }
      scrollBottom();
    }

    function streamDone(id, data) {
      const s = streams.get(id);
      if (!s) return;
      const isToolExecution = data.stream_type === 'tool_execution';
      s.header.textContent = isToolExecution
        ? `TOOL DONE step=${data.step} ${data.tool_name || 'tool'} ok=${data.ok === false ? 'false' : 'true'}`
        : `STREAM DONE step=${data.step} finish=${data.finish_reason || '?'} chunks=${data.chunks_seen} tool_calls=${data.tool_calls_count || 0}`;
      // For tool streams the call/result JSON was already streamed into the card.
      // Do not add another final JSON block, because it repeats the same information in the panel.
      if (!isToolExecution) {
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'stream final JSON';
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(data, null, 2);
        details.appendChild(summary); details.appendChild(pre); s.div.appendChild(details);
      }
      scrollBottom(true);
    }

    ws.onopen = () => { wsStatus.textContent = 'connected'; sendResize(); term.focus(); };
    ws.onclose = () => { wsStatus.textContent = 'disconnected'; addAgentLog('error', 'WebSocket disconnected'); };
    ws.onerror = () => { wsStatus.textContent = 'error'; };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'terminal') term.write(msg.data);
      if (msg.type === 'agent') addAgentLog(msg.level, msg.data);
      if (msg.type === 'llm_event') addLlmEvent(msg.item);
      if (msg.type === 'llm_history') { llmLog.innerHTML = ''; streams.clear(); for (const item of msg.events || []) addLlmEvent(item); }
      if (msg.type === 'llm_stream_start') streamStart(msg.data);
      if (msg.type === 'llm_stream_delta') streamDelta(msg.stream_id, msg.kind, msg.text);
      if (msg.type === 'llm_stream_done') streamDone(msg.stream_id, msg.data);
      if (msg.type === 'status') agentStatus.textContent = msg.agent_running ? 'agent pracuje' : 'agent idle';
    };

    window.addEventListener('resize', () => { fitAddon.fit(); sendResize(); });
    term.onData(data => safeSend({type:'terminal_input', data}));
    term.onResize(size => safeSend({type:'resize', cols:size.cols, rows:size.rows}));

    document.getElementById('chatForm').addEventListener('submit', ev => {
      ev.preventDefault();
      const input = document.getElementById('chatInput');
      const text = input.value.trim();
      input.value = '';
      if (!text) return;
      safeSend({type:'chat', text});
    });
    document.getElementById('startBtn').addEventListener('click', () => {
      const promptEl = document.getElementById('prompt');
      const prompt = promptEl.value.trim();
      if (!prompt) return addAgentLog('error', 'Provide a startup prompt.');
      safeSend({type:'agent_prompt', prompt});
      promptEl.value = '';
    });
    document.getElementById('stepBtn').addEventListener('click', () => {
      const promptEl = document.getElementById('prompt');
      const prompt = promptEl.value.trim();
      if (prompt) { safeSend({type:'agent_prompt_step', prompt}); promptEl.value = ''; }
      else { safeSend({type:'agent_step'}); }
    });
    document.getElementById('clearBtn').addEventListener('click', () => {
      llmLog.innerHTML = '';
      streams.clear();
      safeSend({type:'clear_conversation'});
    });
  </script>
</body>
</html>
"""


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    debug_path = log_dir / "agent_debug.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(debug_path, encoding="utf-8"), logging.StreamHandler(sys.stderr)],
    )
    logging.info("Debug log: %s", debug_path)


def make_app(terminal: PtyTerminal, agent: TerminalAgent, hub: WebHub, startup_prompt: Optional[str]) -> FastAPI:
    app = FastAPI(title=APP_FULL_NAME)

    @app.on_event("startup")
    async def on_startup() -> None:
        hub.set_loop(asyncio.get_running_loop())
        terminal.start(hub)
        if startup_prompt:
            await asyncio.sleep(0.5)
            await agent.add_user_message(startup_prompt, source="startup_prompt")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        terminal.stop()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True, "app": APP_FULL_NAME, "model": agent.cfg.model, "base_url": agent.cfg.base_url, "reasoning_effort": agent.cfg.reasoning_effort, "memory_enabled": agent.cfg.memory_enabled, "memory_items": agent.memory.list_recent(1).get("total_memories", 0) if agent.memory else 0})

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        await ws.send_json({"type": "terminal", "data": terminal.read(20_000)})
        await ws.send_json({"type": "llm_history", "events": list(hub.llm_history)})
        await ws.send_json({"type": "status", "agent_running": agent.running})
        try:
            while True:
                msg = await ws.receive_json()
                msg_type = msg.get("type")
                if msg_type == "terminal_input":
                    data = msg.get("data", "")
                    if data:
                        terminal.write(str(data).encode("utf-8", errors="replace"))
                elif msg_type == "resize":
                    cols = int(msg.get("cols", terminal.cols))
                    rows = int(msg.get("rows", terminal.rows))
                    terminal.resize(rows=rows, cols=cols)
                elif msg_type == "chat":
                    await agent.add_user_message(str(msg.get("text", "")), source="chat")
                elif msg_type == "agent_prompt":
                    await agent.add_user_message(str(msg.get("prompt", "")), source="web_start")
                elif msg_type == "agent_prompt_step":
                    await agent.add_user_message(str(msg.get("prompt", "")), source="web_step", step_limit=1)
                elif msg_type == "agent_step":
                    await agent.step_once()
                elif msg_type == "clear_conversation":
                    await agent.reset_conversation()
                else:
                    await agent.emit_agent("warn", f"Unknown WebSocket message type: {msg_type}", msg)
        except WebSocketDisconnect:
            await hub.disconnect(ws)
        except Exception:
            logging.exception("WebSocket error")
            await hub.disconnect(ws)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TALK — Terminal Agent Linux Kit: a single-file terminal agent for FastFlowLM")
    parser.add_argument("--base-url", default=os.getenv("FASTFLOWLM_BASE_URL", "http://127.0.0.1:52625/v1"))
    parser.add_argument("--api-key", default=os.getenv("FASTFLOWLM_API_KEY", "flm"))
    parser.add_argument("--model", default=os.getenv("FASTFLOWLM_MODEL", "qwen3.5:9b"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--shell", default=os.getenv("SHELL", "/bin/bash"))
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--prompt", default=None, help="Startup prompt; if omitted, provide it in the UI")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument(
        "--tool-choice",
        choices=["required", "auto", "none", "omit"],
        default=os.getenv("FASTFLOWLM_TOOL_CHOICE", "required"),
        help="Defaults to required so the model does not end the loop with an empty response without tool_calls. Use auto/omit if the backend rejects required.",
    )
    parser.add_argument("--empty-retries", type=int, default=2, help="How many times to retry when the model returns an empty response without tool_calls. Also used as the retry budget for thinking-only responses without tool_calls, with a minimum of one thinking-only retry.")
    parser.add_argument("--terminal-context-chars", type=int, default=6000, help="How many trailing terminal characters to attach to each LLM request.")
    parser.add_argument("--terminal-tool-delay-ms", type=int, default=500, help="Delay before automatically attaching fresh terminal output after terminal-changing tools.")
    parser.add_argument("--memory-enabled", action=argparse.BooleanOptionalAction, default=True, help="Enable small session RAG memory tools and automatic memory context injection.")
    parser.add_argument("--memory-max-items", type=int, default=200, help="Maximum number of facts kept in session memory.")
    parser.add_argument("--memory-context-items", type=int, default=5, help="How many relevant memory items to attach automatically to each LLM request.")
    parser.add_argument("--max-request-chars", type=int, default=60000, help="Approximate character budget for the full request sent to the LLM; older history is compacted when this is exceeded.")
    parser.add_argument("--compact-keep-recent-messages", type=int, default=30, help="How many newest conversation messages to keep verbatim before inserting a compacted-history summary.")
    parser.add_argument("--compact-tool-result-chars", type=int, default=4000, help="Maximum size of large strings inside tool results in the request copy sent to the LLM.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default=os.getenv("FASTFLOWLM_REASONING_EFFORT", "high"),
        help="Controls thinking/reasoning. Defaults to high, the maximum thinking mode in FastFlowLM.",
    )
    parser.add_argument("--no-think", action="store_true", help="Shortcut that sets --reasoning-effort none.")
    parser.add_argument("--think", action="store_true", help="Kept for compatibility; thinking is high by default anyway.")
    parser.add_argument("--log-dir", default="logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_think:
        args.reasoning_effort = "none"
    log_dir = Path(args.log_dir)
    setup_logging(log_dir)
    logging.info("Starting %s with model=%s base_url=%s shell=%s cwd=%s reasoning_effort=%s memory_enabled=%s", APP_FULL_NAME, args.model, args.base_url, args.shell, args.cwd, args.reasoning_effort, args.memory_enabled)
    logging.info("Run command hint: flm serve %s", shlex.quote(args.model))

    hub = WebHub()
    terminal = PtyTerminal(shell=args.shell, cwd=args.cwd)
    convo_log = JsonlLogger(log_dir / "llm_conversation.jsonl")
    memory = SessionMemory(max_items=max(1, args.memory_max_items), log_path=log_dir / "session_memory.jsonl") if args.memory_enabled else None
    agent = TerminalAgent(
        AgentConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            tool_choice=args.tool_choice,
            empty_retries=max(0, args.empty_retries),
            terminal_context_chars=max(100, min(args.terminal_context_chars, 50_000)),
            terminal_tool_delay_ms=max(0, min(args.terminal_tool_delay_ms, 10_000)),
            memory_enabled=bool(args.memory_enabled),
            memory_max_items=max(1, args.memory_max_items),
            memory_context_items=max(0, min(args.memory_context_items, 20)),
            max_request_chars=max(8000, min(args.max_request_chars, 500_000)),
            compact_keep_recent_messages=max(4, min(args.compact_keep_recent_messages, 200)),
            compact_tool_result_chars=max(500, min(args.compact_tool_result_chars, 50_000)),
            reasoning_effort=args.reasoning_effort,
        ),
        terminal=terminal,
        hub=hub,
        convo_log=convo_log,
        memory=memory,
    )
    app = make_app(terminal=terminal, agent=agent, hub=hub, startup_prompt=args.prompt)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
