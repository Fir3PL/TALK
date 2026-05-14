#!/usr/bin/env python3
"""
TALK — Terminal Agent Linux Kit

A single-file terminal agent for FastFlowLM / OpenAI-compatible APIs.

Features:
- FastFlowLM API compatible with OpenAI Chat Completions
- streams LLM responses to the conversation panel, including thinking/reasoning fields when the backend exposes them
- maximum thinking mode by default: reasoning_effort=high
- native tool/function calling: terminal_read, terminal_send_text, terminal_send_keys,
  terminal_resize, sleep, finish_task
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
4. After sending a command to the terminal, usually use sleep and then terminal_read.
5. Do not assume a command has finished until you read the terminal.
6. Do not run destructive commands such as rm -rf, disk formatting, repository deletion, or global installations without an explicit user request.
7. When the task is finished, call finish_task with a short summary.
8. If the human types in the terminal, treat the visible terminal snapshot as shared work context.
9. If you need the human, write a short text message and wait for the next message from the conversation panel.
10. Be concise. The terminal is the source of truth.

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
    reasoning_effort: str


class TerminalAgent:
    def __init__(self, cfg: AgentConfig, terminal: PtyTerminal, hub: WebHub, convo_log: JsonlLogger) -> None:
        self.cfg = cfg
        self.terminal = terminal
        self.hub = hub
        self.convo_log = convo_log
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
        payload = {"generation": self.generation, "message": "The LLM context has been cleared. The terminal remains unchanged."}
        self.convo_log.write("conversation_cleared", payload)
        await self.hub.clear_llm_history()
        await self.hub.emit_llm("conversation_cleared", payload)

    def _terminal_context_block(self) -> str:
        text = self.terminal.read(self.cfg.terminal_context_chars)
        if not text:
            return "[terminal_snapshot]\nThe terminal does not have visible output yet.\n[/terminal_snapshot]"
        return f"[terminal_snapshot: last {min(len(text), self.cfg.terminal_context_chars)} characters of the visible terminal]\n{text}\n[/terminal_snapshot]"

    def _terminal_context_message(self) -> Dict[str, str]:
        return {
            "role": "user",
            "content": (
                "[auto_terminal_snapshot]\n"
                "This is not a new command from the human; it is the current shared terminal context. "
                "You can also see manual keystrokes/commands entered by the human and their output. "
                "If the state may be incomplete or the task concerns the terminal, use terminal_read.\n"
                f"{self._terminal_context_block()}\n"
                "[/auto_terminal_snapshot]"
            ),
        }

    def _build_request_messages(self) -> List[Dict[str, Any]]:
        return [*self.messages, self._terminal_context_message()]

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
            await self.hub.broadcast({"type": "status", "agent_running": True})
            try:
                for step in range(1, steps_limit + 1):
                    if run_generation != self.generation:
                        await self.emit_agent("info", "The previous agent run was interrupted because the conversation context was cleared.")
                        break
                    logging.info("LLM step %s/%s source=%s", step, steps_limit, source)
                    request_messages = self._build_request_messages()
                    await self.emit_llm("request", {
                        "step": step,
                        "source": source,
                        "model": self.cfg.model,
                        "stream": True,
                        "tool_choice": self.cfg.tool_choice,
                        "reasoning_effort": self.cfg.reasoning_effort,
                        "extra_body": self._extra_body(),
                        "messages_tail": request_messages[-20:],
                    })

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
                        if not content.strip() and not thinking.strip() and empty_retries_left > 0:
                            empty_retries_left -= 1
                            nudge = (
                                "Your previous response was empty and did not contain tool_calls. "
                                "You must use native tool calling. If you do not know what to do, call terminal_read. "
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

        stream = await self._open_stream(request_messages)
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
                    preview = self._tool_delta_preview(tcd)
                    if preview:
                        await self.hub.broadcast({"type": "llm_stream_delta", "stream_id": stream_id, "kind": "tool", "text": preview})

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
        await self.emit_llm("tool_call", {"step": step, "id": tc.get("id"), "name": name, "raw_arguments": raw_args, "parsed_arguments": args, "parse_meta": parse_meta})
        result = await self._call_tool(name, args)
        if parse_meta.get("ignored_args"):
            result["ignored_args"] = parse_meta["ignored_args"]
        if parse_meta.get("parse_notes"):
            result["arg_parse_notes"] = parse_meta["parse_notes"]
        result_payload = {"step": step, "tool_call_id": tc.get("id"), "name": name, "args": args, "result": result}
        await self.emit_llm("tool_result", result_payload)
        self.messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result, ensure_ascii=False)})
        if name == "finish_task":
            self.finished = True

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
      if (ev === 'fake_tool_calls_recovered') return `WARNING step=${d.step}: Model returned fake tool_calls in content, recovered via fallback parser (${d.recovered_tool_calls_count || 0})`;
      if (ev === 'tool_choice_retry') return `TOOL_CHOICE RETRY: backend rejected required, retrying with auto`;
      if (ev === 'conversation_cleared') return d.message || 'LLM context cleared.';
      if (ev === 'max_steps') return `MAX STEPS: ${d.max_steps}`;
      if (ev === 'error') return `ERROR: ${d.error || ''}`;
      if (ev === 'agent_log') return d.text || '';
      return `${ev}: ${JSON.stringify(d).slice(0, 500)}`;
    }

    function addLlmEvent(item) {
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
      const div = document.createElement('div');
      div.className = 'entry stream-entry';
      div.dataset.streamId = id;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.innerHTML = `<span>assistant stream</span><span>${asTime()}</span>`;
      const header = document.createElement('div');
      header.className = 'summary';
      header.textContent = `STREAM step=${data.step} model=${data.model} reasoning=${data.reasoning_effort}`;
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
      tool.innerHTML = '<span class="label">tool_call stream</span><span class="tool-text"></span>';
      div.appendChild(meta); div.appendChild(header); div.appendChild(thinking); div.appendChild(content); div.appendChild(tool);
      llmLog.appendChild(div);
      streams.set(id, {div, meta, header, thinking, content, tool,
        thinkingText: thinking.querySelector('.thinking-text'),
        contentText: content.querySelector('.content-text'),
        toolText: tool.querySelector('.tool-text')});
      scrollBottom(true);
    }

    function streamDelta(id, kind, text) {
      const s = streams.get(id);
      if (!s || !text) return;
      if (kind === 'thinking') { s.thinking.style.display = ''; s.thinkingText.textContent += text; }
      else if (kind === 'tool') { s.tool.style.display = ''; s.toolText.textContent += text; }
      else { s.content.style.display = ''; s.contentText.textContent += text; }
      scrollBottom();
    }

    function streamDone(id, data) {
      const s = streams.get(id);
      if (!s) return;
      s.header.textContent = `STREAM DONE step=${data.step} finish=${data.finish_reason || '?'} chunks=${data.chunks_seen} tool_calls=${data.tool_calls_count || 0}`;
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = 'stream final JSON';
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(data, null, 2);
      details.appendChild(summary); details.appendChild(pre); s.div.appendChild(details);
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
        return JSONResponse({"ok": True, "app": APP_FULL_NAME, "model": agent.cfg.model, "base_url": agent.cfg.base_url, "reasoning_effort": agent.cfg.reasoning_effort})

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
    parser.add_argument("--empty-retries", type=int, default=2, help="How many times to retry when the model returns an empty response without tool_calls.")
    parser.add_argument("--terminal-context-chars", type=int, default=6000, help="How many trailing terminal characters to attach to each LLM request.")
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
    logging.info("Starting %s with model=%s base_url=%s shell=%s cwd=%s reasoning_effort=%s", APP_FULL_NAME, args.model, args.base_url, args.shell, args.cwd, args.reasoning_effort)
    logging.info("Run command hint: flm serve %s", shlex.quote(args.model))

    hub = WebHub()
    terminal = PtyTerminal(shell=args.shell, cwd=args.cwd)
    convo_log = JsonlLogger(log_dir / "llm_conversation.jsonl")
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
            reasoning_effort=args.reasoning_effort,
        ),
        terminal=terminal,
        hub=hub,
        convo_log=convo_log,
    )
    app = make_app(terminal=terminal, agent=agent, hub=hub, startup_prompt=args.prompt)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
