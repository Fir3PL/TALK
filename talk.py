#!/usr/bin/env python3
"""
TALK — Terminal Agent Linux Kit
v1.2

A single-file terminal agent for FastFlowLM / OpenAI-compatible APIs and native Ollama API.

Features:
- FastFlowLM / OpenAI Chat Completions compatible mode
- native Ollama /api/chat compatible mode with streaming, thinking, and tools
- streams LLM responses to the conversation panel, including thinking/reasoning fields when the backend exposes them
- streams tool execution start, parsed arguments, delayed terminal capture, results, and errors to the GUI
- maximum thinking mode by default: reasoning_effort=high
- native tool/function calling: terminal_read, terminal_send_text, terminal_send_keys,
  terminal_resize, terminal_create_session, terminal_list_sessions,
  terminal_switch_session, terminal_close_session, sleep, file_search, file_read,
  file_write, finish_task
- multiple PTY terminal sessions: the agent can create, list, switch, close, and
  target sessions with session_id; when the current session is Working and a new
  command is sent without session_id, TALK automatically opens a new terminal session
- after terminal-writing tools, automatically waits 500 ms and forwards fresh terminal output
  from the target session to the next LLM request in the tool result
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
    pip install fastapi uvicorn openai httpx

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
from urllib.parse import urlparse, urlunparse
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
import httpx
import uvicorn


APP_NAME = "TALK"
APP_FULL_NAME = "TALK — Terminal Agent Linux Kit"


SYSTEM_PROMPT = """You are TALK — Terminal Agent Linux Kit. You share a Linux terminal with a human.

Working rules:
1. Complete the task by using the active tool-calling mode provided by TALK.
2. If the task requires the terminal, use a tool. Do not merely describe what you would do.
3. Before ANY terminal operation (terminal_read / terminal_send_text / terminal_send_keys / terminal_resize), call terminal_list_sessions first to confirm which sessions are idle or Working. For a fresh terminal task, terminal_list_sessions is normally the first tool call.
4. Working means a command or interactive application is active in that terminal. It can still receive control keys and app-specific commands when you target it with its session_id.
5. Always use terminal_read first if you do not know the current terminal state of the chosen session.
6. If session_id is not provided, pick an idle session for new shell commands. If all sessions are Working, create a new one with terminal_create_session or wait (sleep), then re-check with terminal_list_sessions.
7. TALK supports multiple terminal sessions. Use terminal_list_sessions to see session IDs, Working/idle state, active_app, active_command, cwd, and prompt state.
8. If a session is Working because msfconsole, a REPL, editor, shell job, or another app is active, keep using that same session_id to interact with that app. Use separate idle/new sessions for unrelated shell tools while other sessions continue Working.
9. After sending text/keys to a terminal session, the runtime automatically waits 500 ms and attaches fresh terminal output from that same session to the tool result. Use explicit sleep/terminal_read only when the command needs more time or more output.
10. Do not assume a long-running command has finished until you read that terminal session.
11. Starting a tool or command is not completion. Before finish_task, verify the result with terminal_read for every session where you sent text/keys; if output is inconclusive, use sleep and terminal_read again.
12. finish_task is allowed only after you have inspected the actual result and can summarize the verified outcome, not merely that an attempt was initiated.
13. Do not run destructive commands such as rm -rf, disk formatting, repository deletion, or global installations without an explicit user request.
14. When coordinating multi-terminal work, declare dependencies with task_declare and mark completion with task_mark_done to avoid races.
15. When the task is finished, call finish_task with a short summary.
16. If the human types in the terminal, treat the visible terminal snapshot as shared work context.
17. If you need the human, write a short text message and wait for the next message from the conversation panel.
18. Be concise. The terminal sessions are the source of truth.
19. Never stop after producing only private thinking/reasoning. If your analysis is not enough to finish, immediately choose and call the next useful tool such as terminal_list_sessions, terminal_read, file_search, file_read, memory_search, sleep, or finish_task.
20. Do not output an ACTION/COMMAND block instead of a tool call. If you write or decide on a COMMAND, execute it with terminal_send_text in the appropriate session immediately.

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
- Use the active tool-calling mode provided by the runtime. In normal mode this is API-native tool_calls; in FLM inline mode emit exactly one <|tool_call>call:tool_name{...}<tool_call|> block.
- Function arguments must be a single JSON object.
- Do not use double braces such as {{...}}.
- Do not use Python dict syntax.
- Do not use XML/HTML-style parameter tags such as <param_value>, <text_value>, <newline_param_value>, or </...>.
- Keys must be quoted.
- Booleans must be JSON: true or false.
- Do not add fields that are not in the schema.
- terminal_send_text requires non-empty text. Put the shell command inside the JSON text value, never outside the argument object.
- A bare tool name is not executable. Never emit only <|tool_call>call:terminal_send_text. If the command text is not fully formed yet, call terminal_list_sessions or terminal_read instead.

Correct examples:
- terminal_list_sessions: {}
- terminal_read: {"max_chars":6000}
- terminal_read specific session: {"session_id":"session_2","max_chars":6000}
- terminal_send_text: {"text":"ls -la /home/user","newline":true}
- terminal_send_text specific session: {"session_id":"session_2","text":"pytest","newline":true}
- terminal_send_keys: {"keys":["CTRL_C"]}
- terminal_resize: {"cols":120,"rows":30}
- terminal_create_session: {}
- terminal_switch_session: {"session_id":"session_2"}
- terminal_close_session: {"session_id":"session_2"}
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


FLM_INLINE_TOOL_MODE_PROMPT = """[FLM INLINE TOOL MODE]
This backend streams reasoning text but may hide OpenAI-native tool_calls from the client. In this mode, call tools by emitting exactly one inline tool call, then stop generating.

Required inline format, exactly:
<|tool_call>call:tool_name{"arg":value}<tool_call|>

Rules:
- Emit at most ONE tool call per assistant turn.
- If you decide that the next step is to read/search/run/wait/write something, do NOT describe that intention in prose. Emit the inline tool call immediately.
- Emit the inline call only when the full JSON argument object is ready. A bare <|tool_call>call:tool_name prefix is invalid and will not run.
- Do not output sentences like "I will now read ..." or "Next I will search ..." unless the task is genuinely complete or you need the human to clarify.
- Do not invent <|tool_response>; the runtime will execute the tool and send the real tool result in the next request.
- Do not continue reasoning after the inline tool call.
- Use JSON-compatible arguments with quoted string values.
- Do not use XML/HTML-style parameter tags such as <param_value>, <text_value>, <newline_param_value>, or </...>.
- terminal_send_text must include non-empty "text"; put the full shell command inside the JSON text value, not before/after the object.
- If you know you need terminal work but have not selected a session or command yet, call terminal_list_sessions{} first.
- Use finish_task only after the user's task is actually complete, not after a plan or partial review.
- Do not call finish_task immediately after terminal_send_text or terminal_send_keys. First call terminal_read for that session and verify the command/app result.
- Always call terminal_list_sessions before choosing any terminal session to operate on.
- Treat Working as "an app/command is active", not "unusable". Use explicit session_id to interact with that active app; use idle/new sessions for unrelated shell commands.
- If session_id is omitted, choose an idle session; if all are Working, create a new one (terminal_create_session) or wait (sleep), then re-check.
- Use only these tool names: terminal_read, terminal_send_text, terminal_send_keys, terminal_resize, terminal_create_session, terminal_list_sessions, terminal_switch_session, terminal_close_session, sleep, file_search, file_read, file_write, memory_save, memory_search, memory_list, memory_forget, task_declare, task_mark_done, task_status, finish_task.

Examples:
<|tool_call>call:terminal_list_sessions{}<tool_call|>
<|tool_call>call:terminal_read{"max_chars":6000}<tool_call|>
<|tool_call>call:terminal_send_text{"text":"ls -la","newline":true}<tool_call|>
<|tool_call>call:terminal_create_session{}<tool_call|>
<|tool_call>call:file_search{"query":"TODO","path":".","search_filenames":true,"search_contents":true}<tool_call|>
<|tool_call>call:finish_task{"summary":"I completed the task."}<tool_call|>
[/FLM INLINE TOOL MODE]"""


TOOLS: List[Dict[str, Any]] = [{'type': 'function',
  'function': {'name': 'terminal_read',
               'description': 'Read recent visible/output text from a terminal session. Optional session_id targets a '
                              'specific session; otherwise the current session is used. Reading refreshes prompt-based '
                              'Working/idle status and active_app. Correct arguments example: {"session_id":"session_2","max_chars":6000}',
               'parameters': {'type': 'object',
                              'additionalProperties': False,
                              'properties': {'max_chars': {'type': 'integer',
                                                           'description': 'Maximum number of trailing characters to '
                                                                          'return.',
                                                           'default': 6000},
                                             'session_id': {'type': 'string',
                                                            'description': 'Optional terminal session id. If omitted, '
                                                                           'the current session is used.'}}}}},
 {'type': 'function',
  'function': {'name': 'terminal_send_text',
               'description': 'Send exact text to a terminal session. Use newline=true to press Enter. Optional '
                              'session_id targets a specific session, including a Working interactive app. If '
                              'session_id is omitted and the current session is Working, TALK automatically creates a '
                              'new session and runs unrelated shell commands there. Correct '
                              'arguments example: {"text":"ls -la /home/user","newline":true}',
               'parameters': {'type': 'object',
                              'required': ['text'],
                              'additionalProperties': False,
                              'properties': {'text': {'type': 'string',
                                                      'description': 'Exact text to type into the terminal. Do not '
                                                                     'omit this field.'},
                                             'newline': {'type': 'boolean',
                                                         'description': 'Append Enter/Return after the text.',
                                                         'default': False},
                                             'session_id': {'type': 'string',
                                                            'description': 'Optional terminal session id. If omitted, '
                                                                           'the current session is used.'}}}}},
 {'type': 'function',
  'function': {'name': 'terminal_send_keys',
               'description': 'Send special keys to a terminal session. Optional session_id targets a specific '
                              'session, including a Working interactive app. Correct arguments example: {"session_id":"session_2","keys":["ENTER"]}. Valid '
                              'keys: ENTER, CTRL_C, UP, DOWN, LEFT, RIGHT, TAB, ESC, BACKSPACE, CTRL_D.',
               'parameters': {'type': 'object',
                              'required': ['keys'],
                              'additionalProperties': False,
                              'properties': {'keys': {'type': 'array',
                                                      'items': {'type': 'string'},
                                                      'description': 'List of special key names to send in order.'},
                                             'session_id': {'type': 'string',
                                                            'description': 'Optional terminal session id. If omitted, '
                                                                           'the current session is used.'}}}}},
 {'type': 'function',
  'function': {'name': 'terminal_resize',
               'description': 'Resize a terminal session PTY. Optional session_id targets a specific session. Correct '
                              'arguments example: {"session_id":"session_2","cols":120,"rows":30}',
               'parameters': {'type': 'object',
                              'required': ['cols', 'rows'],
                              'additionalProperties': False,
                              'properties': {'cols': {'type': 'integer', 'minimum': 20, 'maximum': 300},
                                             'rows': {'type': 'integer', 'minimum': 5, 'maximum': 120},
                                             'session_id': {'type': 'string',
                                                            'description': 'Optional terminal session id. If omitted, '
                                                                           'the current session is used.'}}}}},
 {'type': 'function',
  'function': {'name': 'sleep',
               'description': 'Wait briefly, useful after sending commands before reading terminal output. Correct '
                              'arguments example: {"seconds":1}',
               'parameters': {'type': 'object',
                              'additionalProperties': False,
                              'properties': {'seconds': {'type': 'number',
                                                         'minimum': 0.1,
                                                         'maximum': 10,
                                                         'default': 1}}}}},
 {'type': 'function',
  'function': {'name': 'file_read',
               'description': 'Read a UTF-8 text file from disk, optionally restricted to a 1-based inclusive line '
                              'range. Returns raw text plus numbered_text with line-number prefixes for choosing '
                              "surgical edit ranges. Relative paths are resolved from the terminal's current working "
                              'directory. Correct arguments examples: {"path":"src/app.py"} or '
                              '{"path":"src/app.py","start_line":10,"end_line":40} Optional session_id targets a '
                              'specific terminal session.',
               'parameters': {'type': 'object',
                              'required': ['path'],
                              'additionalProperties': False,
                              'properties': {'path': {'type': 'string',
                                                      'description': 'File path to read. Relative paths are resolved '
                                                                     "from the terminal's current working directory, "
                                                                     'equivalent to pwd in the shared shell.'},
                                             'start_line': {'type': 'integer',
                                                            'minimum': 1,
                                                            'description': 'Optional 1-based first line to include.'},
                                             'end_line': {'type': 'integer',
                                                          'minimum': 1,
                                                          'description': 'Optional 1-based last line to include, '
                                                                         'inclusive.'},
                                             'max_chars': {'type': 'integer',
                                                           'minimum': 1,
                                                           'maximum': 200000,
                                                           'default': 50000,
                                                           'description': 'Maximum characters of file text to return '
                                                                          'after line filtering.'},
                                             'session_id': {'type': 'string',
                                                            'description': 'Optional terminal session id. If omitted, '
                                                                           'the current session is used.'}}}}},
 {'type': 'function',
  'function': {'name': 'file_write',
               'description': "Write UTF-8 text to a file on disk. Relative paths are resolved from the terminal's "
                              'current working directory. Replaces the whole file by default, appends with '
                              'append=true, or surgically replaces a 1-based inclusive line range when '
                              'start_line/end_line are provided. Correct arguments examples: '
                              '{"path":"notes.txt","content":"hello\\n","append":false} or '
                              '{"path":"src/app.py","start_line":42,"end_line":47,"content":"def fixed():\\n    return '
                              'True\\n"} Optional session_id targets a specific terminal session.',
               'parameters': {'type': 'object',
                              'required': ['path', 'content'],
                              'additionalProperties': False,
                              'properties': {'path': {'type': 'string',
                                                      'description': 'File path to write. Relative paths are resolved '
                                                                     "from the terminal's current working directory, "
                                                                     'equivalent to pwd in the shared shell.'},
                                             'content': {'type': 'string',
                                                         'description': 'Complete UTF-8 text content to write, text to '
                                                                        'append when append=true, or replacement '
                                                                        'content for the selected line range. Do not '
                                                                        'include line-number prefixes from '
                                                                        'numbered_text.'},
                                             'start_line': {'type': 'integer',
                                                            'minimum': 1,
                                                            'description': 'Optional 1-based first line to replace. '
                                                                           'When provided without end_line, only this '
                                                                           'line is replaced.'},
                                             'end_line': {'type': 'integer',
                                                          'minimum': 1,
                                                          'description': 'Optional 1-based last line to replace, '
                                                                         'inclusive. When provided without start_line, '
                                                                         'lines 1..end_line are replaced.'},
                                             'append': {'type': 'boolean',
                                                        'default': False,
                                                        'description': 'Append to the file instead of replacing it. '
                                                                       'Cannot be combined with start_line/end_line.'},
                                             'create_dirs': {'type': 'boolean',
                                                             'default': False,
                                                             'description': 'Create missing parent directories before '
                                                                            'writing whole-file or append content.'},
                                             'session_id': {'type': 'string',
                                                            'description': 'Optional terminal session id. If omitted, '
                                                                           'the current session is used.'}}}}},
 {'type': 'function',
  'function': {'name': 'file_search',
               'description': 'Search for a query in file names/relative paths and file contents under a directory '
                              "resolved from the terminal's current working directory. Returns structured path matches "
                              'and content matches with line numbers and matching line text. Prefer this over '
                              'grep/find for ordinary codebase searches. Correct arguments example: '
                              '{"query":"TODO","path":".","search_filenames":true,"search_contents":true,"max_results":50} '
                              'Optional session_id targets a specific terminal session.',
               'parameters': {'type': 'object',
                              'required': ['query'],
                              'additionalProperties': False,
                              'properties': {'query': {'type': 'string',
                                                       'description': 'Plain text or regex pattern to search for in '
                                                                      'file names/relative paths and/or file '
                                                                      'contents.'},
                                             'path': {'type': 'string',
                                                      'default': '.',
                                                      'description': 'Directory or file to search. Relative paths are '
                                                                     "resolved from the terminal's current working "
                                                                     'directory.'},
                                             'regex': {'type': 'boolean',
                                                       'default': False,
                                                       'description': 'Treat query as a Python regular expression '
                                                                      'instead of literal text.'},
                                             'case_sensitive': {'type': 'boolean',
                                                                'default': False,
                                                                'description': 'Whether matching is case-sensitive.'},
                                             'search_filenames': {'type': 'boolean',
                                                                  'default': True,
                                                                  'description': 'Search file names and relative '
                                                                                 'paths.'},
                                             'search_contents': {'type': 'boolean',
                                                                 'default': True,
                                                                 'description': 'Search inside text file contents and '
                                                                                'return matching line text.'},
                                             'include_hidden': {'type': 'boolean',
                                                                'default': False,
                                                                'description': 'Include hidden files and directories. '
                                                                               'Common heavy directories such as .git '
                                                                               'and node_modules are still skipped by '
                                                                               'default unless named explicitly as the '
                                                                               'search path.'},
                                             'max_results': {'type': 'integer',
                                                             'minimum': 1,
                                                             'maximum': 1000,
                                                             'default': 100,
                                                             'description': 'Maximum total matches to return.'},
                                             'max_files': {'type': 'integer',
                                                           'minimum': 1,
                                                           'maximum': 20000,
                                                           'default': 2000,
                                                           'description': 'Maximum files to inspect.'},
                                             'max_file_bytes': {'type': 'integer',
                                                                'minimum': 1,
                                                                'maximum': 10000000,
                                                                'default': 1000000,
                                                                'description': 'Skip content search for files larger '
                                                                               'than this many bytes.'},
                                             'session_id': {'type': 'string',
                                                            'description': 'Optional terminal session id. If omitted, '
                                                                           'the current session is used.'}}}}},
 {'type': 'function',
  'function': {'name': 'memory_save',
               'description': 'Save a durable, task-relevant fact to the session RAG memory. Do not store secrets, '
                              'tokens, passwords, private keys, or transient terminal noise. Correct arguments '
                              'example: {"text":"The repository root is '
                              '/home/user/project.","tags":["repo","path"],"importance":4}',
               'parameters': {'type': 'object',
                              'required': ['text'],
                              'additionalProperties': False,
                              'properties': {'text': {'type': 'string',
                                                      'description': 'The concise fact to remember for this session.'},
                                             'tags': {'type': 'array',
                                                      'items': {'type': 'string'},
                                                      'description': 'Optional short tags such as repo, path, '
                                                                     'decision, error.'},
                                             'importance': {'type': 'integer',
                                                            'minimum': 1,
                                                            'maximum': 5,
                                                            'default': 3}}}}},
 {'type': 'function',
  'function': {'name': 'memory_search',
               'description': 'Search session RAG memory for relevant saved facts. Correct arguments example: '
                              '{"query":"repository root","max_items":5}',
               'parameters': {'type': 'object',
                              'required': ['query'],
                              'additionalProperties': False,
                              'properties': {'query': {'type': 'string',
                                                       'description': 'What to search for in session memory.'},
                                             'max_items': {'type': 'integer',
                                                           'minimum': 1,
                                                           'maximum': 20,
                                                           'default': 5}}}}},
 {'type': 'function',
  'function': {'name': 'memory_list',
               'description': 'List recent saved facts from the session memory. Correct arguments example: '
                              '{"max_items":10}',
               'parameters': {'type': 'object',
                              'additionalProperties': False,
                              'properties': {'max_items': {'type': 'integer',
                                                           'minimum': 1,
                                                           'maximum': 50,
                                                           'default': 10}}}}},
 {'type': 'function',
  'function': {'name': 'memory_forget',
               'description': 'Remove one saved session memory item by id. Correct arguments example: '
                              '{"memory_id":"mem_0001"}',
               'parameters': {'type': 'object',
                              'required': ['memory_id'],
                              'additionalProperties': False,
                              'properties': {'memory_id': {'type': 'string',
                                                           'description': 'Memory id to remove, such as mem_0001.'}}}}},
 {'type': 'function',
  'function': {'name': 'terminal_create_session',
               'description': 'Create a new terminal session (PTY). The session starts with the same shell and current '
                              'working directory as the current session. Returns session_id.',
               'parameters': {'type': 'object', 'additionalProperties': False, 'properties': {}}}},
 {'type': 'function',
  'function': {'name': 'terminal_list_sessions',
               'description': 'List all terminal sessions with session_id, Working/idle state, active_app, '
                              'active_command, pid, cwd, prompt state, and whether they are current.',
               'parameters': {'type': 'object', 'additionalProperties': False, 'properties': {}}}},
 {'type': 'function',
  'function': {'name': 'terminal_switch_session',
               'description': 'Switch the current terminal session. Tools without session_id will use this session. '
                              'Correct arguments example: {"session_id":"session_2"}',
               'parameters': {'type': 'object',
                              'required': ['session_id'],
                              'additionalProperties': False,
                              'properties': {'session_id': {'type': 'string',
                                                            'description': 'Existing terminal session id.'}}}}},
 {'type': 'function',
  'function': {'name': 'terminal_close_session',
               'description': 'Close and remove an existing terminal session. Cannot close the only remaining session. '
                              'Correct arguments example: {"session_id":"session_2"}',
               'parameters': {'type': 'object',
                              'required': ['session_id'],
                              'additionalProperties': False,
                              'properties': {'session_id': {'type': 'string',
                                                            'description': 'Existing terminal session id to '
                                                                           'close.'}}}}},
 {'type': 'function',
  'function': {'name': 'task_declare',
               'description': 'Declare a task and optional dependencies. If task B depends_on task A, the server will '
                              'reject starting B until A is marked done. Correct arguments example: '
                              '{"task_id":"build","depends_on":["deps"],"note":"Run build after deps"}',
               'parameters': {'type': 'object',
                              'required': ['task_id'],
                              'additionalProperties': False,
                              'properties': {'task_id': {'type': 'string', 'description': 'Stable task identifier.'},
                                             'depends_on': {'type': 'array',
                                                            'items': {'type': 'string'},
                                                            'description': 'List of prerequisite task_ids.',
                                                            'default': []},
                                             'note': {'type': 'string',
                                                      'description': 'Optional note about this task.',
                                                      'default': ''}}}}},
 {'type': 'function',
  'function': {'name': 'task_mark_done',
               'description': 'Mark a previously declared task as done. Correct arguments example: {"task_id":"deps"}',
               'parameters': {'type': 'object',
                              'required': ['task_id'],
                              'additionalProperties': False,
                              'properties': {'task_id': {'type': 'string'}}}}},
 {'type': 'function',
  'function': {'name': 'task_status',
               'description': 'Return all declared tasks, their dependencies, and completion state.',
               'parameters': {'type': 'object', 'additionalProperties': False, 'properties': {}}}},
 {'type': 'function',
  'function': {'name': 'finish_task',
               'description': 'Mark the current task as finished only after verifying the result of any terminal '
                              'actions with terminal_read. Provide a concise verified summary for the human. Correct '
                              'arguments example: {"summary":"I completed the task."}',
               'parameters': {'type': 'object',
                              'required': ['summary'],
                              'additionalProperties': False,
                              'properties': {'summary': {'type': 'string'}}}}}]

TOOL_ALLOWED_ARGS: Dict[str, Set[str]] = {'terminal_read': {'session_id', 'max_chars'},
 'terminal_send_text': {'text', 'session_id', 'newline'},
 'terminal_send_keys': {'session_id', 'keys'},
 'terminal_resize': {'cols', 'session_id', 'rows'},
 'sleep': {'seconds'},
 'file_search': {'case_sensitive',
                 'include_hidden',
                 'max_file_bytes',
                 'max_files',
                 'max_results',
                 'path',
                 'query',
                 'regex',
                 'search_contents',
                 'search_filenames',
                 'session_id'},
 'file_read': {'session_id', 'max_chars', 'path', 'start_line', 'end_line'},
 'file_write': {'content', 'session_id', 'create_dirs', 'path', 'append', 'start_line', 'end_line'},
 'memory_save': {'text', 'tags', 'importance'},
 'memory_search': {'max_items', 'query'},
 'memory_list': {'max_items'},
 'memory_forget': {'memory_id'},
 'finish_task': {'summary'},
 'terminal_create_session': set(),
 'terminal_list_sessions': set(),
 'terminal_switch_session': {'session_id'},
 'terminal_close_session': {'session_id'},
 'task_declare': {'task_id', 'depends_on', 'note'},
 'task_mark_done': {'task_id'},
 'task_status': set()}

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
        self.tty_name: Optional[str] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.buffer: Deque[str] = deque(maxlen=scrollback_chars)
        self.buffer_start_abs = 0
        self.buffer_end_abs = 0
        self.buffer_lock = threading.Lock()
        self.hub: Optional[WebHub] = None
        self.session_id: Optional[str] = None

    def start(self, hub: WebHub) -> None:
        self.hub = hub
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd
        try:
            self.tty_name = os.ttyname(slave_fd)
        except Exception:
            self.tty_name = None
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
                    self.hub.broadcast_threadsafe({"type": "terminal", "session_id": self.session_id, "data": text})
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

    @staticmethod
    def _proc_stat_pgrp(pid: int) -> Optional[int]:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            tail = raw.rsplit(")", 1)[1].strip().split()
            return int(tail[2]) if len(tail) >= 3 else None
        except Exception:
            return None

    @staticmethod
    def _proc_cmdline(pid: int) -> str:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            text = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            if text:
                return text
        except Exception:
            pass
        try:
            return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return ""

    @staticmethod
    def _app_name_from_command(command: str) -> str:
        try:
            parts = [p for p in shlex.split(command or "", posix=True) if p]
        except ValueError:
            parts = [p for p in str(command or "").split() if p]
        if not parts:
            return ""
        interpreters = {"python", "python2", "python3", "ruby", "perl", "node", "bash", "sh", "zsh", "fish", "java"}
        first = Path(parts[0]).name
        if first in interpreters and len(parts) > 1:
            for part in parts[1:]:
                if part.startswith("-"):
                    continue
                return Path(part).name or first
        return first

    def foreground_process(self) -> Dict[str, Any]:
        """Return the current foreground process/application for this PTY."""
        proc = self.proc
        shell_pid = proc.pid if proc else None
        shell_pgid = None
        if shell_pid:
            try:
                shell_pgid = os.getpgid(shell_pid)
            except Exception:
                shell_pgid = None
        foreground_pgid = None
        if self.master_fd is not None:
            try:
                foreground_pgid = os.tcgetpgrp(self.master_fd)
            except Exception:
                foreground_pgid = None

        candidates: List[Dict[str, Any]] = []
        if foreground_pgid is not None and Path("/proc").exists():
            try:
                proc_names = [name for name in os.listdir("/proc") if name.isdigit()]
            except Exception:
                proc_names = []
            for name in proc_names:
                pid = int(name)
                pgrp = self._proc_stat_pgrp(pid)
                if pgrp != foreground_pgid:
                    continue
                command = self._proc_cmdline(pid)
                candidates.append({
                    "pid": pid,
                    "pgid": pgrp,
                    "command": command,
                    "app": self._app_name_from_command(command),
                    "is_shell": bool(shell_pid and pid == shell_pid),
                })

        candidates.sort(key=lambda item: (bool(item.get("is_shell")), int(item.get("pid") or 0)))
        selected = candidates[0] if candidates else {}
        app = str(selected.get("app") or "")
        command = str(selected.get("command") or "")
        is_shell = bool(selected.get("is_shell") or (foreground_pgid is not None and shell_pgid is not None and foreground_pgid == shell_pgid))
        if not app and is_shell:
            app = Path(str(self.shell or "shell")).name or "shell"
            command = str(self.shell or app)
        return {
            "pid": selected.get("pid") or (shell_pid if is_shell else None),
            "pgid": foreground_pgid,
            "app": app,
            "command": command,
            "is_shell": is_shell,
            "tty": self.tty_name,
        }

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

    @staticmethod
    def _strip_terminal_control_sequences(text: str) -> str:
        """Return printable-ish terminal text for prompt detection.

        PTY output can end with many non-printing terminal-control sequences after
        the visible prompt, for example Kali/zsh/bash may append bracketed-paste
        mode toggles such as ESC[?2004h, keypad mode ESC=, color SGR sequences,
        cursor movement, erase-line, OSC title updates, or DCS/APC/PM/SOS blocks.
        Prompt detection must therefore ignore terminal protocol bytes and look at
        the last visible characters only.
        """
        if not text:
            return ""

        # OSC: ESC ] ... BEL or ESC \\ ; used by terminals for titles/hyperlinks.
        text = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
        # DCS, SOS, PM, APC: ESC P/^/_/... ESC \\ ; uncommon but possible.
        text = re.sub(r"\x1b[PX^_].*?\x1b\\", "", text, flags=re.DOTALL)
        # CSI: ESC [ ... final-byte. Covers SGR colors, DEC private modes,
        # bracketed paste, cursor addressing, erase-in-line, etc.
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        # SS3 and other two-byte escape/control sequences such as ESC= / ESC>.
        text = re.sub(r"\x1b[ -~]", "", text)
        # Remove remaining C0/C1 controls except newline and tab. Treat carriage
        # return as newline because PTYs frequently emit CRLF before prompts.
        text = text.replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text

    @classmethod
    def _prompt_candidate_from_tail(cls, text: str, max_chars: int = 4000) -> str:
        """Extract the final visible line/region relevant for prompt detection."""
        cleaned = cls._strip_terminal_control_sequences(text[-max_chars:])
        # Collapse excessive trailing whitespace but keep line boundaries so Kali's
        # two-line prompt (`...\n└─$ `) resolves to the final `└─$` line.
        cleaned = cleaned.rstrip(" \t\n")
        if not cleaned:
            return ""
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if not lines:
            return cleaned.strip()[-300:]
        # Most prompts are entirely on the last non-empty line. A small suffix is
        # enough and avoids scanning arbitrary command output.
        return lines[-1][-300:].strip()

    def prompt_visible_candidate(self, max_chars: int = 8000) -> str:
        """Return the final visible prompt candidate after stripping control bytes."""
        raw_tail = self.read(max_chars)
        return self._prompt_candidate_from_tail(raw_tail, max_chars=max_chars)

    def looks_idle_by_prompt(self, max_chars: int = 8000) -> bool:
        """Return True if the terminal tail visibly ends with a shell prompt.

        The heuristic is distribution-agnostic: after removing ANSI/OSC/CSI and
        other control sequences, it checks whether the final visible prompt region
        ends with `$` or `#`. This covers prompts like:
          user@host:~/dir$
          root@host:/dir#
          └─$
          [~]
          └─#
        while allowing trailing spaces and terminal-mode sequences after the prompt.
        """
        candidate = self.prompt_visible_candidate(max_chars=max_chars)
        if not candidate:
            return False
        # User requirement: detect `$` or `#` at the end of the visible terminal
        # prompt, regardless of distro/theme/colors. The control stripping above
        # ensures that terminal mode suffixes such as ESC[?2004h do not hide it.
        return bool(re.search(r"[$#]\s*$", candidate))

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



@dataclass
class TerminalSession:
    """One managed PTY terminal session."""

    id: str
    terminal: PtyTerminal
    busy: bool = False
    created_at: float = 0.0
    last_prompt_idle: bool = False
    last_prompt_candidate: str = ""
    active_app: str = "shell"
    active_command: str = ""
    active_pid: Optional[int] = None
    foreground_pgid: Optional[int] = None
    active_is_shell: bool = True

    def refresh_status(self) -> bool:
        """Refresh and return whether the working/app state changed.

        A session becomes idle when the shell prompt is visible at the end of the
        terminal tail. In the default Bash prompt this means the visible text ends
        with '$' or '#'. If another foreground application owns the PTY, the
        session is Working: it can still receive input for that application.
        """
        old_busy = bool(self.busy)
        old_app = self.active_app
        old_command = self.active_command
        proc = self.terminal.proc
        running = bool(proc and proc.poll() is None)
        candidate = self.terminal.prompt_visible_candidate()
        prompt_idle = bool(re.search(r"[$#]\s*$", candidate)) if candidate else False
        foreground = self.terminal.foreground_process()
        self.last_prompt_candidate = candidate[-300:]
        self.last_prompt_idle = bool(prompt_idle)
        if prompt_idle:
            shell_name = Path(str(self.terminal.shell or "shell")).name or "shell"
            self.active_app = shell_name
            self.active_command = str(self.terminal.shell or shell_name)
            self.active_pid = proc.pid if proc else foreground.get("pid")
            self.foreground_pgid = foreground.get("pgid")
            self.active_is_shell = True
        else:
            self.active_app = str(foreground.get("app") or "unknown")
            self.active_command = str(foreground.get("command") or self.active_app)
            self.active_pid = foreground.get("pid")
            self.foreground_pgid = foreground.get("pgid")
            self.active_is_shell = bool(foreground.get("is_shell"))
        if not running or prompt_idle:
            self.busy = False
        elif self.active_app and not self.active_is_shell:
            self.busy = True
        return old_busy != bool(self.busy) or old_app != self.active_app or old_command != self.active_command

    def info(self, current: bool = False) -> Dict[str, Any]:
        self.refresh_status()
        proc = self.terminal.proc
        pid = proc.pid if proc else None
        running = bool(proc and proc.poll() is None)
        try:
            cwd = str(self.terminal.current_working_directory())
        except Exception:
            cwd = str(self.terminal.cwd or "")
        return {
            "session_id": self.id,
            "busy": bool(self.busy),
            "working": bool(self.busy),
            "idle": not bool(self.busy),
            "state": "working" if self.busy else "idle",
            "prompt_idle": bool(self.last_prompt_idle),
            "prompt_candidate": self.last_prompt_candidate,
            "active_app": self.active_app,
            "active_command": self.active_command,
            "active_pid": self.active_pid,
            "foreground_pgid": self.foreground_pgid,
            "active_is_shell": bool(self.active_is_shell),
            "current": bool(current),
            "pid": pid,
            "running": running,
            "cwd": cwd,
            "rows": self.terminal.rows,
            "cols": self.terminal.cols,
            "created_at": self.created_at,
        }


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
        parsed = cls._sanitize_parsed_args(parsed, meta)
        normalized = cls._normalize(tool_name, parsed, meta)
        allowed = TOOL_ALLOWED_ARGS.get(tool_name)
        if allowed is not None:
            # Session-aware tools accept optional session_id. Preserve it even when
            # an individual normalizer branch did not explicitly copy it.
            if "session_id" in allowed:
                for alias in ("session_id", "session", "sid"):
                    if alias in parsed and parsed.get(alias) is not None:
                        normalized["session_id"] = str(parsed.get(alias))
                        if alias != "session_id":
                            meta.setdefault("consumed_aliases", {})[alias] = "session_id"
                            meta.setdefault("parse_notes", []).append(f"used alias {alias}->session_id")
                        break
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
    def _sanitize_parsed_args(cls, args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        """Clean model/tool-template markup that leaked into argument values.

        Some OpenAI-compatible backends stream malformed tool templates such as
        </param_value></text_value> inside otherwise parseable JSON arguments.
        Keep the recoverable value (for example session_1) and merge any
        explicit <text_value>...</text_value> style parameters into the dict.
        """
        out: Dict[str, Any] = {}
        for key, value in args.items():
            clean_key = cls._canonical_arg_key(str(key))
            clean_value = cls._clean_arg_value(clean_key, value, meta)
            if clean_value is not None:
                out[clean_key] = clean_value
            if isinstance(value, str) and "<" in value and ">" in value:
                tagged = cls._parse_tagged_params(value, meta)
                for tagged_key, tagged_value in tagged.items():
                    if tagged_key not in out and tagged_value is not None:
                        out[tagged_key] = tagged_value
        return out

    @staticmethod
    def _canonical_arg_key(key: str) -> str:
        key = str(key or "").strip().strip("<>/ ").lower().replace("-", "_")
        for suffix in ("_param_value", "_value", "_param"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        aliases = {
            "session": "session_id",
            "sid": "session_id",
            "cmd": "text",
            "command": "text",
            "input": "text",
            "content": "text",
            "enter": "newline",
            "press_enter": "newline",
            "submit": "newline",
            "return": "newline",
        }
        return aliases.get(key, key)

    @classmethod
    def _clean_arg_value(cls, key: str, value: Any, meta: Dict[str, Any]) -> Any:
        if not isinstance(value, str):
            return value
        raw = value
        cleaned = cls._strip_tool_markup(raw).strip()
        if cleaned != raw.strip():
            meta.setdefault("parse_notes", []).append(f"removed tool-template markup from {key}")
        if key == "session_id":
            match = re.search(r"\bsession_\d+\b", cleaned or raw)
            if match:
                sid = match.group(0)
                if sid != raw.strip():
                    meta.setdefault("parse_notes", []).append(f"recovered clean session_id={sid}")
                return sid
            if "<" in raw or ">" in raw:
                meta.setdefault("parse_notes", []).append("dropped invalid session_id containing tool-template markup")
                return None
        return cleaned

    @staticmethod
    def _strip_tool_markup(value: str) -> str:
        text = str(value or "")
        text = text.replace("<tool_call|>", "").replace("<tool_call|", "").replace("<|tool_call>", "")
        text = re.sub(
            r"</?(?:param|parameter|param_name|param_value|name|value|"
            r"[A-Za-z_][A-Za-z0-9_]*(?:_param)?_value)[^>]*>",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip().strip('"').strip("'").strip()

    @classmethod
    def _parse_tagged_params(cls, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        source = str(text or "")
        out: Dict[str, Any] = {}

        def add(key: str, value: str, note: str) -> None:
            canonical = cls._canonical_arg_key(key)
            cleaned = cls._clean_arg_value(canonical, value, meta)
            if cleaned is None or str(cleaned).strip() == "":
                return
            if canonical not in out:
                out[canonical] = cleaned
                meta.setdefault("parse_notes", []).append(note)

        # <text_value>cmd</text_value> or <newline_param_value>true</param_value>
        tag_pattern = re.compile(
            r"<(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?:_param)?_value[^>]*>\s*"
            r"(?P<value>.*?)\s*</(?:param_value|(?P=key)(?:_param)?_value)>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        for match in tag_pattern.finditer(source):
            add(match.group("key"), match.group("value"), f"recovered {match.group('key')} from tagged tool parameter")

        # <param_name>text</param_name><param_value>cmd</param_value>
        pair_pattern = re.compile(
            r"<(?:param_name|name)[^>]*>\s*(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)\s*</(?:param_name|name)>\s*"
            r"<(?:param_value|value)[^>]*>\s*(?P<value>.*?)\s*</(?:param_value|value)>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        for match in pair_pattern.finditer(source):
            add(match.group("key"), match.group("value"), f"recovered {match.group('key')} from tagged name/value parameter")

        # <param name="text">cmd</param>
        attr_pattern = re.compile(
            r"<(?:param|parameter)[^>]*\bname=[\"'](?P<key>[A-Za-z_][A-Za-z0-9_\-]*)[\"'][^>]*>"
            r"\s*(?P<value>.*?)\s*</(?:param|parameter)>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        for match in attr_pattern.finditer(source):
            add(match.group("key"), match.group("value"), f"recovered {match.group('key')} from tagged attribute parameter")

        return out

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
        tagged = cls._parse_tagged_params(s, {"parse_notes": [], "ignored_args": {}})
        if tagged:
            add(json.dumps(tagged, ensure_ascii=False))
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
        out: List[str] = []

        def add(candidate: str) -> None:
            if candidate not in out:
                out.append(candidate)

        add(s)
        add(cls._escape_literal_newlines_in_strings(s))
        repaired = cls._replace_words_outside_strings(s, {"True": "true", "False": "false", "None": "null"})
        repaired = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', lambda m: f'{m.group(1)}"{m.group(2)}":', repaired)
        add(repaired)
        add(cls._escape_literal_newlines_in_strings(repaired))

        # Some backends/UIs can inject a truncation marker into streamed JSON as a
        # bare object key, for example: "_request_truncated_note..." without
        # a colon or value. That makes the whole fake tool_calls payload invalid,
        # so repair this known artifact before giving up on recovery.
        repaired = cls._repair_bare_request_marker_keys(repaired)
        add(repaired)
        add(cls._repair_trailing_comma_object(repaired))

        repaired2 = re.sub(
            r':\s*([^\"\'\{\}\[\],][^,\}]*)',
            lambda m: cls._quote_loose_value(m.group(1)),
            repaired,
        )
        add(repaired2)
        repaired3 = cls._repair_bare_request_marker_keys(repaired2)
        add(repaired3)
        add(cls._repair_trailing_comma_object(repaired3))
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
    def _escape_literal_newlines_in_strings(s: str) -> str:
        out: List[str] = []
        quote: Optional[str] = None
        escape = False
        changed = False
        for ch in s:
            if quote:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    continue
                if ch == quote:
                    out.append(ch)
                    quote = None
                    continue
                if ch == "\n":
                    out.append("\\n")
                    changed = True
                    continue
                if ch == "\r":
                    out.append("\\r")
                    changed = True
                    continue
                out.append(ch)
                continue
            if ch in {'"', "'"}:
                quote = ch
            out.append(ch)
        return "".join(out) if changed else s

    @staticmethod
    def _repair_bare_request_marker_keys(s: str) -> str:
        # Known malformed payload pattern seen when a request-truncation note is
        # spliced into a JSON object as a key without a value. Keep it parseable;
        # ToolArgParser will later ignore this extra argument for the selected tool.
        return re.sub(
            r'([\{,]\s*)("_request_[^"\\]*(?:\\.[^"\\]*)*")(\s*)(?=[,}\]])',
            r'\1\2: true\3',
            s,
        )

    @staticmethod
    def _repair_trailing_comma_object(s: str) -> str:
        stripped = str(s or "").strip()
        if stripped.startswith("{") and stripped.endswith(","):
            return stripped[:-1].rstrip() + "}"
        return s

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
        if tool_name in {"task_declare", "task_mark_done", "task_status"}:
            if tool_name != "task_status":
                task_id = cls._first_present(args, ("task_id", "id", "task"), meta)
                tid = str(task_id or "").strip()
                if tool_name == "task_mark_done":
                    return {"task_id": tid}
                depends_on = args.get("depends_on") or []
                if isinstance(depends_on, str):
                    depends_on = [p.strip() for p in re.split(r"[,;]", depends_on) if p.strip()]
                if not isinstance(depends_on, list):
                    depends_on = []
                depends_on_list = [str(d).strip() for d in depends_on if str(d).strip()]
                return {"task_id": tid, "depends_on": depends_on_list, "note": str(args.get("note", "") or "")}
            return {}
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
    api_provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_steps: int
    tool_choice: str
    empty_retries: int
    llm_error_retries: int
    terminal_context_chars: int
    terminal_tool_delay_ms: int
    memory_enabled: bool
    memory_max_items: int
    memory_context_items: int
    max_request_chars: int
    compact_keep_recent_messages: int
    compact_tool_result_chars: int
    reasoning_effort: str
    flm_tools_mode: str


class TerminalAgent:
    def __init__(self, cfg: AgentConfig, terminal: PtyTerminal, hub: WebHub, convo_log: JsonlLogger, memory: Optional[SessionMemory] = None) -> None:
        self.cfg = cfg
        self.terminal = terminal
        self.terminal.session_id = "session_1"
        self.hub = hub
        self.convo_log = convo_log
        self.memory = memory
        self.sessions: Dict[str, TerminalSession] = {
            "session_1": TerminalSession(id="session_1", terminal=terminal, busy=False, created_at=time.time())
        }
        self.session_locks: Dict[str, asyncio.Lock] = {"session_1": asyncio.Lock()}
        self.current_session_id = "session_1"
        self.next_session_index = 2
        self.api_provider = (cfg.api_provider or "openai").strip().lower()
        self.client: Optional[AsyncOpenAI] = None
        if self.api_provider != "ollama":
            self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY")
        self.tool_call_names: Dict[str, str] = {}
        self.tool_call_recovery_meta: Dict[str, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.lock = asyncio.Lock()
        self.running = False
        self.finished = False
        self.generation = 0
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.active_task_id: Optional[str] = None
        self.pending_terminal_verifications: Dict[str, Dict[str, Any]] = {}


    def get_session(self, session_id: Optional[str] = None) -> Optional[TerminalSession]:
        """Return a managed terminal session by id or the current session."""
        sid = str(session_id or self.current_session_id or "").strip()
        if not sid:
            return None
        return self.sessions.get(sid)

    def _set_current_session(self, session_id: str) -> Optional[TerminalSession]:
        """Switch current session and keep legacy self.terminal aligned."""
        session = self.sessions.get(str(session_id))
        if not session:
            return None
        session.refresh_status()
        self.current_session_id = session.id
        self.terminal = session.terminal
        return session

    def _refresh_session_states(self) -> bool:
        """Refresh working/idle status for all sessions. Returns True if anything changed."""
        changed = False
        for session in self.sessions.values():
            try:
                session_changed = session.refresh_status()
                if session_changed:
                    state = "Working" if session.busy else "idle"
                    self._maybe_memory_save_session_state(session, state, "status_refresh")
                changed = session_changed or changed
            except Exception:
                logging.exception("Could not refresh terminal session status for %s", session.id)
        return changed

    def _create_terminal_session(self, switch_current: bool = False) -> TerminalSession:
        """Create and start a new PTY terminal session."""
        # Allocate a stable session_N id even if older sessions were closed.
        while f"session_{self.next_session_index}" in self.sessions:
            self.next_session_index += 1
        session_id = f"session_{self.next_session_index}"
        self.next_session_index += 1

        base_session = self.get_session() or next(iter(self.sessions.values()))
        base_cwd = str(base_session.terminal.current_working_directory())
        terminal = PtyTerminal(
            shell=base_session.terminal.shell,
            cwd=base_cwd,
            rows=base_session.terminal.rows,
            cols=base_session.terminal.cols,
        )
        terminal.session_id = session_id
        terminal.start(self.hub)
        session = TerminalSession(id=session_id, terminal=terminal, busy=False, created_at=time.time())
        self.sessions[session_id] = session
        self.session_locks[session_id] = asyncio.Lock()
        if switch_current:
            self._set_current_session(session_id)
        return session

    def _best_idle_session(self) -> Optional[TerminalSession]:
        """Return an idle session, preferring the current one if it's idle."""
        self._refresh_session_states()
        current = self.get_session()
        if current and not current.busy:
            return current
        for session in self.sessions.values():
            try:
                session.refresh_status()
            except Exception:
                continue
            if not session.busy:
                return session
        return current

    async def _try_acquire_session_lock(self, session_id: str) -> Tuple[asyncio.Lock, bool]:
        lock = self.session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self.session_locks[session_id] = lock
        if lock.locked():
            return lock, False
        await lock.acquire()
        return lock, True

    def _session_working_result(self, session: TerminalSession, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "ok": False,
            **(meta or {}),
            "error": f"Sesja {session.id} jest w stanie Working; użyj jawnego session_id do sterowania uruchomioną aplikacją albo wybierz sesję idle dla nowych poleceń.",
            "error_code": "session_working",
            "busy": True,
            "working": True,
            "state": "working",
            "prompt_idle": bool(session.last_prompt_idle),
            "active_app": session.active_app,
            "active_command": session.active_command,
            "retryable": True,
            "recovery_hint": "Call terminal_list_sessions. Use this session_id only to interact with its active app, or choose/create an idle session for unrelated shell commands.",
        }

    async def _acquire_available_session(
        self,
        session: TerminalSession,
        *,
        allow_busy_interrupt: bool = False,
        allow_working_interaction: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[asyncio.Lock], Optional[Dict[str, Any]]]:
        lock, acquired = await self._try_acquire_session_lock(session.id)
        if not acquired:
            session.refresh_status()
            return None, self._session_working_result(session, meta)
        try:
            session.refresh_status()
            if session.busy and not (allow_busy_interrupt or allow_working_interaction):
                return_error = self._session_working_result(session, meta)
                lock.release()
                return None, return_error
            return lock, None
        except Exception:
            lock.release()
            raise

    def _maybe_memory_save_session_state(self, session: TerminalSession, state: str, action: str) -> None:
        if not self.cfg.memory_enabled or not self.memory:
            return
        app = session.active_app or "unknown"
        command = session.active_command or app
        text = f"Sesja {session.id} jest teraz {state} po wykonaniu akcji {action}. Aktywna aplikacja: {app}. Komenda/proces: {command}."
        try:
            self.memory.save(text, tags=["session_state", session.id, app], importance=3, source="session_state")
        except Exception:
            logging.exception("Could not save session state memory for %s", session.id)

    def _mark_terminal_verification_required(self, session: TerminalSession, action: str, detail: str = "") -> None:
        self.pending_terminal_verifications[session.id] = {
            "session_id": session.id,
            "action": action,
            "detail": detail,
            "active_app": session.active_app,
            "active_command": session.active_command,
            "created_at": time.time(),
        }

    def _mark_terminal_verified(self, session_id: Optional[str], verifier: str) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        item = self.pending_terminal_verifications.pop(sid, None)
        if item and self.cfg.memory_enabled and self.memory:
            try:
                self.memory.save(
                    f"Sesja {sid}: wynik akcji {item.get('action')} został sprawdzony przez {verifier}.",
                    tags=["session_state", sid, "verified"],
                    importance=3,
                    source="session_state",
                )
            except Exception:
                logging.exception("Could not save terminal verification memory for %s", sid)

    def _finish_blocked_by_unverified_terminal(self) -> Optional[Dict[str, Any]]:
        if not self.pending_terminal_verifications:
            return None
        pending = sorted(self.pending_terminal_verifications.values(), key=lambda item: float(item.get("created_at", 0)))
        return {
            "ok": False,
            "error": "Nie można zakończyć zadania od razu po akcji terminalowej bez sprawdzenia jej wyniku.",
            "error_code": "verification_required",
            "retryable": True,
            "pending_verifications": pending,
            "recovery_hint": "Najpierw wywołaj terminal_read dla wskazanej sesji i oceń wynik. Jeśli proces nadal działa, użyj sleep i terminal_read ponownie albo terminal_list_sessions, zanim użyjesz finish_task.",
        }

    def _task_blocked_by(self, task_id: Any) -> Optional[List[str]]:
        tid = str(task_id or "").strip()
        if not tid:
            return None
        task = self.tasks.get(tid)
        if not task or task.get("done"):
            return None
        deps = list(task.get("depends_on") or [])
        missing = [d for d in deps if not bool(self.tasks.get(d, {}).get("done"))]
        return missing or None

    def _target_session_for_tool(self, args: Dict[str, Any], auto_create_if_busy: bool = False) -> Tuple[Optional[TerminalSession], Dict[str, Any]]:
        """Resolve target session for a terminal/file tool.

        If no explicit session_id is provided and the current session is Working,
        auto_create_if_busy can create a new session for the operation. This
        is used for terminal_send_text / terminal_send_keys so the agent can
        launch unrelated shell commands without disturbing an active app.
        """
        explicit_sid = str(args.get("session_id") or "").strip()
        if explicit_sid:
            session = self.sessions.get(explicit_sid)
            if not session:
                return None, {"ok": False, "error": f"invalid session_id: {explicit_sid}", "received_args": args}
            # Refresh from the terminal prompt before trusting the working flag.
            session.refresh_status()
            # Explicit session_id means the model intentionally targets that PTY.
            # This is allowed even if the session is Working, because interactive
            # programs may need input or CTRL_C/CTRL_D while active. Automatic
            # session creation is only applied when session_id is omitted.
            return session, {"session_id": explicit_sid, "explicit_session_id": True, "auto_created_session": False}

        session = self._best_idle_session()
        if session is None:
            return None, {"ok": False, "error": "no terminal sessions available", "received_args": args}
        # Refresh from the terminal prompt before deciding whether auto-create is needed.
        session.refresh_status()
        if auto_create_if_busy and session.busy:
            session = self._create_terminal_session(switch_current=True)
            return session, {"session_id": session.id, "explicit_session_id": False, "auto_created_session": True}
        return session, {"session_id": session.id, "explicit_session_id": False, "auto_created_session": False}

    def _sessions_context_block(self, max_chars_per_session: int = 1200) -> str:
        """Return a compact summary of all terminal sessions for model context."""
        self._refresh_session_states()
        lines = [
            "[terminal_sessions]",
            "Managed terminal sessions. Use session_id on terminal/file tools to target one session. "
            "Working means an app/command is active and can still receive app-specific input via its session_id. "
            "Prefer an idle/new session for unrelated shell commands. A session is idle when TALK sees a trailing $ or # shell prompt.",
        ]
        for sid, session in self.sessions.items():
            info = session.info(current=(sid == self.current_session_id))
            lines.append(
                f"- {sid} | current={info['current']} | state={info['state']} | working={info['working']} | active_app={info.get('active_app')} | active_command={info.get('active_command')} | prompt_idle={info.get('prompt_idle')} | pid={info['pid']} | cwd={info['cwd']}"
            )
            tail = session.terminal.read(max(200, min(max_chars_per_session, self.cfg.terminal_context_chars)))
            if tail:
                tail = self._trim_text_middle(tail, max_chars_per_session).replace("\n", "\n    ")
                lines.append(f"  tail:\n    {tail}")
            else:
                lines.append("  tail: <empty>")
        lines.append("[/terminal_sessions]")
        return "\n".join(lines)

    def _sessions_payload(self) -> Dict[str, Any]:
        """Return a websocket payload describing all terminal sessions."""
        self._refresh_session_states()
        return {
            "type": "terminal_sessions",
            "current_session_id": self.current_session_id,
            "sessions": [
                session.info(current=(sid == self.current_session_id))
                for sid, session in self.sessions.items()
            ],
        }

    async def _broadcast_sessions(self) -> None:
        """Broadcast the current terminal session list to every connected browser."""
        await self.hub.broadcast(self._sessions_payload())

    async def _send_terminal_snapshot(self, ws: WebSocket, session_id: Optional[str] = None, max_chars: int = 20_000) -> None:
        """Send one terminal session snapshot to a browser websocket."""
        session = self.get_session(session_id)
        if not session:
            await ws.send_json({"type": "terminal_error", "error": f"invalid session_id: {session_id}"})
            return
        await ws.send_json({
            "type": "terminal_snapshot",
            "session_id": session.id,
            "data": session.terminal.read(max_chars),
        })

    async def _broadcast_current_terminal_snapshot(self, max_chars: int = 20_000) -> None:
        """Broadcast a snapshot of the current terminal session to all clients."""
        session = self.get_session()
        if not session:
            return
        await self.hub.broadcast({
            "type": "terminal_snapshot",
            "session_id": session.id,
            "data": session.terminal.read(max_chars),
        })


    def _use_flm_inline_tools(self) -> bool:
        mode = (self.cfg.flm_tools_mode or "auto").strip().lower()
        if mode in {"inline", "text", "raw"}:
            return self.api_provider == "openai"
        if mode in {"native", "openai", "api", "off"}:
            return False
        # Auto-detect the common local FastFlowLM server. This avoids FLM's
        # OpenAI-compatible streaming tool parser swallowing tool calls before
        # the client can execute them.
        base = (self.cfg.base_url or "").lower()
        model = (self.cfg.model or "").lower()
        return self.api_provider == "openai" and ("52625" in base or "fastflow" in base or "flm" in base or "gemma4-it" in model)

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
        self.pending_terminal_verifications.clear()
        self.tool_call_recovery_meta.clear()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        memory_count = self.memory.list_recent(1).get("total_memories", 0) if self.memory else 0
        payload = {"generation": self.generation, "message": "The LLM context has been cleared. The terminal and session memory remain unchanged.", "session_memory_items": memory_count}
        self.convo_log.write("conversation_cleared", payload)
        await self.hub.clear_llm_history()
        await self.hub.emit_llm("conversation_cleared", payload)

    def _terminal_context_block(self, max_chars: Optional[int] = None) -> str:
        limit = self.cfg.terminal_context_chars if max_chars is None else max(100, int(max_chars))
        # Include all managed sessions so the model is aware of parallel terminal state.
        if getattr(self, "sessions", None):
            return self._sessions_context_block(max_chars_per_session=max(300, min(limit, 2000)))
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
                "This is not a new command from the human; it is the current managed terminal context across all sessions. "
                "You can also see manual keystrokes/commands entered by the human and terminal outputs. "
                "If the state may be incomplete or the task concerns the terminal, use terminal_read with the relevant session_id. Use terminal_list_sessions when choosing where to run work and to see each session's state, active_app, and active_command. A Working session can still be used to interact with its active app; use idle/new sessions for unrelated shell commands. If you need to locate files or references, use file_search. If you need file contents, use file_read with an optional line range.\n"
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
        if self._use_flm_inline_tools():
            # Keep the original system prompt intact, but add a provider-specific
            # transport instruction so FLM does not swallow tool calls internally.
            out.insert(1 if out and out[0].get("role") == "system" else 0, {"role": "system", "content": FLM_INLINE_TOOL_MODE_PROMPT})
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

    @classmethod
    def _is_transient_llm_error(cls, exc: BaseException) -> bool:
        """Return True for temporary transport/backend failures that should not stop the agent.

        The OpenAI/httpx/httpcore exception classes vary across versions, so this
        deliberately checks the exception chain and class names instead of relying
        on optional imports. Context-length errors are excluded because they need
        emergency compaction, not ordinary retry.
        """
        if cls._is_context_length_error(exc):
            return False
        seen: Set[int] = set()
        cur: Optional[BaseException] = exc
        markers = (
            "readtimeout",
            "writetimeout",
            "connecttimeout",
            "pooltimeout",
            "timeout",
            "timed out",
            "api connection",
            "apiconnectionerror",
            "apitimeouterror",
            "connection reset",
            "connection aborted",
            "connection refused",
            "server disconnected",
            "remote protocol",
            "broken pipe",
            "temporarily unavailable",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            " 502",
            " 503",
            " 504",
        )
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            name = type(cur).__name__.lower()
            text = f"{name}: {cur!r}".lower()
            if any(marker in text for marker in markers):
                return True
            cur = cur.__cause__ or cur.__context__
        return False

    def _build_transient_llm_error_nudge(self, exc: BaseException, remaining: int) -> str:
        error_name = type(exc).__name__
        return (
            "[RUNTIME TRANSPORT ERROR — DO NOT STOP]\n"
            f"The previous LLM streaming request failed with a transient transport/backend error ({error_name}: {exc!r}). "
            "No assistant action from that failed stream was committed to conversation history. "
            "This is not task completion and not a reason to call finish_task.\n"
            "Continue the same task from the latest terminal snapshot and session memory. "
            "Use the active tool-calling mode for the next concrete action: terminal_read when state may be incomplete, "
            "file_search/file_read when inspecting code, file_write for precise edits, sleep for commands still running, "
            "or finish_task only when the user's task is actually complete.\n"
            f"Remaining transient LLM error retries after this notice: {remaining}."
        )

    async def _emit_request_log(self, step: int, source: str, request_messages: List[Dict[str, Any]], compacted_retry: bool = False, emergency: bool = False) -> None:
        await self.emit_llm("request", {
            "step": step,
            "source": source,
            "api_provider": self.api_provider,
            "base_url": self.cfg.base_url,
            "model": self.cfg.model,
            "stream": True,
            "tool_choice": self.cfg.tool_choice,
            "reasoning_effort": self.cfg.reasoning_effort,
            "extra_body": self._extra_body(),
            "flm_tools_mode": self.cfg.flm_tools_mode,
            "flm_inline_tools_active": self._use_flm_inline_tools(),
            "request_chars_estimate": self._messages_chars(request_messages),
            "request_message_count": len(request_messages),
            "context_budget_chars": self.cfg.max_request_chars,
            "compacted_retry": compacted_retry,
            "emergency_compaction": emergency,
            "messages_tail": request_messages[-20:],
        })

    def _extra_body(self) -> Dict[str, Any]:
        if self.api_provider == "ollama":
            return {}
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
        # Do not silently downgrade the configured step budget to 1.
        # The UI "Step" button now resumes the normal agent loop instead of
        # changing max_steps for this run and causing an early MAX STEPS stop.
        await self.run_loop(source="manual_step")

    async def run_loop(self, max_steps: Optional[int] = None, source: str = "auto") -> None:
        if self.lock.locked():
            await self.emit_agent("info", "The agent is already running; the message was added to the conversation.")
            return
        async with self.lock:
            run_generation = self.generation
            self.running = True
            self.finished = False
            steps_limit = max(1, int(self.cfg.max_steps))
            if max_steps is not None and int(max_steps) != steps_limit:
                await self.emit_llm("step_limit_override_ignored", {
                    "source": source,
                    "requested_max_steps": int(max_steps),
                    "configured_max_steps": steps_limit,
                    "reason": "Runtime step overrides are ignored so the agent cannot accidentally change max_steps and stop early.",
                })
            empty_retries_left = max(0, self.cfg.empty_retries)
            thinking_only_retries_left = max(1, self.cfg.empty_retries)
            malformed_tool_call_retries_left = max(1, self.cfg.empty_retries)
            described_action_retries_left = max(12, self.cfg.empty_retries * 4)
            non_tool_continuation_retries_left = max(8, self.cfg.empty_retries * 3)
            await self.hub.broadcast({"type": "status", "agent_running": True})
            try:
                for step in range(1, steps_limit + 1):
                    if run_generation != self.generation:
                        await self.emit_agent("info", "The previous agent run was interrupted because the conversation context was cleared.")
                        break
                    logging.info("LLM step %s/%s source=%s", step, steps_limit, source)
                    assistant_dict: Optional[Dict[str, Any]] = None
                    stream_payload: Optional[Dict[str, Any]] = None
                    context_retry_done = False
                    stream_attempt = 0
                    llm_error_retry_budget = max(0, int(self.cfg.llm_error_retries))

                    while run_generation == self.generation:
                        stream_attempt += 1
                        request_messages = self._build_request_messages(emergency=context_retry_done)
                        await self._emit_request_log(
                            step,
                            source,
                            request_messages,
                            compacted_retry=(stream_attempt > 1 or context_retry_done),
                            emergency=context_retry_done,
                        )
                        try:
                            assistant_dict, stream_payload = await self._stream_chat_completion(request_messages, step, source)
                            break
                        except Exception as exc:
                            if self._is_context_length_error(exc) and not context_retry_done:
                                context_retry_done = True
                                await self.emit_llm("context_compaction_retry", {
                                    "step": step,
                                    "source": source,
                                    "error": repr(exc),
                                    "action": "Retrying once with emergency context compaction.",
                                })
                                continue
                            if self._is_transient_llm_error(exc) and llm_error_retry_budget > 0:
                                llm_error_retry_budget -= 1
                                backoff_seconds = min(5.0, 0.5 * max(1, int(self.cfg.llm_error_retries) - llm_error_retry_budget))
                                nudge = self._build_transient_llm_error_nudge(exc, llm_error_retry_budget)
                                self.messages.append({"role": "user", "content": nudge})
                                await self.emit_llm("llm_transient_error_retry", {
                                    "step": step,
                                    "source": source,
                                    "attempt": stream_attempt,
                                    "remaining": llm_error_retry_budget,
                                    "backoff_seconds": backoff_seconds,
                                    "error": repr(exc),
                                    "nudge": nudge,
                                })
                                await asyncio.sleep(backoff_seconds)
                                continue
                            if self._is_transient_llm_error(exc):
                                nudge = self._build_transient_llm_error_nudge(exc, 0)
                                self.messages.append({"role": "user", "content": nudge})
                                await self.emit_llm("llm_transient_error_exhausted", {
                                    "step": step,
                                    "source": source,
                                    "attempt": stream_attempt,
                                    "error": repr(exc),
                                    "action": "Transient LLM error retry budget exhausted; stopping this run without marking the task complete.",
                                    "nudge": nudge,
                                })
                                break
                            raise

                    if assistant_dict is None or stream_payload is None:
                        break
                    if run_generation != self.generation:
                        await self.emit_agent("info", "Discarding a response from the old context after the conversation was cleared.")
                        break

                    native_tool_calls = assistant_dict.get("tool_calls") or []
                    content = str(assistant_dict.get("content") or "")
                    thinking = str(stream_payload.get("thinking") or "")
                    tool_calls = native_tool_calls
                    recovery_meta: Dict[str, Any] = {}
                    discipline_events: List[Dict[str, Any]] = []

                    if not tool_calls:
                        recovery_source_text = content
                        if thinking:
                            recovery_source_text = (content + "\n" + thinking).strip()
                        recovered_tool_calls, recovery_meta = self._extract_tool_calls_from_content(recovery_source_text)
                        if recovered_tool_calls:
                            tool_calls = recovered_tool_calls
                            assistant_dict["tool_calls"] = recovered_tool_calls
                            stream_payload["recovered_tool_calls"] = recovered_tool_calls
                            stream_payload["recovered_tool_calls_count"] = len(recovered_tool_calls)
                            stream_payload["tool_calls_count"] = len(recovered_tool_calls)
                            stream_payload["native_tool_calls_count"] = 0
                            stream_payload["tool_call_recovery"] = recovery_meta
                            stream_payload["content_replaced_in_history_after_recovery"] = True
                            self._remember_recovered_tool_call_meta(recovered_tool_calls, recovery_meta)
                            # Keep the raw streamed content in the UI/log payload, but
                            # do not feed the textual fake tool_call back to the model
                            # as assistant content. The history should contain one
                            # assistant tool_calls message followed by tool results.
                            assistant_dict["content"] = None
                            warning_payload = {
                                "step": step,
                                "warning": "Recovered tool_calls from assistant content/thinking via fallback parser.",
                                "native_tool_calls_count": 0,
                                "recovered_tool_calls_count": len(recovered_tool_calls),
                                "recovered_tool_calls": recovered_tool_calls,
                                "recovery_meta": recovery_meta,
                            }
                            await self.emit_llm("fake_tool_calls_recovered", warning_payload)
                            if recovery_meta.get("flm_inline_tool_calls_count") and any("missing closing tag" in str(note) for note in recovery_meta.get("notes", [])):
                                discipline_events.append(self._tool_discipline_event(
                                    kind="malformed_inline_tool_call_recovered",
                                    tool_name=str((recovered_tool_calls[0].get("function") or {}).get("name") or "unknown_tool") if recovered_tool_calls else "unknown_tool",
                                    violation="You emitted an inline tool call without the closing <tool_call|> marker. The runtime recovered and executed it, but the format was malformed.",
                                    correction="On the next inline tool action, emit exactly <|tool_call>call:tool_name{...}<tool_call|> and stop generating immediately after it.",
                                    details={"recovered_count": len(recovered_tool_calls), "recovery_meta": recovery_meta},
                                ))
                            elif not recovery_meta.get("flm_inline_tool_calls_count"):
                                discipline_events.append(self._tool_discipline_event(
                                    kind="fake_tool_call_recovered",
                                    tool_name=str((recovered_tool_calls[0].get("function") or {}).get("name") or "unknown_tool") if recovered_tool_calls else "unknown_tool",
                                    violation="You returned tool_calls as assistant content instead of API-native tool_calls. The runtime recovered it, but this is still a tool-call contract violation.",
                                    correction="On the next action, use the active tool-calling mode correctly: API-native tool_calls in normal mode, or exact FLM inline format in inline mode.",
                                    details={"recovered_count": len(recovered_tool_calls), "recovery_meta": recovery_meta},
                                ))

                    history_msg = {k: v for k, v in assistant_dict.items() if k != "thinking"}
                    if not tool_calls and history_msg.get("content") is None:
                        history_msg["content"] = ""
                    self.messages.append(history_msg)
                    await self.emit_llm("assistant", stream_payload)

                    self._remember_recovered_tool_call_meta(tool_calls, stream_payload.get("inline_tool_call_recovery") or {})
                    self._remember_recovered_tool_call_meta(tool_calls, stream_payload.get("tool_call_recovery") or {})

                    if not tool_calls:
                        no_visible_content = not content.strip()
                        has_private_thinking = bool(thinking.strip())

                        if self._looks_like_fake_tool_call_content(content) and malformed_tool_call_retries_left > 0:
                            malformed_tool_call_retries_left -= 1
                            event = self._tool_discipline_event(
                                kind="malformed_fake_tool_call",
                                tool_name="unknown_tool",
                                violation="You wrote JSON-like textual tool_calls, but the runtime could not recover executable tool_calls from it.",
                                correction="Do not stop. Re-issue the same intended action now using the active tool-calling mode. In FLM inline mode, emit exactly one <|tool_call>call:tool_name{...}<tool_call|> block.",
                                details={"content_preview": content[:1000], "recovery_meta": recovery_meta},
                            )
                            nudge = self._build_tool_discipline_feedback([event])
                            self.messages.append({"role": "user", "content": nudge})
                            await self.emit_llm("malformed_fake_tool_call_retry", {
                                "step": step,
                                "remaining": malformed_tool_call_retries_left,
                                "content_preview": content[:1000],
                                "recovery_meta": recovery_meta,
                                "nudge": nudge,
                            })
                            continue

                        if no_visible_content and has_private_thinking and thinking_only_retries_left > 0:
                            thinking_only_retries_left -= 1
                            nudge = (
                                "Your previous response contained private thinking/reasoning but no visible answer and no tool_calls. "
                                "Do not stop after analysis. Inspect the latest terminal/tool state and convert your analysis into the next valid tool call for the active mode. "
                                "Call terminal_list_sessions before selecting a terminal session, terminal_read if terminal state may be incomplete, file_search/file_read for files, memory_search for saved context, sleep if a command may still be running, or finish_task only when the task is complete."
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
                                "Use the active tool-calling mode. If you do not know what to do, call terminal_list_sessions or terminal_read; if you need files, call file_search/file_read; if you are editing code, use file_write with start_line/end_line for precise replacements when possible. "
                                "If the task is finished, call finish_task."
                            )
                            self.messages.append({"role": "user", "content": nudge})
                            await self.emit_llm("empty_retry", {"step": step, "remaining": empty_retries_left, "nudge": nudge})
                            continue

                        if self._looks_like_described_next_tool_action(content, thinking) and described_action_retries_left > 0:
                            described_action_retries_left -= 1
                            nudge = self._build_described_action_nudge(content, thinking)
                            self.messages.append({"role": "user", "content": nudge})
                            await self.emit_llm("described_action_without_tool_retry", {
                                "step": step,
                                "remaining": described_action_retries_left,
                                "content_preview": content[:1200],
                                "thinking_preview": thinking[:1200],
                                "nudge": nudge,
                            })
                            continue

                        if not self._looks_like_task_completion_content(content):
                            # Prefer continuing until max_steps rather than stopping on prose.
                            # The outer for-loop is the safety budget; assistant prose without
                            # tool_calls is treated as a recoverable stalled step.
                            if non_tool_continuation_retries_left > 0:
                                non_tool_continuation_retries_left -= 1
                            nudge = self._build_generic_continuation_nudge(content, thinking)
                            self.messages.append({"role": "user", "content": nudge})
                            await self.emit_llm("non_tool_continuation_retry", {
                                "step": step,
                                "remaining": non_tool_continuation_retries_left,
                                "content_preview": content[:1200],
                                "thinking_preview": thinking[:1200],
                                "nudge": nudge,
                                "policy": "continue_until_max_steps_unless_task_completion_is_explicit",
                            })
                            continue

                        logging.info("LLM produced no tool calls but content looks like task completion; stopping loop")
                        break

                    for tc in tool_calls:
                        discipline_events.extend(await self._execute_tool_call(tc, step))
                        if self.finished or run_generation != self.generation:
                            break
                    if discipline_events and run_generation == self.generation and not self.finished:
                        await self._append_tool_discipline_feedback(step, discipline_events)
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
        inline_tool_calls: List[Dict[str, Any]] = []
        inline_tool_meta: Dict[str, Any] = {}
        stop_stream_for_inline_tool = False
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
        try:
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
                    if not tool_acc and not inline_tool_calls:
                        # FastFlowLM/Gemma-style tool templates may stream calls inside
                        # the thought/content channel as e.g.
                        # <|tool_call>call:terminal_read{max_chars:6000}<tool_call|>.
                        # Catch the first complete inline call immediately instead of
                        # waiting for the model to continue hallucinating tool responses.
                        combined_stream_text = "\n".join(["".join(content_parts), "".join(thinking_parts)])
                        recovered_inline, recovered_meta = self._extract_tool_calls_from_content(combined_stream_text)
                        if recovered_inline and recovered_meta.get("flm_inline_tool_calls_count"):
                            inline_tool_calls = recovered_inline[:1]
                            inline_tool_meta = recovered_meta
                            finish_reason = "tool_calls"
                            stop_stream_for_inline_tool = True
                            await self.emit_llm("inline_tool_call_recovered_during_stream", {
                                "stream_id": stream_id,
                                "step": step,
                                "tool_call": inline_tool_calls[0],
                                "recovery_meta": inline_tool_meta,
                                "action": "Stopping this assistant stream now; executing the recovered tool and feeding its real result into the next LLM request.",
                            })
                            await self._close_stream_quietly(stream)
                            break
                if stop_stream_for_inline_tool:
                    break
        except Exception as exc:
            partial_content = "".join(content_parts)
            partial_thinking = "".join(thinking_parts)
            partial_tool_calls = inline_tool_calls or self._finalize_tool_calls(tool_acc)
            error_payload = {
                "stream_id": stream_id,
                "step": step,
                "finish_reason": "error",
                "chunks_seen": chunks_seen,
                "content": partial_content,
                "thinking": partial_thinking,
                "tool_calls": partial_tool_calls,
                "tool_calls_count": len(partial_tool_calls),
                "native_tool_calls_count": len(partial_tool_calls),
                "usage": usage,
                "error": repr(exc),
                "partial_stream_discarded_from_history": True,
            }
            self.convo_log.write("stream_error", error_payload)
            await self.hub.broadcast({"type": "llm_stream_delta", "stream_id": stream_id, "kind": "content", "text": f"[stream interrupted; retrying if transient] {exc!r}"})
            await self.hub.broadcast({"type": "llm_stream_done", "stream_id": stream_id, "data": error_payload})
            raise

        content = "".join(content_parts)
        thinking = "".join(thinking_parts)
        tool_calls = inline_tool_calls or self._finalize_tool_calls(tool_acc)
        assistant_dict: Dict[str, Any] = {"role": "assistant", "content": None if inline_tool_calls else (content if content else None)}
        if thinking and self.api_provider == "ollama":
            assistant_dict["thinking"] = thinking
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
            "native_tool_calls_count": 0 if inline_tool_calls else len(tool_calls),
            "inline_tool_calls_count": len(inline_tool_calls),
            "inline_tool_call_recovery": inline_tool_meta if inline_tool_calls else {},
            "stream_interrupted_for_inline_tool_call": bool(inline_tool_calls),
            "usage": usage,
        }
        self.convo_log.write("stream_done", done_payload)
        await self.hub.broadcast({"type": "llm_stream_done", "stream_id": stream_id, "data": done_payload})
        return assistant_dict, done_payload

    @staticmethod
    async def _close_stream_quietly(stream: Any) -> None:
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close is None:
            return
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            pass

    async def _open_stream(self, request_messages: List[Dict[str, Any]]) -> Any:
        if self.api_provider == "ollama":
            if self._ollama_base_url_is_openai_compat():
                await self.emit_llm("ollama_openai_compat_mode", {
                    "base_url": self.cfg.base_url,
                    "openai_base_url": self._ollama_openai_base_url(),
                    "reason": "Configured Ollama base_url already points at /v1, so TALK is using Ollama's OpenAI-compatible endpoint instead of native /api/chat.",
                })
                return await self._open_ollama_openai_compat_stream(request_messages, reason="base_url_points_to_v1")
            return self._open_ollama_stream(request_messages)
        return await self._open_openai_stream(request_messages)

    async def _open_openai_stream(self, request_messages: List[Dict[str, Any]]) -> Any:
        if self.client is None:
            raise RuntimeError("OpenAI-compatible client is not initialized for this provider")
        return await self._open_openai_compatible_stream(
            request_messages,
            client=self.client,
            extra_body=self._extra_body(),
            include_tools=not self._use_flm_inline_tools(),
            tool_choice=self.cfg.tool_choice,
            provider_label="openai",
        )

    @staticmethod
    def _openai_compatible_request_messages(request_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return messages containing only OpenAI-compatible chat fields."""
        out: List[Dict[str, Any]] = []
        for msg in request_messages:
            if not isinstance(msg, dict):
                continue
            item: Dict[str, Any] = {
                "role": str(msg.get("role") or "user"),
                "content": msg.get("content"),
            }
            if item["content"] is None and item["role"] != "assistant":
                item["content"] = ""
            if msg.get("tool_call_id"):
                item["tool_call_id"] = str(msg.get("tool_call_id"))
            if msg.get("name"):
                item["name"] = str(msg.get("name"))
            if msg.get("tool_calls"):
                item["tool_calls"] = to_jsonable(msg.get("tool_calls"))
            out.append(item)
        return out

    async def _open_openai_compatible_stream(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        client: Optional[AsyncOpenAI] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        include_tools: bool = True,
        tool_choice: Optional[str] = None,
        provider_label: str = "openai",
    ) -> Any:
        chat_client = client or AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": self._openai_compatible_request_messages(request_messages),
            "temperature": self.cfg.temperature,
            "stream": True,
        }
        if include_tools:
            kwargs["tools"] = TOOLS
        if extra_body:
            kwargs["extra_body"] = extra_body
        if include_tools and (tool_choice or "auto") != "omit":
            kwargs["tool_choice"] = tool_choice or "auto"
        try:
            return await chat_client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._is_context_length_error(exc):
                raise
            if kwargs.get("tool_choice") == "required":
                await self.emit_llm("tool_choice_retry", {
                    "provider": provider_label,
                    "reason": "backend rejected tool_choice=required",
                    "error": repr(exc),
                })
                kwargs["tool_choice"] = "auto"
                return await chat_client.chat.completions.create(**kwargs)
            raise

    def _ollama_chat_url(self) -> str:
        raw = (self.cfg.base_url or "http://127.0.0.1:11434").rstrip("/")
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            return raw + "/api/chat"
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            new_path = path[: -len("/v1")] + "/api/chat"
        elif path.endswith("/api/chat"):
            new_path = path
        elif path.endswith("/api"):
            new_path = path + "/chat"
        else:
            new_path = path + "/api/chat"
        return urlunparse(parsed._replace(path=new_path, params="", query="", fragment=""))

    def _ollama_base_url_is_openai_compat(self) -> bool:
        raw = (self.cfg.base_url or "").rstrip("/")
        parsed = urlparse(raw)
        return bool(parsed.scheme and parsed.netloc and parsed.path.rstrip("/").endswith("/v1"))

    def _ollama_openai_base_url(self) -> str:
        raw = (self.cfg.base_url or "http://127.0.0.1:11434").rstrip("/")
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            return raw.rstrip("/") + "/v1"
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            new_path = path
        elif path.endswith("/api/chat"):
            new_path = path[: -len("/api/chat")] + "/v1"
        elif path.endswith("/api"):
            new_path = path[: -len("/api")] + "/v1"
        else:
            new_path = path + "/v1"
        return urlunparse(parsed._replace(path=new_path, params="", query="", fragment=""))

    async def _open_ollama_openai_compat_stream(self, request_messages: List[Dict[str, Any]], reason: str) -> Any:
        base_url = self._ollama_openai_base_url()
        await self.emit_llm("ollama_openai_compat_fallback", {
            "reason": reason,
            "native_url": self._ollama_chat_url(),
            "openai_base_url": base_url,
            "hint": "If this also fails with 404, the service at base_url is probably not Ollama or does not expose Ollama's OpenAI-compatible /v1 API.",
        })
        return await self._open_openai_compatible_stream(
            request_messages,
            base_url=base_url,
            api_key=self.cfg.api_key or "ollama",
            extra_body={},
            include_tools=True,
            tool_choice=self.cfg.tool_choice,
            provider_label="ollama-openai-compat",
        )

    def _ollama_think_value(self) -> Any:
        effort = (self.cfg.reasoning_effort or "none").strip().lower()
        if effort == "none":
            return False
        if effort in {"low", "medium", "high"}:
            return effort
        return True

    def _ollama_request_messages(self, request_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in request_messages:
            role = str(msg.get("role") or "")
            if role == "tool":
                tool_call_id = str(msg.get("tool_call_id") or "")
                tool_name = str(msg.get("tool_name") or msg.get("name") or self.tool_call_names.get(tool_call_id) or "tool")
                out.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": self._ollama_content_text(msg.get("content")),
                })
                continue
            item: Dict[str, Any] = {"role": role or "user", "content": self._ollama_content_text(msg.get("content"))}
            if role == "assistant":
                thinking = msg.get("thinking")
                if thinking:
                    item["thinking"] = self._ollama_content_text(thinking)
                tool_calls = self._ollama_tool_calls_from_openai(msg.get("tool_calls") or [])
                if tool_calls:
                    item["tool_calls"] = tool_calls
            out.append(item)
        return out

    @staticmethod
    def _ollama_content_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(to_jsonable(value), ensure_ascii=False, default=str)

    @staticmethod
    def _ollama_tool_calls_from_openai(tool_calls: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(tool_calls, list):
            return out
        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args_obj = json.loads(raw_args) if raw_args.strip() else {}
                except Exception:
                    args_obj = {}
            elif isinstance(raw_args, dict):
                args_obj = raw_args
            else:
                args_obj = to_jsonable(raw_args)
            out.append({
                "type": "function",
                "function": {
                    "index": idx,
                    "name": str(name),
                    "arguments": args_obj,
                },
            })
        return out

    @staticmethod
    def _openai_tool_calls_from_ollama(tool_calls: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(tool_calls, list):
            return out
        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            if not name:
                continue
            args = fn.get("arguments", tc.get("arguments", {}))
            arg_text = args if isinstance(args, str) else json.dumps(to_jsonable(args or {}), ensure_ascii=False, default=str)
            function_delta = {"name": str(name), "arguments": arg_text}
            out.append({
                "index": int(fn.get("index", tc.get("index", idx)) or idx),
                "id": str(tc.get("id") or f"call_ollama_{int(time.time() * 1000)}_{idx}"),
                "type": "function",
                "function": function_delta,
            })
        return out

    def _open_ollama_stream(self, request_messages: List[Dict[str, Any]]) -> Any:
        url = self._ollama_chat_url()
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": self._ollama_request_messages(request_messages),
            "tools": TOOLS,
            "stream": True,
            "think": self._ollama_think_value(),
            "options": {"temperature": self.cfg.temperature},
        }
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        async def generator():
            final_usage: Dict[str, Any] = {}
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        if exc.response is not None and exc.response.status_code == 404:
                            await self.emit_llm("ollama_native_api_not_found", {
                                "native_url": url,
                                "status_code": 404,
                                "error": repr(exc),
                                "action": "Retrying through Ollama/OpenAI-compatible /v1/chat/completions.",
                                "likely_causes": [
                                    "base_url points to an OpenAI-compatible server, not native Ollama",
                                    "Ollama is behind a proxy that exposes /v1 but not /api/chat",
                                    "the host/port is not the Ollama service",
                                ],
                            })
                            fallback_stream = await self._open_ollama_openai_compat_stream(request_messages, reason="native_api_chat_404")
                            async for fallback_chunk in fallback_stream:
                                yield fallback_chunk
                            return
                        raise
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        message = data.get("message") or {}
                        delta: Dict[str, Any] = {}
                        if message.get("content"):
                            delta["content"] = message.get("content")
                        if message.get("thinking"):
                            delta["thinking"] = message.get("thinking")
                        tool_deltas = self._openai_tool_calls_from_ollama(message.get("tool_calls") or [])
                        if tool_deltas:
                            delta["tool_calls"] = tool_deltas
                        usage_keys = (
                            "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
                            "eval_count", "eval_duration", "done_reason",
                        )
                        for key in usage_keys:
                            if key in data:
                                final_usage[key] = data.get(key)
                        finish_reason = data.get("done_reason") if data.get("done") else None
                        if data.get("done") and not finish_reason:
                            finish_reason = "stop"
                        yield {
                            "id": f"ollama-{data.get('created_at', '')}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": data.get("model") or self.cfg.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                            "usage": final_usage if data.get("done") and final_usage else None,
                        }
        return generator()

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

        flm_inline_calls, flm_inline_meta = cls._extract_flm_inline_tool_calls(text)
        if flm_inline_calls:
            add_candidate({"tool_calls": flm_inline_calls}, "parsed FastFlowLM/Gemma inline <|tool_call> block")
            meta["flm_inline_tool_calls_count"] = len(flm_inline_calls)
            meta["flm_inline_tool_calls"] = flm_inline_calls
            meta["notes"].extend(flm_inline_meta.get("notes", []))
        elif "<|tool_call>" in text:
            meta["flm_inline_tool_calls_count"] = 0
            meta["notes"].extend(flm_inline_meta.get("notes", []))
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

    @classmethod
    def _extract_flm_inline_tool_calls(cls, text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        meta: Dict[str, Any] = {"source": "fastflowlm_inline_tool_call", "notes": [], "source_text": text, "call_recovery": {}}
        calls: List[Dict[str, Any]] = []
        if not text or "<|tool_call>" not in text:
            return calls, meta

        def append_call(name: str, raw_args_text: str, idx: int, recovered_from: str) -> None:
            if name not in TOOL_ALLOWED_ARGS:
                meta["notes"].append(f"ignored unknown inline tool: {name}")
                return
            repaired_args_text = cls._repair_flm_inline_arg_text(raw_args_text)
            parse_meta: Dict[str, Any] = {"parse_notes": [], "ignored_args": {}}
            parsed_args = ToolArgParser._parse_any(repaired_args_text, parse_meta)
            if not isinstance(parsed_args, dict):
                meta["notes"].append(f"inline tool {name}: arguments were not an object; using empty arguments")
                parsed_args = {}
            normalized_args, normalize_meta = ToolArgParser.parse_and_normalize(name, parsed_args)
            if (
                name == "terminal_send_text"
                and str(normalized_args.get("text") or "").strip()
                and "newline" not in parsed_args
                and recovered_from != "closed tag"
            ):
                normalized_args["newline"] = True
                normalize_meta.setdefault("parse_notes", []).append("inferred newline=true for recovered shell command")
            raw_arguments = json.dumps(normalized_args, ensure_ascii=False, default=str)
            call_id = f"flm_inline_call_{int(time.time() * 1000)}_{idx}"
            calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": raw_arguments},
            })
            notes = parse_meta.get("parse_notes", []) + normalize_meta.get("parse_notes", [])
            if notes:
                meta["notes"].append(f"inline tool {name} ({recovered_from}): " + "; ".join(str(n) for n in notes))
            else:
                meta["notes"].append(f"recovered inline tool {name} ({recovered_from})")
            meta.setdefault("call_recovery", {})[call_id] = {
                "source": "fastflowlm_inline_tool_call",
                "source_text": text,
                "tool_name": name,
                "raw_inline_args": raw_args_text,
                "normalized_args": normalized_args,
                "recovered_from": recovered_from,
                "recovery_notes": notes,
            }

        pattern = re.compile(
            r"<\|tool_call\>\s*call:([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*?\})\s*<tool_call\|>",
            flags=re.DOTALL,
        )
        for idx, match in enumerate(pattern.finditer(text)):
            name = match.group(1).strip()
            raw_args_text = match.group(2).strip()
            append_call(name, raw_args_text, idx, "closed tag")
        closed_count = len(calls)
        open_pattern = re.compile(r"<\|tool_call\>\s*call:([A-Za-z_][A-Za-z0-9_]*)", flags=re.DOTALL)
        for match in open_pattern.finditer(text):
            name = match.group(1).strip()
            if name not in TOOL_ALLOWED_ARGS:
                meta["notes"].append(f"open inline tool {name}: waiting for complete/known tool name")
                continue
            suffix = text[match.end():]
            next_open_idx = suffix.find("<|tool_call>")
            close_idx = suffix.find("<tool_call|>")
            if close_idx >= 0 and (next_open_idx < 0 or close_idx < next_open_idx):
                continue
            objects = cls._extract_tool_argument_objects_from_text(suffix, name, limit=1)
            if not objects:
                if name == "terminal_send_text":
                    probe_meta = {"source_text": text, "raw_inline_args": suffix.strip()}
                    candidates = cls._recover_terminal_send_text_candidates(probe_meta, suffix)
                    if candidates:
                        append_call(name, suffix.strip(), closed_count + len(calls), "missing argument object")
                    else:
                        meta["notes"].append("open inline terminal_send_text: missing JSON arguments and no unambiguous command text; not executed")
                    continue
                meta["notes"].append(f"open inline tool {name}: no recoverable JSON object")
                continue
            append_call(name, objects[0], closed_count + len(calls), "missing closing tag")
        meta["count"] = len(calls)
        return calls, meta

    @staticmethod
    def _repair_flm_inline_arg_text(text: str) -> str:
        # Gemma/FLM templates encode string quotes in tool args as <|"|>-like
        # tokens; the most common literal in raw output is <|"|>. Convert these
        # back before the tolerant JSON/Python argument parser runs.
        repaired = (text or "").replace('<|"|>', '"')
        repaired = repaired.replace("<|'|>", "'")
        repaired = repaired.replace("<|\"|>", '"')
        return repaired

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

    @classmethod
    def _extract_tool_argument_objects_from_text(cls, text: str, tool_name: str, limit: int = 20) -> List[str]:
        objects = cls._extract_json_objects_from_text(text, limit=max(limit, 10))
        allowed = TOOL_ALLOWED_ARGS.get(tool_name) or set()
        prefix = cls._extract_partial_object_prefix(text, allowed)
        if prefix and prefix not in objects:
            objects.append(prefix)
        scored: List[Tuple[int, int, str]] = []
        for obj in objects:
            parse_meta: Dict[str, Any] = {"parse_notes": [], "ignored_args": {}}
            parsed = ToolArgParser._parse_any(obj, parse_meta)
            key_score = 0
            if isinstance(parsed, dict):
                key_score = sum(1 for key in parsed if key in allowed)
            scored.append((key_score, len(obj), obj))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [obj for key_score, _length, obj in scored if key_score > 0][:limit] or [obj for _key_score, _length, obj in scored[:limit]]

    @staticmethod
    def _extract_partial_object_prefix(text: str, allowed_keys: Optional[Set[str]] = None) -> Optional[str]:
        start = text.find("{")
        if start < 0:
            return None
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
            if ch == ",":
                candidate = text[start:i].rstrip() + "}"
                parse_meta: Dict[str, Any] = {"parse_notes": [], "ignored_args": {}}
                parsed = ToolArgParser._parse_any(candidate, parse_meta)
                if isinstance(parsed, dict):
                    keys = set(parsed.keys())
                    if not allowed_keys or keys.intersection(allowed_keys):
                        return candidate
        return None

    @staticmethod
    def _looks_like_fake_tool_call_content(content: str) -> bool:
        text = (content or "").lower()
        if not text.strip():
            return False
        if "tool_calls" in text or "function_calls" in text or "toolcalls" in text or "<|tool_call>" in text:
            return True
        return any(f'"{name.lower()}"' in text or f"'{name.lower()}'" in text for name in TOOL_ALLOWED_ARGS)

    @staticmethod
    def _looks_like_task_completion_content(content: str) -> bool:
        """Return True only when visible content clearly says the task is complete."""
        text = (content or "").strip().lower()
        if not text:
            return False
        completion_markers = (
            "task complete", "completed the task", "all done", "done.", "finished.",
            "zadanie zakończone", "zadanie zakonczone", "ukończone", "ukonczone",
            "przegląd zakończony", "przeglad zakonczony", "raport końcowy", "raport koncowy",
            "final report", "finish_task",
        )
        return any(marker in text for marker in completion_markers)

    @staticmethod
    def _extract_described_command(content: str) -> Optional[str]:
        """Extract a COMMAND block from model prose, if present."""
        text = content or ""
        patterns = (
            r"(?ims)^\s*(?:COMMAND|KOMENDA)\s*:\s*\[([^\]]{1,1000})\]",
            r"(?ims)^\s*(?:COMMAND|KOMENDA)\s*:\s*`([^`]{1,1000})`",
            r"(?ims)^\s*(?:COMMAND|KOMENDA)\s*:\s*([^\n]{1,1000})",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            command = re.sub(r"\s+", " ", match.group(1)).strip()
            command = command.strip("[]` ")
            if command:
                return command[:1000]
        return None

    @classmethod
    def _looks_like_described_next_tool_action(cls, content: str, thinking: str = "") -> bool:
        """Detect responses that describe a next action instead of executing a tool.

        This catches natural language like "I will now run ..." and structured
        ACTION/COMMAND templates. Such responses are not task completion; they
        are a stalled tool step and should be converted into a native tool call.
        """
        visible = content or ""
        combined = f"{content or ''}\n{thinking or ''}".strip()
        text = combined.lower()
        if not text:
            return False
        if "<|tool_call>" in text or "tool_calls" in text:
            return False
        if cls._looks_like_task_completion_content(visible):
            return False

        # Structured action plans from some models, e.g.:
        # ACTION: ... COMMAND: [cd directory]
        if cls._extract_described_command(visible):
            return True
        if re.search(r"(?im)^\s*(action|akcja)\s*:", visible) and re.search(r"(?im)^\s*(command|komenda)\s*:", visible):
            return True

        future_action_patterns = (
            r"\bi will now\b", r"\bi will use\b", r"\bi will read\b", r"\bi will search\b",
            r"\bi will inspect\b", r"\bi will check\b", r"\bi will run\b", r"\bi will execute\b",
            r"\bi should now\b", r"\bnext step\b", r"\bnext logical step\b",
            r"\bi need to read\b", r"\bi need to search\b", r"\bi need to inspect\b", r"\bi need to run\b",
            r"\bi'll now\b", r"\bi’ll now\b", r"\bi am going to\b", r"\bi'm going to\b", r"\bi’m going to\b",
            r"\bteraz przeczytam\b", r"\bteraz sprawdzę\b", r"\bteraz sprawdze\b", r"\bteraz uruchomię\b", r"\bteraz uruchomie\b",
            r"\bnastępny krok\b", r"\bnastepny krok\b", r"\bmuszę przeczytać\b", r"\bmusze przeczytac\b", r"\bmuszę uruchomić\b", r"\bmusze uruchomic\b",
        )
        action_words = (
            "read", "search", "inspect", "check", "run", "execute", "list", "grep", "command",
            "file_read", "file_search", "terminal_read", "terminal_send_text", "terminal_send_keys",
            "przeczyt", "sprawdz", "wyszuk", "uruchom", "komend", "polecen",
        )
        command_line_hint = bool(re.search(r"(?m)^\s*(?:cd|ls|pwd|cat|sed|grep|find|python|python3|pytest|npm|nmap|curl|git|mkdir|touch)\b", combined))
        path_hint = bool(re.search(r'[`\'\"]?[\w./-]+\.(?:php|py|js|ts|json|yml|yaml|md|txt|sh|ini|env|lock|xml|html|css)[`\'\"]?', text))
        has_future_action = any(re.search(p, text) for p in future_action_patterns)
        has_action_word = any(word in text for word in action_words)
        return has_future_action and (has_action_word or path_hint or command_line_hint)

    def _build_described_action_nudge(self, content: str, thinking: str = "") -> str:
        preview = self._trim_text_middle((content or thinking or "").strip(), 1400)
        command = self._extract_described_command(content or "")
        command_hint = ""
        if command:
            command_hint = (
                "\nYou included this COMMAND block but did not execute it:\n"
                f"{command}\n"
                "Execute that exact command now with terminal_send_text in the correct session.\n"
            )
        if self._use_flm_inline_tools():
            example_command = command or "ls -lia"
            return (
                "[runtime_continuation_required]\n"
                "Your previous response described the next step but did not execute it. This is not task completion. "
                "In FLM inline tool mode, do not write ACTION/COMMAND prose and stop. Emit exactly ONE inline tool call now, then stop generating.\n"
                f"{command_hint}"
                "Use the action you just described. Examples:\n"
                f"<|tool_call>call:terminal_send_text{{\"text\":\"{example_command}\",\"newline\":true}}<tool_call|>\n"
                "<|tool_call>call:file_read{\"path\":\"src/app.py\",\"start_line\":1,\"end_line\":200}<tool_call|>\n"
                "<|tool_call>call:terminal_read{\"max_chars\":6000}<tool_call|>\n"
                "Call finish_task only after the user's task is actually complete, not after a plan.\n"
                f"Previous non-executed response preview:\n{preview}\n"
                "[/runtime_continuation_required]"
            )
        return (
            "Your previous response described the next step but did not execute it. This is not task completion. "
            "Do not output ACTION/COMMAND prose without tool_calls. Use the active tool-calling mode now for the action you described, "
            "or call finish_task only if the user's task is actually complete."
            f"{command_hint}\nPrevious non-executed response preview:\n{preview}"
        )

    def _build_generic_continuation_nudge(self, content: str, thinking: str = "") -> str:
        preview = self._trim_text_middle((content or thinking or "").strip(), 1200)
        if self._use_flm_inline_tools():
            return (
                "[runtime_continue_long_running_task]\n"
                "Your previous assistant message had no executable tool_calls and did not clearly finish the task. "
                "Continue working. Emit exactly ONE inline tool call now. Prefer terminal_read, terminal_list_sessions, "
                "terminal_send_text, file_search, file_read, sleep, or finish_task only when actually complete.\n"
                f"Previous message preview:\n{preview}\n"
                "[/runtime_continue_long_running_task]"
            )
        return (
            "Your previous assistant message had no executable tool_calls and did not clearly finish the task. "
            "Continue working with the active tool-calling mode now. Prefer terminal_list_sessions, terminal_read, terminal_send_text, "
            "file_search, file_read, sleep, or finish_task only when the task is actually complete.\n"
            f"Previous message preview:\n{preview}"
        )

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

    @staticmethod
    def _tool_schema_hint(tool_name: str) -> str:
        examples = {
            "terminal_read": '{"session_id":"session_2","max_chars":6000}',
            "terminal_send_text": '{"session_id":"session_2","text":"ls -la","newline":true}',
            "terminal_send_keys": '{"session_id":"session_2","keys":["CTRL_C"]}',
            "terminal_resize": '{"session_id":"session_2","cols":120,"rows":30}',
            "terminal_create_session": '{}',
            "terminal_list_sessions": '{}',
            "terminal_switch_session": '{"session_id":"session_2"}',
            "terminal_close_session": '{"session_id":"session_2"}',
            "sleep": '{"seconds":1}',
            "file_search": '{"query":"TODO","path":".","search_filenames":true,"search_contents":true,"max_results":50}',
            "file_read": '{"path":"src/app.py","start_line":1,"end_line":80}',
            "file_write": '{"path":"src/app.py","start_line":42,"end_line":47,"content":"replacement\\n"}',
            "memory_save": '{"text":"Concise task-relevant fact.","tags":["repo"],"importance":4}',
            "memory_search": '{"query":"repository root","max_items":5}',
            "memory_list": '{"max_items":10}',
            "memory_forget": '{"memory_id":"mem_0001"}',
            "finish_task": '{"summary":"I completed the task."}',
        }
        allowed = sorted(TOOL_ALLOWED_ARGS.get(tool_name, []))
        allowed_text = ", ".join(allowed) if allowed else "no known arguments"
        example = examples.get(tool_name, "one JSON object with only schema-valid keys")
        return f"Allowed keys for {tool_name or 'the selected tool'}: {allowed_text}. Correct argument example: {example}."

    @staticmethod
    def _tool_discipline_event(kind: str, tool_name: str, violation: str, correction: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "kind": kind,
            "tool_name": tool_name or "unknown_tool",
            "violation": violation,
            "correction": correction,
            "details": to_jsonable(details or {}),
        }

    def _build_tool_discipline_feedback(self, events: List[Dict[str, Any]]) -> str:
        lines = [
            "[tool_discipline]",
            "CORRECTIVE TOOL-CALL FEEDBACK.",
            "The runtime may repair and execute recoverable tool-call mistakes, but the next tool call should use the correct format.",
            "Use the active tool mode: API-native tool_calls in normal mode, or the exact FLM inline format when FLM inline mode is active.",
            "For the next action, correct the behavior immediately while continuing the task.",
        ]
        for idx, event in enumerate(events, start=1):
            tool_name = str(event.get("tool_name") or "unknown_tool")
            lines.append(f"Violation {idx}: {event.get('violation')}")
            lines.append(f"Correction {idx}: {event.get('correction')}")
            lines.append(f"Schema {idx}: {self._tool_schema_hint(tool_name)}")
            details = event.get("details") or {}
            if details:
                lines.append(f"Details {idx}: {json.dumps(to_jsonable(details), ensure_ascii=False, default=str)[:1200]}")
        lines.extend([
            "Hard rules now:",
            "- In FLM inline mode, emit exactly one <|tool_call>call:tool_name{...}<tool_call|> block and stop.",
            "- In normal tool mode, use API-native tool_calls and keep assistant content empty/null for tool calls.",
            "- Tool arguments must be exactly one valid JSON object, with quoted keys, JSON booleans/null, and no comments or debug/truncation artifacts.",
            "- Do not use XML/HTML parameter tags such as <param_value>, <text_value>, <newline_param_value>, or closing tag fragments.",
            "- terminal_send_text requires a non-empty text argument; put the command inside the JSON text value.",
            "- A bare tool prefix such as <|tool_call>call:terminal_send_text is incomplete; use terminal_list_sessions/terminal_read until you have a complete command.",
            "- Do not include unknown keys. Do not include request-truncation notes or parser artifacts.",
            "- Required arguments must be present and non-empty.",
            "- Do not call finish_task after terminal actions until terminal_read verifies the resulting output.",
            "- If unsure, call terminal_list_sessions, terminal_read, or file_search using the active tool mode; if complete, call finish_task.",
            "[/tool_discipline]",
        ])
        return "\n".join(lines)

    async def _append_tool_discipline_feedback(self, step: int, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        # Deduplicate very similar events so the model gets one strong correction
        # instead of many noisy copies after multi-tool responses.
        unique: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for event in events:
            key = json.dumps({
                "kind": event.get("kind"),
                "tool_name": event.get("tool_name"),
                "violation": event.get("violation"),
            }, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            unique.append(event)
        feedback = self._build_tool_discipline_feedback(unique)
        self.messages.append({"role": "user", "content": feedback})
        await self.emit_llm("tool_discipline_feedback", {
            "step": step,
            "count": len(unique),
            "events": unique,
            "feedback": feedback,
        })

    def _remember_recovered_tool_call_meta(self, tool_calls: List[Dict[str, Any]], recovery_meta: Dict[str, Any]) -> None:
        call_recovery = recovery_meta.get("call_recovery") if isinstance(recovery_meta, dict) else None
        if not isinstance(call_recovery, dict):
            return
        for tc in tool_calls or []:
            call_id = str(tc.get("id") or "")
            if call_id and isinstance(call_recovery.get(call_id), dict):
                self.tool_call_recovery_meta[call_id] = dict(call_recovery[call_id])

    @staticmethod
    def _looks_like_shell_command_text(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        if re.match(r"(?i)^(terminal session|the terminal|i need|i will|let me|next i|now i|assistant|thinking)\b", stripped):
            return False
        if "\n" in stripped:
            return True
        first = stripped.split(None, 1)[0]
        if first[:1].isupper() and not any(ch in first for ch in "./:-_"):
            return False
        return bool(re.fullmatch(r"[A-Za-z0-9_./:-]+", first)) and not stripped.startswith(("{", "[", "<|"))

    @classmethod
    def _extract_bare_terminal_command_from_inline_tail(cls, text: str) -> Optional[str]:
        value = str(text or "")
        if not value.strip():
            return None
        match = re.search(r"<\|tool_call\>\s*call:terminal_send_text", value, flags=re.DOTALL)
        if match:
            value = value[match.end():]
        for marker in ("<tool_call|>", "<|tool_call>"):
            marker_idx = value.find(marker)
            if marker_idx >= 0:
                value = value[:marker_idx]
        value = value.strip()
        if not value or value.startswith(("{", "[", '"', "'")):
            return None
        line = next((line.strip() for line in value.splitlines() if line.strip()), "")
        return line if cls._looks_like_shell_command_text(line) else None

    @classmethod
    def _recover_terminal_send_text_candidates(cls, recovery_meta: Dict[str, Any], raw_args: Any) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        def add(text: Any, source: str, newline: Optional[bool] = None) -> None:
            value = str(text or "").strip()
            if not value:
                return
            for existing in candidates:
                if existing.get("text") == value:
                    existing["source"] = f"{existing.get('source')},{source}"
                    if newline is not None and "newline" not in existing:
                        existing["newline"] = bool(newline)
                    return
            item = {"text": value, "source": source}
            if newline is not None:
                item["newline"] = bool(newline)
            if item not in candidates:
                candidates.append(item)

        for source_name, source_value in (
            ("raw_arguments", raw_args),
            ("raw_inline_args", recovery_meta.get("raw_inline_args")),
        ):
            if source_value is None:
                continue
            parsed, _candidate_meta = ToolArgParser.parse_and_normalize("terminal_send_text", source_value)
            if isinstance(parsed, dict) and str(parsed.get("text") or "").strip():
                add(parsed.get("text"), source_name, parsed.get("newline") if "newline" in parsed else None)

        for source_name, source_value in (
            ("source_text", recovery_meta.get("source_text")),
            ("raw_inline_args", recovery_meta.get("raw_inline_args")),
            ("raw_arguments", raw_args),
        ):
            text = str(source_value or "")
            if not text:
                continue
            for obj in cls._extract_tool_argument_objects_from_text(text, "terminal_send_text", limit=5):
                parsed, _candidate_meta = ToolArgParser.parse_and_normalize("terminal_send_text", obj)
                if isinstance(parsed, dict) and str(parsed.get("text") or "").strip():
                    add(parsed.get("text"), f"{source_name}:object", parsed.get("newline") if "newline" in parsed else None)
            for block in re.findall(r"```(?:bash|sh|shell|console)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
                add(block, f"{source_name}:fenced", True)
            for match in re.finditer(r"(?im)^\s*(?:COMMAND|cmd|run|execute)\s*[:=]\s*(.+)$", text):
                add(match.group(1), f"{source_name}:command_line", True)
            bare_command = cls._extract_bare_terminal_command_from_inline_tail(text)
            if bare_command:
                add(bare_command, f"{source_name}:inline_tail", True)

        return [item for item in candidates if cls._looks_like_shell_command_text(str(item.get("text") or ""))]

    async def _repair_tool_args_before_execution(
        self,
        name: str,
        args: Dict[str, Any],
        raw_args: Any,
        tc: Dict[str, Any],
        parse_meta: Dict[str, Any],
        step: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if name != "terminal_send_text" or str(args.get("text") or "").strip():
            return args, parse_meta
        call_id = str(tc.get("id") or "")
        recovery_meta = self.tool_call_recovery_meta.get(call_id, {})
        candidates = self._recover_terminal_send_text_candidates(recovery_meta, raw_args)
        if len(candidates) != 1:
            if candidates:
                parse_meta.setdefault("parse_notes", []).append(f"terminal_send_text text recovery ambiguous: {len(candidates)} candidates")
            return args, parse_meta
        repaired = dict(args)
        candidate = candidates[0]
        repaired["text"] = str(candidate["text"])
        newline_sources = str(
            recovery_meta.get("raw_inline_args")
            if isinstance(recovery_meta, dict) and recovery_meta.get("raw_inline_args") is not None
            else raw_args
        )
        explicit_newline = bool(re.search(r"\b(newline|enter|press_enter|submit|return)\b", newline_sources))
        if "newline" in candidate and (not explicit_newline or "newline" not in repaired):
            repaired["newline"] = bool(candidate["newline"])
        if not explicit_newline and recovery_meta.get("recovered_from") in {"missing closing tag", "missing argument object"}:
            repaired["newline"] = True
        parse_meta.setdefault("parse_notes", []).append(f"recovered missing terminal_send_text.text from {candidate.get('source')}")
        await self.emit_llm("tool_call_repaired", {
            "step": step,
            "tool_name": name,
            "repair": "missing_text_recovered",
            "tool_call_id": call_id,
            "source": candidate.get("source"),
            "text_preview": repaired["text"][:300],
            "newline": repaired.get("newline"),
            "recovery_meta": recovery_meta,
        })
        return repaired, parse_meta

    async def _execute_tool_call(self, tc: Dict[str, Any], step: int) -> List[Dict[str, Any]]:
        fn = tc.get("function") or {}
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments") or "{}"
        args, parse_meta = ToolArgParser.parse_and_normalize(name, raw_args)
        args, parse_meta = await self._repair_tool_args_before_execution(name, args, raw_args, tc, parse_meta, step)
        discipline_events: List[Dict[str, Any]] = []
        if name not in TOOL_ALLOWED_ARGS:
            discipline_events.append(self._tool_discipline_event(
                kind="unknown_tool",
                tool_name=name or "unknown_tool",
                violation=f"You attempted to call an unknown tool: {name or '<empty>'}.",
                correction="Use only one of the declared tools from the system tool list, with a valid JSON argument object.",
                details={"raw_arguments": raw_args},
            ))
        # FLM/Gemma tool templates sometimes arrive through the OpenAI-compatible
        # stream as repairable argument text, e.g. {{...}} or <|"|>quoted<|"|>
        # values. If the tolerant parser produced a valid schema-only argument
        # object, accept it silently. Penalizing benign repairs here can deadlock
        # the loop after finish_task: the task is actually finished, but the model
        # receives corrective feedback and tries to continue.
        ignored_args = parse_meta.get("ignored_args") or {}
        if ignored_args:
            discipline_events.append(self._tool_discipline_event(
                kind="malformed_tool_arguments",
                tool_name=name,
                violation="Your tool arguments contained unknown/ignored keys.",
                correction="Emit exactly one clean JSON object using only the tool schema keys. Do not include aliases, comments, debug fields, or truncation artifacts.",
                details={"parse_meta": parse_meta, "raw_arguments": raw_args, "parsed_arguments": args},
            ))
        elif parse_meta.get("parse_notes"):
            await self.emit_llm("tool_argument_repair_accepted", {
                "step": step,
                "tool_name": name,
                "parse_notes": parse_meta.get("parse_notes"),
                "raw_arguments": raw_args,
                "parsed_arguments": args,
                "reason": "Arguments were repaired into a valid schema-only object, so no negative tool-discipline feedback was sent.",
            })
        tool_stream_id = await self._start_tool_execution_stream(step, tc.get("id"), name)
        capture_terminal_after = self._should_auto_capture_terminal_after_tool(name, args)
        terminal_cursor_before = None
        if capture_terminal_after and name not in {"terminal_send_text", "terminal_send_keys"}:
            target_session, _target_meta = self._target_session_for_tool(args, auto_create_if_busy=False) if name.startswith("terminal_") else (self.get_session(), {})
            terminal_cursor_before = target_session.terminal.cursor() if target_session else self.terminal.cursor()
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
                terminal_cursor_for_capture = result.get("_terminal_cursor_before", terminal_cursor_before)
                session_id_for_capture = result.get("session_id") or args.get("session_id")
                await self._attach_delayed_terminal_output(result, name, terminal_cursor_for_capture, session_id=session_id_for_capture)
                terminal_after = result.get("terminal_output_after_delay") or {}
                new_output = str(terminal_after.get("new_output") or "")
                if new_output:
                    await self._emit_stream_delta(tool_stream_id, "result", "\n[terminal output after delay]\n")
                    await self._emit_stream_text(tool_stream_id, "result", new_output)
            if parse_meta.get("ignored_args"):
                result["ignored_args"] = parse_meta["ignored_args"]
            if parse_meta.get("parse_notes"):
                result["arg_parse_notes"] = parse_meta["parse_notes"]
            if not bool(result.get("ok", True)) and result.get("error_code") in {"session_busy", "session_working"}:
                await self.emit_llm("tool_resource_working", {
                    "step": step,
                    "tool_name": name,
                    "error": result.get("error"),
                    "session_id": result.get("session_id") or args.get("session_id"),
                    "busy": result.get("busy"),
                    "working": result.get("working"),
                    "active_app": result.get("active_app"),
                    "active_command": result.get("active_command"),
                    "prompt_idle": result.get("prompt_idle"),
                    "retryable": result.get("retryable"),
                    "recovery_hint": result.get("recovery_hint"),
                })
            elif not bool(result.get("ok", True)) and result.get("error_code") == "verification_required":
                await self.emit_llm("finish_verification_required", {
                    "step": step,
                    "tool_name": name,
                    "error": result.get("error"),
                    "pending_verifications": result.get("pending_verifications"),
                    "recovery_hint": result.get("recovery_hint"),
                })
                discipline_events.append(self._tool_discipline_event(
                    kind="premature_finish_task",
                    tool_name=name,
                    violation="You tried to finish immediately after terminal action(s) without verifying their output.",
                    correction="Call terminal_read for each pending session, inspect the output, wait/read again if needed, and only then call finish_task with the verified outcome.",
                    details={
                        "error": result.get("error"),
                        "pending_verifications": result.get("pending_verifications"),
                        "recovery_hint": result.get("recovery_hint"),
                    },
                ))
            elif not bool(result.get("ok", True)) and result.get("error_code") == "missing_required_argument":
                await self.emit_llm("tool_call_incomplete", {
                    "step": step,
                    "tool_name": name,
                    "error": result.get("error"),
                    "argument": result.get("argument"),
                    "received_args": result.get("received_args"),
                    "retryable": result.get("retryable"),
                    "recovery_hint": result.get("recovery_hint"),
                })
                discipline_events.append(self._tool_discipline_event(
                    kind="missing_required_argument",
                    tool_name=name,
                    violation=f"Required argument {result.get('argument') or '<unknown>'} was missing, so the tool could not execute the intended action.",
                    correction="Re-issue the same intended action with a complete JSON argument object in the active tool mode. Do not use XML/HTML parameter tags; put command text in terminal_send_text.text.",
                    details={
                        "error": result.get("error"),
                        "args": args,
                        "received_args": result.get("received_args"),
                        "recovery_hint": result.get("recovery_hint"),
                    },
                ))
            elif not bool(result.get("ok", True)):
                discipline_events.append(self._tool_discipline_event(
                    kind="tool_result_error",
                    tool_name=name,
                    violation="The tool call returned ok=false, so the previous tool invocation was not successful.",
                    correction="Read the tool error, fix the arguments or choose the correct next tool. Verify paths and required fields before retrying.",
                    details={"error": result.get("error"), "args": args, "received_args": result.get("received_args")},
                ))
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
            if tc.get("id"):
                self.tool_call_names[str(tc.get("id"))] = name
            self.messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result, ensure_ascii=False)})
            await self._finish_tool_execution_stream(tool_stream_id, step, name, ok=bool(result.get("ok", True)), result=result)
        except Exception as exc:
            logging.exception("Tool execution stream failed for %s", name)
            result = {"ok": False, "error": repr(exc)}
            discipline_events.append(self._tool_discipline_event(
                kind="tool_execution_exception",
                tool_name=name,
                violation="The tool execution raised an exception.",
                correction="Do not repeat the same malformed call. Inspect the error and call the correct tool with valid JSON arguments.",
                details={"error": repr(exc), "args": args},
            ))
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
            if tc.get("id"):
                self.tool_call_names[str(tc.get("id"))] = name
            self.messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(result, ensure_ascii=False)})
            await self._finish_tool_execution_stream(tool_stream_id, step, name, ok=False, result=result)
        if name == "finish_task" and bool(result.get("ok", False)):
            self.finished = True
        elif name == "finish_task":
            self.finished = False
            await self.emit_llm("finish_task_rejected_not_finished", {
                "step": step,
                "error": result.get("error"),
                "error_code": result.get("error_code"),
                "recovery_hint": result.get("recovery_hint"),
                "pending_verifications": result.get("pending_verifications"),
            })
        return discipline_events

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
        parse_meta = payload.get("parse_meta") or {}
        notes = parse_meta.get("parse_notes") or []
        ignored = parse_meta.get("ignored_args") or {}
        out = [f"CALL {name}", "", "raw arguments:", str(raw_args), "", "parsed arguments:", json.dumps(to_jsonable(parsed), ensure_ascii=False, indent=2, default=str)]
        if notes:
            out.extend(["", "argument parser notes:", json.dumps(to_jsonable(notes), ensure_ascii=False, indent=2, default=str)])
        if ignored:
            out.extend(["", "ignored/invalid arguments:", json.dumps(to_jsonable(ignored), ensure_ascii=False, indent=2, default=str)])
        return "\n".join(out) + "\n"

    def _should_auto_capture_terminal_after_tool(self, name: str, args: Dict[str, Any]) -> bool:
        # These tools can change or advance the terminal state. terminal_read already returns
        # terminal text, and finish_task should not delay completion.
        if name == "terminal_send_text":
            return bool(str(args.get("text") or "").strip())
        if name in {"terminal_send_text", "terminal_send_keys", "terminal_resize", "sleep"}:
            return True
        return False

    async def _attach_delayed_terminal_output(self, result: Dict[str, Any], tool_name: str, terminal_cursor_before: Optional[int], session_id: Optional[str] = None) -> None:
        delay_ms = max(0, min(int(self.cfg.terminal_tool_delay_ms), 10_000))
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000.0)
        session = self.get_session(session_id) if session_id else self.get_session()
        terminal = session.terminal if session else self.terminal
        cursor = terminal_cursor_before if terminal_cursor_before is not None else terminal.cursor()
        new_output = terminal.read_since(cursor, self.cfg.terminal_context_chars)
        snapshot = terminal.read(self.cfg.terminal_context_chars)
        prompt_idle = False
        busy = None
        if session:
            session.refresh_status()
            prompt_idle = bool(session.last_prompt_idle)
            busy = bool(session.busy)
            state = "Working" if busy else "idle"
            self._maybe_memory_save_session_state(session, state, f"{tool_name}_after_delay")
        result["terminal_output_after_delay"] = {
            "ok": True,
            "delay_ms": delay_ms,
            "tool_name": tool_name,
            "session_id": session.id if session else session_id,
            "new_output": new_output,
            "snapshot": snapshot,
            "prompt_idle": prompt_idle,
            "busy": busy,
            "working": bool(busy),
            "state": "working" if busy else "idle",
            "active_app": session.active_app if session else None,
            "active_command": session.active_command if session else None,
        }
        await self._broadcast_sessions()

    def _resolve_tool_file_path(self, path_value: Any, session_id: Optional[str] = None) -> Tuple[Optional[Path], Dict[str, Any]]:
        raw_path = str(path_value or "").strip()
        if not raw_path:
            return None, {"ok": False, "error": "missing required argument: path"}
        session = self.get_session(session_id) if session_id else self.get_session()
        if session is None:
            return None, {"ok": False, "error": f"invalid session_id: {session_id}"}
        base_cwd = session.terminal.current_working_directory()
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base_cwd / path
        resolved = path.resolve(strict=False)
        return resolved, {
            "raw_path": raw_path,
            "session_id": session.id,
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

        session_id = args.get("session_id")
        if session_id and self.get_session(session_id) is None:
            return {"ok": False, "error": f"invalid session_id: {session_id}", "received_args": args}
        path_value = args.get("path", ".")
        root, meta = self._resolve_tool_file_path(path_value, session_id=session_id)
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
        session_id = args.get("session_id")
        if session_id and self.get_session(session_id) is None:
            return {"ok": False, "error": f"invalid session_id: {session_id}", "received_args": args}
        path, meta = self._resolve_tool_file_path(args.get("path"), session_id=session_id)
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
        session_id = args.get("session_id")
        if session_id and self.get_session(session_id) is None:
            return {"ok": False, "error": f"invalid session_id: {session_id}", "received_args": args}
        path, meta = self._resolve_tool_file_path(args.get("path"), session_id=session_id)
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
            # Multi-session terminal management.
            if name == "terminal_create_session":
                session = self._create_terminal_session(switch_current=False)
                await self._broadcast_sessions()
                return {
                    "ok": True,
                    "session_id": session.id,
                    "current_session_id": self.current_session_id,
                    "session": session.info(current=False),
                }

            if name == "terminal_list_sessions":
                self._refresh_session_states()
                return {
                    "ok": True,
                    "current_session_id": self.current_session_id,
                    "sessions": [
                        session.info(current=(sid == self.current_session_id))
                        for sid, session in self.sessions.items()
                    ],
                }

            if name == "terminal_switch_session":
                sid = str(args.get("session_id") or "").strip()
                if not sid:
                    return {"ok": False, "error": "missing required argument: session_id", "received_args": args}
                session = self._set_current_session(sid)
                if not session:
                    return {"ok": False, "error": f"invalid session_id: {sid}", "received_args": args}
                await self._broadcast_sessions()
                await self._broadcast_current_terminal_snapshot()
                return {"ok": True, "current_session_id": sid, "session": session.info(current=True)}

            if name == "terminal_close_session":
                sid = str(args.get("session_id") or "").strip()
                if not sid:
                    return {"ok": False, "error": "missing required argument: session_id", "received_args": args}
                if sid not in self.sessions:
                    return {"ok": False, "error": f"invalid session_id: {sid}", "received_args": args}
                if len(self.sessions) <= 1:
                    return {"ok": False, "error": "cannot close the only remaining terminal session", "received_args": args}
                session = self.sessions.pop(sid)
                self.session_locks.pop(sid, None)
                self.pending_terminal_verifications.pop(sid, None)
                try:
                    session.terminal.stop()
                except Exception:
                    logging.exception("Could not stop terminal session %s", sid)
                if self.current_session_id == sid:
                    self._set_current_session(next(iter(self.sessions)))
                await self._broadcast_sessions()
                await self._broadcast_current_terminal_snapshot()
                return {"ok": True, "closed_session_id": sid, "current_session_id": self.current_session_id}

            # Terminal tools.
            if name == "terminal_read":
                session, meta = self._target_session_for_tool(args, auto_create_if_busy=False)
                if not session:
                    return meta
                lock, acquired = await self._try_acquire_session_lock(session.id)
                if not acquired:
                    session.refresh_status()
                    return self._session_working_result(session, meta)
                try:
                    text = session.terminal.read(args.get("max_chars", 6000))
                    session.refresh_status()
                    state = "Working" if session.busy else "idle"
                    self._maybe_memory_save_session_state(session, state, "terminal_read")
                    self._mark_terminal_verified(session.id, "terminal_read")
                    await self._broadcast_sessions()
                    return {"ok": True, **meta, "text": text, "busy": bool(session.busy), "working": bool(session.busy), "state": "working" if session.busy else "idle", "prompt_idle": bool(session.last_prompt_idle), "active_app": session.active_app, "active_command": session.active_command}
                finally:
                    lock.release()

            if name == "terminal_send_text":
                if "text" not in args or args.get("text") is None or str(args.get("text")) == "":
                    return {
                        "ok": False,
                        "error": "missing required argument: text",
                        "error_code": "missing_required_argument",
                        "tool_name": name,
                        "argument": "text",
                        "received_args": args,
                        "retryable": True,
                        "recovery_hint": "terminal_send_text requires a complete JSON argument object with non-empty text, for example {\"session_id\":\"session_1\",\"text\":\"nmap -sV 192.168.0.32\",\"newline\":true}. If the command is not ready, call terminal_list_sessions or terminal_read instead.",
                    }
                session, meta = self._target_session_for_tool(args, auto_create_if_busy=True)
                if not session:
                    return meta
                missing = self._task_blocked_by(self.active_task_id)
                if missing:
                    return {"ok": False, **meta, "error": f"Task {self.active_task_id} is blocked by: {missing}"}
                lock, busy_error = await self._acquire_available_session(
                    session,
                    allow_working_interaction=bool(meta.get("explicit_session_id")),
                    meta=meta,
                )
                if busy_error:
                    return busy_error
                try:
                    cursor_before = session.terminal.cursor()
                    text_value = str(args["text"])
                    result = session.terminal.send_text(text_value, bool(args.get("newline", False)))
                    session.busy = True
                    session.last_prompt_idle = False
                    preview = text_value.strip().replace("\n", "\\n")[:160]
                    self._maybe_memory_save_session_state(session, "Working", f"terminal_send_text({preview})")
                    if bool(args.get("newline", False)):
                        self._mark_terminal_verification_required(session, "terminal_send_text", preview)
                    await self._broadcast_sessions()
                    return {**result, **meta, "busy": session.busy, "working": True, "state": "working", "prompt_idle": False, "active_app": session.active_app, "active_command": session.active_command, "_terminal_cursor_before": cursor_before}
                finally:
                    lock.release()

            if name == "terminal_send_keys":
                keys = args.get("keys", [])
                if not isinstance(keys, list):
                    keys = [str(keys)]
                session, meta = self._target_session_for_tool(args, auto_create_if_busy=True)
                if not session:
                    return meta
                missing = self._task_blocked_by(self.active_task_id)
                if missing:
                    return {"ok": False, **meta, "error": f"Task {self.active_task_id} is blocked by: {missing}"}
                key_list = [str(k) for k in keys]
                interruption = any(k in {"CTRL_C", "CTRL_D"} for k in key_list)
                lock, busy_error = await self._acquire_available_session(
                    session,
                    allow_busy_interrupt=interruption,
                    allow_working_interaction=bool(meta.get("explicit_session_id")),
                    meta=meta,
                )
                if busy_error:
                    return busy_error
                try:
                    cursor_before = session.terminal.cursor()
                    result = session.terminal.send_keys(key_list)
                    session.busy = True
                    session.last_prompt_idle = False
                    self._maybe_memory_save_session_state(session, "Working", f"terminal_send_keys({','.join(key_list)})")
                    self._mark_terminal_verification_required(session, "terminal_send_keys", ",".join(key_list))
                    await self._broadcast_sessions()
                    return {**result, **meta, "busy": session.busy, "working": True, "state": "working", "prompt_idle": False, "active_app": session.active_app, "active_command": session.active_command, "_terminal_cursor_before": cursor_before}
                finally:
                    lock.release()

            if name == "terminal_resize":
                session, meta = self._target_session_for_tool(args, auto_create_if_busy=False)
                if not session:
                    return meta
                lock, busy_error = await self._acquire_available_session(session, allow_working_interaction=True, meta=meta)
                if busy_error:
                    return busy_error
                try:
                    result = session.terminal.resize(int(args.get("rows", 28)), int(args.get("cols", 100)))
                    state = "Working" if session.busy else "idle"
                    self._maybe_memory_save_session_state(session, state, "terminal_resize")
                    await self._broadcast_sessions()
                    return {**result, **meta}
                finally:
                    lock.release()

            if name == "sleep":
                seconds = max(0.1, min(float(args.get("seconds", 1)), 10.0))
                await asyncio.sleep(seconds)
                self._refresh_session_states()
                await self._broadcast_sessions()
                return {"ok": True, "slept": seconds, "sessions": [session.info(current=(sid == self.current_session_id)) for sid, session in self.sessions.items()]}

            if name == "file_search":
                return self._search_files_for_tool(args)
            if name == "file_read":
                return self._read_text_file_for_tool(args)
            if name == "file_write":
                return self._write_text_file_for_tool(args)

            if name == "task_declare":
                tid = str(args.get("task_id", "")).strip()
                depends_on = args.get("depends_on") or []
                if isinstance(depends_on, str):
                    depends_on = [p.strip() for p in re.split(r"[,;]", depends_on) if p.strip()]
                if not isinstance(depends_on, list):
                    depends_on = []
                depends_on = [str(d).strip() for d in depends_on if str(d).strip()]
                note = str(args.get("note", "") or "")
                self.tasks[tid] = {"task_id": tid, "depends_on": depends_on, "note": note, "done": False, "updated_at": time.time()}
                self.active_task_id = tid
                if self.cfg.memory_enabled and self.memory:
                    self.memory.save(f"Task {tid} declared with deps={depends_on}. {note}".strip(), tags=["task", tid], importance=2, source="task_coord")
                return {"ok": True, "task": dict(self.tasks[tid]), "active_task_id": self.active_task_id, "tasks_count": len(self.tasks)}

            if name == "task_mark_done":
                tid = str(args.get("task_id", "")).strip()
                task = self.tasks.get(tid)
                if not task:
                    self.tasks[tid] = {"task_id": tid, "depends_on": [], "note": "", "done": True, "updated_at": time.time()}
                else:
                    task["done"] = True
                    task["updated_at"] = time.time()
                if self.cfg.memory_enabled and self.memory:
                    self.memory.save(f"Task {tid} marked done.", tags=["task", tid, "done"], importance=2, source="task_coord")
                if self.active_task_id == tid:
                    self.active_task_id = None
                return {"ok": True, "task": dict(self.tasks[tid]), "active_task_id": self.active_task_id, "tasks_count": len(self.tasks)}

            if name == "task_status":
                tasks = [dict(v) for v in self.tasks.values()]
                tasks.sort(key=lambda t: (bool(t.get("done")), str(t.get("task_id"))))
                return {"ok": True, "active_task_id": self.active_task_id, "tasks": tasks, "tasks_count": len(tasks)}

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
                blocked = self._finish_blocked_by_unverified_terminal()
                if blocked:
                    return blocked
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
    #terminalTabs { height: 38px; flex: 0 0 auto; display:flex; align-items:flex-end; gap:4px; padding:6px 8px 0 8px; overflow-x:auto; overflow-y:hidden; background:#0b1220; border-bottom:1px solid #374151; }
    .terminal-tab { max-width: 220px; min-width: 112px; height:32px; padding:0 10px; display:flex; align-items:center; gap:7px; border:1px solid #374151; border-bottom-color:#374151; border-radius:10px 10px 0 0; background:#1f2937; color:#cbd5e1; font-size:12px; cursor:pointer; user-select:none; }
    .terminal-tab:hover { background:#263244; color:#fff; }
    .terminal-tab.active { background:#111827; color:#fff; border-bottom-color:#111827; }
    .terminal-tab .name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .terminal-tab .status { margin-left:auto; font-size:10px; opacity:.75; }
    .terminal-tab .dot { width:8px; height:8px; border-radius:999px; background:#64748b; flex:0 0 auto; }
    .terminal-tab.idle .dot { background:#22c55e; box-shadow:0 0 6px rgba(34,197,94,.55); }
    .terminal-tab.working .dot, .terminal-tab.busy .dot { background:#f59e0b; box-shadow:0 0 8px rgba(245,158,11,.85); }
    .terminal-tab.current .name::after { content:" •"; color:#60a5fa; }
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
    .event-empty_retry, .event-assistant_empty, .event-fake_tool_calls_recovered, .event-tool_discipline_feedback, .event-malformed_fake_tool_call_retry { border-left-color: #f97316; color:#fed7aa; }
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
    <section class="left"><div id="terminalTabs" aria-label="Terminal sessions"></div><div id="terminal"></div></section>
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
    const terminalTabs = document.getElementById('terminalTabs');
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    const streams = new Map();
    let autoScroll = true;
    let sessions = [];
    let activeSessionId = 'session_1';

    llmLog.addEventListener('scroll', () => {
      autoScroll = (llmLog.scrollTop + llmLog.clientHeight) >= (llmLog.scrollHeight - 40);
    });
    function scrollBottom(force=false) { if (force || autoScroll) llmLog.scrollTop = llmLog.scrollHeight; }
    function safeSend(obj) { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }
    function sendResize() { safeSend({type:'resize', session_id: activeSessionId, cols: term.cols, rows: term.rows}); }
    function asTime(ts) { return new Date((ts || Date.now()/1000) * 1000).toLocaleTimeString(); }

    function renderTerminalTabs() {
      terminalTabs.innerHTML = '';
      for (const sess of sessions) {
        const sid = sess.session_id;
        const tab = document.createElement('button');
        tab.type = 'button';
        const isWorking = !!(sess.working ?? sess.busy);
        const statusLabel = isWorking ? 'Working' : 'idle';
        const activeApp = sess.active_app ? `\napp: ${sess.active_app}` : '';
        const activeCommand = sess.active_command ? `\ncmd: ${sess.active_command}` : '';
        tab.className = `terminal-tab${sid === activeSessionId ? ' active' : ''}${isWorking ? ' working' : ' idle'}${sess.current ? ' current' : ''}`;
        tab.title = `${sid} — ${statusLabel}${sess.prompt_idle ? ' (prompt detected)' : ''}${activeApp}${activeCommand}\n${sess.cwd || ''}`;
        tab.innerHTML = `<span class="dot"></span><span class="name">${escapeHtml(sid)}</span><span class="status">${statusLabel}</span>`;
        tab.addEventListener('click', () => {
          if (sid === activeSessionId) return;
          activeSessionId = sid;
          renderTerminalTabs();
          term.reset();
          safeSend({type:'switch_session', session_id:sid});
        });
        terminalTabs.appendChild(tab);
      }
    }

    function updateTerminalSessions(payload) {
      sessions = payload.sessions || sessions || [];
      const knownActive = sessions.some(s => s.session_id === activeSessionId);
      const serverCurrent = payload.current_session_id || (sessions[0] && sessions[0].session_id);
      // Browser tab selection is user-owned. Do not forcibly jump back to the
      // agent's current session every time the server broadcasts statuses. Only
      // switch when the active tab disappeared or no tab has been selected yet.
      if (!knownActive) {
        activeSessionId = serverCurrent || (sessions[0] && sessions[0].session_id) || activeSessionId;
        term.reset();
        safeSend({type:'request_session_snapshot', session_id:activeSessionId});
      }
      renderTerminalTabs();
    }
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
      if (ev === 'tool_discipline_feedback') return `TOOL FEEDBACK step=${d.step}: corrective feedback sent for ${d.count || 0} tool-call issue(s)`;
      if (ev === 'malformed_fake_tool_call_retry') return `TOOL RETRY step=${d.step}: malformed textual tool_calls; corrective feedback sent and retry requested`;
      if (ev === 'llm_transient_error_retry') return `LLM TRANSIENT ERROR RETRY step=${d.step}: ${d.error || ''} remaining=${d.remaining}`;
      if (ev === 'llm_transient_error_exhausted') return `LLM TRANSIENT ERROR EXHAUSTED step=${d.step}: ${d.error || ''}`;
      if (ev === 'tool_choice_retry') return `TOOL_CHOICE RETRY: backend rejected required, retrying with auto`;
      if (ev === 'conversation_cleared') return d.message || 'LLM context cleared.';
      if (ev === 'step_limit_override_ignored') return `STEP LIMIT OVERRIDE IGNORED: requested=${d.requested_max_steps}, configured=${d.configured_max_steps}`;
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
      if (msg.type === 'terminal_sessions') updateTerminalSessions(msg);
      if (msg.type === 'terminal_snapshot') {
        activeSessionId = msg.session_id || activeSessionId;
        renderTerminalTabs();
        term.reset();
        term.write(msg.data || '');
        sendResize();
      }
      if (msg.type === 'terminal') {
        const sid = msg.session_id || activeSessionId;
        if (!activeSessionId || sid === activeSessionId) term.write(msg.data);
      }
      if (msg.type === 'terminal_error') addAgentLog('error', msg.error || 'Terminal session error');
      if (msg.type === 'agent') addAgentLog(msg.level, msg.data);
      if (msg.type === 'llm_event') addLlmEvent(msg.item);
      if (msg.type === 'llm_history') { llmLog.innerHTML = ''; streams.clear(); for (const item of msg.events || []) addLlmEvent(item); }
      if (msg.type === 'llm_stream_start') streamStart(msg.data);
      if (msg.type === 'llm_stream_delta') streamDelta(msg.stream_id, msg.kind, msg.text);
      if (msg.type === 'llm_stream_done') streamDone(msg.stream_id, msg.data);
      if (msg.type === 'status') agentStatus.textContent = msg.agent_running ? 'agent pracuje' : 'agent idle';
    };

    window.addEventListener('resize', () => { fitAddon.fit(); sendResize(); });
    term.onData(data => safeSend({type:'terminal_input', session_id: activeSessionId, data}));
    term.onResize(size => safeSend({type:'resize', session_id: activeSessionId, cols:size.cols, rows:size.rows}));

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

    async def session_status_monitor() -> None:
        """Background monitor that keeps GUI tabs in sync with shell prompts."""
        while True:
            try:
                if agent._refresh_session_states():
                    await agent._broadcast_sessions()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Terminal session status monitor error")
                await asyncio.sleep(1.0)

    @app.on_event("startup")
    async def on_startup() -> None:
        hub.set_loop(asyncio.get_running_loop())
        terminal.start(hub)
        app.state.session_status_monitor_task = asyncio.create_task(session_status_monitor())
        if startup_prompt:
            await asyncio.sleep(0.5)
            await agent.add_user_message(startup_prompt, source="startup_prompt")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        task = getattr(app.state, "session_status_monitor_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Stop all managed terminal sessions, not only the primary one.
        for session in list(agent.sessions.values()):
            session.terminal.stop()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({
            "ok": True,
            "app": APP_FULL_NAME,
            "api_provider": agent.cfg.api_provider,
            "model": agent.cfg.model,
            "base_url": agent.cfg.base_url,
            "reasoning_effort": agent.cfg.reasoning_effort,
            "memory_enabled": agent.cfg.memory_enabled,
            "memory_items": agent.memory.list_recent(1).get("total_memories", 0) if agent.memory else 0,
            "current_session_id": agent.current_session_id,
            "terminal_sessions": [session.info(current=(sid == agent.current_session_id)) for sid, session in agent.sessions.items()],
        })

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        await ws.send_json(agent._sessions_payload())
        await agent._send_terminal_snapshot(ws, agent.current_session_id, max_chars=20_000)
        await ws.send_json({"type": "llm_history", "events": list(hub.llm_history)})
        await ws.send_json({"type": "status", "agent_running": agent.running})
        try:
            while True:
                msg = await ws.receive_json()
                msg_type = msg.get("type")
                if msg_type == "terminal_input":
                    data = msg.get("data", "")
                    sid = str(msg.get("session_id") or agent.current_session_id)
                    session = agent.get_session(sid)
                    if data and session:
                        session.terminal.write(str(data).encode("utf-8", errors="replace"))
                        # Human pressing Enter usually advances work in the shell/app;
                        # mark it Working until prompt/app detection refreshes it.
                        if "\r" in str(data) or "\n" in str(data):
                            session.busy = True
                            session.last_prompt_idle = False
                            await agent._broadcast_sessions()
                elif msg_type == "resize":
                    sid = str(msg.get("session_id") or agent.current_session_id)
                    session = agent.get_session(sid) or agent.get_session()
                    if session:
                        cols = int(msg.get("cols", session.terminal.cols))
                        rows = int(msg.get("rows", session.terminal.rows))
                        session.terminal.resize(rows=rows, cols=cols)
                        await agent._broadcast_sessions()
                elif msg_type == "switch_session":
                    sid = str(msg.get("session_id") or "")
                    session = agent._set_current_session(sid)
                    if session:
                        await agent._broadcast_sessions()
                        await agent._send_terminal_snapshot(ws, session.id, max_chars=20_000)
                    else:
                        await ws.send_json({"type": "terminal_error", "error": f"invalid session_id: {sid}"})
                elif msg_type == "request_session_snapshot":
                    sid = str(msg.get("session_id") or agent.current_session_id)
                    await agent._send_terminal_snapshot(ws, sid, max_chars=20_000)
                elif msg_type == "chat":
                    await agent.add_user_message(str(msg.get("text", "")), source="chat")
                elif msg_type == "agent_prompt":
                    await agent.add_user_message(str(msg.get("prompt", "")), source="web_start")
                elif msg_type == "agent_prompt_step":
                    # A prompt should always run with the configured max_steps.
                    # Previously this path forced step_limit=1, which produced
                    # MAX STEPS: 1 and made the agent stop after a single turn.
                    await agent.add_user_message(str(msg.get("prompt", "")), source="web_step")
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
    parser = argparse.ArgumentParser(description="TALK — Terminal Agent Linux Kit: a single-file terminal agent for FastFlowLM, OpenAI-compatible APIs, and native Ollama")
    parser.add_argument(
        "--api-provider",
        choices=["openai", "ollama"],
        default=os.getenv("TALK_API_PROVIDER", os.getenv("LLM_API_PROVIDER", "openai")).lower(),
        help="openai = OpenAI-compatible /v1 chat completions; ollama = native Ollama /api/chat.",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL or Ollama host/API URL. Defaults depend on --api-provider.")
    parser.add_argument("--api-key", default=None, help="API key. For local native Ollama this can be empty; for ollama.com use OLLAMA_API_KEY.")
    parser.add_argument("--model", default=None)
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
    parser.add_argument(
        "--flm-tools-mode",
        choices=["auto", "native", "inline"],
        default=os.getenv("TALK_FLM_TOOLS_MODE", "auto"),
        help="For FastFlowLM OpenAI-compatible streaming: native sends OpenAI tools; inline omits backend tools and lets TALK parse <|tool_call> blocks itself; auto enables inline for common local FLM/gemma4-it setups.",
    )
    parser.add_argument("--empty-retries", type=int, default=2, help="How many times to retry when the model returns an empty response without tool_calls. Also used as the retry budget for thinking-only responses without tool_calls, with a minimum of one thinking-only retry.")
    parser.add_argument("--llm-error-retries", type=int, default=5, help="How many times to retry transient LLM transport/backend errors such as ReadTimeout before stopping the current run without marking the task complete.")
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
        default=os.getenv("TALK_REASONING_EFFORT", os.getenv("FASTFLOWLM_REASONING_EFFORT", "high")),
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
    args.api_provider = (args.api_provider or "openai").strip().lower()
    if args.base_url is None:
        if args.api_provider == "ollama":
            args.base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        else:
            args.base_url = os.getenv("FASTFLOWLM_BASE_URL", "http://127.0.0.1:52625/v1")
    if args.api_key is None:
        if args.api_provider == "ollama":
            args.api_key = os.getenv("OLLAMA_API_KEY", "")
        else:
            args.api_key = os.getenv("FASTFLOWLM_API_KEY", "flm")
    if args.model is None:
        if args.api_provider == "ollama":
            args.model = os.getenv("OLLAMA_MODEL", os.getenv("FASTFLOWLM_MODEL", "qwen3.5:9b"))
        else:
            args.model = os.getenv("FASTFLOWLM_MODEL", "qwen3.5:9b")
    log_dir = Path(args.log_dir)
    setup_logging(log_dir)
    logging.info("Starting %s with provider=%s model=%s base_url=%s shell=%s cwd=%s reasoning_effort=%s memory_enabled=%s", APP_FULL_NAME, args.api_provider, args.model, args.base_url, args.shell, args.cwd, args.reasoning_effort, args.memory_enabled)
    if args.api_provider == "ollama":
        logging.info("Run command hint: ollama serve; ollama pull %s", shlex.quote(args.model))
    else:
        logging.info("Run command hint: flm serve %s", shlex.quote(args.model))

    hub = WebHub()
    terminal = PtyTerminal(shell=args.shell, cwd=args.cwd)
    convo_log = JsonlLogger(log_dir / "llm_conversation.jsonl")
    memory = SessionMemory(max_items=max(1, args.memory_max_items), log_path=log_dir / "session_memory.jsonl") if args.memory_enabled else None
    agent = TerminalAgent(
        AgentConfig(
            api_provider=args.api_provider,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            tool_choice=args.tool_choice,
            empty_retries=max(0, args.empty_retries),
            llm_error_retries=max(0, args.llm_error_retries),
            terminal_context_chars=max(100, min(args.terminal_context_chars, 50_000)),
            terminal_tool_delay_ms=max(0, min(args.terminal_tool_delay_ms, 10_000)),
            memory_enabled=bool(args.memory_enabled),
            memory_max_items=max(1, args.memory_max_items),
            memory_context_items=max(0, min(args.memory_context_items, 20)),
            max_request_chars=max(8000, min(args.max_request_chars, 500_000)),
            compact_keep_recent_messages=max(4, min(args.compact_keep_recent_messages, 200)),
            compact_tool_result_chars=max(500, min(args.compact_tool_result_chars, 50_000)),
            reasoning_effort=args.reasoning_effort,
            flm_tools_mode=args.flm_tools_mode,
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
