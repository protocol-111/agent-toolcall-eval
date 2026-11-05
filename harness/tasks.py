"""Tasks and the two experimental conditions.

Each task has a natural-language request and the tool call that correctly
answers it. The *same* task is presented under two conditions:

  isolated     - only the one relevant tool is registered; short system
                 prompt; no prior conversation.
  full_harness - all tools registered; long operator-style system prompt;
                 several turns of unrelated prior conversation history.

Holding the task and the correct answer constant while varying only the
surrounding context is what isolates the effect of "harness load" on
structured tool-call emission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.tools import TOOL_SCHEMAS


@dataclass
class Task:
    task_id: str
    request: str
    expected_tool: str
    expected_args: dict[str, Any]


TASKS: list[Task] = [
    Task("add", "What is 1847 plus 2963? Use the tool.", "add_numbers",
         {"a": 1847, "b": 2963}),
    Task("lookup", "Look up the owner of record R-002.", "lookup_record",
         {"record_id": "R-002"}),
    Task("date", "Give me November 5th 2026 as an ISO date.", "format_date",
         {"year": 2026, "month": 11, "day": 5}),
]

SHORT_SYSTEM = "You are a helpful assistant. Use a tool when one fits."

# A deliberately long, operator-style prompt to reproduce "full-harness" load.
LONG_SYSTEM = (
    "You are an autonomous operations agent running inside a larger "
    "orchestration harness. You have access to a set of tools. Follow the "
    "operating procedure strictly: think step by step, prefer tools over "
    "guessing, never fabricate results, cite tool outputs, and stop when the "
    "task is complete. You must respect network and permission constraints, "
    "log your reasoning, and avoid side effects. " + ("Policy note. " * 80)
)

# Unrelated prior turns, included only in the full-harness condition.
DISTRACTOR_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "Earlier, summarize our deployment status."},
    {"role": "assistant", "content": "All services nominal; no action needed."},
    {"role": "user", "content": "Thanks. Standby for the next request."},
    {"role": "assistant", "content": "Standing by."},
]


def build_condition(task: Task, condition: str) -> dict[str, Any]:
    """Return {system, messages, tools} for a task under a condition."""
    if condition == "isolated":
        tools = [t for t in TOOL_SCHEMAS if t["name"] == task.expected_tool]
        system = SHORT_SYSTEM
        messages = [{"role": "user", "content": task.request}]
    elif condition == "full_harness":
        tools = list(TOOL_SCHEMAS)
        system = LONG_SYSTEM
        messages = [*DISTRACTOR_HISTORY, {"role": "user", "content": task.request}]
    else:
        raise ValueError(f"unknown condition: {condition!r}")
    return {"system": system, "messages": messages, "tools": tools}


def score(task: Task, result) -> dict[str, Any]:
    """Score one result against the task's expected call."""
    correct_tool = result.tool_name == task.expected_tool
    correct_args = result.tool_args == task.expected_args
    return {
        "emitted_tool_call": result.emitted_tool_call,
        "correct_tool": correct_tool,
        "correct_args": correct_args,
        "fully_correct": bool(
            result.emitted_tool_call and correct_tool and correct_args
        ),
    }
