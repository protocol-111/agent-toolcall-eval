"""Benign, deterministic tools used for the tool-call reliability study.

Each tool is pure and side-effect-free so that results are reproducible and
the study measures *whether the model emits a well-formed structured tool
call*, not whether an external service behaves. The same schemas are used in
the isolated (single-tool) and full-harness (all-tools) conditions.
"""
from __future__ import annotations

import json
from typing import Any, Callable


# --- Tool implementations (deterministic, no I/O) ------------------------

def add_numbers(a: float, b: float) -> dict[str, Any]:
    return {"sum": a + b}


def lookup_record(record_id: str) -> dict[str, Any]:
    # Fixed in-memory table; no disk or network access.
    table = {
        "R-001": {"owner": "alpha", "status": "open"},
        "R-002": {"owner": "bravo", "status": "closed"},
        "R-003": {"owner": "charlie", "status": "open"},
    }
    return {"record": table.get(record_id, None)}


def format_date(year: int, month: int, day: int) -> dict[str, Any]:
    return {"iso": f"{year:04d}-{month:02d}-{day:02d}"}


TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "add_numbers": add_numbers,
    "lookup_record": lookup_record,
    "format_date": format_date,
}


# --- Schemas (Anthropic tool-use format) ---------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "add_numbers",
        "description": "Add two numbers and return their sum.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First addend"},
                "b": {"type": "number", "description": "Second addend"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "lookup_record",
        "description": "Look up a record by its identifier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string", "description": "e.g. R-001"},
            },
            "required": ["record_id"],
        },
    },
    {
        "name": "format_date",
        "description": "Format a calendar date as an ISO-8601 string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "month": {"type": "integer"},
                "day": {"type": "integer"},
            },
            "required": ["year", "month", "day"],
        },
    },
]


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool call and return its result (or a structured error)."""
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return impl(**arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}


if __name__ == "__main__":
    # Smoke test the tools without any model in the loop.
    print(json.dumps(dispatch("add_numbers", {"a": 2, "b": 3})))
    print(json.dumps(dispatch("lookup_record", {"record_id": "R-002"})))
    print(json.dumps(dispatch("format_date", {"year": 2026, "month": 11, "day": 5})))
