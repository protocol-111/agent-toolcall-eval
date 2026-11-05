"""Two parsers for raw model output that is meant to be a tool call.

The production collapse (zero accepted tool calls across 25 turns) was not the
model refusing to act — it was the model emitting tool-call JSON wrapped in
stray control tokens or followed by a second JSON object, which a *strict*
harness parser rejected outright. A *sanitizing* parser recovers the intended
call. Running both over the same raw output measures the gap between "the model
intended a tool call" and "the harness accepted one".

A call is represented as {"tool": <name>, "arguments": {...}}.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Chat/control tokens various local models leak into content.
_CONTROL_TOKEN = re.compile(r"<\|[^|>]*\|>")
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9]*\n?|```")


@dataclass
class ParseOutcome:
    call: dict[str, Any] | None          # the recovered {"tool","arguments"} or None
    ok: bool                             # was a well-formed call obtained?
    reasons: list[str] = field(default_factory=list)  # what had to be repaired


def _valid_call(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("tool"), str)
        and isinstance(obj.get("arguments"), dict)
    )


def strict_parse(text: str) -> ParseOutcome:
    """Accept only if the ENTIRE output is exactly one valid call object.

    Any stray token, prose, code fence, or trailing content makes it fail —
    this models a harness that does json.loads on the raw turn output.
    """
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return ParseOutcome(None, False, ["not_pure_json"])
    if _valid_call(obj):
        return ParseOutcome(obj, True, [])
    return ParseOutcome(None, False, ["wrong_shape"])


def _first_json_object(s: str) -> tuple[str | None, bool]:
    """Return the first balanced {...} substring and whether more followed."""
    start = s.find("{")
    if start == -1:
        return None, False
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    rest = s[i + 1:].strip()
                    return s[start:i + 1], ("{" in rest)
    return None, False


def sanitizing_parse(text: str) -> ParseOutcome:
    """Recover the intended call the way a tolerant proxy would.

    Strips control tokens and code fences, extracts the first balanced JSON
    object, ignores any trailing/second object, and validates the shape.
    Records each repair it had to make.
    """
    reasons: list[str] = []
    s = text

    if _CONTROL_TOKEN.search(s):
        s = _CONTROL_TOKEN.sub("", s)
        reasons.append("control_tokens")
    if "```" in s:
        s = _CODE_FENCE.sub("", s)
        reasons.append("code_fence")

    candidate, had_more = _first_json_object(s)
    if candidate is None:
        return ParseOutcome(None, False, reasons + ["no_json_object"])
    if had_more:
        reasons.append("second_object")
    # Was there prose around the JSON (beyond whitespace)?
    if s.strip() != candidate.strip():
        reasons.append("extra_content")

    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return ParseOutcome(None, False, reasons + ["unparseable_object"])
    if _valid_call(obj):
        return ParseOutcome(obj, True, reasons)
    return ParseOutcome(None, False, reasons + ["wrong_shape"])
