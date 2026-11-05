"""A deterministic mock client so the harness runs end-to-end with no API key.

It mimics the finding under study: high tool-call emission in the isolated
condition and degraded emission under full-harness load. It is used for smoke
tests and CI only; real runs use AnthropicClient / OpenAICompatClient.
"""
from __future__ import annotations

import random
import time
from typing import Any

from harness.clients import ToolCallResult
from harness.tasks import DISTRACTOR_HISTORY, LONG_SYSTEM


class MockClient:
    def __init__(self, model: str, seed: int = 0,
                 iso_rate: float = 0.98, full_rate: float = 0.60) -> None:
        self.model = model
        self._rng = random.Random(seed + hash(model) % 10_000)
        self._iso_rate = iso_rate
        self._full_rate = full_rate

    def run_turn(self, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]], max_tokens: int = 1024) -> ToolCallResult:
        t0 = time.perf_counter()
        is_full = system == LONG_SYSTEM or any(
            m in DISTRACTOR_HISTORY for m in messages
        )
        rate = self._full_rate if is_full else self._iso_rate
        emit = self._rng.random() < rate

        # Recover the correct call from the single registered tool (isolated)
        # or infer from the user's request text (full-harness).
        user_text = messages[-1]["content"]
        name, args = _infer_expected(user_text, tools)

        time.sleep(0.001)
        if emit:
            return ToolCallResult(
                emitted_tool_call=True, tool_name=name, tool_args=args,
                raw_text="", stop_reason="tool_use",
                latency_s=time.perf_counter() - t0,
            )
        return ToolCallResult(
            emitted_tool_call=False, tool_name=None, tool_args=None,
            raw_text="(answered in prose instead of calling the tool)",
            stop_reason="end_turn", latency_s=time.perf_counter() - t0,
        )


def _infer_expected(user_text: str, tools: list[dict[str, Any]]):
    names = {t["name"] for t in tools}
    if "plus" in user_text and "add_numbers" in names:
        return "add_numbers", {"a": 1847, "b": 2963}
    if "record" in user_text and "lookup_record" in names:
        return "lookup_record", {"record_id": "R-002"}
    if "ISO" in user_text and "format_date" in names:
        return "format_date", {"year": 2026, "month": 11, "day": 5}
    # Fallback: first registered tool, empty args.
    first = next(iter(names)) if names else None
    return first, {}
