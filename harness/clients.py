"""Model clients behind one small interface.

Two backends:
  * AnthropicClient   - Claude via the Anthropic Messages API (native tool use)
  * OpenAICompatClient - local models served with an OpenAI-compatible API
                         (vLLM, Ollama, or a LiteLLM proxy in front of them)

Both return a normalized ToolCallResult so the harness can score them the same
way. API keys and base URLs come from the environment at run time and are
never written to disk or committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallResult:
    """Normalized outcome of one model turn."""
    emitted_tool_call: bool           # did the model emit a well-formed tool call?
    tool_name: str | None             # name of the first tool call, if any
    tool_args: dict[str, Any] | None  # parsed arguments, if any
    raw_text: str                     # any assistant text returned
    stop_reason: str | None
    latency_s: float
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class AnthropicClient:
    def __init__(self, model: str) -> None:
        import anthropic  # imported lazily so local-only runs need no dep

        self.model = model
        # Reads ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic()

    def run_turn(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> ToolCallResult:
        import time

        t0 = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - record, don't crash the sweep
            return ToolCallResult(
                emitted_tool_call=False, tool_name=None, tool_args=None,
                raw_text="", stop_reason=None,
                latency_s=time.perf_counter() - t0, error=repr(exc),
            )
        latency = time.perf_counter() - t0

        tool_name = tool_args = None
        text_parts: list[str] = []
        for block in resp.content:
            if block.type == "tool_use" and tool_name is None:
                tool_name, tool_args = block.name, dict(block.input)
            elif block.type == "text":
                text_parts.append(block.text)

        return ToolCallResult(
            emitted_tool_call=tool_name is not None,
            tool_name=tool_name,
            tool_args=tool_args,
            raw_text="".join(text_parts),
            stop_reason=resp.stop_reason,
            latency_s=latency,
        )


class OpenAICompatClient:
    """For vLLM / Ollama / LiteLLM. Base URL and key from the environment."""

    def __init__(self, model: str) -> None:
        from openai import OpenAI  # lazy import

        self.model = model
        self._client = OpenAI(
            base_url=os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LOCAL_OPENAI_API_KEY", "not-needed"),
        )

    def run_turn(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> ToolCallResult:
        import json
        import time

        # Convert Anthropic-style tool schemas to OpenAI "function" format.
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        oai_messages = [{"role": "system", "content": system}, *messages]

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=oai_messages,
                tools=oai_tools,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolCallResult(
                emitted_tool_call=False, tool_name=None, tool_args=None,
                raw_text="", stop_reason=None,
                latency_s=time.perf_counter() - t0, error=repr(exc),
            )
        latency = time.perf_counter() - t0

        choice = resp.choices[0]
        msg = choice.message
        tool_name = tool_args = None
        if msg.tool_calls:
            call = msg.tool_calls[0]
            tool_name = call.function.name
            try:
                tool_args = json.loads(call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                # Emitted a call but arguments were not valid JSON — this is
                # itself a finding, recorded via malformed_args in extra.
                tool_args = None
                return ToolCallResult(
                    emitted_tool_call=True, tool_name=tool_name, tool_args=None,
                    raw_text=msg.content or "", stop_reason=choice.finish_reason,
                    latency_s=latency, extra={"malformed_args": call.function.arguments},
                )

        return ToolCallResult(
            emitted_tool_call=tool_name is not None,
            tool_name=tool_name,
            tool_args=tool_args,
            raw_text=msg.content or "",
            stop_reason=choice.finish_reason,
            latency_s=latency,
        )


def build_client(backend: str, model: str):
    if backend == "anthropic":
        return AnthropicClient(model)
    if backend == "openai_compat":
        return OpenAICompatClient(model)
    if backend == "mock":
        from harness.mock_client import MockClient  # lazy: avoids import cycle

        return MockClient(model)
    raise ValueError(f"unknown backend: {backend!r}")
