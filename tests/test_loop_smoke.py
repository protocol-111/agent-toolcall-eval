"""Exercise the real run_loop code paths with fake clients (no network).

These catch runtime bugs inside the loop (missing imports, message assembly)
that the mock client can't, because the mock overrides run_loop entirely.
"""
from __future__ import annotations

from harness.clients import AnthropicClient, OpenAICompatClient
from harness.tasks import build_depth_turns, make_padded_tools
from tools.tools import dispatch


# --- minimal fakes mimicking each SDK's response shape -------------------

class _Blk:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _AnthResp:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeAnthropicSDK:
    class messages:
        @staticmethod
        def create(**kw):
            return _AnthResp([_Blk("tool_use", name="add_numbers",
                                   input={"a": 1, "b": 2}, id="tu_1")])


class _OAIFn:
    def __init__(self):
        self.name = "add_numbers"
        self.arguments = '{"a": 1, "b": 2}'


class _OAICall:
    def __init__(self):
        self.id = "call_1"
        self.function = _OAIFn()


class _OAIMsg:
    content = ""
    tool_calls = [_OAICall()]


class _OAIChoice:
    message = _OAIMsg()


class _OAIResp:
    choices = [_OAIChoice()]


class _FakeOpenAISDK:
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                return _OAIResp()


def test_anthropic_run_loop_executes_without_error():
    c = AnthropicClient.__new__(AnthropicClient)   # bypass __init__ (no key/network)
    c.model = "fake"
    c._client = _FakeAnthropicSDK()
    turns = build_depth_turns(3)
    obs = c.run_loop("sys", turns, make_padded_tools(5), dispatch)
    assert len(obs) == 3
    assert all(o.emitted_tool_call for o in obs)   # fake always emits a tool call


def test_openai_run_loop_executes_without_error():
    c = OpenAICompatClient.__new__(OpenAICompatClient)
    c.model = "fake"
    c._client = _FakeOpenAISDK()
    turns = build_depth_turns(3)
    obs = c.run_loop("sys", turns, make_padded_tools(5), dispatch)
    assert len(obs) == 3
    assert all(o.emitted_tool_call for o in obs)
