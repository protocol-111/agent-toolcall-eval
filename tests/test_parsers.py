"""Tests for strict vs sanitizing tool-call parsing.

The dirty inputs mirror real leaked output: control tokens (<|im_end|>,
<|tool_call_end|>), code fences, prose preambles, and a second JSON object —
the exact cases the production sanitizer logged.
"""
from __future__ import annotations

from harness.parsers import strict_parse, sanitizing_parse

CALL = '{"tool": "add_numbers", "arguments": {"a": 1, "b": 2}}'


def test_strict_accepts_clean_call():
    assert strict_parse(CALL).ok is True


def test_strict_rejects_trailing_control_token():
    # This is the production failure: strict parser sees invalid JSON.
    out = strict_parse(CALL + "<|im_end|>")
    assert out.ok is False


def test_strict_rejects_code_fence_and_prose():
    assert strict_parse("```json\n" + CALL + "\n```").ok is False
    assert strict_parse("Sure! " + CALL).ok is False


def test_sanitizer_recovers_control_token():
    out = sanitizing_parse(CALL + "<|im_end|>")
    assert out.ok is True
    assert out.call["tool"] == "add_numbers"
    assert "control_tokens" in out.reasons


def test_sanitizer_recovers_code_fence():
    out = sanitizing_parse("```json\n" + CALL + "\n```")
    assert out.ok is True
    assert "code_fence" in out.reasons


def test_sanitizer_recovers_and_flags_second_object():
    out = sanitizing_parse(CALL + '\n{"tool": "noop_0", "arguments": {}}')
    assert out.ok is True
    assert out.call["tool"] == "add_numbers"   # takes the FIRST call
    assert "second_object" in out.reasons


def test_sanitizer_flags_prose_as_extra_content():
    out = sanitizing_parse("Sure, here you go: " + CALL)
    assert out.ok is True
    assert "extra_content" in out.reasons


def test_sanitizer_reports_failure_when_no_json():
    out = sanitizing_parse("I cannot do that.")
    assert out.ok is False
    assert "no_json_object" in out.reasons


def test_both_agree_on_clean_input():
    assert strict_parse(CALL).ok is sanitizing_parse(CALL).ok is True
    assert sanitizing_parse(CALL).reasons == []   # nothing to repair


def test_balanced_braces_handles_nested_arguments():
    nested = '{"tool": "f", "arguments": {"obj": {"k": "v"}, "n": 3}}'
    out = sanitizing_parse(nested + "<|im_end|>")
    assert out.ok is True
    assert out.call["arguments"]["obj"]["k"] == "v"
