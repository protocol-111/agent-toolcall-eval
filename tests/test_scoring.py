"""Tests for tool dispatch and scoring — the parts that decide correctness."""
from __future__ import annotations

from harness.clients import ToolCallResult
from harness.tasks import TASKS, build_condition, score
from tools.tools import dispatch


def _result(name, args, emitted=True):
    return ToolCallResult(
        emitted_tool_call=emitted, tool_name=name, tool_args=args,
        raw_text="", stop_reason="tool_use", latency_s=0.0,
    )


def test_dispatch_known_tools():
    assert dispatch("add_numbers", {"a": 2, "b": 3}) == {"sum": 5}
    assert dispatch("lookup_record", {"record_id": "R-002"})["record"]["owner"] == "bravo"
    assert dispatch("format_date", {"year": 2026, "month": 11, "day": 5}) == {"iso": "2026-11-05"}


def test_dispatch_unknown_tool_is_structured_error():
    out = dispatch("nope", {})
    assert "error" in out


def test_dispatch_bad_arguments_is_caught():
    out = dispatch("add_numbers", {"a": 1})  # missing b
    assert "error" in out


def test_dispatch_coerces_string_args_from_models():
    # Models often return tool args as strings; tools must tolerate that.
    assert dispatch("format_date", {"year": "2026", "month": "11", "day": "5"}) == {"iso": "2026-11-05"}
    assert dispatch("add_numbers", {"a": "100", "b": "200"}) == {"sum": 300.0}


def test_dispatch_unparseable_arg_is_structured_error_not_crash():
    out = dispatch("format_date", {"year": "nope", "month": 1, "day": 1})
    assert "error" in out  # structured, not an exception


def test_score_fully_correct():
    task = next(t for t in TASKS if t.task_id == "add")
    s = score(task, _result(task.expected_tool, task.expected_args))
    assert s["fully_correct"] is True
    assert s["correct_tool"] and s["correct_args"]


def test_score_wrong_tool_is_not_correct():
    task = next(t for t in TASKS if t.task_id == "add")
    s = score(task, _result("format_date", {"year": 2026, "month": 1, "day": 1}))
    assert s["correct_tool"] is False
    assert s["fully_correct"] is False


def test_score_right_tool_wrong_args():
    task = next(t for t in TASKS if t.task_id == "add")
    s = score(task, _result("add_numbers", {"a": 0, "b": 0}))
    assert s["correct_tool"] is True
    assert s["correct_args"] is False
    assert s["fully_correct"] is False


def test_score_no_emission_is_not_correct():
    task = next(t for t in TASKS if t.task_id == "add")
    s = score(task, _result(None, None, emitted=False))
    assert s["emitted_tool_call"] is False
    assert s["fully_correct"] is False


def test_conditions_differ_only_in_context():
    """Isolated registers one tool; full-harness registers all and adds history."""
    task = TASKS[0]
    iso = build_condition(task, "isolated")
    full = build_condition(task, "full_harness")
    assert len(iso["tools"]) == 1
    assert len(full["tools"]) >= len(iso["tools"])
    assert len(full["messages"]) > len(iso["messages"])
    # The task request itself is identical across conditions.
    assert iso["messages"][-1] == full["messages"][-1]
