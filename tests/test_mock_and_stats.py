"""Tests for the Wilson interval and the mock backend's behavior."""
from __future__ import annotations

from harness.run_eval import wilson_interval
from harness.tasks import TASKS, build_condition
from harness.mock_client import MockClient


def test_wilson_bounds_are_valid():
    for succ, n in [(0, 0), (0, 10), (10, 10), (5, 10), (1, 100)]:
        lo, hi = wilson_interval(succ, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_brackets_point_estimate():
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi


def test_mock_is_deterministic_for_seed():
    task = TASKS[0]
    built = build_condition(task, "isolated")
    a = MockClient("m", seed=1).run_turn(**built)
    b = MockClient("m", seed=1).run_turn(**built)
    assert a.emitted_tool_call == b.emitted_tool_call


def test_mock_degrades_under_full_harness():
    """Averaged over many trials, full-harness emission < isolated emission."""
    task = TASKS[0]
    iso = build_condition(task, "isolated")
    full = build_condition(task, "full_harness")

    def rate(built, n=300):
        c = MockClient("m", seed=7)
        return sum(c.run_turn(**built).emitted_tool_call for _ in range(n)) / n

    assert rate(iso) > rate(full)


def test_mock_emits_correct_call_when_it_emits():
    task = next(t for t in TASKS if t.task_id == "add")
    built = build_condition(task, "isolated")
    # Force emission with a high-rate client.
    c = MockClient("m", seed=0, iso_rate=1.0)
    r = c.run_turn(**built)
    assert r.emitted_tool_call
    assert r.tool_name == task.expected_tool
