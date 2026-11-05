"""Tests for the egress-fenced execution sandbox.

Tests that need the network namespace are skipped (not failed) where rootless
`unshare -rn` is unavailable, so the suite stays green on CI runners that
disallow it while still exercising isolation where it exists.
"""
from __future__ import annotations

import pytest

from harness.sandbox import run_tool_sandboxed, verify_fence, _unshare_available

requires_ns = pytest.mark.skipif(
    not _unshare_available(),
    reason="rootless network namespace (unshare -rn) unavailable here",
)


def test_tool_runs_correctly_inside_sandbox():
    r = run_tool_sandboxed("add_numbers", {"a": 2, "b": 3}, require_isolation=False)
    assert r.ok
    assert r.result == {"sum": 5}


@requires_ns
def test_fence_blocks_outbound_and_reports_truthfully():
    fence = verify_fence()
    assert fence["net_isolation_enforced"] is True
    assert fence["outbound_blocked_verified"] is True


@requires_ns
def test_network_isolation_is_enforced_flag():
    r = run_tool_sandboxed("add_numbers", {"a": 1, "b": 1})
    assert r.net_isolation_enforced is True
    assert r.sandboxed is True


def test_fails_closed_when_isolation_required_but_unavailable(monkeypatch):
    # Force "unavailable" and require isolation -> must refuse, not run unfenced.
    monkeypatch.setattr("harness.sandbox._unshare_available", lambda: False)
    r = run_tool_sandboxed("add_numbers", {"a": 1, "b": 1}, require_isolation=True)
    assert r.ok is False
    assert r.net_isolation_enforced is False
    assert "refused" in r.stderr


def test_unknown_tool_returns_structured_error_inside_sandbox():
    r = run_tool_sandboxed("does_not_exist", {}, require_isolation=False)
    assert r.ok
    assert "error" in r.result
