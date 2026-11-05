"""Tests for the multi-turn depth sweep and its task/tool builders."""
from __future__ import annotations

from harness.tasks import build_depth_turns, make_padded_tools, TOOL_SCHEMAS
from harness.mock_client import MockClient
from tools.tools import dispatch, TOOL_SCHEMAS as REAL


def test_build_depth_turns_length_and_shape():
    turns = build_depth_turns(25)
    assert len(turns) == 25
    for request, expected_tool in turns:
        assert isinstance(request, str) and request
        assert expected_tool in {t["name"] for t in REAL}


def test_make_padded_tools_keeps_real_and_pads():
    tools = make_padded_tools(10)
    names = {t["name"] for t in tools}
    assert len(tools) == 10
    for real in REAL:
        assert real["name"] in names          # real tools always present
    assert any(n.startswith("noop_") for n in names)  # padded with dummies


def test_make_padded_tools_never_shrinks_below_real():
    tools = make_padded_tools(1)              # fewer than the real count
    assert len(tools) >= len(REAL)            # does not drop real tools


def test_mock_loop_returns_one_obs_per_turn():
    turns = build_depth_turns(12)
    tools = make_padded_tools(5)
    obs = MockClient("m", seed=1).run_loop("sys", turns, tools, dispatch)
    assert len(obs) == 12
    assert [o.turn for o in obs] == list(range(12))


def test_mock_loop_emission_degrades_with_depth():
    """Averaged over trials, late turns emit less than early turns."""
    turns = build_depth_turns(20)
    tools = make_padded_tools(5)

    early = late = 0
    for seed in range(40):
        obs = MockClient("m", seed=seed, full_rate=0.3).run_loop("sys", turns, tools, dispatch)
        early += int(obs[0].emitted_tool_call)
        late += int(obs[-1].emitted_tool_call)
    assert early > late
