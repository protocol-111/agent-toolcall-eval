"""Multi-turn depth sweep: does structured tool-call emission hold as an
agentic conversation gets deeper and as more tools are registered?

This is the condition the single-turn study could not exercise. It runs a real
agentic loop of `--depth` turns (the model's own tool calls and their results
accumulate in context) with `--n-tools` tools registered, repeated `--trials`
times, and reports emission rate *by turn index* plus a first-failure summary.

    python -m harness.depth_eval --config configs/models.yaml \
        --depth 25 --n-tools 10 --trials 5 --out results/depth.jsonl

Tool results are fed back using in-process dispatch (fast); the egress-fence
study is separate (run_eval --sandbox).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from harness.clients import build_client
from harness.run_eval import wilson_interval
from harness.tasks import LONG_SYSTEM, build_depth_turns, make_padded_tools
from tools.tools import dispatch


def run(config_path: str, depth: int, n_tools: int, trials: int, out_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    turns = build_depth_turns(depth)
    tools = make_padded_tools(n_tools)

    rows: list[dict[str, Any]] = []
    with out.open("w") as fh:
        for m in cfg["models"]:
            client = build_client(m["backend"], m["model"])
            label = m.get("label", m["model"])
            for trial in range(trials):
                obs = client.run_loop(LONG_SYSTEM, turns, tools, dispatch)
                for o in obs:
                    row = {
                        "ts": time.time(), "model": label, "backend": m["backend"],
                        "n_tools": n_tools, "depth": depth, "trial": trial,
                        "turn": o.turn, "emitted": o.emitted_tool_call,
                        "correct_tool": o.correct_tool,
                        "latency_s": round(o.latency_s, 4), "error": o.error,
                    }
                    fh.write(json.dumps(row) + "\n")
                    rows.append(row)
                fh.flush()

    summarize(rows, out.with_suffix(".summary.json"), depth, n_tools)


def summarize(rows, out_path: Path, depth: int, n_tools: int) -> None:
    # Emission rate by (model, turn), and the mean first turn at which a
    # trial stopped emitting (the "collapse depth"), per model.
    by_mt: dict[tuple[str, int], list[int]] = defaultdict(list)
    first_fail: dict[str, list[int]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    by_trial: dict[tuple[str, int], dict[int, bool]] = defaultdict(dict)

    for r in rows:
        by_mt[(r["model"], r["turn"])].append(int(r["emitted"]))
        by_trial[(r["model"], r["trial"])][r["turn"]] = r["emitted"]
        if r["error"]:
            errors[r["model"]] += 1

    for (model, trial), turns_map in by_trial.items():
        miss = [t for t in sorted(turns_map) if not turns_map[t]]
        first_fail[model].append(miss[0] if miss else depth)  # depth = never failed

    summary = []
    for (model, turn), vals in sorted(by_mt.items()):
        lo, hi = wilson_interval(sum(vals), len(vals))
        summary.append({
            "model": model, "turn": turn, "n": len(vals),
            "emission_rate": round(sum(vals) / len(vals), 3),
            "ci95": [round(lo, 3), round(hi, 3)],
        })
    out_path.write_text(json.dumps({
        "depth": depth, "n_tools": n_tools,
        "by_turn": summary,
        "mean_first_fail_turn": {m: round(sum(v) / len(v), 2) for m, v in first_fail.items()},
        "errors": dict(errors),
    }, indent=2))

    # Console: emission at turn 0, mid, last, and mean collapse depth.
    print(f"depth={depth}  n_tools={n_tools}")
    print(f"{'model':<22}{'t0':>6}{'t'+str(depth//2):>6}{'t'+str(depth-1):>6}"
          f"{'meanFail':>10}{'errs':>6}")
    rate = {(s['model'], s['turn']): s['emission_rate'] for s in summary}
    mids = depth // 2
    for model in sorted({s["model"] for s in summary}):
        print(f"{model:<22}"
              f"{rate.get((model,0),0):>6.2f}{rate.get((model,mids),0):>6.2f}"
              f"{rate.get((model,depth-1),0):>6.2f}"
              f"{first_fail and (sum(first_fail[model])/len(first_fail[model])):>10.1f}"
              f"{errors.get(model,0):>6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--depth", type=int, default=25)
    ap.add_argument("--n-tools", type=int, default=3, dest="n_tools")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="results/depth.jsonl")
    args = ap.parse_args()
    run(args.config, args.depth, args.n_tools, args.trials, args.out)


if __name__ == "__main__":
    main()
