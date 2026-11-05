"""Run the tool-call reliability sweep and write JSONL + a summary.

Usage:
    python -m harness.run_eval --config configs/models.yaml --trials 20 \
        --out results/run.jsonl

Every (model, task, condition) cell is repeated `--trials` times so that the
reported tool-call emission rate carries a confidence interval rather than
being a single observation. Nothing here requires a GPU; the local models are
reached over an OpenAI-compatible endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml

from harness.clients import build_client
from harness.sandbox import run_tool_sandboxed
from harness.tasks import TASKS, build_condition, score

CONDITIONS = ["isolated", "full_harness"]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (robust at small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run(config_path: str, trials: int, out_path: str,
        execute: bool = False, sandbox: bool = False) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    models = cfg["models"]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with out.open("w") as fh:
        for m in models:
            client = build_client(m["backend"], m["model"])
            label = m.get("label", m["model"])
            for task in TASKS:
                for condition in CONDITIONS:
                    built = build_condition(task, condition)
                    for trial in range(trials):
                        result = client.run_turn(**built)
                        scored = score(task, result)
                        row = {
                            "ts": time.time(),
                            "model": label,
                            "backend": m["backend"],
                            "task": task.task_id,
                            "condition": condition,
                            "trial": trial,
                            "latency_s": round(result.latency_s, 4),
                            "stop_reason": result.stop_reason,
                            "error": result.error,
                            **scored,
                            "extra": result.extra,
                        }
                        # Optionally execute the emitted call in the fence and
                        # record the outcome, so tool-failure behavior can be
                        # studied downstream.
                        if execute and result.emitted_tool_call and result.tool_name:
                            sb = run_tool_sandboxed(
                                result.tool_name, result.tool_args or {},
                                network=not sandbox, require_isolation=sandbox,
                            )
                            row["exec"] = {
                                "sandboxed": sb.sandboxed,
                                "net_isolation_enforced": sb.net_isolation_enforced,
                                "ok": sb.ok,
                                "result": sb.result,
                                "timed_out": sb.timed_out,
                            }
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        rows.append(row)

    summarize(rows, out.with_suffix(".summary.json"))


def summarize(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Aggregate emission / correctness rates per (model, task, condition)."""
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        key = (r["model"], r["task"], r["condition"])
        c = cells.setdefault(key, {"n": 0, "emitted": 0, "fully_correct": 0,
                                   "errors": 0, "latencies": []})
        c["n"] += 1
        c["emitted"] += int(r["emitted_tool_call"])
        c["fully_correct"] += int(r["fully_correct"])
        c["errors"] += int(bool(r["error"]))
        if r["error"] is None:
            c["latencies"].append(r["latency_s"])

    summary = []
    for (model, task, condition), c in sorted(cells.items()):
        lo, hi = wilson_interval(c["emitted"], c["n"])
        lat = sorted(c["latencies"])
        summary.append({
            "model": model, "task": task, "condition": condition,
            "n": c["n"],
            "emission_rate": round(c["emitted"] / c["n"], 3),
            "emission_ci95": [round(lo, 3), round(hi, 3)],
            "fully_correct_rate": round(c["fully_correct"] / c["n"], 3),
            "error_rate": round(c["errors"] / c["n"], 3),
            "latency_median_s": round(lat[len(lat) // 2], 3) if lat else None,
        })

    out_path.write_text(json.dumps(summary, indent=2))
    # Also print a compact isolated-vs-full delta per model/task.
    print(f"{'model':<22}{'task':<10}{'iso':>7}{'full':>7}{'delta':>8}")
    by_mt: dict[tuple[str, str], dict[str, float]] = {}
    for s in summary:
        by_mt.setdefault((s["model"], s["task"]), {})[s["condition"]] = s["emission_rate"]
    for (model, task), d in sorted(by_mt.items()):
        iso, full = d.get("isolated"), d.get("full_harness")
        if iso is not None and full is not None:
            print(f"{model:<22}{task:<10}{iso:>7.2f}{full:>7.2f}{full - iso:>+8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="results/run.jsonl")
    ap.add_argument("--execute", action="store_true",
                    help="execute each emitted tool call and record the outcome")
    ap.add_argument("--sandbox", action="store_true",
                    help="with --execute, run tools egress-fenced (fail closed)")
    args = ap.parse_args()
    run(args.config, args.trials, args.out, execute=args.execute, sandbox=args.sandbox)


if __name__ == "__main__":
    main()
