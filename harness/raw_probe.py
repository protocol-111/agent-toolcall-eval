"""Raw-output tool-call probe: strict vs sanitizing parser.

Bypasses the OpenAI-compatible endpoint (which pre-normalizes tool calls) and
hits the model's RAW generation endpoint, so the leaked control tokens and
multi-object output that a strict harness parser rejects are visible. Each
sample is parsed two ways and both emission rates are reported, so the gap —
what a sanitizing proxy recovers — is explicit.

    python -m harness.raw_probe --config configs/models.yaml --trials 10 \
        --out results/raw.jsonl
    python -m harness.raw_probe --mock --trials 10 --out results/raw_mock.jsonl

Targets Ollama's native /api/generate (host from --raw-url or OLLAMA_URL,
default derived from LOCAL_OPENAI_BASE_URL). Anthropic models are skipped: they
use native structured tool use and don't emit raw text to parse.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml

from harness.parsers import strict_parse, sanitizing_parse
from harness.tasks import TASKS

_PROMPT = (
    "You can call tools. Respond with ONLY a single JSON object of the form "
    '{{"tool": <name>, "arguments": {{...}}}} and nothing else.\n'
    "Available tools: add_numbers(a, b), lookup_record(record_id), "
    "format_date(year, month, day).\n\nRequest: {request}\n"
)


def _raw_url() -> str:
    env = os.environ.get("OLLAMA_URL")
    if env:
        return env.rstrip("/")
    base = os.environ.get("LOCAL_OPENAI_BASE_URL", "http://localhost:11434/v1")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base.rstrip("/")


def generate_raw(host: str, model: str, prompt: str, timeout: int = 120) -> str:
    """One raw completion from Ollama /api/generate (no tool normalization)."""
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read()).get("response", "")


def probe(models, trials: int, out_path: str, generate_fn: Callable[[str, str], str]) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with out.open("w") as fh:
        for m in models:
            label = m.get("label", m["model"])
            for task in TASKS:
                prompt = _PROMPT.format(request=task.request)
                for trial in range(trials):
                    t0 = time.perf_counter()
                    try:
                        raw = generate_fn(m["model"], prompt)
                        err = None
                    except Exception as exc:  # noqa: BLE001
                        raw, err = "", repr(exc)
                    strict = strict_parse(raw)
                    sani = sanitizing_parse(raw)
                    row = {
                        "model": label, "task": task.task_id, "trial": trial,
                        "latency_s": round(time.perf_counter() - t0, 3),
                        "error": err,
                        "strict_ok": strict.ok,
                        "sanitized_ok": sani.ok,
                        "strict_correct": bool(strict.ok and strict.call.get("tool") == task.expected_tool),
                        "sanitized_correct": bool(sani.ok and sani.call.get("tool") == task.expected_tool),
                        "repairs": sani.reasons,
                        "raw_sample": raw[:300],
                    }
                    fh.write(json.dumps(row) + "\n")
                    rows.append(row)
            fh.flush()
    summarize(rows, out.with_suffix(".summary.json"))


def summarize(rows, out_path: Path) -> None:
    agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "strict": 0, "sani": 0, "errors": 0,
                 "repairs": defaultdict(int)})
    for r in rows:
        a = agg[r["model"]]
        a["n"] += 1
        a["strict"] += int(r["strict_ok"])
        a["sani"] += int(r["sanitized_ok"])
        a["errors"] += int(bool(r["error"]))
        for reason in r["repairs"]:
            a["repairs"][reason] += 1

    summary = {}
    for model, a in agg.items():
        n = a["n"]
        summary[model] = {
            "n": n,
            "strict_rate": round(a["strict"] / n, 3),
            "sanitized_rate": round(a["sani"] / n, 3),
            "recovered_by_sanitizer": round((a["sani"] - a["strict"]) / n, 3),
            "error_rate": round(a["errors"] / n, 3),
            "repairs": dict(a["repairs"]),
        }
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"{'model':<22}{'strict':>8}{'sanitized':>11}{'recovered':>11}{'errs':>6}")
    for model, s in sorted(summary.items()):
        print(f"{model:<22}{s['strict_rate']:>8.2f}{s['sanitized_rate']:>11.2f}"
              f"{s['recovered_by_sanitizer']:>+11.2f}{s['error_rate']:>6.2f}")
    # Which repairs the sanitizer most often had to make.
    allr: dict[str, int] = defaultdict(int)
    for s in summary.values():
        for k, v in s["repairs"].items():
            allr[k] += v
    if allr:
        print("repairs:", ", ".join(f"{k}={v}" for k, v in sorted(allr.items(), key=lambda x: -x[1])))


# --- mock dirty-output generator (reproduces the leaked-token phenomenon) ---

def _mock_generator():
    import random
    rng = random.Random(0)
    clean = {
        "add_numbers": '{"tool": "add_numbers", "arguments": {"a": 1847, "b": 2963}}',
        "lookup_record": '{"tool": "lookup_record", "arguments": {"record_id": "R-002"}}',
        "format_date": '{"tool": "format_date", "arguments": {"year": 2026, "month": 11, "day": 5}}',
    }

    def which(prompt: str) -> str:
        if "plus" in prompt:
            return clean["add_numbers"]
        if "record" in prompt:
            return clean["lookup_record"]
        return clean["format_date"]

    def gen(model: str, prompt: str) -> str:
        base = which(prompt)
        # Dirty it the way local models do: control tokens, fences, 2nd object.
        roll = rng.random()
        if roll < 0.35:
            return base + "<|im_end|>"                       # trailing control token
        if roll < 0.55:
            return "```json\n" + base + "\n```"              # code fence
        if roll < 0.70:
            return base + '\n{"tool": "noop_0", "arguments": {}}'  # second object
        if roll < 0.80:
            return "Sure! Here is the call:\n" + base         # prose preamble
        return base                                           # clean
    return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", default="results/raw.jsonl")
    ap.add_argument("--raw-url", default=None, dest="raw_url")
    ap.add_argument("--mock", action="store_true", help="use a dirty-output generator, no network")
    args = ap.parse_args()

    if args.mock:
        models = [{"label": "mock-local", "model": "mock", "backend": "openai_compat"}]
        probe(models, args.trials, args.out, _mock_generator())
        return

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = [m for m in cfg["models"] if m["backend"] != "anthropic"]
    skipped = [m["label"] for m in cfg["models"] if m["backend"] == "anthropic"]
    if skipped:
        print(f"(skipping native-tool-use models: {', '.join(skipped)})")
    host = args.raw_url or _raw_url()
    print(f"raw endpoint: {host}/api/generate")
    probe(models, args.trials, args.out, lambda model, prompt: generate_raw(host, model, prompt))


if __name__ == "__main__":
    main()
