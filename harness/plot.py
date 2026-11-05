"""Turn a run summary into the report's figures.

    python -m harness.plot --summary results/run.summary.json --out results/figs

Produces one grouped bar chart per task: isolated vs full-harness emission
rate for each model, with 95% Wilson intervals as error bars. Reads only the
summary JSON produced by run_eval.summarize; makes no model calls.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def plot(summary_path: str, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless; safe on a server or in CI
    import matplotlib.pyplot as plt

    data = json.loads(Path(summary_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group by task -> model -> condition -> (rate, ci)
    by_task: dict[str, dict[str, dict[str, tuple[float, list[float]]]]] = \
        defaultdict(lambda: defaultdict(dict))
    for row in data:
        by_task[row["task"]][row["model"]][row["condition"]] = (
            row["emission_rate"], row["emission_ci95"],
        )

    conditions = ["isolated", "full_harness"]
    written = []
    for task, models in sorted(by_task.items()):
        model_names = sorted(models)
        x = range(len(model_names))
        width = 0.38
        fig, ax = plt.subplots(figsize=(1.6 * len(model_names) + 2, 4))
        for i, cond in enumerate(conditions):
            rates, lo_err, hi_err = [], [], []
            for m in model_names:
                rate, ci = models[m].get(cond, (0.0, [0.0, 0.0]))
                rates.append(rate)
                lo_err.append(max(0.0, rate - ci[0]))
                hi_err.append(max(0.0, ci[1] - rate))
            offs = [xi + (i - 0.5) * width for xi in x]
            ax.bar(offs, rates, width, label=cond.replace("_", "-"),
                   yerr=[lo_err, hi_err], capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("structured tool-call emission rate")
        ax.set_title(f"Task: {task} (isolated vs full-harness)")
        ax.legend()
        fig.tight_layout()
        path = out / f"emission_{task}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    print("wrote:")
    for p in written:
        print(" ", p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/run.summary.json")
    ap.add_argument("--out", default="results/figs")
    args = ap.parse_args()
    plot(args.summary, args.out)


if __name__ == "__main__":
    main()
