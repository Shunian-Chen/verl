#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib


def _read_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # skip broken lines
                continue
    return records


def _infer_numeric_keys(records: List[Dict]) -> List[str]:
    numeric_keys = set()
    for rec in records:
        for k, v in rec.items():
            if k == "step":
                continue
            if isinstance(v, (int, float)):
                numeric_keys.add(k)
    return sorted(numeric_keys)


def _plot(records: List[Dict], output: str, metrics: List[str] | None) -> None:
    if not records:
        raise SystemExit("No records to plot.")

    steps = [rec.get("step", i) for i, rec in enumerate(records)]
    if metrics is None or len(metrics) == 0:
        metrics = _infer_numeric_keys(records)
    if len(metrics) == 0:
        raise SystemExit("No numeric metrics found.")

    # Non-interactive backend for headless servers
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_plots = len(metrics)
    cols = 2 if num_plots > 1 else 1
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ys = [rec.get(metric, None) for rec in records]
        xs = steps
        # filter None
        filtered = [(x, y) for x, y in zip(xs, ys) if isinstance(y, (int, float))]
        if not filtered:
            ax.set_title(f"{metric} (no data)")
            continue
        xs2, ys2 = zip(*filtered)
        ax.plot(xs2, ys2, label=metric)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved figure to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from local JSONL metrics file.")
    parser.add_argument("jsonl", type=str, help="Path to metrics.jsonl")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). Default next to JSONL.")
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metric keys to plot; default: all numeric keys.",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"File not found: {jsonl_path}")

    out = args.out
    if out is None:
        out = str(jsonl_path.with_suffix(".png"))

    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    records = _read_jsonl(str(jsonl_path))
    _plot(records, output=out, metrics=metrics)


if __name__ == "__main__":
    main()


