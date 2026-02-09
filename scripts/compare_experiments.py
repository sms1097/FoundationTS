#!/usr/bin/env python3
import csv
import glob
import os
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RunSeries:
    name: str
    train_steps: List[int]
    train_time_s: List[float]
    val_points_time: List[Tuple[float, float, int]]
    val_points_tokens: List[Tuple[float, float, int]]
    total_time_s: float
    total_tokens: float


def _load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _build_series(
    rows: List[Dict[str, str]],
    metric_key: str,
    include_stages: set[str],
) -> RunSeries:
    train_rows = [r for r in rows if r.get("stage") == "train"]
    train_rows.sort(key=lambda r: int(r["step"]))

    train_steps: List[int] = []
    train_time_s: List[float] = []
    token_points: List[float] = []
    cumulative_s = 0.0
    prev_step = 0
    for r in train_rows:
        step = int(r["step"])
        step_time_ms = _parse_float(r.get("step_time_ms", ""))
        budget_val = _parse_float(r.get("cumulative_model_tokens", ""))
        if step_time_ms is None:
            continue
        delta_steps = max(1, step - prev_step)
        cumulative_s += (step_time_ms / 1000.0) * delta_steps
        train_steps.append(step)
        train_time_s.append(cumulative_s)
        token_points.append(0.0 if budget_val is None else budget_val)
        prev_step = step

    val_points_time: List[Tuple[float, float, int]] = []
    val_points_tokens: List[Tuple[float, float, int]] = []
    for r in rows:
        stage = r.get("stage") or ""
        if stage not in include_stages:
            continue
        metric_val = _parse_float(r.get(metric_key, ""))
        if metric_val is None:
            continue
        step = int(r["step"])
        t_s = _budget_at_step(train_steps, train_time_s, step)
        tokens = _budget_at_step(train_steps, token_points, step)
        if t_s is not None:
            val_points_time.append((t_s / 3600.0, metric_val, step))
        if tokens is not None:
            val_points_tokens.append((tokens, metric_val, step))

    total_time_s = train_time_s[-1] if train_time_s else 0.0
    total_tokens = token_points[-1] if token_points else 0.0
    name = "unknown"
    return RunSeries(
        name=name,
        train_steps=train_steps,
        train_time_s=train_time_s,
        val_points_time=val_points_time,
        val_points_tokens=val_points_tokens,
        total_time_s=total_time_s,
        total_tokens=total_tokens,
    )


def _budget_at_step(train_steps: List[int], budget_points: List[float], step: int) -> float | None:
    if not train_steps or not budget_points:
        return None
    idx = bisect_right(train_steps, step) - 1
    if idx < 0:
        return None
    return budget_points[idx]


def _select_metric_at_budget(points: List[Tuple[float, float, int]], budget_value: float) -> float | None:
    eligible = [p for p in points if p[0] <= budget_value]
    if not eligible:
        return None
    eligible.sort(key=lambda p: p[0])
    return eligible[-1][1]


def _time_to_target(points: List[Tuple[float, float, int]], target: float, higher_better: bool) -> float | None:
    if not points:
        return None
    for t_h, metric, _ in sorted(points, key=lambda p: p[0]):
        if higher_better:
            if metric >= target:
                return t_h
        else:
            if metric <= target:
                return t_h
    return None


def main() -> None:
    runs_glob = "checkpoints/*/train_metrics.csv"
    metric_key = "val_mse"
    baseline_name = "baseline"
    out_dir = "reports/experiments"
    include_stages = {"val", "val_budget"}
    higher_better = False

    paths = sorted(glob.glob(runs_glob))
    if not paths:
        raise SystemExit(f"No CSVs matched: {runs_glob}")

    runs: List[RunSeries] = []
    baseline_series: RunSeries | None = None
    for path in paths:
        rows = _load_csv(path)
        series = _build_series(rows, metric_key, include_stages)
        series.name = os.path.basename(os.path.dirname(path))
        runs.append(series)
        if series.name == baseline_name:
            baseline_series = series

    os.makedirs(out_dir, exist_ok=True)

    # Plot metric vs GPU-hours.
    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(7, 4.5))
    for series in runs:
        if not series.val_points_time:
            continue
        xs = [p[0] for p in series.val_points_time]
        ys = [p[1] for p in series.val_points_time]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=series.name)
    plt.xlabel("GPU-hours")
    plt.ylabel(metric_key)
    plt.title(f"{metric_key} vs GPU-hours")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "metric_vs_gpu_hours.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Plot metric vs total tokens.
    plt.figure(figsize=(7, 4.5))
    for series in runs:
        if not series.val_points_tokens:
            continue
        xs = [p[0] for p in series.val_points_tokens]
        ys = [p[1] for p in series.val_points_tokens]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=series.name)
    plt.xlabel("Cumulative model tokens")
    plt.ylabel(metric_key)
    plt.title(f"{metric_key} vs total tokens")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "metric_vs_tokens.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Fixed compute points (GPU-hours).
    fixed_points = [0.1, 0.25, 0.5, 1.0]
    fixed_rows: List[Dict[str, str]] = []
    for series in runs:
        total_budget = series.total_time_s / 3600.0 if series.total_time_s else 0.0
        for frac in fixed_points:
            budget = total_budget * frac
            metric_val = _select_metric_at_budget(series.val_points_time, budget)
            fixed_rows.append(
                {
                    "run": series.name,
                    "fraction": f"{frac:.2f}",
                    "budget_value": f"{budget:.6f}",
                    "metric": "" if metric_val is None else f"{metric_val:.6f}",
                }
            )

    fixed_path = os.path.join(out_dir, "metrics_at_compute_points.csv")
    with open(fixed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "fraction", "budget_value", "metric"])
        writer.writeheader()
        writer.writerows(fixed_rows)

    # Compute-to-target table (GPU-hours).
    target_metric = None
    if baseline_series and baseline_series.val_points_time:
        target_metric = baseline_series.val_points_time[-1][1]

    target_rows: List[Dict[str, str]] = []
    for series in runs:
        time_to_target = None
        if target_metric is not None:
            time_to_target = _time_to_target(series.val_points_time, target_metric, higher_better)
        target_rows.append(
            {
                "run": series.name,
                "target_metric": "" if target_metric is None else f"{target_metric:.6f}",
                "budget_to_target": "" if time_to_target is None else f"{time_to_target:.6f}",
            }
        )

    target_path = os.path.join(out_dir, "compute_to_target.csv")
    with open(target_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "target_metric", "budget_to_target"])
        writer.writeheader()
        writer.writerows(target_rows)

    print(f"Wrote: {fixed_path}")
    print(f"Wrote: {target_path}")


if __name__ == "__main__":
    main()
