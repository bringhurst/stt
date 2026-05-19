"""Analyze persisted STT experiment records.

The analyzer reads `results.json` files produced by `stt-lora --output-dir` or
`stt-continual --output-dir` and prints compact baseline-relative deltas. It is
intentionally simple: the goal is to make dose-response and multi-seed runs easy
to inspect without requiring notebooks or plotting libraries.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

GEOMETRY_METRICS = {
    "head_similarity": "lower",
    "isotropy": "lower",
    "active_fraction": "lower",
    "effective_rank": "higher",
}

CONTINUAL_METRICS = {
    "backward_transfer_a": "lower",
    "learning_b": "higher",
    "eval_b_after_b": "lower",
    "retention_ratio": "higher",
}


def load_record(path: str) -> dict[str, Any]:
    """Load a persisted STT run record from JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent_delta(value: float, baseline: float) -> float:
    """Return percent delta from baseline, guarding against zero baselines."""
    if baseline == 0.0:
        return 0.0
    return ((value - baseline) / baseline) * 100.0


def baseline_variant(summary: dict[str, dict[str, float]]) -> str:
    """Return the preferred baseline variant name from a summary object."""
    if "baseline" in summary:
        return "baseline"
    for name in summary:
        if name.startswith("baseline_"):
            return name
    raise ValueError("no baseline variant found in run summary")


def analyze_record(
    record: dict[str, Any],
    max_loss_delta: float,
    min_geometry_delta: float,
) -> list[str]:
    """Return formatted analysis lines for one run record."""
    summary: dict[str, dict[str, float]] = record["summary"]
    baseline_name = baseline_variant(summary)
    baseline = summary[baseline_name]
    baseline_loss = baseline["eval_lm_loss_mean"]
    lines = [
        f"baseline={baseline_name} eval_lm_loss={baseline_loss:.4f}",
        "variant metric value delta_vs_baseline pass",
    ]
    for variant_name, values in sorted(summary.items()):
        if variant_name == baseline_name:
            continue
        loss_delta = percent_delta(values["eval_lm_loss_mean"], baseline_loss)
        for metric, direction in GEOMETRY_METRICS.items():
            metric_key = f"{metric}_mean"
            delta = percent_delta(values[metric_key], baseline[metric_key])
            improves = (
                delta <= -min_geometry_delta
                if direction == "lower"
                else delta >= min_geometry_delta
            )
            tolerable_loss = loss_delta <= max_loss_delta
            passed = improves and tolerable_loss
            lines.append(
                f"{variant_name} {metric} {values[metric_key]:.4f} "
                f"{delta:+.2f}% {'yes' if passed else 'no'}"
            )
    return lines


def is_continual_record(record: dict[str, Any]) -> bool:
    """Return whether a run record has continual-learning summary metrics."""
    summary: dict[str, dict[str, float]] = record["summary"]
    baseline_name = baseline_variant(summary)
    return "forgetting_a_mean" in summary[baseline_name]


def analyze_continual_record(
    record: dict[str, Any],
    max_learning_b_delta: float,
) -> list[str]:
    """Return formatted baseline-relative lines for a continual run record."""
    summary: dict[str, dict[str, float]] = record["summary"]
    baseline_name = baseline_variant(summary)
    baseline = summary[baseline_name]
    baseline_transfer = baseline.get("backward_transfer_a_mean", baseline["forgetting_a_mean"])
    baseline_learning = baseline["learning_b_mean"]
    lines = [
        f"baseline={baseline_name} backward_transfer_a={baseline_transfer:.4f} "
        f"learning_b={baseline_learning:.4f}",
        "variant metric value delta_vs_baseline pass",
    ]
    for variant_name, values in sorted(summary.items()):
        if variant_name == baseline_name:
            continue
        learning_delta = percent_delta(values["learning_b_mean"], baseline_learning)
        for metric, direction in CONTINUAL_METRICS.items():
            metric_key = f"{metric}_mean"
            baseline_key = metric_key
            if metric == "backward_transfer_a" and baseline_key not in baseline:
                baseline_key = "forgetting_a_mean"
            value_key = metric_key
            if metric == "backward_transfer_a" and value_key not in values:
                value_key = "forgetting_a_mean"
            delta = percent_delta(values[value_key], baseline[baseline_key])
            preserves_learning = learning_delta >= -max_learning_b_delta
            if metric == "learning_b":
                passed = preserves_learning
            else:
                improves = (
                    values[value_key] <= baseline[baseline_key]
                    if direction == "lower"
                    else values[value_key] >= baseline[baseline_key]
                )
                passed = improves and preserves_learning
            lines.append(
                f"{variant_name} {metric} {values[value_key]:.4f} "
                f"{delta:+.2f}% {'yes' if passed else 'no'}"
            )
    return lines


def result_metric(result: dict[str, Any], metric: str) -> float:
    """Return a metric value from a raw result, handling compatibility aliases."""
    if metric == "backward_transfer_a" and metric not in result:
        metric = "forgetting_a"
    return float(result[metric])


def paired_continual_deltas(record: dict[str, Any]) -> list[str]:
    """Return per-seed paired deltas against same-seed baseline runs.

    Aggregate means can hide seed-specific effects when between-seed variance is
    larger than the intervention effect. Paired deltas compare each variant to
    the baseline from the same seed.
    """
    results: list[dict[str, Any]] = record["results"]
    baselines = {
        int(result["seed"]): result for result in results if result["variant"] == "baseline"
    }
    variants = sorted({result["variant"] for result in results if result["variant"] != "baseline"})
    metrics = ["backward_transfer_a", "learning_b", "eval_b_after_b", "retention_ratio"]
    lines = ["", "paired_seed_deltas", "variant metric mean_delta all_seed_deltas"]
    for variant in variants:
        variant_results = [result for result in results if result["variant"] == variant]
        for metric in metrics:
            deltas = []
            for result in variant_results:
                seed = int(result["seed"])
                if seed not in baselines:
                    continue
                deltas.append(
                    result_metric(result, metric) - result_metric(baselines[seed], metric)
                )
            if not deltas:
                continue
            joined = ",".join(f"{delta:+.4f}" for delta in deltas)
            lines.append(f"{variant} {metric} {statistics.fmean(deltas):+.4f} [{joined}]")
    return lines


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-analyze`."""
    parser = argparse.ArgumentParser(description="Analyze persisted STT experiment results.")
    parser.add_argument("results_json", help="Path to a stt-lora results.json file")
    parser.add_argument("--max-loss-delta", type=float, default=10.0)
    parser.add_argument("--max-learning-b-delta", type=float, default=10.0)
    parser.add_argument("--min-geometry-delta", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for result analysis."""
    args = parse_args()
    record = load_record(args.results_json)
    if is_continual_record(record):
        lines = analyze_continual_record(record, args.max_learning_b_delta)
        lines.extend(paired_continual_deltas(record))
    else:
        lines = analyze_record(record, args.max_loss_delta, args.min_geometry_delta)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
