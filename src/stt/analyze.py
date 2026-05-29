"""Analyze persisted STT experiment records.

The analyzer reads `results.json` files produced by `stt-lora --output-dir` or
`stt-continual --output-dir` and prints compact baseline-relative deltas. It is
intentionally simple: the goal is to make dose-response and multi-seed runs easy
to inspect without requiring notebooks or plotting libraries.
"""

from __future__ import annotations

import argparse
import json
import math
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

ACCRETION_METRICS = {
    "accretion_a_after_b": "higher",
    "interference_a_after_c": "lower",
    "interference_b_after_c": "lower",
    "learning_b": "higher",
    "learning_c": "higher",
    "retention_a_after_c": "higher",
    "retention_b_after_c": "higher",
    "lora_cosine_a_b_mean": "higher",
    "lora_cosine_a_c_mean": "lower",
    "lora_cosine_b_c_mean": "lower",
    "grad_cosine_a_b_after_a": "higher",
    "grad_cosine_a_c_after_b": "lower",
}

ACCRETION_PREDICTOR_TARGETS = [
    "accretion_a_after_b",
    "retention_a_after_c",
    "learning_b",
    "learning_c",
]

ORACLE_METRICS = [
    "accretion_a",
    "interference_a",
    "interference_b",
    "learning_b",
    "learning_c",
    "eval_c",
]

ROUTED_METRICS = [*ORACLE_METRICS, "frontier_score"]

ORACLE_METHOD_KEYS = {
    "sequential": {
        "accretion_a": "sequential_accretion_a_after_b",
        "interference_a": "sequential_interference_a_after_c",
        "interference_b": "sequential_interference_b_after_c",
        "learning_b": "sequential_learning_b",
        "learning_c": "sequential_learning_c",
        "eval_c": "sequential_eval_c",
    },
    "fixed": {
        "accretion_a": "fixed_accretion_a",
        "interference_a": "fixed_interference_a",
        "interference_b": "fixed_interference_b",
        "learning_b": "fixed_learning_b",
        "learning_c": "fixed_learning_c",
        "eval_c": "fixed_eval_c",
    },
    "oracle": {
        "accretion_a": "oracle_accretion_a",
        "interference_a": "oracle_interference_a",
        "interference_b": "oracle_interference_b",
        "learning_b": "oracle_learning_b",
        "learning_c": "oracle_learning_c",
        "eval_c": "oracle_eval_c",
    },
}

ROUTED_METHOD_KEYS = {
    "sequential": {
        "accretion_a": "sequential_accretion_a",
        "interference_a": "sequential_interference_a",
        "interference_b": "sequential_interference_b",
        "learning_b": "sequential_learning_b",
        "learning_c": "sequential_learning_c",
        "eval_c": "sequential_eval_c",
    },
    "routed": {
        "accretion_a": "routed_accretion_a",
        "interference_a": "routed_interference_a",
        "interference_b": "routed_interference_b",
        "learning_b": "routed_learning_b",
        "learning_c": "routed_learning_c",
        "eval_c": "routed_eval_c",
        "frontier_score": "frontier_score",
    },
    "delta": {
        "accretion_a": "delta_accretion_a",
        "interference_a": "delta_interference_a",
        "interference_b": "delta_interference_b",
        "learning_b": "delta_learning_b",
        "learning_c": "delta_learning_c",
        "frontier_score": "frontier_score",
    },
}


def load_record(path: str) -> dict[str, Any]:
    """Load a persisted STT run record from JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent_delta(value: float, baseline: float) -> float:
    """Return percent delta from baseline, guarding against zero baselines."""
    if baseline == 0.0:
        return 0.0
    return ((value - baseline) / baseline) * 100.0


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Return Pearson correlation, or None when variance is zero."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_deltas = [value - x_mean for value in xs]
    y_deltas = [value - y_mean for value in ys]
    denominator = math.sqrt(
        sum(delta * delta for delta in x_deltas) * sum(delta * delta for delta in y_deltas)
    )
    if denominator == 0.0:
        return None
    numerator = sum(
        x_delta * y_delta for x_delta, y_delta in zip(x_deltas, y_deltas, strict=True)
    )
    return numerator / denominator


def ranks(values: list[float]) -> list[float]:
    """Return average ranks for values, using one-based ranks."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranked = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = statistics.fmean(range(index + 1, end + 1))
        for original_index, _ in indexed[index:end]:
            ranked[original_index] = rank
        index = end
    return ranked


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Return Spearman rank correlation, or None when variance is zero."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson_correlation(ranks(xs), ranks(ys))


def format_optional_float(value: float | None) -> str:
    """Format an optional float for compact CLI output."""
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


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


def is_accretion_record(record: dict[str, Any]) -> bool:
    """Return whether a run record has A-to-B-to-C accretion metrics."""
    summary: dict[str, dict[str, float]] = record["summary"]
    baseline_name = baseline_variant(summary)
    return "accretion_a_after_b_mean" in summary[baseline_name]


def is_oracle_record(record: dict[str, Any]) -> bool:
    """Return whether a run record has oracle LoRA composition metrics."""
    summary: dict[str, dict[str, float]] = record["summary"]
    if not summary:
        return False
    first_values = next(iter(summary.values()))
    return "oracle_accretion_a_mean" in first_values


def is_routed_record(record: dict[str, Any]) -> bool:
    """Return whether a run record has fixed routed-update metrics."""
    summary: dict[str, dict[str, float]] = record["summary"]
    if not summary:
        return False
    first_values = next(iter(summary.values()))
    return "routed_accretion_a_mean" in first_values


def is_memory_bank_record(record: dict[str, Any]) -> bool:
    """Return whether a run record has contextual memory-bank metrics."""
    summary: dict[str, dict[str, float]] = record["summary"]
    if not summary:
        return False
    first_values = next(iter(summary.values()))
    return "contextual_eval_loss_mean" in first_values and "route_accuracy_mean" in first_values


def record_condition(record: dict[str, Any]) -> str:
    """Return a compact condition label from a run record config."""
    config = record.get("config", {})
    task_b_file = config.get("task_b_file")
    if task_b_file is None and config.get("task_files"):
        task_files = config["task_files"]
        task_b_file = task_files[1] if len(task_files) > 1 else task_files[0]
    if task_b_file is None:
        task_b_file = "unknown"
    return Path(task_b_file).stem


def analyze_accretion_record(
    record: dict[str, Any],
    max_learning_c_delta: float,
) -> list[str]:
    """Return formatted baseline-relative lines for an accretion run record."""
    summary: dict[str, dict[str, float]] = record["summary"]
    baseline_name = baseline_variant(summary)
    baseline = summary[baseline_name]
    lines = [
        f"baseline={baseline_name} accretion_a_after_b="
        f"{baseline['accretion_a_after_b_mean']:.4f} learning_c="
        f"{baseline['learning_c_mean']:.4f}",
        "variant metric value delta_vs_baseline pass",
    ]
    for variant_name, values in sorted(summary.items()):
        if variant_name == baseline_name:
            continue
        learning_c_delta = percent_delta(values["learning_c_mean"], baseline["learning_c_mean"])
        preserves_learning_c = learning_c_delta >= -max_learning_c_delta
        for metric, direction in ACCRETION_METRICS.items():
            metric_key = f"{metric}_mean"
            if metric_key not in baseline or metric_key not in values:
                continue
            delta = percent_delta(values[metric_key], baseline[metric_key])
            if metric == "learning_c":
                passed = preserves_learning_c
            else:
                improves = (
                    values[metric_key] <= baseline[metric_key]
                    if direction == "lower"
                    else values[metric_key] >= baseline[metric_key]
                )
                passed = improves and preserves_learning_c
            lines.append(
                f"{variant_name} {metric} {values[metric_key]:.4f} "
                f"{delta:+.2f}% {'yes' if passed else 'no'}"
            )
    lines.extend(paired_accretion_deltas(record))
    return lines


def analyze_oracle_record(record: dict[str, Any]) -> list[str]:
    """Return formatted summaries for an oracle composition run record."""
    summary: dict[str, dict[str, float]] = record["summary"]
    condition = record_condition(record)
    lines = [
        f"oracle_record condition={condition} variants={','.join(sorted(summary))}",
        "variant method metric value",
    ]
    for variant_name, values in sorted(summary.items()):
        for method, metric_keys in ORACLE_METHOD_KEYS.items():
            for metric in ORACLE_METRICS:
                key = f"{metric_keys[metric]}_mean"
                if key not in values:
                    continue
                lines.append(f"{variant_name} {method} {metric} {values[key]:+.4f}")
        lines.extend(oracle_win_count_lines(variant_name, values))
    return lines


def oracle_win_count_lines(variant_name: str, values: dict[str, float]) -> list[str]:
    """Return formatted oracle/fixed win-count lines for one summary."""
    count = int(values.get("count", 0.0))
    lines = ["", f"win_counts variant={variant_name} seeds={count}"]
    metrics = [
        ("accretion", "accretion_win_count"),
        ("a_interference", "interference_a_win_count"),
        ("b_interference", "interference_b_win_count"),
        ("c_learning_preserved", "learning_c_preserved_count"),
    ]
    for method in ["fixed", "oracle"]:
        parts = []
        for label, suffix in metrics:
            key = f"{method}_{suffix}"
            if key in values:
                parts.append(f"{label}={int(values[key])}/{count}")
        if parts:
            lines.append(f"{method} {' '.join(parts)}")
    return lines


def analyze_routed_record(record: dict[str, Any]) -> list[str]:
    """Return formatted summaries for a fixed routed-update run record."""
    summary: dict[str, dict[str, float]] = record["summary"]
    condition = record_condition(record)
    lines = [
        f"routed_record condition={condition} variants={','.join(sorted(summary))}",
        "variant method metric value",
    ]
    for variant_name, values in sorted(summary.items()):
        for method, metric_keys in ROUTED_METHOD_KEYS.items():
            for metric in ROUTED_METRICS:
                metric_key = metric_keys.get(metric)
                if metric_key is None:
                    continue
                key = f"{metric_key}_mean"
                if key not in values:
                    continue
                lines.append(f"{variant_name} {method} {metric} {values[key]:+.4f}")
        lines.extend(routed_win_count_lines(variant_name, values))
    lines.extend(best_routed_frontier_lines(summary))
    return lines


def routed_win_count_lines(variant_name: str, values: dict[str, float]) -> list[str]:
    """Return formatted routed win-count lines for one summary."""
    count = int(values.get("count", 0.0))
    lines = ["", f"win_counts variant={variant_name} seeds={count}"]
    metrics = [
        ("accretion", "accretion_a_win_count", "routed_accretion_win_count"),
        ("a_interference", "interference_a_win_count", "routed_interference_a_win_count"),
        ("b_interference", "interference_b_win_count", "routed_interference_b_win_count"),
        (
            "c_learning_preserved",
            "learning_c_preserved_count",
            "routed_learning_c_preserved_count",
        ),
        ("frontier_score", "frontier_score_win_count", "frontier_score_win_count"),
    ]
    parts = []
    for label, key, fallback_key in metrics:
        value = values.get(key, values.get(fallback_key))
        if value is not None:
            parts.append(f"{label}={int(value)}/{count}")
    if parts:
        lines.append(f"routed {' '.join(parts)}")
    return lines


def best_routed_frontier_lines(summary: dict[str, dict[str, float]]) -> list[str]:
    """Return the best fixed route by mean frontier score."""
    candidates = [
        (variant_name, values)
        for variant_name, values in summary.items()
        if "frontier_score_mean" in values
    ]
    if not candidates:
        return []
    variant_name, values = max(candidates, key=lambda item: item[1]["frontier_score_mean"])
    c_scales = f"c={values.get('route_c_scale_mean', 0.0):g}"
    if (
        "route_c_early_scale_mean" in values
        and "route_c_middle_scale_mean" in values
        and "route_c_late_scale_mean" in values
        and len(
            {
                values["route_c_early_scale_mean"],
                values["route_c_middle_scale_mean"],
                values["route_c_late_scale_mean"],
            }
        )
        > 1
    ):
        c_scales = (
            f"ce={values['route_c_early_scale_mean']:g} "
            f"cm={values['route_c_middle_scale_mean']:g} "
            f"cl={values['route_c_late_scale_mean']:g}"
        )
    elif "route_c_lora_a_scale_mean" in values and "route_c_lora_b_scale_mean" in values:
        c_scales = (
            f"ca={values['route_c_lora_a_scale_mean']:g} "
            f"cb={values['route_c_lora_b_scale_mean']:g}"
        )
    return [
        "",
        "best_by_frontier",
        f"{variant_name} score={values['frontier_score_mean']:+.4f} "
        f"b={values.get('route_b_scale_mean', 0.0):g} "
        f"{c_scales}",
    ]


def memory_bank_selection(record: dict[str, Any], variant_name: str) -> str:
    """Return the route-selection label for a memory-bank variant."""
    for result in record.get("results", []):
        if result.get("variant") == variant_name:
            selection = str(result.get("route_selection", "unknown"))
            if selection == "global" and result.get("global_route") is not None:
                return f"global:{result['global_route']}"
            return selection
    return "unknown"


def analyze_memory_bank_record(record: dict[str, Any]) -> list[str]:
    """Return formatted summaries for a contextual memory-bank run record."""
    summary: dict[str, dict[str, float]] = record["summary"]
    phase_names = record.get("config", {}).get("phase_names", [])
    lines = [
        f"memory_bank_record phases={','.join(phase_names)} variants={','.join(sorted(summary))}",
        "variant selection eval_loss sequential_eval_loss loss_delta frontier_score "
        "route_accuracy optimal_rate selected_gap expected_gap ambiguous_rate",
    ]
    for variant_name, values in sorted(summary.items()):
        lines.append(
            f"{variant_name} {memory_bank_selection(record, variant_name)} "
            f"{format_optional_float(values.get('contextual_eval_loss_mean'))} "
            f"{format_optional_float(values.get('sequential_eval_loss_mean'))} "
            f"{format_optional_float(values.get('loss_delta_vs_sequential_mean'))} "
            f"{format_optional_float(values.get('frontier_score_mean'))} "
            f"{format_optional_float(values.get('route_accuracy_mean'))} "
            f"{format_optional_float(values.get('optimal_route_rate_mean'))} "
            f"{format_optional_float(values.get('selected_loss_gap_mean'))} "
            f"{format_optional_float(values.get('expected_loss_gap_mean'))} "
            f"{format_optional_float(values.get('ambiguous_rate_mean'))}"
        )
        lines.extend(memory_bank_win_count_lines(variant_name, values))
        lines.extend(memory_bank_domain_lines(record, variant_name))
    return lines


def memory_bank_win_count_lines(variant_name: str, values: dict[str, float]) -> list[str]:
    """Return contextual memory-bank win-count lines for one summary."""
    count = int(values.get("count", 0.0))
    if "contextual_win_count" not in values:
        return []
    return [
        "",
        f"win_counts variant={variant_name} seeds={count}",
        f"contextual frontier_score={int(values['contextual_win_count'])}/{count}",
    ]


def memory_bank_domain_lines(record: dict[str, Any], variant_name: str) -> list[str]:
    """Return route-choice rows aggregated by prompt domain."""
    results = [
        result for result in record.get("results", []) if result.get("variant") == variant_name
    ]
    domains = sorted({domain for result in results for domain in result.get("per_domain", {})})
    if not domains:
        return []

    lines = [
        "",
        "variant domain most_selected_route best_route selection_count accuracy optimal_rate "
        "eval_loss best_eval_loss selected_gap expected_gap learning_retained interference",
    ]
    for domain in domains:
        domain_results = [
            result["per_domain"][domain]
            for result in results
            if domain in result.get("per_domain", {})
        ]
        route_counts: dict[str, int] = {}
        selection_count = 0
        for domain_result in domain_results:
            for route, count in domain_result.get("selected_route_counts", {}).items():
                route_counts[route] = route_counts.get(route, 0) + int(count)
                selection_count += int(count)
        most_selected = (
            max(route_counts.items(), key=lambda item: (item[1], item[0]))[0]
            if route_counts
            else "none"
        )
        best_route_counts: dict[str, int] = {}
        for domain_result in domain_results:
            for route, count in domain_result.get("best_route_counts", {}).items():
                best_route_counts[route] = best_route_counts.get(route, 0) + int(count)
        best_route = (
            max(best_route_counts.items(), key=lambda item: (item[1], item[0]))[0]
            if best_route_counts
            else "none"
        )
        accuracy = weighted_memory_domain_metric(domain_results, selection_count, "route_accuracy")
        optimal_rate = weighted_memory_domain_metric(
            domain_results,
            selection_count,
            "optimal_route_rate",
        )
        eval_loss_value = weighted_memory_domain_metric(
            domain_results,
            selection_count,
            "eval_loss",
        )
        best_eval_loss = weighted_memory_domain_metric(
            domain_results,
            selection_count,
            "best_eval_loss",
        )
        selected_gap = weighted_memory_domain_metric(
            domain_results,
            selection_count,
            "selected_loss_gap",
        )
        expected_gap = weighted_memory_domain_metric(
            domain_results,
            selection_count,
            "expected_loss_gap",
        )
        learning_retained = weighted_memory_domain_metric(
            domain_results,
            selection_count,
            "learning_retained",
        )
        interference = weighted_memory_domain_metric(
            domain_results,
            selection_count,
            "interference",
        )

        lines.append(
            f"{variant_name} {domain} {most_selected} {best_route} {selection_count} "
            f"{accuracy:+.4f} {optimal_rate:+.4f} {eval_loss_value:+.4f} "
            f"{best_eval_loss:+.4f} {selected_gap:+.4f} {expected_gap:+.4f} "
            f"{learning_retained:+.4f} {interference:+.4f}"
        )
    return lines


def weighted_memory_domain_metric(
    domain_results: list[dict[str, Any]],
    selection_count: int,
    metric: str,
) -> float:
    """Return a selection-count weighted per-domain memory-bank metric."""
    if selection_count == 0:
        return 0.0
    return sum(
        float(domain_result.get(metric, 0.0)) * int(domain_result["selection_count"])
        for domain_result in domain_results
    ) / selection_count


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
    if result[metric] is None:
        raise ValueError(f"metric {metric} is None")
    return float(result[metric])


def paired_accretion_deltas(record: dict[str, Any]) -> list[str]:
    """Return per-seed paired accretion deltas against same-seed baselines."""
    results: list[dict[str, Any]] = record["results"]
    baselines = {
        int(result["seed"]): result for result in results if result["variant"] == "baseline"
    }
    variants = sorted({result["variant"] for result in results if result["variant"] != "baseline"})
    lines = ["", "paired_seed_deltas", "variant metric mean_delta all_seed_deltas"]
    for variant in variants:
        variant_results = [result for result in results if result["variant"] == variant]
        for metric in ACCRETION_METRICS:
            deltas = []
            for result in variant_results:
                seed = int(result["seed"])
                if seed not in baselines:
                    continue
                try:
                    deltas.append(
                        result_metric(result, metric) - result_metric(baselines[seed], metric)
                    )
                except (KeyError, ValueError):
                    continue
            if not deltas:
                continue
            joined = ",".join(f"{delta:+.4f}" for delta in deltas)
            lines.append(f"{variant} {metric} {statistics.fmean(deltas):+.4f} [{joined}]")
    return lines


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


def aggregate_continual_records(
    records: list[dict[str, Any]],
    max_learning_b_delta: float,
) -> list[str]:
    """Return aggregate continual metrics across multiple run records."""
    results = [result for record in records for result in record["results"]]
    baseline_results = [result for result in results if result["variant"] == "baseline"]
    if not baseline_results:
        raise ValueError("no baseline results found")
    variants = sorted({result["variant"] for result in results if result["variant"] != "baseline"})
    metrics = ["backward_transfer_a", "learning_b", "eval_b_after_b", "retention_ratio"]
    baseline_means = {
        metric: statistics.fmean(result_metric(result, metric) for result in baseline_results)
        for metric in metrics
    }
    seeds = sorted({int(result["seed"]) for result in results})
    lines = [
        f"combined_continual_records={len(records)} seeds={','.join(str(seed) for seed in seeds)}",
        f"baseline=baseline backward_transfer_a={baseline_means['backward_transfer_a']:.4f} "
        f"learning_b={baseline_means['learning_b']:.4f}",
        "variant metric value delta_vs_baseline pass",
    ]
    for variant in variants:
        variant_results = [result for result in results if result["variant"] == variant]
        variant_means = {
            metric: statistics.fmean(result_metric(result, metric) for result in variant_results)
            for metric in metrics
        }
        learning_delta = percent_delta(variant_means["learning_b"], baseline_means["learning_b"])
        preserves_learning = learning_delta >= -max_learning_b_delta
        for metric, direction in CONTINUAL_METRICS.items():
            delta = percent_delta(variant_means[metric], baseline_means[metric])
            if metric == "learning_b":
                passed = preserves_learning
            else:
                improves = (
                    variant_means[metric] <= baseline_means[metric]
                    if direction == "lower"
                    else variant_means[metric] >= baseline_means[metric]
                )
                passed = improves and preserves_learning
            lines.append(
                f"{variant} {metric} {variant_means[metric]:.4f} "
                f"{delta:+.2f}% {'yes' if passed else 'no'}"
            )
    lines.extend(aggregate_paired_continual_deltas(results))
    return lines


def aggregate_paired_continual_deltas(results: list[dict[str, Any]]) -> list[str]:
    """Return paired deltas across combined continual results."""
    baselines = {
        int(result["seed"]): result for result in results if result["variant"] == "baseline"
    }
    variants = sorted({result["variant"] for result in results if result["variant"] != "baseline"})
    metrics = ["backward_transfer_a", "learning_b", "eval_b_after_b", "retention_ratio"]
    lines = ["", "paired_seed_deltas", "variant metric mean_delta all_seed_deltas"]
    for variant in variants:
        variant_results = sorted(
            [result for result in results if result["variant"] == variant],
            key=lambda result: int(result["seed"]),
        )
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


def aggregate_accretion_predictors(records: list[dict[str, Any]]) -> list[str]:
    """Return paired predictor correlations across accretion records.

    The predictor table asks whether same-seed changes in A-B LoRA cosine track
    same-seed changes in accretion and retention. Pairing against each run's
    baseline reduces condition and seed effects before computing correlations.
    """
    rows = []
    for record_index, record in enumerate(records):
        results: list[dict[str, Any]] = record["results"]
        baselines = {
            int(result["seed"]): result for result in results if result["variant"] == "baseline"
        }
        task_b_file = record.get("config", {}).get("task_b_file", f"record_{record_index}")
        condition = Path(task_b_file).stem
        for result in results:
            if result["variant"] == "baseline":
                continue
            seed = int(result["seed"])
            if seed not in baselines:
                continue
            try:
                predictor_delta = result_metric(result, "lora_cosine_a_b_mean") - result_metric(
                    baselines[seed], "lora_cosine_a_b_mean"
                )
            except (KeyError, ValueError):
                continue
            for target in ACCRETION_PREDICTOR_TARGETS:
                try:
                    target_delta = result_metric(result, target) - result_metric(
                        baselines[seed], target
                    )
                except (KeyError, ValueError):
                    continue
                rows.append(
                    {
                        "condition": condition,
                        "variant": result["variant"],
                        "seed": seed,
                        "target": target,
                        "predictor_delta": predictor_delta,
                        "target_delta": target_delta,
                    }
                )
    if not rows:
        raise ValueError("no paired accretion predictor rows found")

    seeds = sorted({row["seed"] for row in rows})
    conditions = sorted({row["condition"] for row in rows})
    lines = [
        f"combined_accretion_records={len(records)} "
        f"conditions={','.join(conditions)} seeds={','.join(str(seed) for seed in seeds)}",
        "predictor=lora_cosine_a_b_mean paired_delta_correlations",
        "scope target n pearson spearman mean_predictor_delta mean_target_delta",
    ]

    def append_correlation_lines(scope: str, scoped_rows: list[dict[str, Any]]) -> None:
        for target in ACCRETION_PREDICTOR_TARGETS:
            target_rows = [row for row in scoped_rows if row["target"] == target]
            if not target_rows:
                continue
            predictor_deltas = [row["predictor_delta"] for row in target_rows]
            target_deltas = [row["target_delta"] for row in target_rows]
            lines.append(
                f"{scope} {target} {len(target_rows)} "
                f"{format_optional_float(pearson_correlation(predictor_deltas, target_deltas))} "
                f"{format_optional_float(spearman_correlation(predictor_deltas, target_deltas))} "
                f"{statistics.fmean(predictor_deltas):+.4f} "
                f"{statistics.fmean(target_deltas):+.4f}"
            )

    scopes = [
        "all",
        *sorted({f"variant:{row['variant']}" for row in rows}),
        *sorted({f"condition:{row['condition']}" for row in rows}),
        *[f"loo_without:{condition}" for condition in conditions],
    ]
    for scope in scopes:
        if scope == "all":
            scoped_rows = rows
        elif scope.startswith("variant:"):
            variant = scope.removeprefix("variant:")
            scoped_rows = [row for row in rows if row["variant"] == variant]
        elif scope.startswith("loo_without:"):
            condition = scope.removeprefix("loo_without:")
            scoped_rows = [row for row in rows if row["condition"] != condition]
        else:
            condition = scope.removeprefix("condition:")
            scoped_rows = [row for row in rows if row["condition"] == condition]
        append_correlation_lines(scope, scoped_rows)

    centered_rows = []
    group_keys = sorted({(row["condition"], row["variant"], row["target"]) for row in rows})
    for condition, variant, target in group_keys:
        group = [
            row
            for row in rows
            if row["condition"] == condition
            and row["variant"] == variant
            and row["target"] == target
        ]
        predictor_mean = statistics.fmean(row["predictor_delta"] for row in group)
        target_mean = statistics.fmean(row["target_delta"] for row in group)
        for row in group:
            centered_rows.append(
                {
                    **row,
                    "predictor_delta": row["predictor_delta"] - predictor_mean,
                    "target_delta": row["target_delta"] - target_mean,
                }
            )
    append_correlation_lines("centered:condition_variant", centered_rows)
    return lines


def aggregate_oracle_records(records: list[dict[str, Any]]) -> list[str]:
    """Return compact metric rows across multiple oracle composition records."""
    lines = [
        f"combined_oracle_records={len(records)}",
        "condition variant method accretion_a interference_a interference_b "
        "learning_b learning_c eval_c",
    ]
    for record in records:
        condition = record_condition(record)
        summary: dict[str, dict[str, float]] = record["summary"]
        for variant_name, values in sorted(summary.items()):
            for method, metric_keys in ORACLE_METHOD_KEYS.items():
                formatted = []
                for metric in ORACLE_METRICS:
                    key = f"{metric_keys[metric]}_mean"
                    formatted.append(format_optional_float(values.get(key)))
                lines.append(f"{condition} {variant_name} {method} {' '.join(formatted)}")
            for win_line in oracle_win_count_lines(variant_name, values):
                if win_line:
                    lines.append(f"{condition} {win_line}")
    return lines


def aggregate_routed_records(records: list[dict[str, Any]]) -> list[str]:
    """Return compact metric rows across multiple fixed routed-update records."""
    lines = [
        f"combined_routed_records={len(records)}",
        "condition variant method accretion_a interference_a interference_b "
        "learning_b learning_c eval_c frontier_score",
    ]
    for record in records:
        condition = record_condition(record)
        summary: dict[str, dict[str, float]] = record["summary"]
        for variant_name, values in sorted(summary.items()):
            for method, metric_keys in ROUTED_METHOD_KEYS.items():
                formatted = []
                for metric in ROUTED_METRICS:
                    metric_key = metric_keys.get(metric)
                    if metric_key is None:
                        formatted.append(format_optional_float(None))
                        continue
                    key = f"{metric_key}_mean"
                    formatted.append(format_optional_float(values.get(key)))
                lines.append(f"{condition} {variant_name} {method} {' '.join(formatted)}")
            for win_line in routed_win_count_lines(variant_name, values):
                if win_line:
                    lines.append(f"{condition} {win_line}")
        for best_line in best_routed_frontier_lines(summary):
            if best_line:
                lines.append(f"{condition} {best_line}")
    return lines


def aggregate_memory_bank_records(records: list[dict[str, Any]]) -> list[str]:
    """Return compact metric rows across multiple contextual memory-bank records."""
    lines = [
        f"combined_memory_bank_records={len(records)}",
        "condition variant selection eval_loss sequential_eval_loss loss_delta "
        "frontier_score route_accuracy optimal_rate selected_gap expected_gap ambiguous_rate",
    ]
    for record in records:
        condition = record_condition(record)
        summary: dict[str, dict[str, float]] = record["summary"]
        for variant_name, values in sorted(summary.items()):
            lines.append(
                f"{condition} {variant_name} {memory_bank_selection(record, variant_name)} "
                f"{format_optional_float(values.get('contextual_eval_loss_mean'))} "
                f"{format_optional_float(values.get('sequential_eval_loss_mean'))} "
                f"{format_optional_float(values.get('loss_delta_vs_sequential_mean'))} "
                f"{format_optional_float(values.get('frontier_score_mean'))} "
                f"{format_optional_float(values.get('route_accuracy_mean'))} "
                f"{format_optional_float(values.get('optimal_route_rate_mean'))} "
                f"{format_optional_float(values.get('selected_loss_gap_mean'))} "
                f"{format_optional_float(values.get('expected_loss_gap_mean'))} "
                f"{format_optional_float(values.get('ambiguous_rate_mean'))}"
            )
            for win_line in memory_bank_win_count_lines(variant_name, values):
                if win_line:
                    lines.append(f"{condition} {win_line}")
            for domain_line in memory_bank_domain_lines(record, variant_name):
                if domain_line and not domain_line.startswith("variant domain"):
                    lines.append(f"{condition} {domain_line}")
    return lines



def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-analyze`."""
    parser = argparse.ArgumentParser(description="Analyze persisted STT experiment results.")
    parser.add_argument(
        "results_json",
        nargs="+",
        help="Path(s) to stt-lora or stt-continual results.json files",
    )
    parser.add_argument("--max-loss-delta", type=float, default=10.0)
    parser.add_argument("--max-learning-b-delta", type=float, default=10.0)
    parser.add_argument("--max-learning-c-delta", type=float, default=10.0)
    parser.add_argument("--min-geometry-delta", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for result analysis."""
    args = parse_args()
    records = [load_record(path) for path in args.results_json]
    if len(records) > 1:
        if all(is_memory_bank_record(record) for record in records):
            lines = aggregate_memory_bank_records(records)
        elif all(is_routed_record(record) for record in records):
            lines = aggregate_routed_records(records)
        elif all(is_oracle_record(record) for record in records):
            lines = aggregate_oracle_records(records)
        elif all(is_accretion_record(record) for record in records):
            lines = aggregate_accretion_predictors(records)
        elif all(is_continual_record(record) for record in records):
            lines = aggregate_continual_records(records, args.max_learning_b_delta)
        else:
            raise ValueError("multi-file analysis requires all records to have the same type")
    else:
        record = records[0]
        if is_memory_bank_record(record):
            lines = analyze_memory_bank_record(record)
            print("\n".join(lines))
            return
        if is_routed_record(record):
            lines = analyze_routed_record(record)
            print("\n".join(lines))
            return
        if is_oracle_record(record):
            lines = analyze_oracle_record(record)
            print("\n".join(lines))
            return
        if is_accretion_record(record):
            lines = analyze_accretion_record(record, args.max_learning_c_delta)
            print("\n".join(lines))
            return
        if not is_continual_record(record):
            lines = analyze_record(record, args.max_loss_delta, args.min_geometry_delta)
            print("\n".join(lines))
            return
        lines = analyze_continual_record(record, args.max_learning_b_delta)
        lines.extend(paired_continual_deltas(record))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
