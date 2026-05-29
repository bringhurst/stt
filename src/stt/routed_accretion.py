"""Fixed routed-update LoRA accretion experiments.

This module turns the oracle-composition finding into a deployed baseline: train
A then B then C once, but publish a final adapter formed by a predeclared route
through the observed update deltas instead of blindly keeping the sequential C
state.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, NotRequired, TypedDict

import torch

from stt.accretion import (
    gradient_cosine,
    lora_effective_deltas,
    mean_lora_cosine,
    ratio,
    subtract_lora_deltas,
)
from stt.continual import eval_loss, prepare_encoded_splits, train_steps
from stt.experiment import Variant, resolve_device
from stt.lora_experiment import (
    LoraSettings,
    build_lora_model,
    build_variants,
    git_status,
    load_texts,
    load_tokenizer,
    parameter_counts,
    write_run_record,
)
from stt.oracle_compose import (
    apply_trainable_state,
    compose_state,
    snapshot_trainable_state,
    split_eval_encoded,
    subtract_state,
)


@dataclass(frozen=True)
class RouteSpec:
    """Predeclared routed-update scales for one published adapter."""

    b_scale: float
    c_lora_a_scale: float
    c_lora_b_scale: float
    c_early_scale: float
    c_middle_scale: float
    c_late_scale: float


class RoutedAccretionResult(TypedDict):
    """Serializable result for one fixed routed-update run."""

    variant: str
    model: str
    device: str
    seed: int
    diversity_weight: float
    repulsion_weight: float
    sparse_weight: float
    gossip_weight: float
    gossip_tau: float
    gossip_k: int
    max_gossip_vectors: int
    route_b_scale: float
    route_c_scale: float
    route_c_lora_a_scale: float
    route_c_lora_b_scale: float
    route_c_early_scale: float
    route_c_middle_scale: float
    route_c_late_scale: float
    heldout_report: NotRequired[bool]
    selection_candidate_count: NotRequired[int]
    selection_constraint_passed: NotRequired[bool]
    selection_objective_score: NotRequired[float]
    selection_frontier_score: NotRequired[float]
    selection_delta_accretion_a: NotRequired[float]
    selection_delta_interference_a: NotRequired[float]
    selection_delta_interference_b: NotRequired[float]
    selection_delta_learning_b: NotRequired[float]
    selection_delta_learning_c: NotRequired[float]
    selection_routed_learning_c: NotRequired[float]
    selection_sequential_learning_c: NotRequired[float]
    selection_candidates: NotRequired[list[dict[str, float | bool | str]]]
    trainable_parameters: int
    total_parameters: int
    trainable_fraction: float
    eval_a_before: float
    eval_b_before: float
    eval_c_before: float
    eval_a_after_a: float
    eval_b_after_a: float
    eval_c_after_a: float
    eval_a_after_b: float
    eval_b_after_b: float
    eval_c_after_b: float
    sequential_eval_a: float
    sequential_eval_b: float
    sequential_eval_c: float
    routed_eval_a: float
    routed_eval_b: float
    routed_eval_c: float
    sequential_accretion_a: float
    sequential_interference_a: float
    sequential_interference_a_after_c: float
    sequential_interference_b: float
    sequential_interference_b_after_c: float
    sequential_learning_b: float
    sequential_learning_c: float
    sequential_retention_a: float
    sequential_retention_b: float
    routed_accretion_a: float
    routed_interference_a: float
    routed_interference_a_after_c: float
    routed_interference_b: float
    routed_interference_b_after_c: float
    routed_learning_b: float
    routed_learning_c: float
    routed_retention_a: float
    routed_retention_b: float
    delta_accretion_a: float
    delta_interference_a: float
    delta_interference_b: float
    delta_learning_b: float
    delta_learning_c: float
    frontier_score: float
    lora_cosine_a_b_mean: float | None
    lora_cosine_a_c_mean: float | None
    lora_cosine_b_c_mean: float | None
    grad_cosine_a_b_after_a: float | None
    grad_cosine_a_c_after_b: float | None


class RoutedAccretionRunRecord(TypedDict):
    """Persisted fixed routed-update experiment record."""

    created_at: str
    config: dict[str, Any]
    git_status: str
    results: list[RoutedAccretionResult]
    summary: dict[str, dict[str, float]]


def run_routed_accretion_variant(
    variant: Variant,
    settings: LoraSettings,
    task_a_texts: list[str],
    task_b_texts: list[str],
    task_c_texts: list[str],
    phase_steps: int,
    seed: int,
    device: str,
    route_b_scale: float,
    route_c_scale: float,
    compat_batches: int = 0,
) -> RoutedAccretionResult:
    """Train A/B/C and evaluate a predeclared routed final adapter state."""
    return run_routed_accretion_variants(
        variant,
        settings=settings,
        task_a_texts=task_a_texts,
        task_b_texts=task_b_texts,
        task_c_texts=task_c_texts,
        phase_steps=phase_steps,
        seed=seed,
        device=device,
        route_pairs=[scalar_route_spec(route_b_scale, route_c_scale)],
        compat_batches=compat_batches,
    )[0]


def run_routed_accretion_variants(
    variant: Variant,
    settings: LoraSettings,
    task_a_texts: list[str],
    task_b_texts: list[str],
    task_c_texts: list[str],
    phase_steps: int,
    seed: int,
    device: str,
    route_pairs: list[RouteSpec],
    compat_batches: int = 0,
) -> list[RoutedAccretionResult]:
    """Train A/B/C once and evaluate one or more fixed routed final states."""
    resolved_device = resolve_device(device)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(settings.model_name)
    model = build_lora_model(settings, resolved_device)
    trainable, total = parameter_counts(model)

    train_a, eval_a = prepare_encoded_splits(
        tokenizer, task_a_texts, settings, seed, resolved_device
    )
    train_b, eval_b = prepare_encoded_splits(
        tokenizer, task_b_texts, settings, seed + 10_000, resolved_device
    )
    train_c, eval_c = prepare_encoded_splits(
        tokenizer, task_c_texts, settings, seed + 20_000, resolved_device
    )

    eval_a_before = eval_loss(model, eval_a, settings)
    eval_b_before = eval_loss(model, eval_b, settings)
    eval_c_before = eval_loss(model, eval_c, settings)
    lora_initial = lora_effective_deltas(model)

    train_steps(model, train_a, variant, settings, phase_steps)
    eval_a_after_a = eval_loss(model, eval_a, settings)
    eval_b_after_a = eval_loss(model, eval_b, settings)
    eval_c_after_a = eval_loss(model, eval_c, settings)
    state_a = snapshot_trainable_state(model)
    lora_after_a = lora_effective_deltas(model)
    grad_cosine_a_b_after_a = gradient_cosine(
        model, train_a, train_b, settings, compat_batches
    )

    train_steps(model, train_b, variant, settings, phase_steps)
    eval_a_after_b = eval_loss(model, eval_a, settings)
    eval_b_after_b = eval_loss(model, eval_b, settings)
    eval_c_after_b = eval_loss(model, eval_c, settings)
    state_b = snapshot_trainable_state(model)
    lora_after_b = lora_effective_deltas(model)
    grad_cosine_a_c_after_b = gradient_cosine(
        model, train_a, train_c, settings, compat_batches
    )

    train_steps(model, train_c, variant, settings, phase_steps)
    sequential_eval_a = eval_loss(model, eval_a, settings)
    sequential_eval_b = eval_loss(model, eval_b, settings)
    sequential_eval_c = eval_loss(model, eval_c, settings)
    state_c = snapshot_trainable_state(model)
    lora_after_c = lora_effective_deltas(model)

    delta_b_state = subtract_state(state_b, state_a)
    delta_c_state = subtract_state(state_c, state_b)
    lora_delta_a = subtract_lora_deltas(lora_after_a, lora_initial)
    lora_delta_b = subtract_lora_deltas(lora_after_b, lora_after_a)
    lora_delta_c = subtract_lora_deltas(lora_after_c, lora_after_b)
    lora_cosine_a_b_mean = mean_lora_cosine(lora_delta_a, lora_delta_b)
    lora_cosine_a_c_mean = mean_lora_cosine(lora_delta_a, lora_delta_c)
    lora_cosine_b_c_mean = mean_lora_cosine(lora_delta_b, lora_delta_c)
    max_layer_index = max_trainable_layer_index(delta_c_state)

    results = []
    multi_route = len(route_pairs) > 1
    for route in route_pairs:
        route_c_scale = mean_route_c_scale(route)
        routed_state = compose_grouped_route_state(
            state_a,
            delta_b_state=delta_b_state,
            delta_c_state=delta_c_state,
            route=route,
            max_layer_index=max_layer_index,
        )
        apply_trainable_state(model, routed_state)
        routed_eval_a = eval_loss(model, eval_a, settings)
        routed_eval_b = eval_loss(model, eval_b, settings)
        routed_eval_c = eval_loss(model, eval_c, settings)
        sequential_accretion_a = eval_a_after_a - eval_a_after_b
        sequential_interference_a = sequential_eval_a - eval_a_after_b
        sequential_interference_b = sequential_eval_b - eval_b_after_b
        sequential_learning_b = eval_b_after_a - eval_b_after_b
        sequential_learning_c = eval_c_after_b - sequential_eval_c
        routed_accretion_a = eval_a_after_a - routed_eval_a
        routed_interference_a = routed_eval_a - eval_a_after_b
        routed_interference_b = routed_eval_b - eval_b_after_b
        routed_learning_b = eval_b_after_a - routed_eval_b
        routed_learning_c = eval_c_after_b - routed_eval_c
        delta_accretion_a = routed_accretion_a - sequential_accretion_a
        delta_interference_a = sequential_interference_a - routed_interference_a
        delta_interference_b = sequential_interference_b - routed_interference_b
        delta_learning_b = routed_learning_b - sequential_learning_b
        delta_learning_c = routed_learning_c - sequential_learning_c
        results.append(
            {
                "variant": route_variant_name(
                    variant.name,
                    route,
                    multi_route,
                ),
                "model": settings.model_name,
                "device": resolved_device,
                "seed": seed,
                "diversity_weight": variant.diversity,
                "repulsion_weight": variant.repulsion,
                "sparse_weight": variant.sparse,
                "gossip_weight": variant.gossip,
                "gossip_tau": variant.gossip_tau,
                "gossip_k": variant.gossip_k,
                "max_gossip_vectors": variant.max_gossip_vectors,
                "route_b_scale": route.b_scale,
                "route_c_scale": route_c_scale,
                "route_c_lora_a_scale": route.c_lora_a_scale,
                "route_c_lora_b_scale": route.c_lora_b_scale,
                "route_c_early_scale": route.c_early_scale,
                "route_c_middle_scale": route.c_middle_scale,
                "route_c_late_scale": route.c_late_scale,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "trainable_fraction": trainable / total,
                "eval_a_before": eval_a_before,
                "eval_b_before": eval_b_before,
                "eval_c_before": eval_c_before,
                "eval_a_after_a": eval_a_after_a,
                "eval_b_after_a": eval_b_after_a,
                "eval_c_after_a": eval_c_after_a,
                "eval_a_after_b": eval_a_after_b,
                "eval_b_after_b": eval_b_after_b,
                "eval_c_after_b": eval_c_after_b,
                "sequential_eval_a": sequential_eval_a,
                "sequential_eval_b": sequential_eval_b,
                "sequential_eval_c": sequential_eval_c,
                "routed_eval_a": routed_eval_a,
                "routed_eval_b": routed_eval_b,
                "routed_eval_c": routed_eval_c,
                "sequential_accretion_a": sequential_accretion_a,
                "sequential_interference_a": sequential_interference_a,
                "sequential_interference_a_after_c": sequential_interference_a,
                "sequential_interference_b": sequential_interference_b,
                "sequential_interference_b_after_c": sequential_interference_b,
                "sequential_learning_b": sequential_learning_b,
                "sequential_learning_c": sequential_learning_c,
                "sequential_retention_a": ratio(eval_a_after_a, sequential_eval_a),
                "sequential_retention_b": ratio(eval_b_after_b, sequential_eval_b),
                "routed_accretion_a": routed_accretion_a,
                "routed_interference_a": routed_interference_a,
                "routed_interference_a_after_c": routed_interference_a,
                "routed_interference_b": routed_interference_b,
                "routed_interference_b_after_c": routed_interference_b,
                "routed_learning_b": routed_learning_b,
                "routed_learning_c": routed_learning_c,
                "routed_retention_a": ratio(eval_a_after_a, routed_eval_a),
                "routed_retention_b": ratio(eval_b_after_b, routed_eval_b),
                "delta_accretion_a": delta_accretion_a,
                "delta_interference_a": delta_interference_a,
                "delta_interference_b": delta_interference_b,
                "delta_learning_b": delta_learning_b,
                "delta_learning_c": delta_learning_c,
                "frontier_score": frontier_score(
                    sequential_accretion_a=sequential_accretion_a,
                    sequential_interference_a=sequential_interference_a,
                    sequential_interference_b=sequential_interference_b,
                    sequential_learning_b=sequential_learning_b,
                    sequential_learning_c=sequential_learning_c,
                    routed_accretion_a=routed_accretion_a,
                    routed_interference_a=routed_interference_a,
                    routed_interference_b=routed_interference_b,
                    routed_learning_b=routed_learning_b,
                    routed_learning_c=routed_learning_c,
                ),
                "lora_cosine_a_b_mean": lora_cosine_a_b_mean,
                "lora_cosine_a_c_mean": lora_cosine_a_c_mean,
                "lora_cosine_b_c_mean": lora_cosine_b_c_mean,
                "grad_cosine_a_b_after_a": grad_cosine_a_b_after_a,
                "grad_cosine_a_c_after_b": grad_cosine_a_c_after_b,
            }
        )
    return results


def run_calibrated_routed_accretion_variant(
    variant: Variant,
    settings: LoraSettings,
    task_a_texts: list[str],
    task_b_texts: list[str],
    task_c_texts: list[str],
    phase_steps: int,
    seed: int,
    device: str,
    route_pairs: list[RouteSpec],
    compat_batches: int = 0,
    c_retention_min: float = 0.9,
    learning_b_tolerance: float = 0.05,
) -> RoutedAccretionResult:
    """Train A/B/C once, select a route on calibration probes, report held-out metrics."""
    resolved_device = resolve_device(device)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(settings.model_name)
    model = build_lora_model(settings, resolved_device)
    trainable, total = parameter_counts(model)

    train_a, eval_a_full = prepare_encoded_splits(
        tokenizer, task_a_texts, settings, seed, resolved_device
    )
    train_b, eval_b_full = prepare_encoded_splits(
        tokenizer, task_b_texts, settings, seed + 10_000, resolved_device
    )
    train_c, eval_c_full = prepare_encoded_splits(
        tokenizer, task_c_texts, settings, seed + 20_000, resolved_device
    )
    eval_a_select, eval_a, heldout_a = split_eval_encoded(eval_a_full, settings.batch_size)
    eval_b_select, eval_b, heldout_b = split_eval_encoded(eval_b_full, settings.batch_size)
    eval_c_select, eval_c, heldout_c = split_eval_encoded(eval_c_full, settings.batch_size)
    heldout_report = heldout_a and heldout_b and heldout_c
    select_settings = settings_for_eval_split(
        settings,
        eval_a_select,
        eval_b_select,
        eval_c_select,
    )
    report_settings = settings_for_eval_split(settings, eval_a, eval_b, eval_c)

    eval_a_before = eval_loss(model, eval_a, report_settings)
    eval_b_before = eval_loss(model, eval_b, report_settings)
    eval_c_before = eval_loss(model, eval_c, report_settings)
    lora_initial = lora_effective_deltas(model)

    train_steps(model, train_a, variant, settings, phase_steps)
    eval_a_after_a = eval_loss(model, eval_a, report_settings)
    eval_b_after_a = eval_loss(model, eval_b, report_settings)
    eval_c_after_a = eval_loss(model, eval_c, report_settings)
    eval_a_after_a_select = eval_loss(model, eval_a_select, select_settings)
    eval_b_after_a_select = eval_loss(model, eval_b_select, select_settings)
    state_a = snapshot_trainable_state(model)
    lora_after_a = lora_effective_deltas(model)
    grad_cosine_a_b_after_a = gradient_cosine(
        model, train_a, train_b, settings, compat_batches
    )

    train_steps(model, train_b, variant, settings, phase_steps)
    eval_a_after_b = eval_loss(model, eval_a, report_settings)
    eval_b_after_b = eval_loss(model, eval_b, report_settings)
    eval_c_after_b = eval_loss(model, eval_c, report_settings)
    eval_a_after_b_select = eval_loss(model, eval_a_select, select_settings)
    eval_b_after_b_select = eval_loss(model, eval_b_select, select_settings)
    eval_c_after_b_select = eval_loss(model, eval_c_select, select_settings)
    state_b = snapshot_trainable_state(model)
    lora_after_b = lora_effective_deltas(model)
    grad_cosine_a_c_after_b = gradient_cosine(
        model, train_a, train_c, settings, compat_batches
    )

    train_steps(model, train_c, variant, settings, phase_steps)
    sequential_eval_a = eval_loss(model, eval_a, report_settings)
    sequential_eval_b = eval_loss(model, eval_b, report_settings)
    sequential_eval_c = eval_loss(model, eval_c, report_settings)
    sequential_eval_a_select = eval_loss(model, eval_a_select, select_settings)
    sequential_eval_b_select = eval_loss(model, eval_b_select, select_settings)
    sequential_eval_c_select = eval_loss(model, eval_c_select, select_settings)
    state_c = snapshot_trainable_state(model)
    lora_after_c = lora_effective_deltas(model)

    delta_b_state = subtract_state(state_b, state_a)
    delta_c_state = subtract_state(state_c, state_b)
    lora_delta_a = subtract_lora_deltas(lora_after_a, lora_initial)
    lora_delta_b = subtract_lora_deltas(lora_after_b, lora_after_a)
    lora_delta_c = subtract_lora_deltas(lora_after_c, lora_after_b)
    lora_cosine_a_b_mean = mean_lora_cosine(lora_delta_a, lora_delta_b)
    lora_cosine_a_c_mean = mean_lora_cosine(lora_delta_a, lora_delta_c)
    lora_cosine_b_c_mean = mean_lora_cosine(lora_delta_b, lora_delta_c)
    max_layer_index = max_trainable_layer_index(delta_c_state)

    route_by_spec = {route_spec_string(route): route for route in route_pairs}
    selection_candidates: list[dict[str, float | bool | str]] = []
    for route in route_pairs:
        route_spec = route_spec_string(route)
        routed_state = compose_grouped_route_state(
            state_a,
            delta_b_state=delta_b_state,
            delta_c_state=delta_c_state,
            route=route,
            max_layer_index=max_layer_index,
        )
        routed_eval_a_select, routed_eval_b_select, routed_eval_c_select = evaluate_route_state(
            model,
            routed_state,
            eval_a_select,
            eval_b_select,
            eval_c_select,
            select_settings,
        )
        selection_metrics = routed_metric_values(
            eval_a_after_a=eval_a_after_a_select,
            eval_b_after_a=eval_b_after_a_select,
            eval_a_after_b=eval_a_after_b_select,
            eval_b_after_b=eval_b_after_b_select,
            eval_c_after_b=eval_c_after_b_select,
            sequential_eval_a=sequential_eval_a_select,
            sequential_eval_b=sequential_eval_b_select,
            sequential_eval_c=sequential_eval_c_select,
            routed_eval_a=routed_eval_a_select,
            routed_eval_b=routed_eval_b_select,
            routed_eval_c=routed_eval_c_select,
        )
        constraint_passed = route_passes_selection_constraints(
            selection_metrics,
            c_retention_min=c_retention_min,
            learning_b_tolerance=learning_b_tolerance,
        )
        selection_candidates.append(
            selection_candidate_record(
                route_spec=route_spec,
                route=route,
                metrics=selection_metrics,
                constraint_passed=constraint_passed,
            )
        )

    selected_candidate = select_calibrated_candidate(selection_candidates)
    selected_route = route_by_spec[str(selected_candidate["route_spec"])]
    selected_state = compose_grouped_route_state(
        state_a,
        delta_b_state=delta_b_state,
        delta_c_state=delta_c_state,
        route=selected_route,
        max_layer_index=max_layer_index,
    )
    routed_eval_a, routed_eval_b, routed_eval_c = evaluate_route_state(
        model,
        selected_state,
        eval_a,
        eval_b,
        eval_c,
        report_settings,
    )
    metrics = routed_metric_values(
        eval_a_after_a=eval_a_after_a,
        eval_b_after_a=eval_b_after_a,
        eval_a_after_b=eval_a_after_b,
        eval_b_after_b=eval_b_after_b,
        eval_c_after_b=eval_c_after_b,
        sequential_eval_a=sequential_eval_a,
        sequential_eval_b=sequential_eval_b,
        sequential_eval_c=sequential_eval_c,
        routed_eval_a=routed_eval_a,
        routed_eval_b=routed_eval_b,
        routed_eval_c=routed_eval_c,
    )
    route_c_scale = mean_route_c_scale(selected_route)

    return {
        "variant": f"{variant.name}_calibrated",
        "model": settings.model_name,
        "device": resolved_device,
        "seed": seed,
        "diversity_weight": variant.diversity,
        "repulsion_weight": variant.repulsion,
        "sparse_weight": variant.sparse,
        "gossip_weight": variant.gossip,
        "gossip_tau": variant.gossip_tau,
        "gossip_k": variant.gossip_k,
        "max_gossip_vectors": variant.max_gossip_vectors,
        "route_b_scale": selected_route.b_scale,
        "route_c_scale": route_c_scale,
        "route_c_lora_a_scale": selected_route.c_lora_a_scale,
        "route_c_lora_b_scale": selected_route.c_lora_b_scale,
        "route_c_early_scale": selected_route.c_early_scale,
        "route_c_middle_scale": selected_route.c_middle_scale,
        "route_c_late_scale": selected_route.c_late_scale,
        "heldout_report": heldout_report,
        "selection_candidate_count": len(selection_candidates),
        "selection_constraint_passed": bool(selected_candidate["selection_constraint_passed"]),
        "selection_objective_score": float(selected_candidate["selection_objective_score"]),
        "selection_frontier_score": float(selected_candidate["frontier_score"]),
        "selection_delta_accretion_a": float(selected_candidate["delta_accretion_a"]),
        "selection_delta_interference_a": float(selected_candidate["delta_interference_a"]),
        "selection_delta_interference_b": float(selected_candidate["delta_interference_b"]),
        "selection_delta_learning_b": float(selected_candidate["delta_learning_b"]),
        "selection_delta_learning_c": float(selected_candidate["delta_learning_c"]),
        "selection_routed_learning_c": float(selected_candidate["routed_learning_c"]),
        "selection_sequential_learning_c": float(selected_candidate["sequential_learning_c"]),
        "selection_candidates": selection_candidates,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "eval_a_before": eval_a_before,
        "eval_b_before": eval_b_before,
        "eval_c_before": eval_c_before,
        "eval_a_after_a": eval_a_after_a,
        "eval_b_after_a": eval_b_after_a,
        "eval_c_after_a": eval_c_after_a,
        "eval_a_after_b": eval_a_after_b,
        "eval_b_after_b": eval_b_after_b,
        "eval_c_after_b": eval_c_after_b,
        "sequential_eval_a": sequential_eval_a,
        "sequential_eval_b": sequential_eval_b,
        "sequential_eval_c": sequential_eval_c,
        "routed_eval_a": routed_eval_a,
        "routed_eval_b": routed_eval_b,
        "routed_eval_c": routed_eval_c,
        **metrics,
        "lora_cosine_a_b_mean": lora_cosine_a_b_mean,
        "lora_cosine_a_c_mean": lora_cosine_a_c_mean,
        "lora_cosine_b_c_mean": lora_cosine_b_c_mean,
        "grad_cosine_a_b_after_a": grad_cosine_a_b_after_a,
        "grad_cosine_a_c_after_b": grad_cosine_a_c_after_b,
    }


def evaluate_route_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    eval_a: dict[str, torch.Tensor],
    eval_b: dict[str, torch.Tensor],
    eval_c: dict[str, torch.Tensor],
    settings: LoraSettings,
) -> tuple[float, float, float]:
    """Apply and evaluate one routed adapter state on A/B/C probes."""
    apply_trainable_state(model, state)
    return (
        eval_loss(model, eval_a, settings),
        eval_loss(model, eval_b, settings),
        eval_loss(model, eval_c, settings),
    )


def settings_for_eval_split(
    settings: LoraSettings,
    *encoded_splits: dict[str, torch.Tensor],
) -> LoraSettings:
    """Return settings that evaluate each sample in the smallest provided split once."""
    if not encoded_splits:
        return settings
    eval_batches = min(
        max(1, encoded["input_ids"].shape[0] // settings.batch_size)
        for encoded in encoded_splits
    )
    return replace(settings, eval_batches=eval_batches)


def routed_metric_values(
    *,
    eval_a_after_a: float,
    eval_b_after_a: float,
    eval_a_after_b: float,
    eval_b_after_b: float,
    eval_c_after_b: float,
    sequential_eval_a: float,
    sequential_eval_b: float,
    sequential_eval_c: float,
    routed_eval_a: float,
    routed_eval_b: float,
    routed_eval_c: float,
) -> dict[str, float]:
    """Return route metrics using the phase-local pre-C baseline for C learning."""
    sequential_accretion_a = eval_a_after_a - eval_a_after_b
    sequential_interference_a = sequential_eval_a - eval_a_after_b
    sequential_interference_b = sequential_eval_b - eval_b_after_b
    sequential_learning_b = eval_b_after_a - eval_b_after_b
    sequential_learning_c = eval_c_after_b - sequential_eval_c
    routed_accretion_a = eval_a_after_a - routed_eval_a
    routed_interference_a = routed_eval_a - eval_a_after_b
    routed_interference_b = routed_eval_b - eval_b_after_b
    routed_learning_b = eval_b_after_a - routed_eval_b
    routed_learning_c = eval_c_after_b - routed_eval_c
    delta_accretion_a = routed_accretion_a - sequential_accretion_a
    delta_interference_a = sequential_interference_a - routed_interference_a
    delta_interference_b = sequential_interference_b - routed_interference_b
    delta_learning_b = routed_learning_b - sequential_learning_b
    delta_learning_c = routed_learning_c - sequential_learning_c
    score = frontier_score(
        sequential_accretion_a=sequential_accretion_a,
        sequential_interference_a=sequential_interference_a,
        sequential_interference_b=sequential_interference_b,
        sequential_learning_b=sequential_learning_b,
        sequential_learning_c=sequential_learning_c,
        routed_accretion_a=routed_accretion_a,
        routed_interference_a=routed_interference_a,
        routed_interference_b=routed_interference_b,
        routed_learning_b=routed_learning_b,
        routed_learning_c=routed_learning_c,
    )
    return {
        "sequential_accretion_a": sequential_accretion_a,
        "sequential_interference_a": sequential_interference_a,
        "sequential_interference_a_after_c": sequential_interference_a,
        "sequential_interference_b": sequential_interference_b,
        "sequential_interference_b_after_c": sequential_interference_b,
        "sequential_learning_b": sequential_learning_b,
        "sequential_learning_c": sequential_learning_c,
        "sequential_retention_a": ratio(eval_a_after_a, sequential_eval_a),
        "sequential_retention_b": ratio(eval_b_after_b, sequential_eval_b),
        "routed_accretion_a": routed_accretion_a,
        "routed_interference_a": routed_interference_a,
        "routed_interference_a_after_c": routed_interference_a,
        "routed_interference_b": routed_interference_b,
        "routed_interference_b_after_c": routed_interference_b,
        "routed_learning_b": routed_learning_b,
        "routed_learning_c": routed_learning_c,
        "routed_retention_a": ratio(eval_a_after_a, routed_eval_a),
        "routed_retention_b": ratio(eval_b_after_b, routed_eval_b),
        "delta_accretion_a": delta_accretion_a,
        "delta_interference_a": delta_interference_a,
        "delta_interference_b": delta_interference_b,
        "delta_learning_b": delta_learning_b,
        "delta_learning_c": delta_learning_c,
        "frontier_score": score,
    }


def route_passes_selection_constraints(
    metrics: dict[str, float],
    *,
    c_retention_min: float,
    learning_b_tolerance: float,
) -> bool:
    """Return whether a candidate keeps enough B and C learning on calibration probes."""
    return (
        metrics["routed_learning_c"] >= c_retention_min * metrics["sequential_learning_c"]
        and metrics["routed_learning_b"]
        >= metrics["sequential_learning_b"] - learning_b_tolerance
    )


def route_selection_objective(metrics: dict[str, float]) -> float:
    """Score calibration-passing routes by old-task protection."""
    return (
        metrics["delta_accretion_a"]
        + metrics["delta_interference_a"]
        + metrics["delta_interference_b"]
    )


def selection_candidate_record(
    *,
    route_spec: str,
    route: RouteSpec,
    metrics: dict[str, float],
    constraint_passed: bool,
) -> dict[str, float | bool | str]:
    """Return a compact serializable calibration candidate record."""
    return {
        "route_spec": route_spec,
        "route_b_scale": route.b_scale,
        "route_c_scale": mean_route_c_scale(route),
        "selection_constraint_passed": constraint_passed,
        "selection_objective_score": route_selection_objective(metrics),
        "frontier_score": metrics["frontier_score"],
        "delta_accretion_a": metrics["delta_accretion_a"],
        "delta_interference_a": metrics["delta_interference_a"],
        "delta_interference_b": metrics["delta_interference_b"],
        "delta_learning_b": metrics["delta_learning_b"],
        "delta_learning_c": metrics["delta_learning_c"],
        "routed_learning_b": metrics["routed_learning_b"],
        "routed_learning_c": metrics["routed_learning_c"],
        "sequential_learning_b": metrics["sequential_learning_b"],
        "sequential_learning_c": metrics["sequential_learning_c"],
    }


def select_calibrated_candidate(
    candidates: list[dict[str, float | bool | str]],
) -> dict[str, float | bool | str]:
    """Select a calibration route, preferring constraint-passing old-task protection."""
    if not candidates:
        raise ValueError("calibration route selection requires at least one candidate")
    passing = [candidate for candidate in candidates if candidate["selection_constraint_passed"]]
    if passing:
        return max(
            passing,
            key=lambda candidate: (
                float(candidate["selection_objective_score"]),
                float(candidate["routed_learning_c"]),
                float(candidate["frontier_score"]),
            ),
        )
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["frontier_score"]),
            float(candidate["selection_objective_score"]),
            float(candidate["routed_learning_c"]),
        ),
    )


def route_variant_name(
    variant_name: str,
    route: RouteSpec,
    multi_route: bool,
) -> str:
    """Return a stable variant label for single routes and route sweeps."""
    if not multi_route:
        return variant_name
    if is_layer_route(route):
        return (
            f"{variant_name}_b{route.b_scale:g}"
            f"_ce{route.c_early_scale:g}"
            f"_cm{route.c_middle_scale:g}"
            f"_cl{route.c_late_scale:g}"
        )
    if route.c_lora_a_scale == route.c_lora_b_scale:
        return f"{variant_name}_b{route.b_scale:g}_c{route.c_lora_a_scale:g}"
    return (
        f"{variant_name}_b{route.b_scale:g}"
        f"_ca{route.c_lora_a_scale:g}_cb{route.c_lora_b_scale:g}"
    )


def scalar_route_spec(b_scale: float, c_scale: float) -> RouteSpec:
    """Return a route spec with one scalar C scale."""
    return RouteSpec(b_scale, c_scale, c_scale, c_scale, c_scale, c_scale)


def grouped_route_spec(b_scale: float, c_lora_a_scale: float, c_lora_b_scale: float) -> RouteSpec:
    """Return a route spec with LoRA A/B tensor-family C scales."""
    mean_c_scale = (c_lora_a_scale + c_lora_b_scale) / 2.0
    return RouteSpec(
        b_scale,
        c_lora_a_scale,
        c_lora_b_scale,
        mean_c_scale,
        mean_c_scale,
        mean_c_scale,
    )


def layer_route_spec(
    b_scale: float,
    c_early_scale: float,
    c_middle_scale: float,
    c_late_scale: float,
) -> RouteSpec:
    """Return a route spec with early/middle/late layer-band C scales."""
    mean_c_scale = (c_early_scale + c_middle_scale + c_late_scale) / 3.0
    return RouteSpec(
        b_scale,
        mean_c_scale,
        mean_c_scale,
        c_early_scale,
        c_middle_scale,
        c_late_scale,
    )


def mean_route_c_scale(route: RouteSpec) -> float:
    """Return a scalar compatibility view of C route scales."""
    return (route.c_early_scale + route.c_middle_scale + route.c_late_scale) / 3.0


def is_layer_route(route: RouteSpec) -> bool:
    """Return whether a route uses non-uniform layer-band C scales."""
    return len({route.c_early_scale, route.c_middle_scale, route.c_late_scale}) > 1


def c_scale_for_parameter(
    name: str,
    *,
    route: RouteSpec,
    max_layer_index: int | None = None,
) -> float:
    """Return the C-update scale for a LoRA trainable parameter name."""
    layer_index = extract_layer_index(name)
    if layer_index is not None and max_layer_index is not None and is_layer_route(route):
        return c_scale_for_layer(
            layer_index,
            max_layer_index=max_layer_index,
            route=route,
        )
    if ".lora_A." in name:
        return route.c_lora_a_scale
    if ".lora_B." in name:
        return route.c_lora_b_scale
    return mean_route_c_scale(route)


def extract_layer_index(name: str) -> int | None:
    """Extract a transformer block index from common HF parameter names."""
    for pattern in (r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.layer\.(\d+)\."):
        match = re.search(pattern, name)
        if match is not None:
            return int(match.group(1))
    return None


def max_trainable_layer_index(state: dict[str, torch.Tensor]) -> int | None:
    """Return the largest detected layer index in a trainable state."""
    layer_indices = [index for name in state if (index := extract_layer_index(name)) is not None]
    return max(layer_indices) if layer_indices else None


def c_scale_for_layer(layer_index: int, *, max_layer_index: int, route: RouteSpec) -> float:
    """Return the C scale for a layer using early/middle/late thirds."""
    layer_count = max_layer_index + 1
    if layer_index < layer_count / 3:
        return route.c_early_scale
    if layer_index < (2 * layer_count) / 3:
        return route.c_middle_scale
    return route.c_late_scale


def compose_grouped_route_state(
    base: dict[str, torch.Tensor],
    *,
    delta_b_state: dict[str, torch.Tensor],
    delta_c_state: dict[str, torch.Tensor],
    route: RouteSpec,
    max_layer_index: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compose a routed state with grouped or layer-banded C scales."""
    composed = compose_state(base, [(route.b_scale, delta_b_state)])
    for name in composed.keys() & delta_c_state.keys():
        c_scale = c_scale_for_parameter(
            name,
            route=route,
            max_layer_index=max_layer_index,
        )
        composed[name] = composed[name] + (delta_c_state[name] * c_scale)
    return composed


def frontier_score(
    *,
    sequential_accretion_a: float,
    sequential_interference_a: float,
    sequential_interference_b: float,
    sequential_learning_b: float,
    sequential_learning_c: float,
    routed_accretion_a: float,
    routed_interference_a: float,
    routed_interference_b: float,
    routed_learning_b: float,
    routed_learning_c: float,
) -> float:
    """Score a route by balanced improvement over blind sequential."""
    return (
        (routed_accretion_a - sequential_accretion_a)
        + (sequential_interference_a - routed_interference_a)
        + (sequential_interference_b - routed_interference_b)
        + (routed_learning_c - sequential_learning_c)
        + (0.25 * (routed_learning_b - sequential_learning_b))
    )


def summarize_routed_accretion(
    results: list[RoutedAccretionResult],
) -> dict[str, dict[str, float]]:
    """Aggregate fixed routed-update metrics by variant."""
    metric_names = [
        "route_b_scale",
        "route_c_scale",
        "route_c_lora_a_scale",
        "route_c_lora_b_scale",
        "route_c_early_scale",
        "route_c_middle_scale",
        "route_c_late_scale",
        "heldout_report",
        "selection_candidate_count",
        "selection_constraint_passed",
        "selection_objective_score",
        "selection_frontier_score",
        "selection_delta_accretion_a",
        "selection_delta_interference_a",
        "selection_delta_interference_b",
        "selection_delta_learning_b",
        "selection_delta_learning_c",
        "selection_routed_learning_c",
        "selection_sequential_learning_c",
        "sequential_eval_a",
        "sequential_eval_b",
        "sequential_eval_c",
        "routed_eval_a",
        "routed_eval_b",
        "routed_eval_c",
        "sequential_accretion_a",
        "sequential_interference_a",
        "sequential_interference_b",
        "sequential_learning_b",
        "sequential_learning_c",
        "sequential_retention_a",
        "sequential_retention_b",
        "routed_accretion_a",
        "routed_interference_a",
        "routed_interference_b",
        "routed_learning_b",
        "routed_learning_c",
        "routed_retention_a",
        "routed_retention_b",
        "sequential_interference_a_after_c",
        "sequential_interference_b_after_c",
        "routed_interference_a_after_c",
        "routed_interference_b_after_c",
        "delta_accretion_a",
        "delta_interference_a",
        "delta_interference_b",
        "delta_learning_b",
        "delta_learning_c",
        "frontier_score",
        "lora_cosine_a_b_mean",
        "lora_cosine_a_c_mean",
        "lora_cosine_b_c_mean",
        "grad_cosine_a_b_after_a",
        "grad_cosine_a_c_after_b",
    ]
    summary: dict[str, dict[str, float]] = {}
    for variant in sorted({result["variant"] for result in results}):
        group = [result for result in results if result["variant"] == variant]
        values: dict[str, float] = {"count": float(len(group))}
        for metric in metric_names:
            metric_values = [
                float(value)
                for result in group
                if (value := dict(result).get(metric)) is not None
            ]
            if not metric_values:
                continue
            values[f"{metric}_mean"] = statistics.fmean(metric_values)
            values[f"{metric}_std"] = (
                statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            )
        values["routed_accretion_win_count"] = float(
            sum(result["routed_accretion_a"] > result["sequential_accretion_a"] for result in group)
        )
        values["routed_interference_a_win_count"] = float(
            sum(
                result["routed_interference_a"] < result["sequential_interference_a"]
                for result in group
            )
        )
        values["routed_interference_b_win_count"] = float(
            sum(
                result["routed_interference_b"] < result["sequential_interference_b"]
                for result in group
            )
        )
        values["routed_learning_c_preserved_count"] = float(
            sum(result["routed_learning_c"] >= result["sequential_learning_c"] for result in group)
        )
        values["accretion_a_win_count"] = values["routed_accretion_win_count"]
        values["interference_a_win_count"] = values["routed_interference_a_win_count"]
        values["interference_b_win_count"] = values["routed_interference_b_win_count"]
        values["learning_c_preserved_count"] = values["routed_learning_c_preserved_count"]
        values["frontier_score_win_count"] = float(
            sum(result["frontier_score"] > 0.0 for result in group)
        )
        summary[variant] = values
    return summary


def parse_route_pairs(values: list[str]) -> list[RouteSpec]:
    """Parse route pair specs like `0.9:0.25`."""
    route_pairs = []
    for value in values:
        b_scale, c_scale = value.split(":", maxsplit=1)
        route_pairs.append(scalar_route_spec(float(b_scale), float(c_scale)))
    return route_pairs


def parse_group_route_pairs(values: list[str]) -> list[RouteSpec]:
    """Parse grouped route specs like `0.9:0.4:0.8` as `B:C_A:C_B`."""
    route_pairs = []
    for value in values:
        b_scale, c_lora_a_scale, c_lora_b_scale = value.split(":", maxsplit=2)
        route_pairs.append(
            grouped_route_spec(float(b_scale), float(c_lora_a_scale), float(c_lora_b_scale))
        )
    return route_pairs


def parse_layer_route_pairs(values: list[str]) -> list[RouteSpec]:
    """Parse layer-band route specs like `0.9:0.25:0.4:0.9` as `B:C_E:C_M:C_L`."""
    route_pairs = []
    for value in values:
        b_scale, c_early_scale, c_middle_scale, c_late_scale = value.split(":", maxsplit=3)
        route_pairs.append(
            layer_route_spec(
                float(b_scale),
                float(c_early_scale),
                float(c_middle_scale),
                float(c_late_scale),
            )
        )
    return route_pairs


def route_spec_string(route: RouteSpec) -> str:
    """Return a compact serializable route specification."""
    if is_layer_route(route):
        return (
            f"{route.b_scale:g}:{route.c_early_scale:g}:"
            f"{route.c_middle_scale:g}:{route.c_late_scale:g}"
        )
    if route.c_lora_a_scale != route.c_lora_b_scale:
        return f"{route.b_scale:g}:{route.c_lora_a_scale:g}:{route.c_lora_b_scale:g}"
    return f"{route.b_scale:g}:{route.c_lora_a_scale:g}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-routed-accretion`."""
    parser = argparse.ArgumentParser(description="Run fixed routed-update accretion tests.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, or cuda")
    parser.add_argument("--task-a-file", required=True)
    parser.add_argument("--task-b-file", required=True)
    parser.add_argument("--task-c-file", required=True)
    parser.add_argument("--phase-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--compat-batches", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--variant", default="gossip")
    parser.add_argument("--diversity-weight", type=float, default=None)
    parser.add_argument("--repulsion-weight", type=float, default=None)
    parser.add_argument("--sparse-weight", type=float, default=None)
    parser.add_argument("--gossip-weight", type=float, default=None)
    parser.add_argument("--gossip-tau", type=float, default=None)
    parser.add_argument("--gossip-k", type=int, default=None)
    parser.add_argument("--max-gossip-vectors", type=int, default=None)
    parser.add_argument("--route-b-scale", type=float, default=0.9)
    parser.add_argument("--route-c-scale", type=float, default=0.25)
    parser.add_argument(
        "--calibrate-route",
        action="store_true",
        help="Select one route on calibration probes and report only held-out metrics.",
    )
    parser.add_argument(
        "--selection-c-retention-min",
        type=float,
        default=0.9,
        help="Minimum routed/sequential C-learning ratio required during route selection.",
    )
    parser.add_argument(
        "--selection-learning-b-tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute B-learning drop during route selection.",
    )
    parser.add_argument(
        "--route-pairs",
        nargs="*",
        default=None,
        help="Optional routed scale sweep as B:C pairs, e.g. 0.9:0.25 1.0:0.15.",
    )
    parser.add_argument(
        "--group-route-pairs",
        nargs="*",
        default=None,
        help=(
            "Optional grouped route sweep as B:C_A:C_B triples, e.g. 0.9:0.4:0.8. "
            "C_A scales LoRA A tensors and C_B scales LoRA B tensors."
        ),
    )
    parser.add_argument(
        "--layer-route-pairs",
        nargs="*",
        default=None,
        help=(
            "Optional layer-band route sweep as B:C_E:C_M:C_L quadruples, "
            "e.g. 0.9:0.25:0.4:0.9. C_E/C_M/C_L scale early/middle/late layers."
        ),
    )
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for fixed routed-update accretion experiments."""
    args = parse_args()
    settings = LoraSettings(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tuple(args.target_modules or ()),
    )
    variant = build_variants(
        [args.variant],
        diversity=args.diversity_weight,
        repulsion=args.repulsion_weight,
        sparse=args.sparse_weight,
        gossip=args.gossip_weight,
        gossip_tau=args.gossip_tau,
        gossip_k=args.gossip_k,
        max_gossip_vectors=args.max_gossip_vectors,
    )[0]
    seeds = args.seeds or [args.seed]
    route_modes = [
        args.route_pairs is not None,
        args.group_route_pairs is not None,
        args.layer_route_pairs is not None,
    ]
    if sum(route_modes) > 1:
        raise ValueError(
            "Use only one of --route-pairs, --group-route-pairs, or --layer-route-pairs."
        )
    if args.layer_route_pairs is not None:
        route_pairs = parse_layer_route_pairs(args.layer_route_pairs)
    elif args.group_route_pairs is not None:
        route_pairs = parse_group_route_pairs(args.group_route_pairs)
    elif args.route_pairs is not None:
        route_pairs = parse_route_pairs(args.route_pairs)
    else:
        route_pairs = [scalar_route_spec(args.route_b_scale, args.route_c_scale)]
    if not route_pairs:
        raise ValueError("at least one route candidate is required")
    task_a_texts = load_texts(args.task_a_file)
    task_b_texts = load_texts(args.task_b_file)
    task_c_texts = load_texts(args.task_c_file)
    if args.calibrate_route:
        results = [
            run_calibrated_routed_accretion_variant(
                variant,
                settings=settings,
                task_a_texts=task_a_texts,
                task_b_texts=task_b_texts,
                task_c_texts=task_c_texts,
                phase_steps=args.phase_steps,
                seed=seed,
                device=args.device,
                route_pairs=route_pairs,
                compat_batches=args.compat_batches,
                c_retention_min=args.selection_c_retention_min,
                learning_b_tolerance=args.selection_learning_b_tolerance,
            )
            for seed in seeds
        ]
    else:
        results = [
            result
            for seed in seeds
            for result in run_routed_accretion_variants(
                variant,
                settings=settings,
                task_a_texts=task_a_texts,
                task_b_texts=task_b_texts,
                task_c_texts=task_c_texts,
                phase_steps=args.phase_steps,
                seed=seed,
                device=args.device,
                route_pairs=route_pairs,
                compat_batches=args.compat_batches,
            )
        ]
    record: RoutedAccretionRunRecord = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "model": args.model,
            "device": args.device,
            "task_a_file": args.task_a_file,
            "task_b_file": args.task_b_file,
            "task_c_file": args.task_c_file,
            "phase_steps": args.phase_steps,
            "seeds": seeds,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "compat_batches": args.compat_batches,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(settings.target_modules),
            "variant": variant.name,
            "gossip_weight": args.gossip_weight,
            "gossip_tau": args.gossip_tau,
            "gossip_k": args.gossip_k,
            "max_gossip_vectors": args.max_gossip_vectors,
            "route_b_scale": args.route_b_scale,
            "route_c_scale": args.route_c_scale,
            "calibrate_route": args.calibrate_route,
            "selection_c_retention_min": args.selection_c_retention_min,
            "selection_learning_b_tolerance": args.selection_learning_b_tolerance,
            "route_pairs": [
                route_spec_string(route)
                for route in route_pairs
            ],
            "frontier_score": (
                "accretion_delta + a_interference_reduction + b_interference_reduction "
                "+ learning_c_delta + 0.25 * learning_b_delta"
            ),
        },
        "git_status": git_status(),
        "results": results,
        "summary": summarize_routed_accretion(results),
    }
    if args.output_dir is not None:
        write_run_record(record, args.output_dir)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
