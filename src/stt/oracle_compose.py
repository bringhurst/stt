"""Oracle post-hoc LoRA composition for accretion experiments.

This module intentionally uses task evaluation losses to choose adapter routes.
It is an unfair upper-bound experiment: if oracle composition cannot improve the
A/B/C tradeoff, a learned non-oracle router is unlikely to be useful yet.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

import torch
from torch import Tensor

from stt.accretion import (
    gradient_cosine,
    lora_effective_deltas,
    mean_lora_cosine,
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

Route = Literal["shared", "private", "conflict_private", "reject_or_downweight"]


class CompositionResult(TypedDict):
    """Serializable evaluation for one composed adapter state."""

    variant: str
    seed: int
    composition: str
    b_scale: float
    c_scale: float
    route: str
    eval_a: float
    eval_b: float
    eval_c: float
    accretion_a: float
    learning_b: float
    learning_c: float
    interference_a: float | None
    interference_b: float | None


class OracleSeedResult(TypedDict):
    """Serializable oracle composition result for one seed."""

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
    heldout_report: bool
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
    eval_a_after_c: float
    eval_b_after_c: float
    eval_c_after_c: float
    sequential_accretion_a_after_b: float
    sequential_interference_a_after_c: float
    sequential_interference_b_after_c: float
    sequential_learning_b: float
    sequential_learning_c: float
    sequential_eval_c: float
    selected_b_scale: float
    selected_b_route: str
    selected_c_scale: float
    selected_c_route: str
    oracle_eval_a: float
    oracle_eval_b: float
    oracle_eval_c: float
    oracle_accretion_a: float
    oracle_learning_b: float
    oracle_learning_c: float
    oracle_interference_a: float
    oracle_interference_b: float
    fixed_b_scale: float
    fixed_c_scale: float
    fixed_eval_a: float
    fixed_eval_b: float
    fixed_eval_c: float
    fixed_accretion_a: float
    fixed_learning_b: float
    fixed_learning_c: float
    fixed_interference_a: float
    fixed_interference_b: float
    lora_cosine_a_b_mean: float | None
    lora_cosine_a_c_mean: float | None
    lora_cosine_b_c_mean: float | None
    grad_cosine_a_b_after_a: float | None
    grad_cosine_a_c_after_b: float | None
    compositions: list[CompositionResult]


class OracleRunRecord(TypedDict):
    """Persisted oracle composition experiment record."""

    created_at: str
    config: dict[str, Any]
    git_status: str
    results: list[OracleSeedResult]
    summary: dict[str, dict[str, float]]


def snapshot_trainable_state(model: torch.nn.Module) -> dict[str, Tensor]:
    """Return detached CPU copies of trainable parameters keyed by name."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def subtract_state(
    later: dict[str, Tensor],
    earlier: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Return parameter-wise state difference for shared keys."""
    return {name: later[name] - earlier[name] for name in later.keys() & earlier.keys()}


def compose_state(
    base: dict[str, Tensor],
    updates: list[tuple[float, dict[str, Tensor]]],
) -> dict[str, Tensor]:
    """Return `base` plus scaled update states."""
    composed = {name: value.clone() for name, value in base.items()}
    for scale, update in updates:
        for name in composed.keys() & update.keys():
            composed[name] = composed[name] + (update[name] * scale)
    return composed


def apply_trainable_state(model: torch.nn.Module, state: dict[str, Tensor]) -> None:
    """Copy a trainable parameter state into a model."""
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def split_eval_encoded(
    encoded: dict[str, Tensor],
    batch_size: int,
) -> tuple[dict[str, Tensor], dict[str, Tensor], bool]:
    """Split encoded eval tensors into selection and held-out report halves."""
    size = encoded["input_ids"].shape[0]
    if size < batch_size * 2:
        return encoded, encoded, False
    midpoint = size // 2
    select = {name: value[:midpoint] for name, value in encoded.items()}
    report = {name: value[midpoint:] for name, value in encoded.items()}
    return select, report, True


def classify_route(
    old_delta: float,
    new_delta: float,
    eps_old: float,
    eps_new: float,
) -> Route:
    """Classify an oracle candidate from behavioral old/new loss deltas."""
    if old_delta > eps_old and new_delta > eps_new:
        return "shared"
    if abs(old_delta) <= eps_old and new_delta > eps_new:
        return "private"
    if old_delta < -eps_old and new_delta > eps_new:
        return "conflict_private"
    return "reject_or_downweight"


def select_b_candidate(candidates: list[CompositionResult]) -> CompositionResult:
    """Select the best B candidate, preferring non-conflicting B learning."""
    viable = [
        candidate
        for candidate in candidates
        if candidate["route"] in {"shared", "private"} and candidate["learning_b"] > 0.0
    ]
    if viable:
        return max(viable, key=lambda candidate: candidate["learning_b"])
    return max(
        candidates,
        key=lambda candidate: (candidate["accretion_a"], candidate["learning_b"]),
    )


def select_c_candidate(candidates: list[CompositionResult]) -> CompositionResult:
    """Select the best safe C candidate, otherwise keep C private at scale zero."""
    viable = [
        candidate
        for candidate in candidates
        if candidate["route"] in {"shared", "private"} and candidate["learning_c"] > 0.0
    ]
    if viable:
        return max(viable, key=lambda candidate: candidate["learning_c"])
    zero = [candidate for candidate in candidates if candidate["c_scale"] == 0.0]
    if zero:
        return zero[0]
    return min(candidates, key=lambda candidate: candidate["c_scale"])


def evaluate_composition(
    model: torch.nn.Module,
    state: dict[str, Tensor],
    eval_a: dict[str, Tensor],
    eval_b: dict[str, Tensor],
    eval_c: dict[str, Tensor],
    settings: LoraSettings,
) -> tuple[float, float, float]:
    """Apply and evaluate one composed adapter state on A/B/C."""
    apply_trainable_state(model, state)
    return (
        eval_loss(model, eval_a, settings),
        eval_loss(model, eval_b, settings),
        eval_loss(model, eval_c, settings),
    )


def composition_result(
    variant: Variant,
    seed: int,
    composition: str,
    b_scale: float,
    c_scale: float,
    route: str,
    evals: tuple[float, float, float],
    eval_a_after_a: float,
    eval_b_before: float,
    eval_c_before: float,
    without_c: tuple[float, float, float] | None = None,
) -> CompositionResult:
    """Build a serializable composition result from raw losses."""
    eval_a, eval_b, eval_c = evals
    return {
        "variant": variant.name,
        "seed": seed,
        "composition": composition,
        "b_scale": b_scale,
        "c_scale": c_scale,
        "route": route,
        "eval_a": eval_a,
        "eval_b": eval_b,
        "eval_c": eval_c,
        "accretion_a": eval_a_after_a - eval_a,
        "learning_b": eval_b_before - eval_b,
        "learning_c": eval_c_before - eval_c,
        "interference_a": None if without_c is None else eval_a - without_c[0],
        "interference_b": None if without_c is None else eval_b - without_c[1],
    }


def run_oracle_seed(
    variant: Variant,
    settings: LoraSettings,
    task_a_texts: list[str],
    task_b_texts: list[str],
    task_c_texts: list[str],
    phase_steps: int,
    seed: int,
    device: str,
    b_scales: list[float],
    c_scales: list[float],
    eps_old: float,
    eps_new: float,
    compat_batches: int = 0,
    fixed_compositions: list[tuple[float, float]] | None = None,
) -> OracleSeedResult:
    """Train A/B/C once and evaluate oracle post-hoc LoRA compositions."""
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

    fixed_compositions = fixed_compositions or [(0.9, 0.25)]

    eval_a_before = eval_loss(model, eval_a, settings)
    eval_b_before = eval_loss(model, eval_b, settings)
    eval_c_before = eval_loss(model, eval_c, settings)
    eval_b_before_select = eval_loss(model, eval_b_select, settings)
    eval_c_before_select = eval_loss(model, eval_c_select, settings)
    lora_initial = lora_effective_deltas(model)

    train_steps(model, train_a, variant, settings, phase_steps)
    eval_a_after_a = eval_loss(model, eval_a, settings)
    eval_b_after_a = eval_loss(model, eval_b, settings)
    eval_c_after_a = eval_loss(model, eval_c, settings)
    eval_a_after_a_select = eval_loss(model, eval_a_select, settings)
    eval_b_after_a_select = eval_loss(model, eval_b_select, settings)
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
    eval_a_after_c = eval_loss(model, eval_a, settings)
    eval_b_after_c = eval_loss(model, eval_b, settings)
    eval_c_after_c = eval_loss(model, eval_c, settings)
    state_c = snapshot_trainable_state(model)
    lora_after_c = lora_effective_deltas(model)

    delta_b_state = subtract_state(state_b, state_a)
    delta_c_state = subtract_state(state_c, state_b)
    lora_delta_a = subtract_lora_deltas(lora_after_a, lora_initial)
    lora_delta_b = subtract_lora_deltas(lora_after_b, lora_after_a)
    lora_delta_c = subtract_lora_deltas(lora_after_c, lora_after_b)

    compositions: list[CompositionResult] = []
    a_only_evals = evaluate_composition(model, state_a, eval_a, eval_b, eval_c, settings)
    compositions.append(
        composition_result(
            variant,
            seed,
            "A_only",
            0.0,
            0.0,
            "shared",
            a_only_evals,
            eval_a_after_a,
            eval_b_before,
            eval_c_before,
        )
    )

    b_select_candidates = []
    b_report_by_scale = {}
    for b_scale in b_scales:
        state = compose_state(state_a, [(b_scale, delta_b_state)])
        select_evals = evaluate_composition(
            model, state, eval_a_select, eval_b_select, eval_c_select, settings
        )
        report_evals = evaluate_composition(model, state, eval_a, eval_b, eval_c, settings)
        old_delta = eval_a_after_a_select - select_evals[0]
        new_delta = eval_b_after_a_select - select_evals[1]
        route = classify_route(old_delta, new_delta, eps_old, eps_new)
        select_candidate = composition_result(
            variant,
            seed,
            f"A_plus_{b_scale:g}B",
            b_scale,
            0.0,
            route,
            select_evals,
            eval_a_after_a_select,
            eval_b_before_select,
            eval_c_before_select,
        )
        report_candidate = composition_result(
            variant,
            seed,
            f"A_plus_{b_scale:g}B",
            b_scale,
            0.0,
            route,
            report_evals,
            eval_a_after_a,
            eval_b_before,
            eval_c_before,
        )
        b_select_candidates.append(select_candidate)
        b_report_by_scale[b_scale] = report_candidate
        compositions.append(report_candidate)
    selected_b_select = select_b_candidate(b_select_candidates)
    selected_b = b_report_by_scale[selected_b_select["b_scale"]]
    selected_b_scale = selected_b["b_scale"]
    selected_b_state = compose_state(state_a, [(selected_b_scale, delta_b_state)])
    selected_b_evals_select = evaluate_composition(
        model, selected_b_state, eval_a_select, eval_b_select, eval_c_select, settings
    )
    selected_b_evals_report = evaluate_composition(
        model, selected_b_state, eval_a, eval_b, eval_c, settings
    )

    c_select_candidates = []
    c_report_by_scale = {}
    for c_scale in c_scales:
        state = compose_state(selected_b_state, [(c_scale, delta_c_state)])
        select_evals = evaluate_composition(
            model, state, eval_a_select, eval_b_select, eval_c_select, settings
        )
        report_evals = evaluate_composition(model, state, eval_a, eval_b, eval_c, settings)
        old_before = statistics.fmean([selected_b_evals_select[0], selected_b_evals_select[1]])
        old_after = statistics.fmean([select_evals[0], select_evals[1]])
        old_delta = old_before - old_after
        new_delta = selected_b_evals_select[2] - select_evals[2]
        route = classify_route(old_delta, new_delta, eps_old, eps_new)
        select_candidate = composition_result(
            variant,
            seed,
            f"oracle_B_plus_{c_scale:g}C",
            selected_b_scale,
            c_scale,
            route,
            select_evals,
            eval_a_after_a_select,
            eval_b_before_select,
            eval_c_before_select,
            without_c=selected_b_evals_select,
        )
        report_candidate = composition_result(
            variant,
            seed,
            f"oracle_B_plus_{c_scale:g}C",
            selected_b_scale,
            c_scale,
            route,
            report_evals,
            eval_a_after_a,
            eval_b_before,
            eval_c_before,
            without_c=selected_b_evals_report,
        )
        c_select_candidates.append(select_candidate)
        c_report_by_scale[c_scale] = report_candidate
        compositions.append(report_candidate)

    fixed_results = []
    for fixed_b_scale, fixed_c_scale in fixed_compositions:
        state = compose_state(
            state_a,
            [(fixed_b_scale, delta_b_state), (fixed_c_scale, delta_c_state)],
        )
        report_evals = evaluate_composition(model, state, eval_a, eval_b, eval_c, settings)
        fixed_b_state = compose_state(state_a, [(fixed_b_scale, delta_b_state)])
        fixed_b_evals = evaluate_composition(model, fixed_b_state, eval_a, eval_b, eval_c, settings)
        fixed_result = composition_result(
            variant,
            seed,
            f"fixed_{fixed_b_scale:g}B_{fixed_c_scale:g}C",
            fixed_b_scale,
            fixed_c_scale,
            "fixed",
            report_evals,
            eval_a_after_a,
            eval_b_before,
            eval_c_before,
            without_c=fixed_b_evals,
        )
        fixed_results.append(fixed_result)
        compositions.append(fixed_result)

    selected_c_select = select_c_candidate(c_select_candidates)
    selected_c = c_report_by_scale[selected_c_select["c_scale"]]
    fixed_reference = fixed_results[0]
    sequential_learning_b = eval_b_before - eval_b_after_b
    sequential_learning_c = eval_c_after_b - eval_c_after_c

    return {
        "variant": variant.name,
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
        "heldout_report": heldout_report,
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
        "eval_a_after_c": eval_a_after_c,
        "eval_b_after_c": eval_b_after_c,
        "eval_c_after_c": eval_c_after_c,
        "sequential_accretion_a_after_b": eval_a_after_a - eval_a_after_b,
        "sequential_interference_a_after_c": eval_a_after_c - eval_a_after_b,
        "sequential_interference_b_after_c": eval_b_after_c - eval_b_after_b,
        "sequential_learning_b": sequential_learning_b,
        "sequential_learning_c": sequential_learning_c,
        "sequential_eval_c": eval_c_after_c,
        "selected_b_scale": selected_b_scale,
        "selected_b_route": selected_b["route"],
        "selected_c_scale": selected_c["c_scale"],
        "selected_c_route": selected_c["route"],
        "oracle_eval_a": selected_c["eval_a"],
        "oracle_eval_b": selected_c["eval_b"],
        "oracle_eval_c": selected_c["eval_c"],
        "oracle_accretion_a": selected_c["accretion_a"],
        "oracle_learning_b": selected_c["learning_b"],
        "oracle_learning_c": selected_c["learning_c"],
        "oracle_interference_a": selected_c["interference_a"] or 0.0,
        "oracle_interference_b": selected_c["interference_b"] or 0.0,
        "fixed_b_scale": fixed_reference["b_scale"],
        "fixed_c_scale": fixed_reference["c_scale"],
        "fixed_eval_a": fixed_reference["eval_a"],
        "fixed_eval_b": fixed_reference["eval_b"],
        "fixed_eval_c": fixed_reference["eval_c"],
        "fixed_accretion_a": fixed_reference["accretion_a"],
        "fixed_learning_b": fixed_reference["learning_b"],
        "fixed_learning_c": fixed_reference["learning_c"],
        "fixed_interference_a": fixed_reference["interference_a"] or 0.0,
        "fixed_interference_b": fixed_reference["interference_b"] or 0.0,
        "lora_cosine_a_b_mean": mean_lora_cosine(lora_delta_a, lora_delta_b),
        "lora_cosine_a_c_mean": mean_lora_cosine(lora_delta_a, lora_delta_c),
        "lora_cosine_b_c_mean": mean_lora_cosine(lora_delta_b, lora_delta_c),
        "grad_cosine_a_b_after_a": grad_cosine_a_b_after_a,
        "grad_cosine_a_c_after_b": grad_cosine_a_c_after_b,
        "compositions": compositions,
    }


def summarize_oracle(results: list[OracleSeedResult]) -> dict[str, dict[str, float]]:
    """Aggregate oracle seed-level metrics by variant."""
    metrics = [
        "sequential_accretion_a_after_b",
        "sequential_interference_a_after_c",
        "sequential_interference_b_after_c",
        "sequential_learning_b",
        "sequential_learning_c",
        "sequential_eval_c",
        "selected_b_scale",
        "selected_c_scale",
        "oracle_eval_a",
        "oracle_eval_b",
        "oracle_eval_c",
        "oracle_accretion_a",
        "oracle_learning_b",
        "oracle_learning_c",
        "oracle_interference_a",
        "oracle_interference_b",
        "fixed_b_scale",
        "fixed_c_scale",
        "fixed_eval_a",
        "fixed_eval_b",
        "fixed_eval_c",
        "fixed_accretion_a",
        "fixed_learning_b",
        "fixed_learning_c",
        "fixed_interference_a",
        "fixed_interference_b",
        "lora_cosine_a_b_mean",
        "lora_cosine_a_c_mean",
        "lora_cosine_b_c_mean",
    ]
    summary: dict[str, dict[str, float]] = {}
    for variant in sorted({result["variant"] for result in results}):
        group = [result for result in results if result["variant"] == variant]
        values: dict[str, float] = {"count": float(len(group))}
        for metric in metrics:
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
        values["oracle_accretion_win_count"] = float(
            sum(
                result["oracle_accretion_a"] > result["sequential_accretion_a_after_b"]
                for result in group
            )
        )
        values["oracle_interference_a_win_count"] = float(
            sum(
                result["oracle_interference_a"] < result["sequential_interference_a_after_c"]
                for result in group
            )
        )
        values["oracle_interference_b_win_count"] = float(
            sum(
                result["oracle_interference_b"] < result["sequential_interference_b_after_c"]
                for result in group
            )
        )
        values["oracle_learning_c_preserved_count"] = float(
            sum(result["oracle_learning_c"] >= result["sequential_learning_c"] for result in group)
        )
        values["fixed_accretion_win_count"] = float(
            sum(
                result["fixed_accretion_a"] > result["sequential_accretion_a_after_b"]
                for result in group
            )
        )
        values["fixed_interference_a_win_count"] = float(
            sum(
                result["fixed_interference_a"] < result["sequential_interference_a_after_c"]
                for result in group
            )
        )
        values["fixed_interference_b_win_count"] = float(
            sum(
                result["fixed_interference_b"] < result["sequential_interference_b_after_c"]
                for result in group
            )
        )
        values["fixed_learning_c_preserved_count"] = float(
            sum(result["fixed_learning_c"] >= result["sequential_learning_c"] for result in group)
        )
        summary[variant] = values
    return summary


def parse_fixed_compositions(values: list[str]) -> list[tuple[float, float]]:
    """Parse fixed composition specs like `0.9:0.25`."""
    compositions = []
    for value in values:
        b_scale, c_scale = value.split(":", maxsplit=1)
        compositions.append((float(b_scale), float(c_scale)))
    return compositions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-oracle-compose`."""
    parser = argparse.ArgumentParser(description="Run oracle post-hoc LoRA composition.")
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
    parser.add_argument("--b-scales", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--c-scales", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--fixed-compositions",
        nargs="*",
        default=["0.9:0.25"],
        help="Fixed composer baselines as B:C scale pairs, e.g. 0.9:0.25.",
    )
    parser.add_argument("--eps-old", type=float, default=0.01)
    parser.add_argument("--eps-new", type=float, default=0.01)
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for oracle composition experiments."""
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
    variants = build_variants(
        [args.variant],
        diversity=args.diversity_weight,
        repulsion=args.repulsion_weight,
        sparse=args.sparse_weight,
        gossip=args.gossip_weight,
        gossip_tau=args.gossip_tau,
        gossip_k=args.gossip_k,
        max_gossip_vectors=args.max_gossip_vectors,
    )
    variant = variants[0]
    seeds = args.seeds or [args.seed]
    task_a_texts = load_texts(args.task_a_file)
    task_b_texts = load_texts(args.task_b_file)
    task_c_texts = load_texts(args.task_c_file)
    results = [
        run_oracle_seed(
            variant,
            settings=settings,
            task_a_texts=task_a_texts,
            task_b_texts=task_b_texts,
            task_c_texts=task_c_texts,
            phase_steps=args.phase_steps,
            seed=seed,
            device=args.device,
            b_scales=args.b_scales,
            c_scales=args.c_scales,
            eps_old=args.eps_old,
            eps_new=args.eps_new,
            compat_batches=args.compat_batches,
            fixed_compositions=parse_fixed_compositions(args.fixed_compositions),
        )
        for seed in seeds
    ]
    record: OracleRunRecord = {
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
            "b_scales": args.b_scales,
            "c_scales": args.c_scales,
            "fixed_compositions": args.fixed_compositions,
            "eps_old": args.eps_old,
            "eps_new": args.eps_new,
        },
        "git_status": git_status(),
        "results": results,
        "summary": summarize_oracle(results),
    }
    if args.output_dir is not None:
        write_run_record(record, args.output_dir)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
