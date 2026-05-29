"""Contextual routed memory-bank experiments for sequential LoRA deltas.

This experiment generalizes the A/B/C routed-accretion scaffold to N phases.
It snapshots the adapter after every phase, stores post-A phase deltas in a
small memory bank, then composes route expressions per prompt/domain.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import torch
from torch import Tensor

from stt.continual import eval_loss, prepare_encoded_splits, train_steps
from stt.experiment import Variant, resolve_device
from stt.lora_experiment import (
    LoraSettings,
    build_lora_model,
    build_variants,
    encode_texts,
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

RouteSelection = Literal["oracle", "loss_probe", "calibration"]

ROUTE_TERM_RE = re.compile(
    r"^\s*(?:(?P<scale>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\*?)?"
    r"(?P<name>[A-Za-z][A-Za-z0-9_-]*)\s*$"
)


@dataclass(frozen=True)
class Phase:
    """One sequential memory-learning phase."""

    name: str
    path: str


@dataclass(frozen=True)
class Probe:
    """Eval-only semantic boundary probe with an expected route label."""

    name: str
    path: str
    expected_route: str


@dataclass
class MemoryDelta:
    """One trainable-parameter delta between adjacent adapter snapshots."""

    name: str
    from_phase: str
    to_phase: str
    tensors: dict[str, Tensor]
    metadata: dict[str, Any]


@dataclass
class MemoryBank:
    """Stable adapter plus named phase deltas that can be routed per prompt."""

    stable_phase: str
    stable_adapter: dict[str, Tensor]
    deltas: dict[str, MemoryDelta]


@dataclass(frozen=True)
class RouteExpression:
    """Compiled route expression such as `A+0.9B+0.4C`."""

    expression: str
    stable_phase: str
    scales: dict[str, float]


@dataclass(frozen=True)
class RouteChoice:
    """Selected route for one probe, with ambiguity bookkeeping."""

    selected_route: str
    eval_route: str
    loss: float
    ambiguous: bool


class DomainRouteResult(TypedDict):
    """Serializable contextual-routing result for one held-out domain."""

    selected_route_counts: dict[str, int]
    expected_route: str
    most_selected_route: str
    selection_count: int
    route_accuracy: float
    ambiguous_count: int
    ambiguous_rate: float
    eval_loss: float
    sequential_eval_loss: float
    learned_eval_loss: float
    initial_eval_loss: float
    loss_delta_vs_sequential: float
    learning_retained: float
    interference: float
    best_route_counts: NotRequired[dict[str, int]]
    most_best_route: NotRequired[str]
    best_eval_loss: NotRequired[float]
    selected_loss_gap: NotRequired[float]
    expected_route_loss: NotRequired[float]
    expected_loss_gap: NotRequired[float]
    optimal_route_count: NotRequired[int]
    optimal_route_rate: NotRequired[float]


class MemoryBankResult(TypedDict):
    """Serializable result for one contextual memory-bank run."""

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
    route_selection: str
    contextual_route: bool
    global_route: NotRequired[str]
    phase_names: list[str]
    candidate_routes: list[str]
    audit_routes: NotRequired[list[str]]
    heldout_report: bool
    trainable_parameters: int
    total_parameters: int
    trainable_fraction: float
    contextual_eval_loss: float
    sequential_eval_loss: float
    loss_delta_vs_sequential: float
    mean_learning_retained: float
    mean_interference: float
    route_accuracy: float
    ambiguous_rate: float
    optimal_route_rate: NotRequired[float]
    selected_loss_gap: NotRequired[float]
    expected_loss_gap: NotRequired[float]
    probe_eval_loss: NotRequired[float]
    probe_route_accuracy: NotRequired[float]
    probe_ambiguous_rate: NotRequired[float]
    probe_optimal_route_rate: NotRequired[float]
    probe_selected_loss_gap: NotRequired[float]
    probe_expected_loss_gap: NotRequired[float]
    frontier_score: float
    per_domain: dict[str, DomainRouteResult]
    per_probe: NotRequired[dict[str, DomainRouteResult]]
    phase_eval_losses: dict[str, dict[str, float]]


class MemoryBankRunRecord(TypedDict):
    """Persisted contextual memory-bank run record."""

    created_at: str
    config: dict[str, Any]
    git_status: str
    results: list[MemoryBankResult]
    summary: dict[str, dict[str, float]]


def parse_route_expr(value: str, *, stable_phase: str = "A") -> RouteExpression:
    """Parse `A+B`, `A+0.9B`, or `A+0.9*B+0.4*C` into delta scales."""
    terms = [term.strip() for term in value.split("+") if term.strip()]
    if not terms:
        raise ValueError("route expression cannot be empty")

    stable_seen = False
    scales: dict[str, float] = {}
    for term in terms:
        match = ROUTE_TERM_RE.match(term)
        if match is None:
            raise ValueError(f"invalid route term: {term!r}")
        name = match.group("name")
        scale = float(match.group("scale")) if match.group("scale") is not None else 1.0
        if name == stable_phase:
            if scale != 1.0:
                raise ValueError(f"stable phase {stable_phase!r} must use implicit scale 1.0")
            stable_seen = True
            continue
        if name in scales:
            raise ValueError(f"duplicate route delta: {name}")
        scales[name] = scale

    if not stable_seen:
        raise ValueError(f"route expression must include stable phase {stable_phase!r}")
    route = RouteExpression(value, stable_phase, scales)
    return RouteExpression(route_expr_string(route), stable_phase, scales)


def route_expr_string(route: RouteExpression) -> str:
    """Return a normalized route expression string."""
    parts = [route.stable_phase]
    for name, scale in route.scales.items():
        parts.append(name if scale == 1.0 else f"{scale:g}{name}")
    return "+".join(parts)


def parse_route_exprs(values: list[str], *, stable_phase: str) -> list[RouteExpression]:
    """Parse multiple CLI route expressions."""
    return [parse_route_expr(value, stable_phase=stable_phase) for value in values]


def merge_route_exprs(
    route_exprs: list[RouteExpression],
    audit_route_exprs: list[RouteExpression] | None,
) -> list[RouteExpression]:
    """Return audit routes with contextual candidates included first."""
    merged: dict[str, RouteExpression] = {}
    for route in [*route_exprs, *(audit_route_exprs or [])]:
        merged.setdefault(route.expression, route)
    return list(merged.values())


def default_phase_names(count: int) -> list[str]:
    """Return A/B/C-style default phase names."""
    names = []
    for index in range(count):
        names.append(chr(ord("A") + index) if index < 26 else f"P{index}")
    return names


def default_route_exprs(phase_names: list[str]) -> list[RouteExpression]:
    """Return a small default contextual-route set for the provided phases."""
    stable = phase_names[0]
    raw_routes = [stable]
    raw_routes.extend(f"{stable}+{name}" for name in phase_names[1:])
    if len(phase_names) >= 3:
        raw_routes.append(f"{stable}+0.9{phase_names[1]}+0.4{phase_names[2]}")
    return parse_route_exprs(raw_routes, stable_phase=stable)


def compose_memory_route_state(
    bank: MemoryBank,
    route: RouteExpression,
) -> dict[str, Tensor]:
    """Compose the stable adapter plus scaled named memory deltas."""
    updates = []
    for delta_name, scale in route.scales.items():
        if delta_name not in bank.deltas:
            raise ValueError(f"route references unknown memory delta: {delta_name}")
        updates.append((scale, bank.deltas[delta_name].tensors))
    return compose_state(bank.stable_adapter, updates)


def build_memory_bank(
    phase_names: list[str],
    snapshots: list[dict[str, Tensor]],
) -> MemoryBank:
    """Build a memory bank from per-phase adapter snapshots."""
    if len(phase_names) != len(snapshots):
        raise ValueError("phase_names and snapshots must have the same length")
    if not snapshots:
        raise ValueError("at least one phase snapshot is required")

    deltas = {}
    for index in range(1, len(snapshots)):
        name = phase_names[index]
        from_phase = phase_names[index - 1]
        deltas[name] = MemoryDelta(
            name=name,
            from_phase=from_phase,
            to_phase=name,
            tensors=subtract_state(snapshots[index], snapshots[index - 1]),
            metadata={"phase_index": index},
        )
    return MemoryBank(
        stable_phase=phase_names[0],
        stable_adapter=snapshots[0],
        deltas=deltas,
    )


def settings_for_encoded(settings: LoraSettings, encoded: dict[str, Tensor]) -> LoraSettings:
    """Return settings that evaluate an encoded split once."""
    eval_batches = max(1, encoded["input_ids"].shape[0] // settings.batch_size)
    return replace(settings, eval_batches=eval_batches)


def prepare_probe_encoded(
    tokenizer: Any,
    texts: list[str],
    settings: LoraSettings,
    seed: int,
    device: str,
) -> dict[str, Tensor]:
    """Create a bounded shuffled encoded split from eval-only probe texts."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=generator).tolist()
    sample_count = max(settings.batch_size * 2, settings.batch_size * settings.eval_batches * 2)
    probe_texts = [texts[index] for index in indices[:sample_count]]
    return encode_texts(tokenizer, probe_texts, settings.max_length, device)


def one_example(encoded: dict[str, Tensor], index: int) -> dict[str, Tensor]:
    """Return a single encoded example preserving batch dimensions."""
    return {name: value[index : index + 1] for name, value in encoded.items()}


def score_route_states(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded: dict[str, Tensor],
    settings: LoraSettings,
) -> list[tuple[str, float]]:
    """Evaluate every candidate route state on one encoded probe."""
    scored = []
    for route_name, state in route_states.items():
        apply_trainable_state(model, state)
        scored.append((route_name, eval_loss(model, encoded, settings)))
    return sorted(scored, key=lambda item: item[1])


def route_scores_for_examples(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded: dict[str, Tensor],
    settings: LoraSettings,
) -> list[list[tuple[str, float]]]:
    """Evaluate every route for each example in a held-out domain split."""
    one_settings = replace(settings, batch_size=1, eval_batches=1)
    return [
        score_route_states(model, route_states, one_example(encoded, index), one_settings)
        for index in range(encoded["input_ids"].shape[0])
    ]


def route_choice_from_scored(
    scored: list[tuple[str, float]],
    *,
    ambiguity_margin: float,
) -> RouteChoice:
    """Return a route choice from already-scored route losses."""
    if not scored:
        raise ValueError("at least one route candidate is required")
    best_route, best_loss = scored[0]
    ambiguous = (
        ambiguity_margin > 0.0
        and len(scored) > 1
        and scored[1][1] - best_loss < ambiguity_margin
    )
    return RouteChoice(
        selected_route="uncertain" if ambiguous else best_route,
        eval_route=best_route,
        loss=best_loss,
        ambiguous=ambiguous,
    )


def select_route_by_loss(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded: dict[str, Tensor],
    settings: LoraSettings,
    *,
    ambiguity_margin: float,
) -> RouteChoice:
    """Select the lowest-loss route for one probe, with optional uncertainty."""
    scored = score_route_states(model, route_states, encoded, settings)
    return route_choice_from_scored(scored, ambiguity_margin=ambiguity_margin)


def expected_route_for_domain(
    domain: str,
    phase_names: list[str],
    candidate_routes: list[str],
) -> str:
    """Return the hand-authored oracle route expected for a prompt domain."""
    stable = phase_names[0]
    route_set = set(candidate_routes)
    if domain == stable:
        if len(phase_names) > 1 and f"{stable}+{phase_names[1]}" in route_set:
            return f"{stable}+{phase_names[1]}"
        if stable in route_set:
            return stable
    direct = f"{stable}+{domain}"
    if direct in route_set:
        return direct
    if stable in route_set:
        return stable
    return candidate_routes[0]


def most_selected_route(counts: dict[str, int]) -> str:
    """Return the most frequently selected route label."""
    if not counts:
        return "none"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def route_variant_suffix(route: RouteExpression) -> str:
    """Return a filesystem/variant friendly route suffix."""
    return route.expression.replace("+", "_").replace(".", "p")


def route_eval_loss(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    route_name: str,
    encoded: dict[str, Tensor],
    settings: LoraSettings,
) -> float:
    """Evaluate one named route on an encoded split."""
    if route_name not in route_states:
        raise ValueError(f"unknown route: {route_name}")
    apply_trainable_state(model, route_states[route_name])
    return eval_loss(model, encoded, settings_for_encoded(settings, encoded))


def learning_retention(
    *,
    initial_loss: float,
    learned_loss: float,
    routed_loss: float,
) -> float:
    """Return retained learning relative to the domain's own post-phase snapshot."""
    learned_delta = initial_loss - learned_loss
    if learned_delta <= 0.0:
        return 0.0
    return (initial_loss - routed_loss) / learned_delta


def domain_result(
    *,
    selected_route_counts: dict[str, int],
    expected_route: str,
    eval_loss_value: float,
    sequential_eval_loss: float,
    learned_eval_loss: float,
    initial_eval_loss: float,
    correct_count: int,
    ambiguous_count: int,
    audit: dict[str, float | int | dict[str, int] | str] | None = None,
) -> DomainRouteResult:
    """Build one serializable per-domain result record."""
    selection_count = sum(selected_route_counts.values())
    route_accuracy = correct_count / selection_count if selection_count else 0.0
    ambiguous_rate = ambiguous_count / selection_count if selection_count else 0.0
    result: DomainRouteResult = {
        "selected_route_counts": selected_route_counts,
        "expected_route": expected_route,
        "most_selected_route": most_selected_route(selected_route_counts),
        "selection_count": selection_count,
        "route_accuracy": route_accuracy,
        "ambiguous_count": ambiguous_count,
        "ambiguous_rate": ambiguous_rate,
        "eval_loss": eval_loss_value,
        "sequential_eval_loss": sequential_eval_loss,
        "learned_eval_loss": learned_eval_loss,
        "initial_eval_loss": initial_eval_loss,
        "loss_delta_vs_sequential": sequential_eval_loss - eval_loss_value,
        "learning_retained": learning_retention(
            initial_loss=initial_eval_loss,
            learned_loss=learned_eval_loss,
            routed_loss=eval_loss_value,
        ),
        "interference": eval_loss_value - learned_eval_loss,
    }
    if audit is not None:
        result.update(audit)  # type: ignore[typeddict-item]
    return result


def route_optimality_audit(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded: dict[str, Tensor],
    settings: LoraSettings,
    *,
    expected_route: str,
    selected_eval_routes: list[str],
) -> dict[str, float | int | dict[str, int] | str]:
    """Compare selected and expected routes against per-example best-loss routes."""
    scored_examples = route_scores_for_examples(model, route_states, encoded, settings)
    return route_optimality_from_scores(
        scored_examples,
        expected_route=expected_route,
        selected_eval_routes=selected_eval_routes,
    )


def route_optimality_from_scores(
    scored_examples: list[list[tuple[str, float]]],
    *,
    expected_route: str,
    selected_eval_routes: list[str],
) -> dict[str, float | int | dict[str, int] | str]:
    """Compare selected and expected routes against cached per-example route losses."""
    sample_count = len(scored_examples)
    if sample_count == 0:
        return {
            "best_route_counts": {},
            "most_best_route": "none",
            "best_eval_loss": 0.0,
            "selected_loss_gap": 0.0,
            "expected_route_loss": 0.0,
            "expected_loss_gap": 0.0,
            "optimal_route_count": 0,
            "optimal_route_rate": 0.0,
        }
    if len(selected_eval_routes) == 1:
        selected_eval_routes = selected_eval_routes * sample_count
    if len(selected_eval_routes) != sample_count:
        raise ValueError("selected_eval_routes must have one route or one route per example")

    first_scores = dict(scored_examples[0])
    expected_eval_route = (
        expected_route if expected_route in first_scores else next(iter(first_scores))
    )
    best_route_counts: dict[str, int] = {}
    best_loss_total = 0.0
    selected_gap_total = 0.0
    expected_loss_total = 0.0
    expected_gap_total = 0.0
    optimal_count = 0
    for index, scored in enumerate(scored_examples):
        best_route, best_loss = scored[0]
        score_by_route = dict(scored)
        selected_route = selected_eval_routes[index]
        if selected_route not in score_by_route:
            selected_route = best_route
        selected_loss = score_by_route[selected_route]
        expected_loss = score_by_route[expected_eval_route]
        best_route_counts[best_route] = best_route_counts.get(best_route, 0) + 1
        best_loss_total += best_loss
        selected_gap_total += selected_loss - best_loss
        expected_loss_total += expected_loss
        expected_gap_total += expected_loss - best_loss
        optimal_count += int(selected_loss <= best_loss + 1e-8)

    return {
        "best_route_counts": best_route_counts,
        "most_best_route": most_selected_route(best_route_counts),
        "best_eval_loss": best_loss_total / sample_count,
        "selected_loss_gap": selected_gap_total / sample_count,
        "expected_route_loss": expected_loss_total / sample_count,
        "expected_loss_gap": expected_gap_total / sample_count,
        "optimal_route_count": optimal_count,
        "optimal_route_rate": optimal_count / sample_count,
    }


def evaluate_oracle_domain(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded: dict[str, Tensor],
    settings: LoraSettings,
    *,
    expected_route: str,
    sequential_eval_loss: float,
    learned_eval_loss: float,
    initial_eval_loss: float,
    audit_scores: list[list[tuple[str, float]]] | None = None,
) -> DomainRouteResult:
    """Evaluate a domain with the known oracle route."""
    route_name = expected_route if expected_route in route_states else next(iter(route_states))
    apply_trainable_state(model, route_states[route_name])
    eval_loss_value = eval_loss(model, encoded, settings_for_encoded(settings, encoded))
    selection_count = encoded["input_ids"].shape[0]
    audit = (
        route_optimality_from_scores(
            audit_scores,
            expected_route=expected_route,
            selected_eval_routes=[route_name],
        )
        if audit_scores is not None
        else route_optimality_audit(
            model,
            route_states,
            encoded,
            settings,
            expected_route=expected_route,
            selected_eval_routes=[route_name],
        )
    )
    return domain_result(
        selected_route_counts={route_name: selection_count},
        expected_route=expected_route,
        eval_loss_value=eval_loss_value,
        sequential_eval_loss=sequential_eval_loss,
        learned_eval_loss=learned_eval_loss,
        initial_eval_loss=initial_eval_loss,
        correct_count=selection_count if route_name == expected_route else 0,
        ambiguous_count=0,
        audit=audit,
    )


def evaluate_calibration_domain(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    select_encoded: dict[str, Tensor],
    report_encoded: dict[str, Tensor],
    settings: LoraSettings,
    *,
    expected_route: str,
    sequential_eval_loss: float,
    learned_eval_loss: float,
    initial_eval_loss: float,
    ambiguity_margin: float,
    audit_scores: list[list[tuple[str, float]]] | None = None,
) -> DomainRouteResult:
    """Pick one route on calibration probes and evaluate it on held-out probes."""
    choice = select_route_by_loss(
        model,
        route_states,
        select_encoded,
        settings_for_encoded(settings, select_encoded),
        ambiguity_margin=ambiguity_margin,
    )
    apply_trainable_state(model, route_states[choice.eval_route])
    report_settings = settings_for_encoded(settings, report_encoded)
    eval_loss_value = eval_loss(model, report_encoded, report_settings)
    selection_count = report_encoded["input_ids"].shape[0]
    audit = (
        route_optimality_from_scores(
            audit_scores,
            expected_route=expected_route,
            selected_eval_routes=[choice.eval_route],
        )
        if audit_scores is not None
        else route_optimality_audit(
            model,
            route_states,
            report_encoded,
            settings,
            expected_route=expected_route,
            selected_eval_routes=[choice.eval_route],
        )
    )
    return domain_result(
        selected_route_counts={choice.selected_route: selection_count},
        expected_route=expected_route,
        eval_loss_value=eval_loss_value,
        sequential_eval_loss=sequential_eval_loss,
        learned_eval_loss=learned_eval_loss,
        initial_eval_loss=initial_eval_loss,
        correct_count=selection_count if choice.selected_route == expected_route else 0,
        ambiguous_count=selection_count if choice.ambiguous else 0,
        audit=audit,
    )


def evaluate_loss_probe_domain(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded: dict[str, Tensor],
    settings: LoraSettings,
    *,
    expected_route: str,
    sequential_eval_loss: float,
    learned_eval_loss: float,
    initial_eval_loss: float,
    ambiguity_margin: float,
    audit_scores: list[list[tuple[str, float]]] | None = None,
    selection_scores: list[list[tuple[str, float]]] | None = None,
) -> DomainRouteResult:
    """Select and evaluate routes per held-out example using probe loss."""
    selected_route_counts: dict[str, int] = {}
    total_loss = 0.0
    correct_count = 0
    ambiguous_count = 0
    selected_eval_routes = []
    sample_count = encoded["input_ids"].shape[0]

    scored_examples = selection_scores
    if scored_examples is None:
        scored_examples = route_scores_for_examples(model, route_states, encoded, settings)

    for scored in scored_examples:
        choice = route_choice_from_scored(scored, ambiguity_margin=ambiguity_margin)
        selected_eval_routes.append(choice.eval_route)
        selected_route_counts[choice.selected_route] = (
            selected_route_counts.get(choice.selected_route, 0) + 1
        )
        total_loss += choice.loss
        correct_count += int(choice.selected_route == expected_route)
        ambiguous_count += int(choice.ambiguous)

    eval_loss_value = total_loss / sample_count if sample_count else 0.0
    audit = route_optimality_from_scores(
        audit_scores or scored_examples,
        expected_route=expected_route,
        selected_eval_routes=selected_eval_routes,
    )
    return domain_result(
        selected_route_counts=selected_route_counts,
        expected_route=expected_route,
        eval_loss_value=eval_loss_value,
        sequential_eval_loss=sequential_eval_loss,
        learned_eval_loss=learned_eval_loss,
        initial_eval_loss=initial_eval_loss,
        correct_count=correct_count,
        ambiguous_count=ambiguous_count,
        audit=audit,
    )


def evaluate_contextual_domains(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    eval_select: dict[str, dict[str, Tensor]],
    eval_report: dict[str, dict[str, Tensor]],
    settings: LoraSettings,
    *,
    phase_names: list[str],
    route_selection: RouteSelection,
    initial_eval_losses: dict[str, float],
    phase_eval_losses: dict[str, dict[str, float]],
    ambiguity_margin: float,
    audit_scores_by_domain: dict[str, list[list[tuple[str, float]]]] | None = None,
) -> dict[str, DomainRouteResult]:
    """Evaluate all domains with the configured contextual routing strategy."""
    candidate_routes = list(route_states.keys())
    final_phase = phase_names[-1]
    per_domain = {}
    for domain in phase_names:
        expected_route = expected_route_for_domain(domain, phase_names, candidate_routes)
        common = {
            "expected_route": expected_route,
            "sequential_eval_loss": phase_eval_losses[final_phase][domain],
            "learned_eval_loss": phase_eval_losses[domain][domain],
            "initial_eval_loss": initial_eval_losses[domain],
            "audit_scores": None
            if audit_scores_by_domain is None
            else audit_scores_by_domain[domain],
        }
        if route_selection == "oracle":
            per_domain[domain] = evaluate_oracle_domain(
                model,
                route_states,
                eval_report[domain],
                settings,
                **common,
            )
        elif route_selection == "calibration":
            per_domain[domain] = evaluate_calibration_domain(
                model,
                route_states,
                eval_select[domain],
                eval_report[domain],
                settings,
                ambiguity_margin=ambiguity_margin,
                **common,
            )
        else:
            per_domain[domain] = evaluate_loss_probe_domain(
                model,
                route_states,
                eval_report[domain],
                settings,
                ambiguity_margin=ambiguity_margin,
                **common,
            )
    return per_domain


def evaluate_contextual_probes(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    probe_select: dict[str, dict[str, Tensor]],
    probe_report: dict[str, dict[str, Tensor]],
    settings: LoraSettings,
    *,
    probes: list[Probe],
    route_selection: RouteSelection,
    initial_eval_losses: dict[str, float],
    sequential_eval_losses: dict[str, float],
    learned_eval_losses: dict[str, float],
    ambiguity_margin: float,
    audit_scores_by_probe: dict[str, list[list[tuple[str, float]]]] | None = None,
) -> dict[str, DomainRouteResult]:
    """Evaluate eval-only boundary probes with expected semantic routes."""
    per_probe = {}
    for probe in probes:
        common = {
            "expected_route": probe.expected_route,
            "sequential_eval_loss": sequential_eval_losses[probe.name],
            "learned_eval_loss": learned_eval_losses[probe.name],
            "initial_eval_loss": initial_eval_losses[probe.name],
            "audit_scores": None
            if audit_scores_by_probe is None
            else audit_scores_by_probe[probe.name],
        }
        if route_selection == "oracle":
            per_probe[probe.name] = evaluate_oracle_domain(
                model,
                route_states,
                probe_report[probe.name],
                settings,
                **common,
            )
        elif route_selection == "calibration":
            per_probe[probe.name] = evaluate_calibration_domain(
                model,
                route_states,
                probe_select[probe.name],
                probe_report[probe.name],
                settings,
                ambiguity_margin=ambiguity_margin,
                **common,
            )
        else:
            per_probe[probe.name] = evaluate_loss_probe_domain(
                model,
                route_states,
                probe_report[probe.name],
                settings,
                ambiguity_margin=ambiguity_margin,
                **common,
            )
    return per_probe


def evaluate_global_domain(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded: dict[str, Tensor],
    settings: LoraSettings,
    *,
    route_name: str,
    expected_route: str,
    sequential_eval_loss: float,
    learned_eval_loss: float,
    initial_eval_loss: float,
    audit_scores: list[list[tuple[str, float]]] | None = None,
) -> DomainRouteResult:
    """Evaluate one domain with the same globally selected route."""
    apply_trainable_state(model, route_states[route_name])
    eval_loss_value = eval_loss(model, encoded, settings_for_encoded(settings, encoded))
    selection_count = encoded["input_ids"].shape[0]
    audit = (
        route_optimality_from_scores(
            audit_scores,
            expected_route=expected_route,
            selected_eval_routes=[route_name],
        )
        if audit_scores is not None
        else route_optimality_audit(
            model,
            route_states,
            encoded,
            settings,
            expected_route=expected_route,
            selected_eval_routes=[route_name],
        )
    )
    return domain_result(
        selected_route_counts={route_name: selection_count},
        expected_route=expected_route,
        eval_loss_value=eval_loss_value,
        sequential_eval_loss=sequential_eval_loss,
        learned_eval_loss=learned_eval_loss,
        initial_eval_loss=initial_eval_loss,
        correct_count=selection_count if route_name == expected_route else 0,
        ambiguous_count=0,
        audit=audit,
    )


def evaluate_global_domains(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    eval_report: dict[str, dict[str, Tensor]],
    settings: LoraSettings,
    *,
    phase_names: list[str],
    route_name: str,
    expected_routes: dict[str, str] | None = None,
    initial_eval_losses: dict[str, float],
    phase_eval_losses: dict[str, dict[str, float]],
    audit_scores_by_domain: dict[str, list[list[tuple[str, float]]]] | None = None,
) -> dict[str, DomainRouteResult]:
    """Evaluate all domains using one fixed global route."""
    candidate_routes = list(route_states.keys())
    final_phase = phase_names[-1]
    per_domain = {}
    for domain in phase_names:
        expected_route = (
            expected_routes[domain]
            if expected_routes is not None
            else expected_route_for_domain(domain, phase_names, candidate_routes)
        )
        per_domain[domain] = evaluate_global_domain(
            model,
            route_states,
            eval_report[domain],
            settings,
            route_name=route_name,
            expected_route=expected_route,
            sequential_eval_loss=phase_eval_losses[final_phase][domain],
            learned_eval_loss=phase_eval_losses[domain][domain],
            initial_eval_loss=initial_eval_losses[domain],
            audit_scores=None
            if audit_scores_by_domain is None
            else audit_scores_by_domain[domain],
        )
    return per_domain


def evaluate_global_probes(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    probe_report: dict[str, dict[str, Tensor]],
    settings: LoraSettings,
    *,
    probes: list[Probe],
    route_name: str,
    initial_eval_losses: dict[str, float],
    sequential_eval_losses: dict[str, float],
    learned_eval_losses: dict[str, float],
    audit_scores_by_probe: dict[str, list[list[tuple[str, float]]]] | None = None,
) -> dict[str, DomainRouteResult]:
    """Evaluate eval-only probes using one fixed global route."""
    per_probe = {}
    for probe in probes:
        per_probe[probe.name] = evaluate_global_domain(
            model,
            route_states,
            probe_report[probe.name],
            settings,
            route_name=route_name,
            expected_route=probe.expected_route,
            sequential_eval_loss=sequential_eval_losses[probe.name],
            learned_eval_loss=learned_eval_losses[probe.name],
            initial_eval_loss=initial_eval_losses[probe.name],
            audit_scores=None
            if audit_scores_by_probe is None
            else audit_scores_by_probe[probe.name],
        )
    return per_probe


def aggregate_domain_metrics(
    per_domain: dict[str, DomainRouteResult],
) -> dict[str, float]:
    """Return weighted aggregate contextual-routing metrics."""
    total = sum(domain["selection_count"] for domain in per_domain.values())
    if total == 0:
        return {
            "contextual_eval_loss": 0.0,
            "sequential_eval_loss": 0.0,
            "loss_delta_vs_sequential": 0.0,
            "mean_learning_retained": 0.0,
            "mean_interference": 0.0,
            "route_accuracy": 0.0,
            "ambiguous_rate": 0.0,
            "optimal_route_rate": 0.0,
            "selected_loss_gap": 0.0,
            "expected_loss_gap": 0.0,
            "frontier_score": 0.0,
        }

    def weighted(metric: str) -> float:
        return sum(
            float(domain.get(metric, 0.0)) * domain["selection_count"]
            for domain in per_domain.values()
        ) / total

    contextual_eval_loss = weighted("eval_loss")
    sequential_eval_loss = weighted("sequential_eval_loss")
    return {
        "contextual_eval_loss": contextual_eval_loss,
        "sequential_eval_loss": sequential_eval_loss,
        "loss_delta_vs_sequential": sequential_eval_loss - contextual_eval_loss,
        "mean_learning_retained": weighted("learning_retained"),
        "mean_interference": weighted("interference"),
        "route_accuracy": weighted("route_accuracy"),
        "ambiguous_rate": weighted("ambiguous_rate"),
        "optimal_route_rate": weighted("optimal_route_rate"),
        "selected_loss_gap": weighted("selected_loss_gap"),
        "expected_loss_gap": weighted("expected_loss_gap"),
        "frontier_score": sequential_eval_loss - contextual_eval_loss,
    }


def memory_bank_result_record(
    *,
    variant: Variant,
    variant_name: str,
    settings: LoraSettings,
    resolved_device: str,
    seed: int,
    route_selection: str,
    contextual_route: bool,
    phase_names: list[str],
    route_exprs: list[RouteExpression],
    audit_route_exprs: list[RouteExpression],
    heldout_report: bool,
    trainable: int,
    total: int,
    per_domain: dict[str, DomainRouteResult],
    phase_eval_losses: dict[str, dict[str, float]],
    per_probe: dict[str, DomainRouteResult] | None = None,
    global_route: str | None = None,
) -> MemoryBankResult:
    """Build a serializable memory-bank result from per-domain metrics."""
    aggregate = aggregate_domain_metrics(per_domain)
    result: MemoryBankResult = {
        "variant": variant_name,
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
        "route_selection": route_selection,
        "contextual_route": contextual_route,
        "phase_names": phase_names,
        "candidate_routes": [route.expression for route in route_exprs],
        "audit_routes": [route.expression for route in audit_route_exprs],
        "heldout_report": heldout_report,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "contextual_eval_loss": aggregate["contextual_eval_loss"],
        "sequential_eval_loss": aggregate["sequential_eval_loss"],
        "loss_delta_vs_sequential": aggregate["loss_delta_vs_sequential"],
        "mean_learning_retained": aggregate["mean_learning_retained"],
        "mean_interference": aggregate["mean_interference"],
        "route_accuracy": aggregate["route_accuracy"],
        "ambiguous_rate": aggregate["ambiguous_rate"],
        "optimal_route_rate": aggregate["optimal_route_rate"],
        "selected_loss_gap": aggregate["selected_loss_gap"],
        "expected_loss_gap": aggregate["expected_loss_gap"],
        "frontier_score": aggregate["frontier_score"],
        "per_domain": per_domain,
        "phase_eval_losses": phase_eval_losses,
    }
    if per_probe:
        probe_aggregate = aggregate_domain_metrics(per_probe)
        result.update(
            {
                "probe_eval_loss": probe_aggregate["contextual_eval_loss"],
                "probe_route_accuracy": probe_aggregate["route_accuracy"],
                "probe_ambiguous_rate": probe_aggregate["ambiguous_rate"],
                "probe_optimal_route_rate": probe_aggregate["optimal_route_rate"],
                "probe_selected_loss_gap": probe_aggregate["selected_loss_gap"],
                "probe_expected_loss_gap": probe_aggregate["expected_loss_gap"],
                "per_probe": per_probe,
            }
        )
    if global_route is not None:
        result["global_route"] = global_route
    return result


def run_memory_bank_seed_results(
    variant: Variant,
    settings: LoraSettings,
    phases: list[Phase],
    phase_texts: list[list[str]],
    phase_steps: int,
    seed: int,
    device: str,
    route_exprs: list[RouteExpression],
    route_selection: RouteSelection,
    contextual_route: bool,
    probes: list[Probe] | None = None,
    probe_texts: list[list[str]] | None = None,
    audit_route_exprs: list[RouteExpression] | None = None,
    ambiguity_margin: float = 0.0,
    include_global_routes: bool = False,
) -> list[MemoryBankResult]:
    """Train N phases once and evaluate contextual plus optional global routes."""
    if len(phases) != len(phase_texts):
        raise ValueError("phases and phase_texts must have the same length")
    if not phases:
        raise ValueError("at least one phase is required")
    if not route_exprs:
        raise ValueError("at least one route expression is required")

    probes = probes or []
    probe_texts = probe_texts or []
    if len(probes) != len(probe_texts):
        raise ValueError("probes and probe_texts must have the same length")
    audit_route_exprs = merge_route_exprs(route_exprs, audit_route_exprs)
    candidate_route_names = [route.expression for route in route_exprs]
    missing_probe_routes = sorted(
        {
            probe.expected_route
            for probe in probes
            if probe.expected_route not in candidate_route_names
        }
    )
    if missing_probe_routes:
        raise ValueError(
            "probe expected routes must be included as --route-expr: "
            + ", ".join(missing_probe_routes)
        )

    resolved_device = resolve_device(device)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(settings.model_name)
    model = build_lora_model(settings, resolved_device)
    trainable, total = parameter_counts(model)

    train_splits: dict[str, dict[str, Tensor]] = {}
    eval_select: dict[str, dict[str, Tensor]] = {}
    eval_report: dict[str, dict[str, Tensor]] = {}
    heldout_flags = []
    for index, (phase, texts) in enumerate(zip(phases, phase_texts, strict=True)):
        train_split, eval_full = prepare_encoded_splits(
            tokenizer,
            texts,
            settings,
            seed + (index * 10_000),
            resolved_device,
        )
        select_split, report_split, heldout = split_eval_encoded(eval_full, settings.batch_size)
        train_splits[phase.name] = train_split
        eval_select[phase.name] = select_split
        eval_report[phase.name] = report_split
        heldout_flags.append(heldout)

    probe_select: dict[str, dict[str, Tensor]] = {}
    probe_report: dict[str, dict[str, Tensor]] = {}
    for index, (probe, texts) in enumerate(zip(probes, probe_texts, strict=True)):
        eval_full = prepare_probe_encoded(
            tokenizer,
            texts,
            settings,
            seed + 1_000_000 + (index * 10_000),
            resolved_device,
        )
        select_split, report_split, heldout = split_eval_encoded(eval_full, settings.batch_size)
        probe_select[probe.name] = select_split
        probe_report[probe.name] = report_split
        heldout_flags.append(heldout)

    phase_names = [phase.name for phase in phases]
    initial_eval_losses = {
        phase.name: eval_loss(
            model,
            eval_report[phase.name],
            settings_for_encoded(settings, eval_report[phase.name]),
        )
        for phase in phases
    }
    initial_probe_eval_losses = {
        probe.name: eval_loss(
            model,
            probe_report[probe.name],
            settings_for_encoded(settings, probe_report[probe.name]),
        )
        for probe in probes
    }
    snapshots = []
    phase_eval_losses: dict[str, dict[str, float]] = {}
    for phase in phases:
        train_steps(model, train_splits[phase.name], variant, settings, phase_steps)
        snapshots.append(snapshot_trainable_state(model))
        phase_eval_losses[phase.name] = {
            domain: eval_loss(model, encoded, settings_for_encoded(settings, encoded))
            for domain, encoded in eval_report.items()
        }

    sequential_probe_eval_losses = {
        probe.name: eval_loss(
            model,
            probe_report[probe.name],
            settings_for_encoded(settings, probe_report[probe.name]),
        )
        for probe in probes
    }

    bank = build_memory_bank(phase_names, snapshots)
    route_states = {
        route.expression: compose_memory_route_state(bank, route) for route in route_exprs
    }
    audit_route_states = {
        route.expression: compose_memory_route_state(bank, route) for route in audit_route_exprs
    }
    audit_scores_by_domain = {
        domain: route_scores_for_examples(model, audit_route_states, encoded, settings)
        for domain, encoded in eval_report.items()
    }
    audit_scores_by_probe = {
        probe.name: route_scores_for_examples(
            model,
            audit_route_states,
            probe_report[probe.name],
            settings,
        )
        for probe in probes
    }
    learned_probe_eval_losses = {
        probe.name: route_eval_loss(
            model,
            audit_route_states,
            probe.expected_route,
            probe_report[probe.name],
            settings,
        )
        for probe in probes
    }
    per_domain = evaluate_contextual_domains(
        model,
        route_states,
        eval_select,
        eval_report,
        settings,
        phase_names=phase_names,
        route_selection=route_selection,
        initial_eval_losses=initial_eval_losses,
        phase_eval_losses=phase_eval_losses,
        ambiguity_margin=ambiguity_margin,
        audit_scores_by_domain=audit_scores_by_domain,
    )
    per_probe = evaluate_contextual_probes(
        model,
        route_states,
        probe_select,
        probe_report,
        settings,
        probes=probes,
        route_selection=route_selection,
        initial_eval_losses=initial_probe_eval_losses,
        sequential_eval_losses=sequential_probe_eval_losses,
        learned_eval_losses=learned_probe_eval_losses,
        ambiguity_margin=ambiguity_margin,
        audit_scores_by_probe=audit_scores_by_probe,
    )
    results = [
        memory_bank_result_record(
            variant=variant,
            variant_name=f"{variant.name}_contextual_memory_bank_{route_selection}",
            settings=settings,
            resolved_device=resolved_device,
            seed=seed,
            route_selection=route_selection,
            contextual_route=contextual_route,
            phase_names=phase_names,
            route_exprs=route_exprs,
            audit_route_exprs=audit_route_exprs,
            heldout_report=all(heldout_flags),
            trainable=trainable,
            total=total,
            per_domain=per_domain,
            phase_eval_losses=phase_eval_losses,
            per_probe=per_probe,
        )
    ]
    if include_global_routes:
        expected_routes_by_domain = {
            domain: expected_route_for_domain(domain, phase_names, candidate_route_names)
            for domain in phase_names
        }
        for route in audit_route_exprs:
            global_per_domain = evaluate_global_domains(
                model,
                audit_route_states,
                eval_report,
                settings,
                phase_names=phase_names,
                route_name=route.expression,
                expected_routes=expected_routes_by_domain,
                initial_eval_losses=initial_eval_losses,
                phase_eval_losses=phase_eval_losses,
                audit_scores_by_domain=audit_scores_by_domain,
            )
            global_per_probe = evaluate_global_probes(
                model,
                audit_route_states,
                probe_report,
                settings,
                probes=probes,
                route_name=route.expression,
                initial_eval_losses=initial_probe_eval_losses,
                sequential_eval_losses=sequential_probe_eval_losses,
                learned_eval_losses=learned_probe_eval_losses,
                audit_scores_by_probe=audit_scores_by_probe,
            )
            results.append(
                memory_bank_result_record(
                    variant=variant,
                    variant_name=(
                        f"{variant.name}_global_memory_bank_{route_variant_suffix(route)}"
                    ),
                    settings=settings,
                    resolved_device=resolved_device,
                    seed=seed,
                    route_selection="global",
                    contextual_route=False,
                    phase_names=phase_names,
                    route_exprs=route_exprs,
                    audit_route_exprs=audit_route_exprs,
                    heldout_report=all(heldout_flags),
                    trainable=trainable,
                    total=total,
                    per_domain=global_per_domain,
                    phase_eval_losses=phase_eval_losses,
                    per_probe=global_per_probe,
                    global_route=route.expression,
                )
            )
    return results


def run_memory_bank_seed(
    variant: Variant,
    settings: LoraSettings,
    phases: list[Phase],
    phase_texts: list[list[str]],
    phase_steps: int,
    seed: int,
    device: str,
    route_exprs: list[RouteExpression],
    route_selection: RouteSelection,
    contextual_route: bool,
    ambiguity_margin: float = 0.0,
) -> MemoryBankResult:
    """Train N phases once and evaluate one contextual routed memory-bank result."""
    return run_memory_bank_seed_results(
        variant,
        settings=settings,
        phases=phases,
        phase_texts=phase_texts,
        phase_steps=phase_steps,
        seed=seed,
        device=device,
        route_exprs=route_exprs,
        route_selection=route_selection,
        contextual_route=contextual_route,
        ambiguity_margin=ambiguity_margin,
    )[0]


def summarize_memory_bank(results: list[MemoryBankResult]) -> dict[str, dict[str, float]]:
    """Aggregate contextual memory-bank metrics by variant."""
    metric_names = [
        "heldout_report",
        "contextual_eval_loss",
        "sequential_eval_loss",
        "loss_delta_vs_sequential",
        "mean_learning_retained",
        "mean_interference",
        "route_accuracy",
        "ambiguous_rate",
        "optimal_route_rate",
        "selected_loss_gap",
        "expected_loss_gap",
        "probe_eval_loss",
        "probe_route_accuracy",
        "probe_ambiguous_rate",
        "probe_optimal_route_rate",
        "probe_selected_loss_gap",
        "probe_expected_loss_gap",
        "frontier_score",
    ]
    summary: dict[str, dict[str, float]] = {}
    for variant in sorted({result["variant"] for result in results}):
        group = [result for result in results if result["variant"] == variant]
        values: dict[str, float] = {"count": float(len(group))}
        for metric in metric_names:
            metric_values = [float(result.get(metric, 0.0)) for result in group]
            values[f"{metric}_mean"] = statistics.fmean(metric_values)
            values[f"{metric}_std"] = (
                statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            )
        values["contextual_win_count"] = float(
            sum(result["loss_delta_vs_sequential"] > 0.0 for result in group)
        )
        domains = sorted({domain for result in group for domain in result["per_domain"]})
        for domain in domains:
            domain_group = [
                result["per_domain"][domain]
                for result in group
                if domain in result["per_domain"]
            ]
            for metric in [
                "eval_loss",
                "sequential_eval_loss",
                "loss_delta_vs_sequential",
                "learning_retained",
                "interference",
                "route_accuracy",
                "ambiguous_rate",
                "optimal_route_rate",
                "selected_loss_gap",
                "expected_loss_gap",
            ]:
                metric_values = [
                    float(domain_result.get(metric, 0.0)) for domain_result in domain_group
                ]
                values[f"domain_{domain}_{metric}_mean"] = statistics.fmean(metric_values)
                values[f"domain_{domain}_{metric}_std"] = (
                    statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
                )
        probes = sorted({probe for result in group for probe in result.get("per_probe", {})})
        for probe in probes:
            probe_group = [
                result["per_probe"][probe]
                for result in group
                if probe in result.get("per_probe", {})
            ]
            selection_count = sum(
                int(probe_result["selection_count"]) for probe_result in probe_group
            )
            for metric in [
                "eval_loss",
                "sequential_eval_loss",
                "loss_delta_vs_sequential",
                "route_accuracy",
                "ambiguous_rate",
                "optimal_route_rate",
                "selected_loss_gap",
                "expected_loss_gap",
            ]:
                if selection_count == 0:
                    metric_values = [0.0]
                else:
                    metric_values = [
                        float(probe_result.get(metric, 0.0))
                        for probe_result in probe_group
                    ]
                values[f"probe_{probe}_{metric}_mean"] = statistics.fmean(metric_values)
                values[f"probe_{probe}_{metric}_std"] = (
                    statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
                )
        summary[variant] = values
    return summary


def phase_paths_from_args(args: argparse.Namespace) -> list[str]:
    """Return task files from list-based args or A/B/C compatibility shims."""
    if args.task_files is not None:
        return args.task_files
    shim_paths = [args.task_a_file, args.task_b_file, args.task_c_file]
    if all(path is not None for path in shim_paths):
        return [str(path) for path in shim_paths]
    raise ValueError("provide --task-files or all of --task-a-file/--task-b-file/--task-c-file")


def probes_from_args(args: argparse.Namespace, *, stable_phase: str) -> list[Probe]:
    """Return eval-only probe specs from aligned CLI lists."""
    if args.probe_files is None:
        return []
    if args.probe_routes is None:
        raise ValueError("--probe-routes is required when --probe-files is provided")
    if len(args.probe_files) != len(args.probe_routes):
        raise ValueError("--probe-files and --probe-routes must have the same length")
    probe_names = args.probe_names or [Path(path).stem for path in args.probe_files]
    if len(probe_names) != len(args.probe_files):
        raise ValueError("--probe-names must match --probe-files length")
    if len(set(probe_names)) != len(probe_names):
        raise ValueError("--probe-names must be unique")
    return [
        Probe(
            name=name,
            path=path,
            expected_route=parse_route_expr(route, stable_phase=stable_phase).expression,
        )
        for name, path, route in zip(
            probe_names,
            args.probe_files,
            args.probe_routes,
            strict=True,
        )
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-memory-bank`."""
    parser = argparse.ArgumentParser(description="Run contextual routed memory-bank LoRA tests.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, or cuda")
    parser.add_argument("--task-files", nargs="+", default=None)
    parser.add_argument("--phase-names", nargs="*", default=None)
    parser.add_argument("--task-a-file", default=None)
    parser.add_argument("--task-b-file", default=None)
    parser.add_argument("--task-c-file", default=None)
    parser.add_argument("--probe-files", nargs="*", default=None)
    parser.add_argument("--probe-names", nargs="*", default=None)
    parser.add_argument("--probe-routes", nargs="*", default=None)
    parser.add_argument("--phase-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=16)
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
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--route-expr", action="append", default=None)
    parser.add_argument("--audit-route-expr", action="append", default=None)
    parser.add_argument("--contextual-route", action="store_true")
    parser.add_argument(
        "--route-selection",
        choices=["oracle", "loss_probe", "calibration"],
        default="oracle",
    )
    parser.add_argument("--ambiguity-margin", type=float, default=0.0)
    parser.add_argument(
        "--global-route-baseline",
        action="store_true",
        help=(
            "Also evaluate every route expression as a fixed global route after one "
            "training pass."
        ),
    )
    parser.add_argument("--snapshot-each-phase", action="store_true")
    parser.add_argument("--emit-deltas", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for contextual memory-bank experiments."""
    args = parse_args()
    task_files = phase_paths_from_args(args)
    phase_names = args.phase_names or default_phase_names(len(task_files))
    if len(phase_names) != len(task_files):
        raise ValueError("--phase-names must match --task-files length")
    phases = [
        Phase(name=name, path=path)
        for name, path in zip(phase_names, task_files, strict=True)
    ]
    stable_phase = phase_names[0]
    route_exprs = (
        parse_route_exprs(args.route_expr, stable_phase=stable_phase)
        if args.route_expr is not None
        else default_route_exprs(phase_names)
    )
    audit_route_exprs = (
        parse_route_exprs(args.audit_route_expr, stable_phase=stable_phase)
        if args.audit_route_expr is not None
        else None
    )
    merged_audit_route_exprs = merge_route_exprs(route_exprs, audit_route_exprs)
    probes = probes_from_args(args, stable_phase=stable_phase)
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
    phase_texts = [load_texts(path) for path in task_files]
    probe_texts = [load_texts(probe.path) for probe in probes]
    seeds = args.seeds or [args.seed]
    results = [
        result
        for seed in seeds
        for result in run_memory_bank_seed_results(
            variant,
            settings=settings,
            phases=phases,
            phase_texts=phase_texts,
            phase_steps=args.phase_steps,
            seed=seed,
            device=args.device,
            route_exprs=route_exprs,
            route_selection=args.route_selection,
            contextual_route=args.contextual_route,
            probes=probes,
            probe_texts=probe_texts,
            audit_route_exprs=audit_route_exprs,
            ambiguity_margin=args.ambiguity_margin,
            include_global_routes=args.global_route_baseline,
        )
    ]
    record: MemoryBankRunRecord = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "experiment": "memory_bank",
            "model": args.model,
            "device": args.device,
            "task_files": task_files,
            "phase_names": phase_names,
            "probe_files": [probe.path for probe in probes],
            "probe_names": [probe.name for probe in probes],
            "probe_routes": [probe.expected_route for probe in probes],
            "phase_steps": args.phase_steps,
            "seeds": seeds,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
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
            "route_exprs": [route.expression for route in route_exprs],
            "audit_route_exprs": [route.expression for route in merged_audit_route_exprs],
            "contextual_route": args.contextual_route,
            "route_selection": args.route_selection,
            "ambiguity_margin": args.ambiguity_margin,
            "global_route_baseline": args.global_route_baseline,
            "snapshot_each_phase": args.snapshot_each_phase,
            "emit_deltas": args.emit_deltas,
            "frontier_score": "sequential_eval_loss - contextual_eval_loss",
        },
        "git_status": git_status(),
        "results": results,
        "summary": summarize_memory_bank(results),
    }
    if args.output_dir is not None:
        write_run_record(record, args.output_dir)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
