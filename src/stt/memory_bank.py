"""Contextual routed memory-bank experiments for sequential LoRA deltas.

This experiment generalizes the A/B/C routed-accretion scaffold to N phases.
It snapshots the adapter after every phase, stores post-A phase deltas in a
small memory bank, then composes route expressions per prompt/domain.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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
    split_corpus,
    write_run_record,
)
from stt.oracle_compose import (
    apply_trainable_state,
    compose_state,
    snapshot_trainable_state,
    split_eval_encoded,
    subtract_state,
)

RouteSelection = Literal["oracle", "loss_probe", "calibration", "distilled", "micro_probe"]
ResidualRouteMode = Literal["full", "axis", "pairs"]
DistilledSelectorMethod = Literal["centroid", "knn"]

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
    abstained: bool


@dataclass(frozen=True)
class DistilledRouteSelector:
    """Cheap token selector distilled from loss-probe route labels."""

    method: DistilledSelectorMethod
    centroids: dict[str, dict[int, float]]
    example_vectors: list[dict[int, float]]
    example_routes: list[str]
    route_counts: dict[str, int]
    default_route: str
    selector_margin: float
    knn_k: int


@dataclass(frozen=True)
class ResidualRouteCandidate:
    """One residual deformation of a base sleep route."""

    route: RouteExpression
    base_route_expr: str
    residual: dict[str, float]


class DomainRouteResult(TypedDict):
    """Serializable contextual-routing result for one held-out domain."""

    selected_route_counts: dict[str, int]
    expected_route: str
    most_selected_route: str
    selection_count: int
    route_accuracy: float
    ambiguous_count: int
    ambiguous_rate: float
    abstained_count: NotRequired[int]
    abstention_rate: NotRequired[float]
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
    selected_route_rank: NotRequired[float]
    selected_top_k_count: NotRequired[int]
    selected_top_k_rate: NotRequired[float]
    expected_top_k_count: NotRequired[int]
    expected_top_k_rate: NotRequired[float]
    top_k_loss_gap: NotRequired[float]
    top_k_boundary_margin: NotRequired[float]
    top_k_route_counts: NotRequired[dict[str, int]]
    route_margin: NotRequired[float]
    route_entropy: NotRequired[float]
    low_margin_count: NotRequired[int]
    low_margin_rate: NotRequired[float]
    ambiguity_abstention_rate: NotRequired[float]
    false_confident_route_rate: NotRequired[float]
    expected_route_loss: NotRequired[float]
    expected_loss_gap: NotRequired[float]
    optimal_route_count: NotRequired[int]
    optimal_route_rate: NotRequired[float]
    selected_route_expr: NotRequired[str]
    selected_residual: NotRequired[dict[str, float]]
    best_route_expr: NotRequired[str]
    best_residual: NotRequired[dict[str, float]]
    residual_gap: NotRequired[float]


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
    residual_contextual_route: NotRequired[bool]
    residual_route_base_expr: NotRequired[str]
    residual_route_grid: NotRequired[list[float]]
    residual_route_mode: NotRequired[str]
    residual_route_phases: NotRequired[list[str]]
    residual_candidate_count: NotRequired[int]
    route_top_k: NotRequired[int]
    distilled_selector_method: NotRequired[str]
    distilled_selector_margin: NotRequired[float]
    distilled_knn_k: NotRequired[int]
    micro_probe_prefix_words: NotRequired[int]
    micro_probe_max_length: NotRequired[int]
    micro_probe_template: NotRequired[str]
    micro_probe_margin: NotRequired[float]
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
    selected_route_rank: NotRequired[float]
    selected_top_k_rate: NotRequired[float]
    expected_top_k_rate: NotRequired[float]
    top_k_loss_gap: NotRequired[float]
    top_k_boundary_margin: NotRequired[float]
    route_margin: NotRequired[float]
    route_entropy: NotRequired[float]
    abstention_rate: NotRequired[float]
    low_margin_rate: NotRequired[float]
    ambiguity_abstention_rate: NotRequired[float]
    false_confident_route_rate: NotRequired[float]
    residual_selected_gap: NotRequired[float]
    residual_optimal_rate: NotRequired[float]
    residual_route_margin: NotRequired[float]
    residual_route_entropy: NotRequired[float]
    expected_loss_gap: NotRequired[float]
    probe_eval_loss: NotRequired[float]
    probe_route_accuracy: NotRequired[float]
    probe_ambiguous_rate: NotRequired[float]
    probe_optimal_route_rate: NotRequired[float]
    probe_selected_loss_gap: NotRequired[float]
    probe_selected_top_k_rate: NotRequired[float]
    probe_expected_top_k_rate: NotRequired[float]
    probe_top_k_loss_gap: NotRequired[float]
    probe_top_k_boundary_margin: NotRequired[float]
    probe_expected_loss_gap: NotRequired[float]
    probe_abstention_rate: NotRequired[float]
    probe_low_margin_rate: NotRequired[float]
    probe_ambiguity_abstention_rate: NotRequired[float]
    probe_false_confident_route_rate: NotRequired[float]
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


def clamp_scale(value: float, *, min_scale: float, max_scale: float) -> float:
    """Clamp a route scale into the configured residual coefficient range."""
    return min(max(value, min_scale), max_scale)


def route_expr_from_scales(
    *,
    stable_phase: str,
    scales: dict[str, float],
    phase_order: list[str] | None = None,
) -> RouteExpression:
    """Build a normalized route expression from named non-stable scales."""
    ordered = phase_order or list(scales)
    parts = [stable_phase]
    emitted = set()
    for phase in ordered:
        if phase == stable_phase or phase not in scales:
            continue
        scale = scales[phase]
        if scale == 0.0:
            continue
        parts.append(phase if scale == 1.0 else f"{scale:g}{phase}")
        emitted.add(phase)
    for phase, scale in scales.items():
        if phase in emitted or phase == stable_phase or scale == 0.0:
            continue
        parts.append(phase if scale == 1.0 else f"{scale:g}{phase}")
    return parse_route_expr("+".join(parts), stable_phase=stable_phase)


def residual_offsets(
    phases: list[str],
    grid: list[float],
    mode: ResidualRouteMode,
) -> list[dict[str, float]]:
    """Return residual offset dictionaries for full, axis, or pair mode."""
    if not phases:
        return [{}]
    if not grid:
        raise ValueError("residual route grid must include at least one value")

    zero = {phase: 0.0 for phase in phases}
    if mode == "full":
        return [
            dict(zip(phases, values, strict=True))
            for values in itertools.product(grid, repeat=len(phases))
        ]

    offsets = [zero]
    for phase in phases:
        for value in grid:
            if value == 0.0:
                continue
            offsets.append({**zero, phase: value})
    if mode == "pairs":
        for start in range(0, len(phases) - 1, 2):
            pair = phases[start : start + 2]
            for values in itertools.product(grid, repeat=2):
                if all(value == 0.0 for value in values):
                    continue
                offsets.append({**zero, **dict(zip(pair, values, strict=True))})
    return offsets


def generate_residual_routes(
    base_route: RouteExpression,
    *,
    phases: list[str],
    grid: list[float],
    mode: ResidualRouteMode = "full",
    min_scale: float = 0.0,
    max_scale: float = 1.5,
) -> list[ResidualRouteCandidate]:
    """Generate residual deformations around a base sleep route."""
    if min_scale > max_scale:
        raise ValueError("residual min scale cannot exceed max scale")
    if any(phase == base_route.stable_phase for phase in phases):
        raise ValueError("residual phases must not include the stable phase")

    candidates: dict[str, ResidualRouteCandidate] = {}
    for residual in residual_offsets(phases, grid, mode):
        scales = dict(base_route.scales)
        clamped_residual = {}
        for phase in phases:
            base_scale = base_route.scales.get(phase, 0.0)
            scale = clamp_scale(
                base_scale + residual.get(phase, 0.0),
                min_scale=min_scale,
                max_scale=max_scale,
            )
            scales[phase] = scale
            clamped_residual[phase] = scale - base_scale
        route = route_expr_from_scales(
            stable_phase=base_route.stable_phase,
            scales=scales,
            phase_order=phases,
        )
        candidates.setdefault(
            route.expression,
            ResidualRouteCandidate(
                route=route,
                base_route_expr=base_route.expression,
                residual=clamped_residual,
            ),
        )
    return list(candidates.values())


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


MICRO_PROBE_BOUNDARY_RE = re.compile(
    r"\b(?:answer\s+should|expected\s+(?:semantic\s+)?route)\b",
    re.IGNORECASE,
)
MICRO_PROBE_PROJECT_RE = re.compile(
    r"\bProject\s+([A-Za-z][A-Za-z0-9_-]*)\b",
    re.IGNORECASE,
)
MICRO_PROBE_NAMED_SCOPE_RE = re.compile(
    r"\b(Alpha|Beta|Gamma|Delta|Epsilon)\b",
    re.IGNORECASE,
)


def bounded_phase_eval_texts(
    texts: list[str],
    settings: LoraSettings,
    seed: int,
) -> list[str]:
    """Return the raw eval texts encoded by `prepare_encoded_splits`."""
    _, eval_texts = split_corpus(texts, seed)
    eval_sample_count = max(settings.batch_size, settings.batch_size * settings.eval_batches)
    return eval_texts[:eval_sample_count]


def split_eval_texts(
    texts: list[str],
    batch_size: int,
) -> tuple[list[str], list[str], bool]:
    """Split raw eval texts the same way `split_eval_encoded` splits tensors."""
    if len(texts) < batch_size * 2:
        return texts, texts, False
    midpoint = len(texts) // 2
    return texts[:midpoint], texts[midpoint:], True


def prepare_probe_texts(
    texts: list[str],
    settings: LoraSettings,
    seed: int,
) -> list[str]:
    """Return the bounded shuffled raw probe texts encoded for eval-only probes."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=generator).tolist()
    sample_count = max(settings.batch_size * 2, settings.batch_size * settings.eval_batches * 2)
    return [texts[index] for index in indices[:sample_count]]


def prepare_probe_encoded(
    tokenizer: Any,
    texts: list[str],
    settings: LoraSettings,
    seed: int,
    device: str,
) -> dict[str, Tensor]:
    """Create a bounded shuffled encoded split from eval-only probe texts."""
    probe_texts = prepare_probe_texts(texts, settings, seed)
    return encode_texts(tokenizer, probe_texts, settings.max_length, device)


def micro_probe_scope(text: str) -> str:
    """Infer a scope-only selector answer from answer-stripped prompt text."""
    lowered = text.lower()
    if "ambiguous scope" in lowered or "unscoped" in lowered:
        return "scope unclear"
    project_match = MICRO_PROBE_PROJECT_RE.search(text)
    if project_match:
        return f"Project {project_match.group(1).capitalize()}"
    named_match = MICRO_PROBE_NAMED_SCOPE_RE.search(text)
    if named_match:
        return f"Project {named_match.group(1).capitalize()}"
    return "scope unclear"


def build_micro_probe_text(
    text: str,
    *,
    prefix_words: int,
    template: str,
) -> str:
    """Build an answer-stripped route-selection probe from one report example."""
    if prefix_words < 1:
        raise ValueError("micro-probe prefix words must be at least 1")
    if "{prefix}" not in template and "{scope}" not in template:
        raise ValueError("micro-probe template must contain {prefix} or {scope}")
    collapsed = " ".join(text.split())
    boundary = MICRO_PROBE_BOUNDARY_RE.search(collapsed)
    safe_text = collapsed[: boundary.start()].strip(" .,:;-") if boundary else collapsed
    if not safe_text:
        safe_text = collapsed
    prefix = " ".join(safe_text.split()[:prefix_words])
    scope = micro_probe_scope(safe_text)
    try:
        return template.format(prefix=prefix, scope=scope)
    except (IndexError, KeyError) as exc:
        raise ValueError("micro-probe template may only use {prefix} and {scope}") from exc


def build_micro_probe_texts(
    texts: list[str],
    *,
    prefix_words: int,
    template: str,
) -> list[str]:
    """Build aligned micro-probe texts for a report split."""
    return [
        build_micro_probe_text(text, prefix_words=prefix_words, template=template)
        for text in texts
    ]


def prepare_micro_probe_encoded(
    tokenizer: Any,
    texts: list[str],
    settings: LoraSettings,
    device: str,
    *,
    prefix_words: int,
    max_length: int,
    template: str,
) -> dict[str, Tensor]:
    """Encode short route-selection probes aligned to held-out report examples."""
    if max_length < 1:
        raise ValueError("micro-probe max length must be at least 1")
    micro_texts = build_micro_probe_texts(
        texts,
        prefix_words=prefix_words,
        template=template,
    )
    return encode_texts(tokenizer, micro_texts, max_length, device)


def one_example(encoded: dict[str, Tensor], index: int) -> dict[str, Tensor]:
    """Return a single encoded example preserving batch dimensions."""
    return {name: value[index : index + 1] for name, value in encoded.items()}


def select_examples(encoded: dict[str, Tensor], indices: list[int]) -> dict[str, Tensor]:
    """Return selected encoded examples preserving tensor devices."""
    index_tensor = torch.tensor(indices, dtype=torch.long, device=encoded["input_ids"].device)
    return {name: value.index_select(0, index_tensor) for name, value in encoded.items()}


def token_count_vector(encoded: dict[str, Tensor], index: int) -> dict[int, float]:
    """Return an L2-normalized token-count vector for one encoded example."""
    input_ids = encoded["input_ids"][index].detach().cpu().tolist()
    if "attention_mask" in encoded:
        attention_mask = encoded["attention_mask"][index].detach().cpu().tolist()
    else:
        attention_mask = [1 for _ in input_ids]

    counts: dict[int, float] = {}
    for token_id, mask_value in zip(input_ids, attention_mask, strict=True):
        if int(mask_value) == 0:
            continue
        counts[int(token_id)] = counts.get(int(token_id), 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0.0:
        return {}
    return {token_id: value / norm for token_id, value in counts.items()}


def dot_sparse(left: dict[int, float], right: dict[int, float]) -> float:
    """Return a dot product for sparse token vectors."""
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token_id, 0.0) for token_id, value in left.items())


def build_distilled_route_selector_from_scores(
    encoded: dict[str, Tensor],
    scored_examples: list[list[tuple[str, float]]],
    *,
    selector_margin: float = 0.0,
    method: DistilledSelectorMethod = "centroid",
    knn_k: int = 3,
) -> DistilledRouteSelector:
    """Train a token-centroid route selector from loss-probe oracle labels."""
    if selector_margin < 0.0:
        raise ValueError("selector margin cannot be negative")
    if knn_k < 1:
        raise ValueError("distilled kNN k must be at least 1")
    sample_count = encoded["input_ids"].shape[0]
    if len(scored_examples) != sample_count:
        raise ValueError("scored_examples must align with encoded examples")
    if sample_count == 0:
        raise ValueError("at least one encoded selector example is required")

    sums: dict[str, dict[int, float]] = {}
    route_counts: dict[str, int] = {}
    example_vectors = []
    example_routes = []
    for index, scored in enumerate(scored_examples):
        if not scored:
            raise ValueError("each selector example needs at least one route score")
        route = scored[0][0]
        vector = token_count_vector(encoded, index)
        example_vectors.append(vector)
        example_routes.append(route)
        route_counts[route] = route_counts.get(route, 0) + 1
        route_sum = sums.setdefault(route, {})
        for token_id, value in vector.items():
            route_sum[token_id] = route_sum.get(token_id, 0.0) + value

    centroids = {}
    for route, vector in sums.items():
        count = route_counts[route]
        averaged = {token_id: value / count for token_id, value in vector.items()}
        norm = math.sqrt(sum(value * value for value in averaged.values()))
        centroids[route] = (
            {token_id: value / norm for token_id, value in averaged.items()}
            if norm > 0.0
            else {}
        )
    default_route = most_selected_route(route_counts)
    return DistilledRouteSelector(
        method=method,
        centroids=centroids,
        example_vectors=example_vectors,
        example_routes=example_routes,
        route_counts=route_counts,
        default_route=default_route,
        selector_margin=selector_margin,
        knn_k=knn_k,
    )


def build_distilled_route_selector(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded_splits: list[dict[str, Tensor]],
    settings: LoraSettings,
    *,
    selector_margin: float = 0.0,
    method: DistilledSelectorMethod = "centroid",
    knn_k: int = 3,
) -> DistilledRouteSelector:
    """Score calibration examples once, then train a cheap token selector."""
    scored_examples = []
    encoded_examples = []
    for encoded in encoded_splits:
        split_scores = route_scores_for_examples(model, route_states, encoded, settings)
        scored_examples.extend(split_scores)
        encoded_examples.extend(one_example(encoded, index) for index in range(len(split_scores)))

    if not encoded_examples:
        raise ValueError("at least one selector split is required")
    concatenated = {
        name: torch.cat([example[name] for example in encoded_examples], dim=0)
        for name in encoded_examples[0]
    }
    return build_distilled_route_selector_from_scores(
        concatenated,
        scored_examples,
        selector_margin=selector_margin,
        method=method,
        knn_k=knn_k,
    )


def distilled_selector_scores(
    selector: DistilledRouteSelector,
    vector: dict[int, float],
) -> list[tuple[str, float]]:
    """Return route scores from the configured distilled selector backend."""
    if selector.method == "centroid":
        return sorted(
            (
                (route, dot_sparse(vector, centroid))
                for route, centroid in selector.centroids.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

    neighbors = sorted(
        (
            (route, dot_sparse(vector, example_vector))
            for route, example_vector in zip(
                selector.example_routes,
                selector.example_vectors,
                strict=True,
            )
        ),
        key=lambda item: item[1],
        reverse=True,
    )[: selector.knn_k]
    route_scores: dict[str, float] = {}
    for route, score in neighbors:
        route_scores[route] = route_scores.get(route, 0.0) + score
    if not route_scores:
        return []
    return sorted(route_scores.items(), key=lambda item: item[1], reverse=True)


def distilled_route_choice(
    selector: DistilledRouteSelector,
    encoded: dict[str, Tensor],
    index: int,
) -> RouteChoice:
    """Select one route with the distilled token-centroid selector."""
    vector = token_count_vector(encoded, index)
    scored = distilled_selector_scores(selector, vector)
    if not scored:
        best_route = selector.default_route
        ambiguous = False
    else:
        best_route, best_score = scored[0]
        ambiguous = (
            selector.selector_margin > 0.0
            and len(scored) > 1
            and best_score - scored[1][1] < selector.selector_margin
        )
    return RouteChoice(
        selected_route="uncertain" if ambiguous else best_route,
        eval_route=best_route,
        loss=0.0,
        ambiguous=ambiguous,
        abstained=ambiguous,
    )


def distilled_route_choices(
    selector: DistilledRouteSelector,
    encoded: dict[str, Tensor],
) -> list[RouteChoice]:
    """Select routes for every encoded example with a distilled selector."""
    return [
        distilled_route_choice(selector, encoded, index)
        for index in range(encoded["input_ids"].shape[0])
    ]


def selected_route_eval_loss(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    encoded: dict[str, Tensor],
    settings: LoraSettings,
    selected_eval_routes: list[str],
) -> float:
    """Evaluate a per-example route assignment by grouping examples by route."""
    sample_count = encoded["input_ids"].shape[0]
    if len(selected_eval_routes) != sample_count:
        raise ValueError("selected_eval_routes must align with encoded examples")
    if sample_count == 0:
        return 0.0

    indices_by_route: dict[str, list[int]] = {}
    for index, route in enumerate(selected_eval_routes):
        if route not in route_states:
            raise ValueError(f"unknown route: {route}")
        indices_by_route.setdefault(route, []).append(index)

    total_loss = 0.0
    for route, indices in indices_by_route.items():
        apply_trainable_state(model, route_states[route])
        routed_encoded = select_examples(encoded, indices)
        total_loss += eval_loss(
            model,
            routed_encoded,
            settings_for_encoded(settings, routed_encoded),
        ) * len(indices)
    return total_loss / sample_count


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
        abstained=ambiguous,
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
        "abstained_count": ambiguous_count,
        "abstention_rate": ambiguous_rate,
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
    selected_routes: list[str] | None = None,
    top_k: int = 3,
    ambiguity_margin: float = 0.0,
) -> dict[str, float | int | dict[str, int] | str]:
    """Compare selected and expected routes against per-example best-loss routes."""
    scored_examples = route_scores_for_examples(model, route_states, encoded, settings)
    return route_optimality_from_scores(
        scored_examples,
        expected_route=expected_route,
        selected_eval_routes=selected_eval_routes,
        selected_routes=selected_routes,
        top_k=top_k,
        ambiguity_margin=ambiguity_margin,
    )


def route_optimality_from_scores(
    scored_examples: list[list[tuple[str, float]]],
    *,
    expected_route: str,
    selected_eval_routes: list[str],
    selected_routes: list[str] | None = None,
    top_k: int = 3,
    ambiguity_margin: float = 0.0,
) -> dict[str, float | int | dict[str, int] | str]:
    """Compare selected and expected routes against cached per-example route losses."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    sample_count = len(scored_examples)
    if sample_count == 0:
        return {
            "best_route_counts": {},
            "most_best_route": "none",
            "best_eval_loss": 0.0,
            "selected_loss_gap": 0.0,
            "selected_route_rank": 0.0,
            "selected_top_k_count": 0,
            "selected_top_k_rate": 0.0,
            "expected_top_k_count": 0,
            "expected_top_k_rate": 0.0,
            "top_k_loss_gap": 0.0,
            "top_k_boundary_margin": 0.0,
            "top_k_route_counts": {},
            "route_margin": 0.0,
            "route_entropy": 0.0,
            "low_margin_count": 0,
            "low_margin_rate": 0.0,
            "ambiguity_abstention_rate": 0.0,
            "false_confident_route_rate": 0.0,
            "expected_route_loss": 0.0,
            "expected_loss_gap": 0.0,
            "optimal_route_count": 0,
            "optimal_route_rate": 0.0,
        }
    if len(selected_eval_routes) == 1:
        selected_eval_routes = selected_eval_routes * sample_count
    if len(selected_eval_routes) != sample_count:
        raise ValueError("selected_eval_routes must have one route or one route per example")
    selected_routes = selected_routes or selected_eval_routes
    if len(selected_routes) == 1:
        selected_routes = selected_routes * sample_count
    if len(selected_routes) != sample_count:
        raise ValueError("selected_routes must have one route or one route per example")

    first_scores = dict(scored_examples[0])
    expected_eval_route = (
        expected_route if expected_route in first_scores else next(iter(first_scores))
    )
    best_route_counts: dict[str, int] = {}
    top_k_route_counts: dict[str, int] = {}
    best_loss_total = 0.0
    selected_gap_total = 0.0
    selected_rank_total = 0.0
    top_k_loss_gap_total = 0.0
    top_k_boundary_margin_total = 0.0
    margin_total = 0.0
    entropy_total = 0.0
    expected_loss_total = 0.0
    expected_gap_total = 0.0
    optimal_count = 0
    selected_top_k_count = 0
    expected_top_k_count = 0
    low_margin_count = 0
    abstained_low_margin_count = 0
    false_confident_count = 0
    for index, scored in enumerate(scored_examples):
        best_route, best_loss = scored[0]
        score_by_route = dict(scored)
        selected_route = selected_eval_routes[index]
        if selected_route not in score_by_route:
            selected_route = best_route
        selected_label = selected_routes[index]
        selected_loss = score_by_route[selected_route]
        selected_rank = next(
            rank
            for rank, (route_name, _) in enumerate(scored, start=1)
            if route_name == selected_route
        )
        expected_loss = score_by_route[expected_eval_route]
        route_margin = scored[1][1] - best_loss if len(scored) > 1 else 0.0
        margin_total += route_margin
        effective_top_k = min(top_k, len(scored))
        top_k_scores = scored[:effective_top_k]
        top_k_routes = {route_name for route_name, _ in top_k_scores}
        for route_name in top_k_routes:
            top_k_route_counts[route_name] = top_k_route_counts.get(route_name, 0) + 1
        selected_top_k_count += int(selected_route in top_k_routes)
        expected_top_k_count += int(expected_eval_route in top_k_routes)
        top_k_loss_gap_total += top_k_scores[-1][1] - best_loss
        top_k_boundary_margin_total += (
            scored[effective_top_k][1] - top_k_scores[-1][1]
            if len(scored) > effective_top_k
            else 0.0
        )
        low_margin = ambiguity_margin > 0.0 and route_margin < ambiguity_margin
        low_margin_count += int(low_margin)
        abstained = selected_label == "uncertain"
        abstained_low_margin_count += int(low_margin and abstained)
        false_confident_count += int(not abstained and selected_label != expected_route)
        weights = [math.exp(-(loss - best_loss)) for _, loss in scored]
        weight_total = sum(weights)
        if weight_total > 0.0:
            entropy_total += -sum(
                (weight / weight_total) * math.log(weight / weight_total)
                for weight in weights
                if weight > 0.0
            )
        best_route_counts[best_route] = best_route_counts.get(best_route, 0) + 1
        best_loss_total += best_loss
        selected_gap_total += selected_loss - best_loss
        selected_rank_total += selected_rank
        expected_loss_total += expected_loss
        expected_gap_total += expected_loss - best_loss
        optimal_count += int(selected_loss <= best_loss + 1e-8)

    ambiguity_abstention_rate = (
        abstained_low_margin_count / low_margin_count if low_margin_count else 0.0
    )
    return {
        "best_route_counts": best_route_counts,
        "most_best_route": most_selected_route(best_route_counts),
        "best_eval_loss": best_loss_total / sample_count,
        "selected_loss_gap": selected_gap_total / sample_count,
        "selected_route_rank": selected_rank_total / sample_count,
        "selected_top_k_count": selected_top_k_count,
        "selected_top_k_rate": selected_top_k_count / sample_count,
        "expected_top_k_count": expected_top_k_count,
        "expected_top_k_rate": expected_top_k_count / sample_count,
        "top_k_loss_gap": top_k_loss_gap_total / sample_count,
        "top_k_boundary_margin": top_k_boundary_margin_total / sample_count,
        "top_k_route_counts": top_k_route_counts,
        "route_margin": margin_total / sample_count,
        "route_entropy": entropy_total / sample_count,
        "low_margin_count": low_margin_count,
        "low_margin_rate": low_margin_count / sample_count,
        "ambiguity_abstention_rate": ambiguity_abstention_rate,
        "false_confident_route_rate": false_confident_count / sample_count,
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
    top_k: int = 3,
    ambiguity_margin: float = 0.0,
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
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
        )
        if audit_scores is not None
        else route_optimality_audit(
            model,
            route_states,
            encoded,
            settings,
            expected_route=expected_route,
            selected_eval_routes=[route_name],
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
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
    top_k: int = 3,
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
            selected_routes=[choice.selected_route],
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
        )
        if audit_scores is not None
        else route_optimality_audit(
            model,
            route_states,
            report_encoded,
            settings,
            expected_route=expected_route,
            selected_eval_routes=[choice.eval_route],
            selected_routes=[choice.selected_route],
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
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
    top_k: int = 3,
) -> DomainRouteResult:
    """Select and evaluate routes per held-out example using probe loss."""
    selected_route_counts: dict[str, int] = {}
    total_loss = 0.0
    correct_count = 0
    ambiguous_count = 0
    selected_eval_routes = []
    selected_routes = []
    sample_count = encoded["input_ids"].shape[0]

    scored_examples = selection_scores
    if scored_examples is None:
        scored_examples = route_scores_for_examples(model, route_states, encoded, settings)

    for scored in scored_examples:
        choice = route_choice_from_scored(scored, ambiguity_margin=ambiguity_margin)
        selected_eval_routes.append(choice.eval_route)
        selected_routes.append(choice.selected_route)
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
        selected_routes=selected_routes,
        top_k=top_k,
        ambiguity_margin=ambiguity_margin,
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


def evaluate_micro_probe_domain(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    micro_probe_encoded: dict[str, Tensor],
    report_encoded: dict[str, Tensor],
    settings: LoraSettings,
    *,
    expected_route: str,
    sequential_eval_loss: float,
    learned_eval_loss: float,
    initial_eval_loss: float,
    micro_probe_margin: float,
    ambiguity_margin: float,
    audit_scores: list[list[tuple[str, float]]] | None = None,
    top_k: int = 3,
) -> DomainRouteResult:
    """Select per-example routes on short probes, then eval full examples."""
    sample_count = report_encoded["input_ids"].shape[0]
    if micro_probe_encoded["input_ids"].shape[0] != sample_count:
        raise ValueError("micro_probe_encoded must align with report_encoded")

    scored_examples = route_scores_for_examples(
        model,
        route_states,
        micro_probe_encoded,
        settings,
    )
    choices = [
        route_choice_from_scored(scored, ambiguity_margin=micro_probe_margin)
        for scored in scored_examples
    ]
    selected_eval_routes = [choice.eval_route for choice in choices]
    selected_routes = [choice.selected_route for choice in choices]
    eval_loss_value = selected_route_eval_loss(
        model,
        route_states,
        report_encoded,
        settings,
        selected_eval_routes,
    )

    selected_route_counts: dict[str, int] = {}
    correct_count = 0
    ambiguous_count = 0
    for choice in choices:
        selected_route_counts[choice.selected_route] = (
            selected_route_counts.get(choice.selected_route, 0) + 1
        )
        correct_count += int(choice.selected_route == expected_route)
        ambiguous_count += int(choice.ambiguous)

    audit = (
        route_optimality_from_scores(
            audit_scores,
            expected_route=expected_route,
            selected_eval_routes=selected_eval_routes,
            selected_routes=selected_routes,
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
        )
        if audit_scores is not None
        else route_optimality_audit(
            model,
            route_states,
            report_encoded,
            settings,
            expected_route=expected_route,
            selected_eval_routes=selected_eval_routes,
            selected_routes=selected_routes,
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
        )
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


def evaluate_distilled_domain(
    model: torch.nn.Module,
    route_states: dict[str, dict[str, Tensor]],
    selector: DistilledRouteSelector,
    encoded: dict[str, Tensor],
    settings: LoraSettings,
    *,
    expected_route: str,
    sequential_eval_loss: float,
    learned_eval_loss: float,
    initial_eval_loss: float,
    ambiguity_margin: float,
    audit_scores: list[list[tuple[str, float]]] | None = None,
    top_k: int = 3,
) -> DomainRouteResult:
    """Evaluate held-out examples with a distilled cheap route selector."""
    choices = distilled_route_choices(selector, encoded)
    selected_eval_routes = [choice.eval_route for choice in choices]
    selected_routes = [choice.selected_route for choice in choices]
    eval_loss_value = selected_route_eval_loss(
        model,
        route_states,
        encoded,
        settings,
        selected_eval_routes,
    )
    selected_route_counts: dict[str, int] = {}
    correct_count = 0
    ambiguous_count = 0
    for choice in choices:
        selected_route_counts[choice.selected_route] = (
            selected_route_counts.get(choice.selected_route, 0) + 1
        )
        correct_count += int(choice.selected_route == expected_route)
        ambiguous_count += int(choice.ambiguous)

    audit = (
        route_optimality_from_scores(
            audit_scores,
            expected_route=expected_route,
            selected_eval_routes=selected_eval_routes,
            selected_routes=selected_routes,
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
        )
        if audit_scores is not None
        else route_optimality_audit(
            model,
            route_states,
            encoded,
            settings,
            expected_route=expected_route,
            selected_eval_routes=selected_eval_routes,
            selected_routes=selected_routes,
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
        )
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
    top_k: int = 3,
    distilled_selector_margin: float = 0.0,
    distilled_selector_method: DistilledSelectorMethod = "centroid",
    distilled_knn_k: int = 3,
    eval_micro_probe: dict[str, dict[str, Tensor]] | None = None,
    micro_probe_margin: float = 0.0,
    expected_routes_by_domain: dict[str, str] | None = None,
    audit_scores_by_domain: dict[str, list[list[tuple[str, float]]]] | None = None,
) -> dict[str, DomainRouteResult]:
    """Evaluate all domains with the configured contextual routing strategy."""
    candidate_routes = list(route_states.keys())
    final_phase = phase_names[-1]
    distilled_selector = (
        build_distilled_route_selector(
            model,
            route_states,
            [eval_select[domain] for domain in phase_names],
            settings,
            selector_margin=distilled_selector_margin,
            method=distilled_selector_method,
            knn_k=distilled_knn_k,
        )
        if route_selection == "distilled"
        else None
    )
    per_domain = {}
    for domain in phase_names:
        expected_route = (
            expected_routes_by_domain[domain]
            if expected_routes_by_domain is not None
            else expected_route_for_domain(domain, phase_names, candidate_routes)
        )
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
                top_k=top_k,
                ambiguity_margin=ambiguity_margin,
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
                top_k=top_k,
                **common,
            )
        elif route_selection == "loss_probe":
            per_domain[domain] = evaluate_loss_probe_domain(
                model,
                route_states,
                eval_report[domain],
                settings,
                ambiguity_margin=ambiguity_margin,
                top_k=top_k,
                **common,
            )
        elif route_selection == "micro_probe":
            if eval_micro_probe is None:
                raise ValueError("micro-probe inputs were not built")
            per_domain[domain] = evaluate_micro_probe_domain(
                model,
                route_states,
                eval_micro_probe[domain],
                eval_report[domain],
                settings,
                micro_probe_margin=micro_probe_margin,
                ambiguity_margin=ambiguity_margin,
                top_k=top_k,
                **common,
            )
        else:
            if distilled_selector is None:
                raise ValueError("distilled selector was not built")
            per_domain[domain] = evaluate_distilled_domain(
                model,
                route_states,
                distilled_selector,
                eval_report[domain],
                settings,
                ambiguity_margin=ambiguity_margin,
                top_k=top_k,
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
    top_k: int = 3,
    distilled_selector_margin: float = 0.0,
    distilled_selector_method: DistilledSelectorMethod = "centroid",
    distilled_knn_k: int = 3,
    probe_micro: dict[str, dict[str, Tensor]] | None = None,
    micro_probe_margin: float = 0.0,
    audit_scores_by_probe: dict[str, list[list[tuple[str, float]]]] | None = None,
) -> dict[str, DomainRouteResult]:
    """Evaluate eval-only boundary probes with expected semantic routes."""
    distilled_selector = (
        build_distilled_route_selector(
            model,
            route_states,
            [probe_select[probe.name] for probe in probes],
            settings,
            selector_margin=distilled_selector_margin,
            method=distilled_selector_method,
            knn_k=distilled_knn_k,
        )
        if route_selection == "distilled" and probes
        else None
    )
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
                top_k=top_k,
                ambiguity_margin=ambiguity_margin,
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
                top_k=top_k,
                **common,
            )
        elif route_selection == "loss_probe":
            per_probe[probe.name] = evaluate_loss_probe_domain(
                model,
                route_states,
                probe_report[probe.name],
                settings,
                ambiguity_margin=ambiguity_margin,
                top_k=top_k,
                **common,
            )
        elif route_selection == "micro_probe":
            if probe_micro is None:
                raise ValueError("micro-probe inputs were not built")
            per_probe[probe.name] = evaluate_micro_probe_domain(
                model,
                route_states,
                probe_micro[probe.name],
                probe_report[probe.name],
                settings,
                micro_probe_margin=micro_probe_margin,
                ambiguity_margin=ambiguity_margin,
                top_k=top_k,
                **common,
            )
        else:
            if distilled_selector is None:
                raise ValueError("distilled selector was not built")
            per_probe[probe.name] = evaluate_distilled_domain(
                model,
                route_states,
                distilled_selector,
                probe_report[probe.name],
                settings,
                ambiguity_margin=ambiguity_margin,
                top_k=top_k,
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
    top_k: int = 3,
    ambiguity_margin: float = 0.0,
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
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
        )
        if audit_scores is not None
        else route_optimality_audit(
            model,
            route_states,
            encoded,
            settings,
            expected_route=expected_route,
            selected_eval_routes=[route_name],
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
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
    top_k: int = 3,
    ambiguity_margin: float = 0.0,
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
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
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
    top_k: int = 3,
    ambiguity_margin: float = 0.0,
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
            top_k=top_k,
            ambiguity_margin=ambiguity_margin,
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
            "abstention_rate": 0.0,
            "optimal_route_rate": 0.0,
            "selected_loss_gap": 0.0,
            "selected_route_rank": 0.0,
            "selected_top_k_rate": 0.0,
            "expected_top_k_rate": 0.0,
            "top_k_loss_gap": 0.0,
            "top_k_boundary_margin": 0.0,
            "route_margin": 0.0,
            "route_entropy": 0.0,
            "low_margin_rate": 0.0,
            "ambiguity_abstention_rate": 0.0,
            "false_confident_route_rate": 0.0,
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
        "abstention_rate": weighted("abstention_rate"),
        "optimal_route_rate": weighted("optimal_route_rate"),
        "selected_loss_gap": weighted("selected_loss_gap"),
        "selected_route_rank": weighted("selected_route_rank"),
        "selected_top_k_rate": weighted("selected_top_k_rate"),
        "expected_top_k_rate": weighted("expected_top_k_rate"),
        "top_k_loss_gap": weighted("top_k_loss_gap"),
        "top_k_boundary_margin": weighted("top_k_boundary_margin"),
        "route_margin": weighted("route_margin"),
        "route_entropy": weighted("route_entropy"),
        "low_margin_rate": weighted("low_margin_rate"),
        "ambiguity_abstention_rate": weighted("ambiguity_abstention_rate"),
        "false_confident_route_rate": weighted("false_confident_route_rate"),
        "expected_loss_gap": weighted("expected_loss_gap"),
        "frontier_score": sequential_eval_loss - contextual_eval_loss,
    }


def residual_metadata_by_route(
    residual_routes: list[ResidualRouteCandidate],
) -> dict[str, ResidualRouteCandidate]:
    """Return residual route metadata keyed by route expression."""
    return {candidate.route.expression: candidate for candidate in residual_routes}


def annotate_residual_route_results(
    route_results: dict[str, DomainRouteResult],
    residual_routes: list[ResidualRouteCandidate],
) -> None:
    """Attach residual selected/best route metadata to domain or probe rows."""
    metadata = residual_metadata_by_route(residual_routes)
    if not metadata:
        return
    for result in route_results.values():
        selected_route = result.get("most_selected_route", "none")
        best_route = str(result.get("most_best_route", "none"))
        result["selected_route_expr"] = selected_route
        result["best_route_expr"] = best_route
        if selected_route in metadata:
            result["selected_residual"] = metadata[selected_route].residual
        if best_route in metadata:
            result["best_residual"] = metadata[best_route].residual
        result["residual_gap"] = float(result.get("selected_loss_gap", 0.0))


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
    residual_contextual_route: bool = False,
    residual_route_base_expr: str | None = None,
    residual_route_grid: list[float] | None = None,
    residual_route_mode: str | None = None,
    residual_route_phases: list[str] | None = None,
    residual_candidate_count: int = 0,
    route_top_k: int = 3,
    distilled_selector_method: str = "centroid",
    distilled_selector_margin: float = 0.0,
    distilled_knn_k: int = 3,
    micro_probe_prefix_words: int = 8,
    micro_probe_max_length: int = 32,
    micro_probe_template: str = "Route selector probe: {prefix}",
    micro_probe_margin: float = 0.0,
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
        "route_top_k": route_top_k,
        "distilled_selector_method": distilled_selector_method,
        "distilled_selector_margin": distilled_selector_margin,
        "distilled_knn_k": distilled_knn_k,
        "micro_probe_prefix_words": micro_probe_prefix_words,
        "micro_probe_max_length": micro_probe_max_length,
        "micro_probe_template": micro_probe_template,
        "micro_probe_margin": micro_probe_margin,
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
        "abstention_rate": aggregate["abstention_rate"],
        "optimal_route_rate": aggregate["optimal_route_rate"],
        "selected_loss_gap": aggregate["selected_loss_gap"],
        "selected_route_rank": aggregate["selected_route_rank"],
        "selected_top_k_rate": aggregate["selected_top_k_rate"],
        "expected_top_k_rate": aggregate["expected_top_k_rate"],
        "top_k_loss_gap": aggregate["top_k_loss_gap"],
        "top_k_boundary_margin": aggregate["top_k_boundary_margin"],
        "route_margin": aggregate["route_margin"],
        "route_entropy": aggregate["route_entropy"],
        "low_margin_rate": aggregate["low_margin_rate"],
        "ambiguity_abstention_rate": aggregate["ambiguity_abstention_rate"],
        "false_confident_route_rate": aggregate["false_confident_route_rate"],
        "expected_loss_gap": aggregate["expected_loss_gap"],
        "frontier_score": aggregate["frontier_score"],
        "per_domain": per_domain,
        "phase_eval_losses": phase_eval_losses,
    }
    if residual_contextual_route:
        result.update(
            {
                "residual_contextual_route": residual_contextual_route,
                "residual_route_base_expr": residual_route_base_expr or "",
                "residual_route_grid": residual_route_grid or [],
                "residual_route_mode": residual_route_mode or "",
                "residual_route_phases": residual_route_phases or [],
                "residual_candidate_count": residual_candidate_count,
                "residual_selected_gap": aggregate["selected_loss_gap"],
                "residual_optimal_rate": aggregate["optimal_route_rate"],
                "residual_route_margin": aggregate["route_margin"],
                "residual_route_entropy": aggregate["route_entropy"],
            }
        )
    if per_probe:
        probe_aggregate = aggregate_domain_metrics(per_probe)
        result.update(
            {
                "probe_eval_loss": probe_aggregate["contextual_eval_loss"],
                "probe_route_accuracy": probe_aggregate["route_accuracy"],
                "probe_ambiguous_rate": probe_aggregate["ambiguous_rate"],
                "probe_abstention_rate": probe_aggregate["abstention_rate"],
                "probe_optimal_route_rate": probe_aggregate["optimal_route_rate"],
                "probe_selected_loss_gap": probe_aggregate["selected_loss_gap"],
                "probe_selected_top_k_rate": probe_aggregate["selected_top_k_rate"],
                "probe_expected_top_k_rate": probe_aggregate["expected_top_k_rate"],
                "probe_top_k_loss_gap": probe_aggregate["top_k_loss_gap"],
                "probe_top_k_boundary_margin": probe_aggregate[
                    "top_k_boundary_margin"
                ],
                "probe_expected_loss_gap": probe_aggregate["expected_loss_gap"],
                "probe_low_margin_rate": probe_aggregate["low_margin_rate"],
                "probe_ambiguity_abstention_rate": probe_aggregate[
                    "ambiguity_abstention_rate"
                ],
                "probe_false_confident_route_rate": probe_aggregate[
                    "false_confident_route_rate"
                ],
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
    probe_select_texts: list[list[str]] | None = None,
    audit_route_exprs: list[RouteExpression] | None = None,
    global_route_exprs: list[RouteExpression] | None = None,
    expected_route_exprs: list[RouteExpression] | None = None,
    residual_routes: list[ResidualRouteCandidate] | None = None,
    residual_contextual_route: bool = False,
    residual_route_grid: list[float] | None = None,
    residual_route_mode: str | None = None,
    residual_route_phases: list[str] | None = None,
    ambiguity_margin: float = 0.0,
    route_top_k: int = 3,
    distilled_selector_method: DistilledSelectorMethod = "centroid",
    distilled_selector_margin: float = 0.0,
    distilled_knn_k: int = 3,
    micro_probe_prefix_words: int = 8,
    micro_probe_max_length: int = 32,
    micro_probe_template: str = "Route selector probe: {prefix}",
    micro_probe_margin: float = 0.0,
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
    has_explicit_probe_select = probe_select_texts is not None
    probe_select_texts = probe_select_texts or []
    if len(probes) != len(probe_texts):
        raise ValueError("probes and probe_texts must have the same length")
    if has_explicit_probe_select and len(probes) != len(probe_select_texts):
        raise ValueError("probes and probe_select_texts must have the same length")
    residual_routes = residual_routes or []
    audit_route_exprs = merge_route_exprs(route_exprs, audit_route_exprs)
    if global_route_exprs is None:
        global_route_exprs = audit_route_exprs
    else:
        global_route_exprs = merge_route_exprs(global_route_exprs, None)
        audit_route_exprs = merge_route_exprs(audit_route_exprs, global_route_exprs)
    expected_route_sources = expected_route_exprs if expected_route_exprs else route_exprs
    expected_route_names = [route.expression for route in expected_route_sources]
    phase_names_for_expected = [phase.name for phase in phases]
    expected_routes_by_domain = {
        domain: expected_route_for_domain(
            domain,
            phase_names_for_expected,
            expected_route_names,
        )
        for domain in phase_names_for_expected
    }
    residual_base_expr = (
        residual_routes[0].base_route_expr
        if residual_routes
        else None
    )

    resolved_device = resolve_device(device)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(settings.model_name)
    model = build_lora_model(settings, resolved_device)
    trainable, total = parameter_counts(model)

    train_splits: dict[str, dict[str, Tensor]] = {}
    eval_select: dict[str, dict[str, Tensor]] = {}
    eval_report: dict[str, dict[str, Tensor]] = {}
    eval_report_texts: dict[str, list[str]] = {}
    heldout_flags = []
    for index, (phase, texts) in enumerate(zip(phases, phase_texts, strict=True)):
        split_seed = seed + (index * 10_000)
        eval_texts = bounded_phase_eval_texts(texts, settings, split_seed)
        train_split, eval_full = prepare_encoded_splits(
            tokenizer,
            texts,
            settings,
            split_seed,
            resolved_device,
        )
        select_split, report_split, heldout = split_eval_encoded(eval_full, settings.batch_size)
        _, report_texts, text_heldout = split_eval_texts(eval_texts, settings.batch_size)
        if text_heldout != heldout:
            raise RuntimeError("encoded/text eval splits disagree")
        train_splits[phase.name] = train_split
        eval_select[phase.name] = select_split
        eval_report[phase.name] = report_split
        eval_report_texts[phase.name] = report_texts
        heldout_flags.append(heldout)

    probe_select: dict[str, dict[str, Tensor]] = {}
    probe_report: dict[str, dict[str, Tensor]] = {}
    probe_report_texts_by_name: dict[str, list[str]] = {}
    for index, (probe, texts) in enumerate(zip(probes, probe_texts, strict=True)):
        report_seed = seed + 1_000_000 + (index * 10_000)
        report_full_texts = prepare_probe_texts(texts, settings, report_seed)
        report_full = encode_texts(
            tokenizer,
            report_full_texts,
            settings.max_length,
            resolved_device,
        )
        if has_explicit_probe_select:
            select_full = prepare_probe_encoded(
                tokenizer,
                probe_select_texts[index],
                settings,
                seed + 2_000_000 + (index * 10_000),
                resolved_device,
            )
            probe_select[probe.name] = select_full
            probe_report[probe.name] = report_full
            probe_report_texts_by_name[probe.name] = report_full_texts
            heldout_flags.append(True)
        else:
            select_split, report_split, heldout = split_eval_encoded(
                report_full,
                settings.batch_size,
            )
            _, report_texts, text_heldout = split_eval_texts(
                report_full_texts,
                settings.batch_size,
            )
            if text_heldout != heldout:
                raise RuntimeError("encoded/text probe splits disagree")
            probe_select[probe.name] = select_split
            probe_report[probe.name] = report_split
            probe_report_texts_by_name[probe.name] = report_texts
            heldout_flags.append(heldout)

    eval_micro_probe = (
        {
            domain: prepare_micro_probe_encoded(
                tokenizer,
                report_texts,
                settings,
                resolved_device,
                prefix_words=micro_probe_prefix_words,
                max_length=micro_probe_max_length,
                template=micro_probe_template,
            )
            for domain, report_texts in eval_report_texts.items()
        }
        if route_selection == "micro_probe"
        else None
    )
    probe_micro = (
        {
            probe.name: prepare_micro_probe_encoded(
                tokenizer,
                probe_report_texts_by_name[probe.name],
                settings,
                resolved_device,
                prefix_words=micro_probe_prefix_words,
                max_length=micro_probe_max_length,
                template=micro_probe_template,
            )
            for probe in probes
        }
        if route_selection == "micro_probe"
        else None
    )

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
            probe.expected_route
            if probe.expected_route in audit_route_states
            else next(iter(audit_route_states)),
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
        top_k=route_top_k,
        distilled_selector_margin=distilled_selector_margin,
        distilled_selector_method=distilled_selector_method,
        distilled_knn_k=distilled_knn_k,
        eval_micro_probe=eval_micro_probe,
        micro_probe_margin=micro_probe_margin,
        expected_routes_by_domain=expected_routes_by_domain,
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
        top_k=route_top_k,
        distilled_selector_margin=distilled_selector_margin,
        distilled_selector_method=distilled_selector_method,
        distilled_knn_k=distilled_knn_k,
        probe_micro=probe_micro,
        micro_probe_margin=micro_probe_margin,
        audit_scores_by_probe=audit_scores_by_probe,
    )
    annotate_residual_route_results(per_domain, residual_routes)
    annotate_residual_route_results(per_probe, residual_routes)
    results = [
        memory_bank_result_record(
            variant=variant,
            variant_name=(
                f"{variant.name}_residual_memory_bank_{route_selection}"
                if residual_contextual_route
                else f"{variant.name}_contextual_memory_bank_{route_selection}"
            ),
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
            residual_contextual_route=residual_contextual_route,
            residual_route_base_expr=residual_base_expr,
            residual_route_grid=residual_route_grid,
            residual_route_mode=residual_route_mode,
            residual_route_phases=residual_route_phases,
            residual_candidate_count=len(residual_routes),
            route_top_k=route_top_k,
            distilled_selector_method=distilled_selector_method,
            distilled_selector_margin=distilled_selector_margin,
            distilled_knn_k=distilled_knn_k,
            micro_probe_prefix_words=micro_probe_prefix_words,
            micro_probe_max_length=micro_probe_max_length,
            micro_probe_template=micro_probe_template,
            micro_probe_margin=micro_probe_margin,
        )
    ]
    if include_global_routes:
        for route in global_route_exprs:
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
                top_k=route_top_k,
                ambiguity_margin=ambiguity_margin,
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
                top_k=route_top_k,
                ambiguity_margin=ambiguity_margin,
                audit_scores_by_probe=audit_scores_by_probe,
            )
            annotate_residual_route_results(global_per_domain, residual_routes)
            annotate_residual_route_results(global_per_probe, residual_routes)
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
                    route_top_k=route_top_k,
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
        "abstention_rate",
        "optimal_route_rate",
        "selected_loss_gap",
        "selected_route_rank",
        "selected_top_k_rate",
        "expected_top_k_rate",
        "top_k_loss_gap",
        "top_k_boundary_margin",
        "route_margin",
        "route_entropy",
        "low_margin_rate",
        "ambiguity_abstention_rate",
        "false_confident_route_rate",
        "residual_selected_gap",
        "residual_optimal_rate",
        "residual_route_margin",
        "residual_route_entropy",
        "residual_candidate_count",
        "expected_loss_gap",
        "probe_eval_loss",
        "probe_route_accuracy",
        "probe_ambiguous_rate",
        "probe_abstention_rate",
        "probe_optimal_route_rate",
        "probe_selected_loss_gap",
        "probe_selected_top_k_rate",
        "probe_expected_top_k_rate",
        "probe_top_k_loss_gap",
        "probe_top_k_boundary_margin",
        "probe_expected_loss_gap",
        "probe_low_margin_rate",
        "probe_ambiguity_abstention_rate",
        "probe_false_confident_route_rate",
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
                "abstention_rate",
                "optimal_route_rate",
                "selected_loss_gap",
                "selected_route_rank",
                "selected_top_k_rate",
                "expected_top_k_rate",
                "top_k_loss_gap",
                "top_k_boundary_margin",
                "route_margin",
                "route_entropy",
                "low_margin_rate",
                "ambiguity_abstention_rate",
                "false_confident_route_rate",
                "residual_gap",
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
                "abstention_rate",
                "optimal_route_rate",
                "selected_loss_gap",
                "selected_route_rank",
                "selected_top_k_rate",
                "expected_top_k_rate",
                "top_k_loss_gap",
                "top_k_boundary_margin",
                "route_margin",
                "route_entropy",
                "low_margin_rate",
                "ambiguity_abstention_rate",
                "false_confident_route_rate",
                "residual_gap",
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
    if getattr(args, "probe_select_files", None) is not None and args.probe_files is None:
        raise ValueError("--probe-select-files requires --probe-files")
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


def probe_select_paths_from_args(
    args: argparse.Namespace,
    probes: list[Probe],
) -> list[str] | None:
    """Return optional route-selection probe files aligned to report probes."""
    probe_select_files = getattr(args, "probe_select_files", None)
    if probe_select_files is None:
        return None
    if not probes:
        raise ValueError("--probe-select-files requires --probe-files")
    if len(probe_select_files) != len(probes):
        raise ValueError("--probe-select-files must match --probe-files length")
    return list(probe_select_files)


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
    parser.add_argument("--probe-select-files", nargs="*", default=None)
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
    parser.add_argument("--residual-contextual-route", action="store_true")
    parser.add_argument("--residual-route-base-expr", default=None)
    parser.add_argument("--residual-route-phases", nargs="*", default=None)
    parser.add_argument("--residual-route-grid", nargs="*", type=float, default=None)
    parser.add_argument(
        "--residual-route-mode",
        choices=["full", "axis", "pairs"],
        default="full",
    )
    parser.add_argument("--residual-route-min-scale", type=float, default=0.0)
    parser.add_argument("--residual-route-max-scale", type=float, default=1.5)
    parser.add_argument("--residual-ambiguity-margin", type=float, default=None)
    parser.add_argument("--contextual-route", action="store_true")
    parser.add_argument(
        "--route-selection",
        choices=["oracle", "loss_probe", "calibration", "distilled", "micro_probe"],
        default="oracle",
    )
    parser.add_argument("--ambiguity-margin", type=float, default=0.0)
    parser.add_argument("--route-top-k", type=int, default=3)
    parser.add_argument(
        "--distilled-selector-method",
        choices=["centroid", "knn"],
        default="centroid",
    )
    parser.add_argument(
        "--distilled-selector-margin",
        type=float,
        default=0.0,
        help=(
            "Cosine margin for abstaining in distilled selector mode. Default 0 keeps "
            "the selector decisive rather than abstaining on every hard prompt."
        ),
    )
    parser.add_argument("--distilled-knn-k", type=int, default=3)
    parser.add_argument("--micro-probe-prefix-words", type=int, default=8)
    parser.add_argument("--micro-probe-max-length", type=int, default=32)
    parser.add_argument(
        "--micro-probe-template",
        default="Route selector probe: {prefix}",
        help="Template for micro-probe selector texts. May contain {prefix} and/or {scope}.",
    )
    parser.add_argument(
        "--micro-probe-margin",
        type=float,
        default=0.0,
        help="Loss margin for abstaining in micro-probe selector mode.",
    )
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
    if args.route_top_k < 1:
        raise ValueError("--route-top-k must be at least 1")
    if args.distilled_selector_margin < 0.0:
        raise ValueError("--distilled-selector-margin cannot be negative")
    if args.distilled_knn_k < 1:
        raise ValueError("--distilled-knn-k must be at least 1")
    if args.micro_probe_prefix_words < 1:
        raise ValueError("--micro-probe-prefix-words must be at least 1")
    if args.micro_probe_max_length < 1:
        raise ValueError("--micro-probe-max-length must be at least 1")
    if args.micro_probe_margin < 0.0:
        raise ValueError("--micro-probe-margin cannot be negative")
    if "{prefix}" not in args.micro_probe_template and "{scope}" not in args.micro_probe_template:
        raise ValueError("--micro-probe-template must contain {prefix} or {scope}")
    task_files = phase_paths_from_args(args)
    phase_names = args.phase_names or default_phase_names(len(task_files))
    if len(phase_names) != len(task_files):
        raise ValueError("--phase-names must match --task-files length")
    phases = [
        Phase(name=name, path=path)
        for name, path in zip(phase_names, task_files, strict=True)
    ]
    stable_phase = phase_names[0]
    clean_route_exprs = (
        parse_route_exprs(args.route_expr, stable_phase=stable_phase)
        if args.route_expr is not None
        else default_route_exprs(phase_names)
    )
    explicit_audit_route_exprs = (
        parse_route_exprs(args.audit_route_expr, stable_phase=stable_phase)
        if args.audit_route_expr is not None
        else None
    )
    residual_routes: list[ResidualRouteCandidate] = []
    residual_route_phases = args.residual_route_phases or phase_names[1:]
    residual_route_grid = args.residual_route_grid or [0.0]
    route_exprs = clean_route_exprs
    if args.residual_contextual_route:
        if args.residual_route_base_expr is None:
            raise ValueError(
                "--residual-route-base-expr is required with --residual-contextual-route"
            )
        residual_base_route = parse_route_expr(
            args.residual_route_base_expr,
            stable_phase=stable_phase,
        )
        residual_routes = generate_residual_routes(
            residual_base_route,
            phases=residual_route_phases,
            grid=residual_route_grid,
            mode=args.residual_route_mode,
            min_scale=args.residual_route_min_scale,
            max_scale=args.residual_route_max_scale,
        )
        route_exprs = [candidate.route for candidate in residual_routes]
    global_route_exprs = merge_route_exprs(clean_route_exprs, explicit_audit_route_exprs)
    audit_route_exprs = merge_route_exprs(
        route_exprs,
        merge_route_exprs(global_route_exprs, clean_route_exprs),
    )
    merged_audit_route_exprs = audit_route_exprs
    ambiguity_margin = (
        args.residual_ambiguity_margin
        if args.residual_contextual_route and args.residual_ambiguity_margin is not None
        else args.ambiguity_margin
    )
    probes = probes_from_args(args, stable_phase=stable_phase)
    probe_select_files = probe_select_paths_from_args(args, probes)
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
    probe_select_texts = (
        [load_texts(path) for path in probe_select_files]
        if probe_select_files is not None
        else None
    )
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
            probe_select_texts=probe_select_texts,
            audit_route_exprs=audit_route_exprs,
            global_route_exprs=global_route_exprs,
            expected_route_exprs=clean_route_exprs,
            residual_routes=residual_routes,
            residual_contextual_route=args.residual_contextual_route,
            residual_route_grid=residual_route_grid,
            residual_route_mode=args.residual_route_mode,
            residual_route_phases=residual_route_phases,
            ambiguity_margin=ambiguity_margin,
            route_top_k=args.route_top_k,
            distilled_selector_method=args.distilled_selector_method,
            distilled_selector_margin=args.distilled_selector_margin,
            distilled_knn_k=args.distilled_knn_k,
            micro_probe_prefix_words=args.micro_probe_prefix_words,
            micro_probe_max_length=args.micro_probe_max_length,
            micro_probe_template=args.micro_probe_template,
            micro_probe_margin=args.micro_probe_margin,
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
            "probe_select_files": probe_select_files or [],
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
            "global_route_exprs": [route.expression for route in global_route_exprs],
            "contextual_route": args.contextual_route,
            "route_selection": args.route_selection,
            "ambiguity_margin": args.ambiguity_margin,
            "route_top_k": args.route_top_k,
            "distilled_selector_method": args.distilled_selector_method,
            "distilled_selector_margin": args.distilled_selector_margin,
            "distilled_knn_k": args.distilled_knn_k,
            "micro_probe_prefix_words": args.micro_probe_prefix_words,
            "micro_probe_max_length": args.micro_probe_max_length,
            "micro_probe_template": args.micro_probe_template,
            "micro_probe_margin": args.micro_probe_margin,
            "residual_contextual_route": args.residual_contextual_route,
            "residual_route_base_expr": args.residual_route_base_expr,
            "residual_route_phases": residual_route_phases,
            "residual_route_grid": residual_route_grid,
            "residual_route_mode": args.residual_route_mode,
            "residual_route_min_scale": args.residual_route_min_scale,
            "residual_route_max_scale": args.residual_route_max_scale,
            "residual_ambiguity_margin": args.residual_ambiguity_margin,
            "residual_candidate_count": len(residual_routes),
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
