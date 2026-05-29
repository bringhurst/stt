"""Oracle group-routed LoRA accretion diagnostic.

This is an intentionally unfair upper-bound test. It trains one A-to-B-to-C
sequence, then greedily chooses C-update scales per parameter group using task
losses on a selection split. The final selected route is reported on a held-out
split when enough evaluation batches are available.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

import torch
from torch import Tensor

from stt.accretion import ratio
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
    snapshot_trainable_state,
    split_eval_encoded,
    subtract_state,
)
from stt.routed_accretion import extract_layer_index, frontier_score

GroupBy = Literal["layer", "module", "tensor"]


class OracleRouteResult(TypedDict):
    """Serializable result for one oracle group route."""

    variant: str
    model: str
    device: str
    seed: int
    group_by: str
    b_scale: float
    c_scales: list[float]
    selected_groups: int
    nonzero_groups: int
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
    sequential_eval_a: float
    sequential_eval_b: float
    sequential_eval_c: float
    oracle_eval_a: float
    oracle_eval_b: float
    oracle_eval_c: float
    sequential_accretion_a: float
    sequential_interference_a: float
    sequential_interference_b: float
    sequential_learning_b: float
    sequential_learning_c: float
    sequential_retention_a: float
    sequential_retention_b: float
    oracle_accretion_a: float
    oracle_interference_a: float
    oracle_interference_b: float
    oracle_learning_b: float
    oracle_learning_c: float
    oracle_retention_a: float
    oracle_retention_b: float
    frontier_score: float
    selected_route: dict[str, float]


class OracleRouteRunRecord(TypedDict):
    """Persisted oracle group-route run."""

    created_at: str
    config: dict[str, Any]
    git_status: str
    results: list[OracleRouteResult]
    summary: dict[str, dict[str, float]]


def group_key(name: str, group_by: GroupBy) -> str:
    """Return the oracle-routing group key for a trainable parameter name."""
    if group_by == "tensor":
        return name
    layer_index = extract_layer_index(name)
    if group_by == "layer":
        return "unlayered" if layer_index is None else f"layer_{layer_index}"
    for marker in (".lora_A.", ".lora_B."):
        if marker in name:
            return name.split(marker, maxsplit=1)[0]
    return name


def group_keys(state: dict[str, Tensor], group_by: GroupBy) -> list[str]:
    """Return stable group keys present in a trainable state."""
    return sorted({group_key(name, group_by) for name in state})


def compose_group_route_state(
    base: dict[str, Tensor],
    *,
    delta_b_state: dict[str, Tensor],
    delta_c_state: dict[str, Tensor],
    b_scale: float,
    group_scales: dict[str, float],
    group_by: GroupBy,
) -> dict[str, Tensor]:
    """Compose `base + b_scale * B + group_scale(name) * C`."""
    composed = {name: value.clone() for name, value in base.items()}
    for name in composed.keys() & delta_b_state.keys():
        composed[name] = composed[name] + (delta_b_state[name] * b_scale)
    for name in composed.keys() & delta_c_state.keys():
        scale = group_scales.get(group_key(name, group_by), 0.0)
        composed[name] = composed[name] + (delta_c_state[name] * scale)
    return composed


def evaluate_state(
    model: torch.nn.Module,
    state: dict[str, Tensor],
    eval_a: dict[str, Tensor],
    eval_b: dict[str, Tensor],
    eval_c: dict[str, Tensor],
    settings: LoraSettings,
) -> tuple[float, float, float]:
    """Apply and evaluate a trainable adapter state."""
    apply_trainable_state(model, state)
    return (
        eval_loss(model, eval_a, settings),
        eval_loss(model, eval_b, settings),
        eval_loss(model, eval_c, settings),
    )


def route_metrics(
    evals: tuple[float, float, float],
    *,
    eval_a_after_a: float,
    eval_b_after_a: float,
    eval_a_after_b: float,
    eval_b_after_b: float,
    eval_c_after_b: float,
    sequential_accretion_a: float,
    sequential_interference_a: float,
    sequential_interference_b: float,
    sequential_learning_b: float,
    sequential_learning_c: float,
) -> dict[str, float]:
    """Return routed metrics and frontier score for one candidate state."""
    eval_a, eval_b, eval_c = evals
    oracle_accretion_a = eval_a_after_a - eval_a
    oracle_interference_a = eval_a - eval_a_after_b
    oracle_interference_b = eval_b - eval_b_after_b
    oracle_learning_b = eval_b_after_a - eval_b
    oracle_learning_c = eval_c_after_b - eval_c
    return {
        "oracle_accretion_a": oracle_accretion_a,
        "oracle_interference_a": oracle_interference_a,
        "oracle_interference_b": oracle_interference_b,
        "oracle_learning_b": oracle_learning_b,
        "oracle_learning_c": oracle_learning_c,
        "frontier_score": frontier_score(
            sequential_accretion_a=sequential_accretion_a,
            sequential_interference_a=sequential_interference_a,
            sequential_interference_b=sequential_interference_b,
            sequential_learning_b=sequential_learning_b,
            sequential_learning_c=sequential_learning_c,
            routed_accretion_a=oracle_accretion_a,
            routed_interference_a=oracle_interference_a,
            routed_interference_b=oracle_interference_b,
            routed_learning_b=oracle_learning_b,
            routed_learning_c=oracle_learning_c,
        ),
    }


def select_group_scales_greedy(
    model: torch.nn.Module,
    *,
    state_a: dict[str, Tensor],
    delta_b_state: dict[str, Tensor],
    delta_c_state: dict[str, Tensor],
    b_scale: float,
    candidate_c_scales: list[float],
    group_by: GroupBy,
    eval_a: dict[str, Tensor],
    eval_b: dict[str, Tensor],
    eval_c: dict[str, Tensor],
    settings: LoraSettings,
    metric_context: dict[str, float],
) -> dict[str, float]:
    """Greedily choose the C scale for each group using frontier score."""
    selected = dict.fromkeys(group_keys(delta_c_state, group_by), 0.0)
    for key in selected:
        best_scale = 0.0
        best_score = float("-inf")
        for scale in candidate_c_scales:
            candidate = dict(selected)
            candidate[key] = scale
            state = compose_group_route_state(
                state_a,
                delta_b_state=delta_b_state,
                delta_c_state=delta_c_state,
                b_scale=b_scale,
                group_scales=candidate,
                group_by=group_by,
            )
            metrics = route_metrics(
                evaluate_state(model, state, eval_a, eval_b, eval_c, settings),
                eval_a_after_a=metric_context["eval_a_after_a"],
                eval_b_after_a=metric_context["eval_b_after_a"],
                eval_a_after_b=metric_context["eval_a_after_b"],
                eval_b_after_b=metric_context["eval_b_after_b"],
                eval_c_after_b=metric_context["eval_c_after_b"],
                sequential_accretion_a=metric_context["sequential_accretion_a"],
                sequential_interference_a=metric_context["sequential_interference_a"],
                sequential_interference_b=metric_context["sequential_interference_b"],
                sequential_learning_b=metric_context["sequential_learning_b"],
                sequential_learning_c=metric_context["sequential_learning_c"],
            )
            if metrics["frontier_score"] > best_score:
                best_scale = scale
                best_score = metrics["frontier_score"]
        selected[key] = best_scale
    return selected


def run_oracle_route_seed(
    variant: Variant,
    settings: LoraSettings,
    task_a_texts: list[str],
    task_b_texts: list[str],
    task_c_texts: list[str],
    phase_steps: int,
    seed: int,
    device: str,
    b_scale: float,
    c_scales: list[float],
    group_by: GroupBy,
) -> OracleRouteResult:
    """Train A/B/C once and run greedy oracle group routing."""
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

    eval_a_before = eval_loss(model, eval_a, settings)
    eval_b_before = eval_loss(model, eval_b, settings)
    eval_c_before = eval_loss(model, eval_c, settings)

    train_steps(model, train_a, variant, settings, phase_steps)
    eval_a_after_a = eval_loss(model, eval_a, settings)
    eval_b_after_a = eval_loss(model, eval_b, settings)
    eval_c_after_a = eval_loss(model, eval_c, settings)
    eval_a_after_a_select = eval_loss(model, eval_a_select, settings)
    eval_b_after_a_select = eval_loss(model, eval_b_select, settings)
    state_a = snapshot_trainable_state(model)

    train_steps(model, train_b, variant, settings, phase_steps)
    eval_a_after_b = eval_loss(model, eval_a, settings)
    eval_b_after_b = eval_loss(model, eval_b, settings)
    eval_c_after_b = eval_loss(model, eval_c, settings)
    eval_a_after_b_select = eval_loss(model, eval_a_select, settings)
    eval_b_after_b_select = eval_loss(model, eval_b_select, settings)
    eval_c_after_b_select = eval_loss(model, eval_c_select, settings)
    state_b = snapshot_trainable_state(model)

    train_steps(model, train_c, variant, settings, phase_steps)
    sequential_eval_a = eval_loss(model, eval_a, settings)
    sequential_eval_b = eval_loss(model, eval_b, settings)
    sequential_eval_c = eval_loss(model, eval_c, settings)
    sequential_eval_a_select = eval_loss(model, eval_a_select, settings)
    sequential_eval_b_select = eval_loss(model, eval_b_select, settings)
    sequential_eval_c_select = eval_loss(model, eval_c_select, settings)
    state_c = snapshot_trainable_state(model)

    delta_b_state = subtract_state(state_b, state_a)
    delta_c_state = subtract_state(state_c, state_b)
    sequential_accretion_a = eval_a_after_a - eval_a_after_b
    sequential_interference_a = sequential_eval_a - eval_a_after_b
    sequential_interference_b = sequential_eval_b - eval_b_after_b
    sequential_learning_b = eval_b_after_a - eval_b_after_b
    sequential_learning_c = eval_c_after_b - sequential_eval_c
    metric_context = {
        "eval_a_after_a": eval_a_after_a,
        "eval_b_after_a": eval_b_after_a,
        "eval_a_after_b": eval_a_after_b,
        "eval_b_after_b": eval_b_after_b,
        "eval_c_after_b": eval_c_after_b,
        "sequential_accretion_a": sequential_accretion_a,
        "sequential_interference_a": sequential_interference_a,
        "sequential_interference_b": sequential_interference_b,
        "sequential_learning_b": sequential_learning_b,
        "sequential_learning_c": sequential_learning_c,
    }
    select_context = dict(metric_context)
    select_context.update(
        {
            "eval_a_after_a": eval_a_after_a_select,
            "eval_b_after_a": eval_b_after_a_select,
            "eval_a_after_b": eval_a_after_b_select,
            "eval_b_after_b": eval_b_after_b_select,
            "eval_c_after_b": eval_c_after_b_select,
            "sequential_accretion_a": eval_a_after_a_select - eval_a_after_b_select,
            "sequential_interference_a": sequential_eval_a_select - eval_a_after_b_select,
            "sequential_interference_b": sequential_eval_b_select - eval_b_after_b_select,
            "sequential_learning_b": eval_b_after_a_select - eval_b_after_b_select,
            "sequential_learning_c": eval_c_after_b_select - sequential_eval_c_select,
        }
    )
    selected_route = select_group_scales_greedy(
        model,
        state_a=state_a,
        delta_b_state=delta_b_state,
        delta_c_state=delta_c_state,
        b_scale=b_scale,
        candidate_c_scales=c_scales,
        group_by=group_by,
        eval_a=eval_a_select,
        eval_b=eval_b_select,
        eval_c=eval_c_select,
        settings=settings,
        metric_context=select_context,
    )
    oracle_state = compose_group_route_state(
        state_a,
        delta_b_state=delta_b_state,
        delta_c_state=delta_c_state,
        b_scale=b_scale,
        group_scales=selected_route,
        group_by=group_by,
    )
    oracle_eval_a, oracle_eval_b, oracle_eval_c = evaluate_state(
        model, oracle_state, eval_a, eval_b, eval_c, settings
    )
    metrics = route_metrics(
        (oracle_eval_a, oracle_eval_b, oracle_eval_c),
        eval_a_after_a=eval_a_after_a,
        eval_b_after_a=eval_b_after_a,
        eval_a_after_b=eval_a_after_b,
        eval_b_after_b=eval_b_after_b,
        eval_c_after_b=eval_c_after_b,
        sequential_accretion_a=sequential_accretion_a,
        sequential_interference_a=sequential_interference_a,
        sequential_interference_b=sequential_interference_b,
        sequential_learning_b=sequential_learning_b,
        sequential_learning_c=sequential_learning_c,
    )
    return {
        "variant": variant.name,
        "model": settings.model_name,
        "device": resolved_device,
        "seed": seed,
        "group_by": group_by,
        "b_scale": b_scale,
        "c_scales": c_scales,
        "selected_groups": len(selected_route),
        "nonzero_groups": sum(scale != 0.0 for scale in selected_route.values()),
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
        "sequential_eval_a": sequential_eval_a,
        "sequential_eval_b": sequential_eval_b,
        "sequential_eval_c": sequential_eval_c,
        "oracle_eval_a": oracle_eval_a,
        "oracle_eval_b": oracle_eval_b,
        "oracle_eval_c": oracle_eval_c,
        "sequential_accretion_a": sequential_accretion_a,
        "sequential_interference_a": sequential_interference_a,
        "sequential_interference_b": sequential_interference_b,
        "sequential_learning_b": sequential_learning_b,
        "sequential_learning_c": sequential_learning_c,
        "sequential_retention_a": ratio(eval_a_after_a, sequential_eval_a),
        "sequential_retention_b": ratio(eval_b_after_b, sequential_eval_b),
        "oracle_accretion_a": metrics["oracle_accretion_a"],
        "oracle_interference_a": metrics["oracle_interference_a"],
        "oracle_interference_b": metrics["oracle_interference_b"],
        "oracle_learning_b": metrics["oracle_learning_b"],
        "oracle_learning_c": metrics["oracle_learning_c"],
        "oracle_retention_a": ratio(eval_a_after_a, oracle_eval_a),
        "oracle_retention_b": ratio(eval_b_after_b, oracle_eval_b),
        "frontier_score": metrics["frontier_score"],
        "selected_route": selected_route,
    }


def summarize_oracle_routes(results: list[OracleRouteResult]) -> dict[str, dict[str, float]]:
    """Aggregate oracle group-route results by variant and grouping mode."""
    metrics = [
        "selected_groups",
        "nonzero_groups",
        "sequential_accretion_a",
        "sequential_interference_a",
        "sequential_interference_b",
        "sequential_learning_b",
        "sequential_learning_c",
        "oracle_accretion_a",
        "oracle_interference_a",
        "oracle_interference_b",
        "oracle_learning_b",
        "oracle_learning_c",
        "frontier_score",
    ]
    summary: dict[str, dict[str, float]] = {}
    keys = sorted({f"{result['variant']}:{result['group_by']}" for result in results})
    for key in keys:
        variant, group_by = key.split(":", maxsplit=1)
        group = [
            result
            for result in results
            if result["variant"] == variant and result["group_by"] == group_by
        ]
        values: dict[str, float] = {"count": float(len(group))}
        for metric in metrics:
            metric_values = [float(dict(result)[metric]) for result in group]
            values[f"{metric}_mean"] = statistics.fmean(metric_values)
            values[f"{metric}_std"] = (
                statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            )
        values["oracle_accretion_win_count"] = float(
            sum(result["oracle_accretion_a"] > result["sequential_accretion_a"] for result in group)
        )
        values["oracle_interference_a_win_count"] = float(
            sum(
                result["oracle_interference_a"] < result["sequential_interference_a"]
                for result in group
            )
        )
        values["oracle_interference_b_win_count"] = float(
            sum(
                result["oracle_interference_b"] < result["sequential_interference_b"]
                for result in group
            )
        )
        values["oracle_learning_c_preserved_count"] = float(
            sum(result["oracle_learning_c"] >= result["sequential_learning_c"] for result in group)
        )
        summary[key] = values
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-oracle-route`."""
    parser = argparse.ArgumentParser(description="Run oracle group-routed LoRA accretion.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task-a-file", required=True)
    parser.add_argument("--task-b-file", required=True)
    parser.add_argument("--task-c-file", required=True)
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
    parser.add_argument("--b-scale", type=float, default=0.9)
    parser.add_argument("--c-scales", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--group-by", choices=["layer", "module", "tensor"], default="layer")
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for oracle group-route diagnostics."""
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
    task_a_texts = load_texts(args.task_a_file)
    task_b_texts = load_texts(args.task_b_file)
    task_c_texts = load_texts(args.task_c_file)
    results = [
        run_oracle_route_seed(
            variant,
            settings=settings,
            task_a_texts=task_a_texts,
            task_b_texts=task_b_texts,
            task_c_texts=task_c_texts,
            phase_steps=args.phase_steps,
            seed=seed,
            device=args.device,
            b_scale=args.b_scale,
            c_scales=args.c_scales,
            group_by=args.group_by,
        )
        for seed in seeds
    ]
    record: OracleRouteRunRecord = {
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
            "b_scale": args.b_scale,
            "c_scales": args.c_scales,
            "group_by": args.group_by,
        },
        "git_status": git_status(),
        "results": results,
        "summary": summarize_oracle_routes(results),
    }
    if args.output_dir is not None:
        write_run_record(record, args.output_dir)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
