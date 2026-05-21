"""Fixed routed-update LoRA accretion experiments.

This module turns the oracle-composition finding into a deployed baseline: train
A then B then C once, but publish a final adapter formed by a predeclared route
through the observed update deltas instead of blindly keeping the sequential C
state.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from typing import Any, TypedDict

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
    subtract_state,
)


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
    sequential_interference_b: float
    sequential_learning_b: float
    sequential_learning_c: float
    sequential_retention_a: float
    sequential_retention_b: float
    routed_accretion_a: float
    routed_interference_a: float
    routed_interference_b: float
    routed_learning_b: float
    routed_learning_c: float
    routed_retention_a: float
    routed_retention_b: float
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
    routed_state = compose_state(
        state_a,
        [(route_b_scale, delta_b_state), (route_c_scale, delta_c_state)],
    )
    apply_trainable_state(model, routed_state)
    routed_eval_a = eval_loss(model, eval_a, settings)
    routed_eval_b = eval_loss(model, eval_b, settings)
    routed_eval_c = eval_loss(model, eval_c, settings)

    lora_delta_a = subtract_lora_deltas(lora_after_a, lora_initial)
    lora_delta_b = subtract_lora_deltas(lora_after_b, lora_after_a)
    lora_delta_c = subtract_lora_deltas(lora_after_c, lora_after_b)

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
        "route_b_scale": route_b_scale,
        "route_c_scale": route_c_scale,
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
        "sequential_accretion_a": eval_a_after_a - eval_a_after_b,
        "sequential_interference_a": sequential_eval_a - eval_a_after_b,
        "sequential_interference_b": sequential_eval_b - eval_b_after_b,
        "sequential_learning_b": eval_b_after_a - eval_b_after_b,
        "sequential_learning_c": eval_c_after_b - sequential_eval_c,
        "sequential_retention_a": ratio(eval_a_after_a, sequential_eval_a),
        "sequential_retention_b": ratio(eval_b_after_b, sequential_eval_b),
        "routed_accretion_a": eval_a_after_a - routed_eval_a,
        "routed_interference_a": routed_eval_a - eval_a_after_b,
        "routed_interference_b": routed_eval_b - eval_b_after_b,
        "routed_learning_b": eval_b_after_a - routed_eval_b,
        "routed_learning_c": eval_c_before - routed_eval_c,
        "routed_retention_a": ratio(eval_a_after_a, routed_eval_a),
        "routed_retention_b": ratio(eval_b_after_b, routed_eval_b),
        "lora_cosine_a_b_mean": mean_lora_cosine(lora_delta_a, lora_delta_b),
        "lora_cosine_a_c_mean": mean_lora_cosine(lora_delta_a, lora_delta_c),
        "lora_cosine_b_c_mean": mean_lora_cosine(lora_delta_b, lora_delta_c),
        "grad_cosine_a_b_after_a": grad_cosine_a_b_after_a,
        "grad_cosine_a_c_after_b": grad_cosine_a_c_after_b,
    }


def summarize_routed_accretion(
    results: list[RoutedAccretionResult],
) -> dict[str, dict[str, float]]:
    """Aggregate fixed routed-update metrics by variant."""
    metric_names = [
        "route_b_scale",
        "route_c_scale",
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
        summary[variant] = values
    return summary


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
    task_a_texts = load_texts(args.task_a_file)
    task_b_texts = load_texts(args.task_b_file)
    task_c_texts = load_texts(args.task_c_file)
    results = [
        run_routed_accretion_variant(
            variant,
            settings=settings,
            task_a_texts=task_a_texts,
            task_b_texts=task_b_texts,
            task_c_texts=task_c_texts,
            phase_steps=args.phase_steps,
            seed=seed,
            device=args.device,
            route_b_scale=args.route_b_scale,
            route_c_scale=args.route_c_scale,
            compat_batches=args.compat_batches,
        )
        for seed in seeds
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
