"""A-to-B-to-C LoRA experiments for compatibility and accretion checks."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from typing import Any, TypedDict

import torch
from torch import Tensor

from stt.continual import batch_slice_for_continual, eval_loss, prepare_encoded_splits, train_steps
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


class AccretionResult(TypedDict):
    """Serializable result for one A-then-B-then-C training run."""

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
    learning_a: float
    learning_b: float
    learning_c: float
    accretion_a_after_b: float
    backward_transfer_a_after_b: float
    interference_a_after_c: float
    interference_b_after_c: float
    retention_a_after_b: float
    retention_a_after_c: float
    retention_b_after_c: float
    lora_cosine_a_b_mean: float | None
    lora_cosine_a_c_mean: float | None
    lora_cosine_b_c_mean: float | None
    grad_cosine_a_b_after_a: float | None
    grad_cosine_a_c_after_b: float | None


class AccretionRunRecord(TypedDict):
    """Persisted A-to-B-to-C experiment record."""

    created_at: str
    config: dict[str, Any]
    git_status: str
    results: list[AccretionResult]
    summary: dict[str, dict[str, float]]


def ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio for positive losses."""
    return numerator / denominator if denominator != 0.0 else 0.0


def tensor_cosine(left: Tensor, right: Tensor) -> float | None:
    """Return cosine similarity for two tensors, or None for zero vectors."""
    left_flat = left.detach().float().cpu().flatten()
    right_flat = right.detach().float().cpu().flatten()
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left_flat, right_flat) / denominator)


def mean_optional(values: list[float | None]) -> float | None:
    """Return the mean of present values, or None when all values are absent."""
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def lora_effective_deltas(model: torch.nn.Module) -> dict[str, Tensor]:
    """Return effective adapter delta matrices keyed by LoRA module name."""
    deltas: dict[str, Tensor] = {}
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        if "default" not in lora_a or "default" not in lora_b:
            continue
        scaling = getattr(module, "scaling", {}).get("default", 1.0)
        effective_delta = lora_b["default"].weight @ lora_a["default"].weight
        deltas[name] = effective_delta.detach().cpu() * scaling
    return deltas


def subtract_lora_deltas(
    later: dict[str, Tensor],
    earlier: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Return per-module adapter delta increments from two snapshots."""
    return {name: later[name] - earlier[name] for name in later.keys() & earlier.keys()}


def mean_lora_cosine(left: dict[str, Tensor], right: dict[str, Tensor]) -> float | None:
    """Return the mean per-module cosine between two adapter delta dictionaries."""
    return mean_optional(
        [tensor_cosine(left[name], right[name]) for name in sorted(left.keys() & right.keys())]
    )


def gradient_vector(
    model: torch.nn.Module,
    encoded: dict[str, Tensor],
    settings: LoraSettings,
    batches: int,
) -> Tensor | None:
    """Return the averaged trainable-parameter LM gradient vector for encoded data."""
    if batches <= 0:
        return None
    model.eval()
    model.zero_grad(set_to_none=True)
    for step in range(batches):
        batch = batch_slice_for_continual(encoded, step * settings.batch_size, settings.batch_size)
        output = model(**batch)
        loss = output.loss / batches
        loss.backward()
    pieces = [
        parameter.grad.detach().float().cpu().flatten()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    model.zero_grad(set_to_none=True)
    if not pieces:
        return None
    return torch.cat(pieces)


def gradient_cosine(
    model: torch.nn.Module,
    left_encoded: dict[str, Tensor],
    right_encoded: dict[str, Tensor],
    settings: LoraSettings,
    batches: int,
) -> float | None:
    """Return cosine similarity between two task gradient vectors."""
    left = gradient_vector(model, left_encoded, settings, batches)
    right = gradient_vector(model, right_encoded, settings, batches)
    if left is None or right is None:
        return None
    return tensor_cosine(left, right)


def run_accretion_variant(
    variant: Variant,
    settings: LoraSettings,
    task_a_texts: list[str],
    task_b_texts: list[str],
    task_c_texts: list[str],
    phase_steps: int,
    seed: int,
    device: str,
    compat_batches: int = 0,
) -> AccretionResult:
    """Train A then B then C and log compatibility across tasks."""
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
    lora_after_a = lora_effective_deltas(model)
    grad_cosine_a_b_after_a = gradient_cosine(
        model, train_a, train_b, settings, compat_batches
    )

    train_steps(model, train_b, variant, settings, phase_steps)
    eval_a_after_b = eval_loss(model, eval_a, settings)
    eval_b_after_b = eval_loss(model, eval_b, settings)
    eval_c_after_b = eval_loss(model, eval_c, settings)
    lora_after_b = lora_effective_deltas(model)
    grad_cosine_a_c_after_b = gradient_cosine(
        model, train_a, train_c, settings, compat_batches
    )

    train_steps(model, train_c, variant, settings, phase_steps)
    eval_a_after_c = eval_loss(model, eval_a, settings)
    eval_b_after_c = eval_loss(model, eval_b, settings)
    eval_c_after_c = eval_loss(model, eval_c, settings)
    lora_after_c = lora_effective_deltas(model)

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
        "learning_a": eval_a_before - eval_a_after_a,
        "learning_b": eval_b_after_a - eval_b_after_b,
        "learning_c": eval_c_after_b - eval_c_after_c,
        "accretion_a_after_b": eval_a_after_a - eval_a_after_b,
        "backward_transfer_a_after_b": eval_a_after_b - eval_a_after_a,
        "interference_a_after_c": eval_a_after_c - eval_a_after_b,
        "interference_b_after_c": eval_b_after_c - eval_b_after_b,
        "retention_a_after_b": ratio(eval_a_after_a, eval_a_after_b),
        "retention_a_after_c": ratio(eval_a_after_a, eval_a_after_c),
        "retention_b_after_c": ratio(eval_b_after_b, eval_b_after_c),
        "lora_cosine_a_b_mean": mean_lora_cosine(lora_delta_a, lora_delta_b),
        "lora_cosine_a_c_mean": mean_lora_cosine(lora_delta_a, lora_delta_c),
        "lora_cosine_b_c_mean": mean_lora_cosine(lora_delta_b, lora_delta_c),
        "grad_cosine_a_b_after_a": grad_cosine_a_b_after_a,
        "grad_cosine_a_c_after_b": grad_cosine_a_c_after_b,
    }


def summarize_accretion(results: list[AccretionResult]) -> dict[str, dict[str, float]]:
    """Aggregate A-to-B-to-C metrics by variant."""
    metric_names = [
        "eval_a_after_a",
        "eval_a_after_b",
        "eval_a_after_c",
        "eval_b_after_b",
        "eval_b_after_c",
        "eval_c_after_c",
        "learning_a",
        "learning_b",
        "learning_c",
        "accretion_a_after_b",
        "backward_transfer_a_after_b",
        "interference_a_after_c",
        "interference_b_after_c",
        "retention_a_after_b",
        "retention_a_after_c",
        "retention_b_after_c",
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
        summary[variant] = values
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-accretion`."""
    parser = argparse.ArgumentParser(description="Run A-to-B-to-C LoRA compatibility tests.")
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
    parser.add_argument(
        "--compat-batches",
        type=int,
        default=0,
        help="Optional train batches for gradient compatibility metrics; 0 disables them.",
    )
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--diversity-weight", type=float, default=None)
    parser.add_argument("--repulsion-weight", type=float, default=None)
    parser.add_argument("--sparse-weight", type=float, default=None)
    parser.add_argument("--gossip-weight", type=float, default=None)
    parser.add_argument("--gossip-tau", type=float, default=None)
    parser.add_argument("--gossip-k", type=int, default=None)
    parser.add_argument("--max-gossip-vectors", type=int, default=None)
    parser.add_argument("--sweep", default=None, help="Dose sweep like gossip=3.0,5.0")
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variants", nargs="+", default=["baseline", "repulsion"])
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for A-to-B-to-C compatibility experiments."""
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
        args.variants,
        diversity=args.diversity_weight,
        repulsion=args.repulsion_weight,
        sparse=args.sparse_weight,
        gossip=args.gossip_weight,
        gossip_tau=args.gossip_tau,
        gossip_k=args.gossip_k,
        max_gossip_vectors=args.max_gossip_vectors,
        sweep=args.sweep,
    )
    seeds = args.seeds or [args.seed]
    task_a_texts = load_texts(args.task_a_file)
    task_b_texts = load_texts(args.task_b_file)
    task_c_texts = load_texts(args.task_c_file)

    results = [
        run_accretion_variant(
            variant,
            settings=settings,
            task_a_texts=task_a_texts,
            task_b_texts=task_b_texts,
            task_c_texts=task_c_texts,
            phase_steps=args.phase_steps,
            seed=seed,
            device=args.device,
            compat_batches=args.compat_batches,
        )
        for seed in seeds
        for variant in variants
    ]
    record: AccretionRunRecord = {
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
            "variants": [variant.name for variant in variants],
            "sweep": args.sweep,
            "gossip_weight": args.gossip_weight,
            "gossip_tau": args.gossip_tau,
            "gossip_k": args.gossip_k,
            "max_gossip_vectors": args.max_gossip_vectors,
        },
        "git_status": git_status(),
        "results": results,
        "summary": summarize_accretion(results),
    }
    if args.output_dir is not None:
        write_run_record(record, args.output_dir)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
