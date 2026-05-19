"""Sequential LoRA continual-learning experiments for STT regularizers.

The experiment trains one adapter on task A, evaluates A and B, then continues
training the same adapter on task B and measures forgetting on task A. It is the
smallest useful continual-learning scaffold after the single-corpus LoRA tests.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from typing import Any, TypedDict

import torch
from torch import Tensor

from stt.experiment import Variant, resolve_device
from stt.lora_experiment import (
    LoraSettings,
    build_lora_model,
    build_variants,
    encode_texts,
    evaluate_model,
    git_status,
    load_texts,
    load_tokenizer,
    parameter_counts,
    split_corpus,
    stt_loss,
    write_run_record,
)


class ContinualResult(TypedDict):
    """Serializable result for one sequential A-then-B training run."""

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
    eval_a_after_a: float
    eval_b_after_a: float
    eval_a_after_b: float
    eval_b_after_b: float
    forgetting_a: float
    backward_transfer_a: float
    learning_b: float
    retention_ratio: float


class ContinualRunRecord(TypedDict):
    """Persisted continual-learning experiment record."""

    created_at: str
    config: dict[str, Any]
    git_status: str
    results: list[ContinualResult]
    summary: dict[str, dict[str, float]]


def train_steps(
    model: torch.nn.Module,
    encoded: dict[str, Tensor],
    variant: Variant,
    settings: LoraSettings,
    steps: int,
) -> float:
    """Train a LoRA model for a bounded number of optimizer micro-steps."""
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings.learning_rate,
    )
    model.train()
    train_lm_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        batch = batch_slice_for_continual(encoded, step * settings.batch_size, settings.batch_size)
        output = model(**batch, output_attentions=True, output_hidden_states=True)
        lm_loss = output.loss
        loss = (
            lm_loss + stt_loss(output, variant, attention_mask=batch.get("attention_mask"))
        ) / settings.grad_accum
        loss.backward()
        if (step + 1) % settings.grad_accum == 0 or step == steps - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_lm_loss = float(lm_loss.detach().cpu())
    return train_lm_loss


def batch_slice_for_continual(
    encoded: dict[str, Tensor],
    start: int,
    batch_size: int,
) -> dict[str, Tensor]:
    """Return a wraparound batch from encoded tensors.

    This local wrapper keeps the continual module independent from CLI internals
    while matching the LoRA experiment batching semantics.
    """
    size = encoded["input_ids"].shape[0]
    indices = torch.arange(start, start + batch_size, device=encoded["input_ids"].device) % size
    return {name: value.index_select(0, indices) for name, value in encoded.items()}


def eval_loss(
    model: torch.nn.Module,
    encoded: dict[str, Tensor],
    settings: LoraSettings,
) -> float:
    """Return averaged LM loss over configured evaluation batches."""
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(
            model,
            encoded,
            batch_size=settings.batch_size,
            eval_batches=settings.eval_batches,
            variant=None,
        )
    return metrics["eval_lm_loss"]


def prepare_encoded_splits(
    tokenizer: Any,
    texts: list[str],
    settings: LoraSettings,
    seed: int,
    device: str,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Create encoded train and bounded eval splits for one corpus."""
    train_texts, eval_texts = split_corpus(texts, seed)
    eval_sample_count = max(settings.batch_size, settings.batch_size * settings.eval_batches)
    return (
        encode_texts(tokenizer, train_texts, settings.max_length, device),
        encode_texts(tokenizer, eval_texts[:eval_sample_count], settings.max_length, device),
    )


def run_continual_variant(
    variant: Variant,
    settings: LoraSettings,
    task_a_texts: list[str],
    task_b_texts: list[str],
    phase_steps: int,
    seed: int,
    device: str,
) -> ContinualResult:
    """Train A then B and measure task-A forgetting."""
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

    eval_a_before = eval_loss(model, eval_a, settings)
    eval_b_before = eval_loss(model, eval_b, settings)
    train_steps(model, train_a, variant, settings, phase_steps)
    eval_a_after_a = eval_loss(model, eval_a, settings)
    eval_b_after_a = eval_loss(model, eval_b, settings)
    train_steps(model, train_b, variant, settings, phase_steps)
    eval_a_after_b = eval_loss(model, eval_a, settings)
    eval_b_after_b = eval_loss(model, eval_b, settings)

    backward_transfer_a = eval_a_after_b - eval_a_after_a
    learning_b = eval_b_before - eval_b_after_b
    retention_ratio = eval_a_after_a / eval_a_after_b if eval_a_after_b != 0.0 else 0.0

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
        "eval_a_after_a": eval_a_after_a,
        "eval_b_after_a": eval_b_after_a,
        "eval_a_after_b": eval_a_after_b,
        "eval_b_after_b": eval_b_after_b,
        "forgetting_a": backward_transfer_a,
        "backward_transfer_a": backward_transfer_a,
        "learning_b": learning_b,
        "retention_ratio": retention_ratio,
    }


def summarize_continual(results: list[ContinualResult]) -> dict[str, dict[str, float]]:
    """Aggregate continual-learning metrics by variant."""
    metric_getters = {
        "eval_a_after_a": lambda result: result["eval_a_after_a"],
        "eval_a_after_b": lambda result: result["eval_a_after_b"],
        "eval_b_after_b": lambda result: result["eval_b_after_b"],
        "forgetting_a": lambda result: result["forgetting_a"],
        "backward_transfer_a": lambda result: result["backward_transfer_a"],
        "learning_b": lambda result: result["learning_b"],
        "retention_ratio": lambda result: result["retention_ratio"],
    }
    summary: dict[str, dict[str, float]] = {}
    for variant in sorted({result["variant"] for result in results}):
        group = [result for result in results if result["variant"] == variant]
        values: dict[str, float] = {"count": float(len(group))}
        for metric, getter in metric_getters.items():
            metric_values = [float(getter(result)) for result in group]
            values[f"{metric}_mean"] = statistics.fmean(metric_values)
            values[f"{metric}_std"] = (
                statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            )
        summary[variant] = values
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-continual`."""
    parser = argparse.ArgumentParser(description="Run sequential LoRA continual-learning tests.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, or cuda")
    parser.add_argument("--task-a-file", required=True)
    parser.add_argument("--task-b-file", required=True)
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
    parser.add_argument("--diversity-weight", type=float, default=None)
    parser.add_argument("--repulsion-weight", type=float, default=None)
    parser.add_argument("--sparse-weight", type=float, default=None)
    parser.add_argument("--gossip-weight", type=float, default=None)
    parser.add_argument("--gossip-tau", type=float, default=None)
    parser.add_argument("--gossip-k", type=int, default=None)
    parser.add_argument("--max-gossip-vectors", type=int, default=None)
    parser.add_argument("--sweep", default=None, help="Dose sweep like repulsion=1.0,1.5")
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variants", nargs="+", default=["baseline", "repulsion"])
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for continual-learning experiments."""
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

    results = [
        run_continual_variant(
            variant,
            settings=settings,
            task_a_texts=task_a_texts,
            task_b_texts=task_b_texts,
            phase_steps=args.phase_steps,
            seed=seed,
            device=args.device,
        )
        for seed in seeds
        for variant in variants
    ]
    record: ContinualRunRecord = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "model": args.model,
            "device": args.device,
            "task_a_file": args.task_a_file,
            "task_b_file": args.task_b_file,
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
            "variants": [variant.name for variant in variants],
            "sweep": args.sweep,
            "gossip_weight": args.gossip_weight,
            "gossip_tau": args.gossip_tau,
            "gossip_k": args.gossip_k,
            "max_gossip_vectors": args.max_gossip_vectors,
        },
        "git_status": git_status(),
        "results": results,
        "summary": summarize_continual(results),
    }
    if args.output_dir is not None:
        write_run_record(record, args.output_dir)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
