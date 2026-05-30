"""TinyTransformer A/B/C accretion scaffold for FFN compartment tests."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor, nn

from stt.experiment import VARIANTS, Variant, resolve_device
from stt.losses import (
    attention_diversity_loss,
    branch_inhibition_loss,
    branch_load_balance_loss,
    branch_output_repulsion_loss,
    representation_repulsion_loss,
    sparse_activation_loss,
)
from stt.metrics import (
    active_fraction,
    branch_active_fraction,
    branch_entropy,
    branch_inhibition_mean,
    branch_score_entropy,
    branch_usage_max,
    branch_usage_min,
    branch_usage_std,
    effective_rank,
    head_similarity,
    isotropy,
)
from stt.model import TinyTransformer


@dataclass(frozen=True)
class MarkedSequenceTask:
    """Synthetic sequence task with a marker token and task-local target offset."""

    name: str
    marker_token: int
    target_offset: int
    seed: int
    vocab_size: int = 32
    seq_len: int = 24
    marker_count: int = 3

    def __post_init__(self) -> None:
        if self.vocab_size <= self.marker_count + 8:
            raise ValueError("vocab_size must leave room for content tokens")
        if not 0 <= self.marker_token < self.marker_count:
            raise ValueError("marker_token must be within marker_count")
        if self.seq_len < 4:
            raise ValueError("seq_len must be at least 4")
        object.__setattr__(self, "generator", torch.Generator().manual_seed(self.seed))

    @property
    def content_vocab_size(self) -> int:
        """Return the non-marker token count."""
        return self.vocab_size - self.marker_count

    def batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        eval_seed: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return marked inputs and task-offset next-token targets."""
        generator = (
            torch.Generator().manual_seed(eval_seed)
            if eval_seed is not None
            else self.generator
        )
        starts = torch.randint(0, self.content_vocab_size, (batch_size, 1), generator=generator)
        offsets = torch.arange(self.seq_len + 1).unsqueeze(0)
        modes = torch.randint(0, 4, (batch_size, 1), generator=generator)

        arithmetic = starts + offsets * (modes + 1)
        motif = (offsets % 4) * (modes + 2)
        noise = torch.randint(0, 3, (batch_size, self.seq_len + 1), generator=generator)
        base = (arithmetic + motif + noise) % self.content_vocab_size

        tokens = (base[:, :-1] + self.marker_count).to(device=device, dtype=torch.long)
        targets = (
            (base[:, 1:] + self.target_offset) % self.content_vocab_size
        ) + self.marker_count
        targets = targets.to(device=device, dtype=torch.long)
        tokens[:, 0] = self.marker_token
        return tokens, targets


class ToyAccretionResult(TypedDict):
    """Serializable output from one TinyTransformer A/B/C run."""

    variant: str
    condition: str
    device: str
    seed: int
    phase_steps: int
    compartments: int
    compartment_top_k: int
    compartment_mode: str
    branch_repulsion_weight: float
    branch_load_balance_weight: float
    branch_inhibition_strength: float
    branch_inhibition_weight: float
    train_task_loss: float
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
    head_similarity: float
    effective_rank: float
    isotropy: float
    active_fraction: float
    branch_entropy: float
    branch_active_fraction: float
    branch_usage_min: float
    branch_usage_max: float
    branch_usage_std: float
    branch_score_entropy: float
    branch_inhibition_mean: float
    branch_repulsion_loss: float
    branch_load_balance_loss: float
    branch_inhibition_loss: float


def ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio for positive losses."""
    return numerator / denominator if denominator != 0.0 else 0.0


def condition_name(compartments: int, compartment_top_k: int, compartment_mode: str) -> str:
    """Return a compact label for the FFN condition."""
    if compartments == 0:
        return "dense"
    return f"{compartment_mode}_top{compartment_top_k}"


def make_tasks(seed: int, vocab_size: int, seq_len: int) -> tuple[MarkedSequenceTask, ...]:
    """Create A/B/C marked tasks with same-rule B and stronger-conflict C."""
    content_vocab_size = vocab_size - 3
    return (
        MarkedSequenceTask(
            "a",
            marker_token=0,
            target_offset=0,
            seed=seed,
            vocab_size=vocab_size,
            seq_len=seq_len,
        ),
        MarkedSequenceTask(
            "b",
            marker_token=1,
            target_offset=0,
            seed=seed + 10_000,
            vocab_size=vocab_size,
            seq_len=seq_len,
        ),
        MarkedSequenceTask(
            "c",
            marker_token=2,
            target_offset=content_vocab_size // 2,
            seed=seed + 20_000,
            vocab_size=vocab_size,
            seq_len=seq_len,
        ),
    )


def regularized_loss(
    task_loss: Tensor,
    output: object,
    variant: Variant,
    branch_repulsion_weight: float,
    branch_load_balance_weight: float,
    branch_inhibition_weight: float,
) -> Tensor:
    """Add STT and branch regularizers to a task loss."""
    loss = task_loss
    logits_output = output  # Keeps attribute access explicit for static readers.
    loss = loss + variant.diversity * attention_diversity_loss(logits_output.attention)
    loss = loss + variant.repulsion * representation_repulsion_loss(logits_output.hidden)
    loss = loss + variant.sparse * sparse_activation_loss(logits_output.hidden)
    if logits_output.branch_outputs is not None:
        loss = loss + branch_repulsion_weight * branch_output_repulsion_loss(
            logits_output.branch_outputs
        )
    if logits_output.branch_gates is not None:
        loss = loss + branch_load_balance_weight * branch_load_balance_loss(
            logits_output.branch_gates
        )
    if logits_output.branch_outputs is not None and logits_output.branch_gates is not None:
        loss = loss + branch_inhibition_weight * branch_inhibition_loss(
            logits_output.branch_outputs,
            logits_output.branch_gates,
        )
    return loss


def train_phase(
    model: TinyTransformer,
    task: MarkedSequenceTask,
    variant: Variant,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    steps: int,
    batch_size: int,
    device: str,
    branch_repulsion_weight: float,
    branch_load_balance_weight: float,
    branch_inhibition_weight: float,
) -> float:
    """Train one sequential phase and return the final task loss."""
    model.train()
    task_loss_value = 0.0
    for _ in range(steps):
        tokens, targets = task.batch(batch_size, device=device)
        output = model(tokens)
        task_loss = criterion(output.logits.reshape(-1, task.vocab_size), targets.reshape(-1))
        loss = regularized_loss(
            task_loss,
            output,
            variant,
            branch_repulsion_weight,
            branch_load_balance_weight,
            branch_inhibition_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        task_loss_value = float(task_loss.detach().cpu())
    return task_loss_value


@torch.no_grad()
def evaluate_task(
    model: TinyTransformer,
    task: MarkedSequenceTask,
    criterion: nn.Module,
    batch_size: int,
    device: str,
    eval_seed: int,
) -> tuple[float, object]:
    """Evaluate one task on a fixed synthetic sample."""
    model.eval()
    tokens, targets = task.batch(batch_size, device=device, eval_seed=eval_seed)
    output = model(tokens)
    loss = criterion(output.logits.reshape(-1, task.vocab_size), targets.reshape(-1))
    return float(loss.detach().cpu()), output


def run_toy_accretion_variant(
    variant: Variant,
    phase_steps: int = 80,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    vocab_size: int = 32,
    seq_len: int = 24,
    dim: int = 48,
    heads: int = 4,
    layers: int = 2,
    compartments: int = 0,
    compartment_top_k: int = 1,
    compartment_mode: str = "router",
    branch_repulsion_weight: float = 0.0,
    branch_load_balance_weight: float = 0.0,
    branch_inhibition_strength: float = 0.5,
    branch_inhibition_weight: float = 0.0,
) -> ToyAccretionResult:
    """Train A then B then C and measure TinyTransformer retention/interference."""
    resolved_device = resolve_device(device)
    torch.manual_seed(seed)
    task_a, task_b, task_c = make_tasks(seed, vocab_size, seq_len)
    eval_seeds = {
        "a": seed + 100_000,
        "b": seed + 110_000,
        "c": seed + 120_000,
    }
    model = TinyTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dim=dim,
        heads=heads,
        layers=layers,
        compartments=compartments,
        compartment_top_k=compartment_top_k,
        compartment_mode=compartment_mode,  # type: ignore[arg-type]
        branch_inhibition_strength=branch_inhibition_strength,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    eval_a_before, _ = evaluate_task(
        model, task_a, criterion, eval_batch_size, resolved_device, eval_seeds["a"]
    )
    eval_b_before, _ = evaluate_task(
        model, task_b, criterion, eval_batch_size, resolved_device, eval_seeds["b"]
    )
    eval_c_before, _ = evaluate_task(
        model, task_c, criterion, eval_batch_size, resolved_device, eval_seeds["c"]
    )

    train_loss = train_phase(
        model,
        task_a,
        variant,
        optimizer,
        criterion,
        phase_steps,
        batch_size,
        resolved_device,
        branch_repulsion_weight,
        branch_load_balance_weight,
        branch_inhibition_weight,
    )
    eval_a_after_a, _ = evaluate_task(
        model, task_a, criterion, eval_batch_size, resolved_device, eval_seeds["a"]
    )
    eval_b_after_a, _ = evaluate_task(
        model, task_b, criterion, eval_batch_size, resolved_device, eval_seeds["b"]
    )
    eval_c_after_a, _ = evaluate_task(
        model, task_c, criterion, eval_batch_size, resolved_device, eval_seeds["c"]
    )

    train_loss = train_phase(
        model,
        task_b,
        variant,
        optimizer,
        criterion,
        phase_steps,
        batch_size,
        resolved_device,
        branch_repulsion_weight,
        branch_load_balance_weight,
        branch_inhibition_weight,
    )
    eval_a_after_b, _ = evaluate_task(
        model, task_a, criterion, eval_batch_size, resolved_device, eval_seeds["a"]
    )
    eval_b_after_b, _ = evaluate_task(
        model, task_b, criterion, eval_batch_size, resolved_device, eval_seeds["b"]
    )
    eval_c_after_b, _ = evaluate_task(
        model, task_c, criterion, eval_batch_size, resolved_device, eval_seeds["c"]
    )

    train_loss = train_phase(
        model,
        task_c,
        variant,
        optimizer,
        criterion,
        phase_steps,
        batch_size,
        resolved_device,
        branch_repulsion_weight,
        branch_load_balance_weight,
        branch_inhibition_weight,
    )
    eval_a_after_c, _ = evaluate_task(
        model, task_a, criterion, eval_batch_size, resolved_device, eval_seeds["a"]
    )
    eval_b_after_c, _ = evaluate_task(
        model, task_b, criterion, eval_batch_size, resolved_device, eval_seeds["b"]
    )
    eval_c_after_c, final_output = evaluate_task(
        model, task_c, criterion, eval_batch_size, resolved_device, eval_seeds["c"]
    )

    branch_repulsion_value = (
        float(branch_output_repulsion_loss(final_output.branch_outputs).detach().cpu())
        if final_output.branch_outputs is not None
        else 0.0
    )
    branch_load_balance_value = (
        float(branch_load_balance_loss(final_output.branch_gates).detach().cpu())
        if final_output.branch_gates is not None
        else 0.0
    )
    branch_inhibition_value = (
        float(
            branch_inhibition_loss(final_output.branch_outputs, final_output.branch_gates)
            .detach()
            .cpu()
        )
        if final_output.branch_outputs is not None and final_output.branch_gates is not None
        else 0.0
    )

    return {
        "variant": variant.name,
        "condition": condition_name(compartments, compartment_top_k, compartment_mode),
        "device": resolved_device,
        "seed": seed,
        "phase_steps": phase_steps,
        "compartments": compartments,
        "compartment_top_k": compartment_top_k,
        "compartment_mode": compartment_mode,
        "branch_repulsion_weight": branch_repulsion_weight,
        "branch_load_balance_weight": branch_load_balance_weight,
        "branch_inhibition_strength": branch_inhibition_strength,
        "branch_inhibition_weight": branch_inhibition_weight,
        "train_task_loss": train_loss,
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
        "head_similarity": head_similarity(final_output.attention),
        "effective_rank": effective_rank(final_output.hidden),
        "isotropy": isotropy(final_output.hidden),
        "active_fraction": active_fraction(final_output.hidden),
        "branch_entropy": branch_entropy(final_output.branch_gates)
        if final_output.branch_gates is not None
        else 0.0,
        "branch_active_fraction": branch_active_fraction(final_output.branch_gates)
        if final_output.branch_gates is not None
        else 0.0,
        "branch_usage_min": branch_usage_min(final_output.branch_gates)
        if final_output.branch_gates is not None
        else 0.0,
        "branch_usage_max": branch_usage_max(final_output.branch_gates)
        if final_output.branch_gates is not None
        else 0.0,
        "branch_usage_std": branch_usage_std(final_output.branch_gates)
        if final_output.branch_gates is not None
        else 0.0,
        "branch_score_entropy": branch_score_entropy(final_output.branch_scores)
        if final_output.branch_scores is not None
        else 0.0,
        "branch_inhibition_mean": branch_inhibition_mean(final_output.branch_inhibition)
        if final_output.branch_inhibition is not None
        else 0.0,
        "branch_repulsion_loss": branch_repulsion_value,
        "branch_load_balance_loss": branch_load_balance_value,
        "branch_inhibition_loss": branch_inhibition_value,
    }


def summarize_toy_accretion(
    results: list[ToyAccretionResult],
) -> dict[str, dict[str, float]]:
    """Aggregate toy accretion metrics by condition and variant."""
    metric_names = [
        "eval_a_after_b",
        "eval_b_after_b",
        "eval_a_after_c",
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
        "effective_rank",
        "isotropy",
        "branch_entropy",
        "branch_score_entropy",
        "branch_inhibition_mean",
    ]
    summary: dict[str, dict[str, float]] = {}
    for key in sorted({f"{result['condition']}:{result['variant']}" for result in results}):
        condition, variant = key.split(":", maxsplit=1)
        group = [
            result
            for result in results
            if result["condition"] == condition and result["variant"] == variant
        ]
        values: dict[str, float] = {"count": float(len(group))}
        for metric in metric_names:
            metric_values = [float(result[metric]) for result in group]
            values[f"{metric}_mean"] = statistics.fmean(metric_values)
            values[f"{metric}_std"] = (
                statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            )
        summary[key] = values
    return summary


def run_toy_accretion(
    variant_names: list[str],
    phase_steps: int,
    seed: int,
    device: str,
    compartments: int = 0,
    compartment_top_k: int = 1,
    compartment_mode: str = "router",
    branch_repulsion_weight: float = 0.0,
    branch_load_balance_weight: float = 0.0,
    branch_inhibition_strength: float = 0.5,
    branch_inhibition_weight: float = 0.0,
) -> list[ToyAccretionResult]:
    """Run one or more variants through the same toy A/B/C setup."""
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    return [
        run_toy_accretion_variant(
            VARIANTS[name],
            phase_steps=phase_steps,
            seed=seed,
            device=device,
            compartments=compartments,
            compartment_top_k=compartment_top_k,
            compartment_mode=compartment_mode,
            branch_repulsion_weight=branch_repulsion_weight,
            branch_load_balance_weight=branch_load_balance_weight,
            branch_inhibition_strength=branch_inhibition_strength,
            branch_inhibition_weight=branch_inhibition_weight,
        )
        for name in variant_names
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-toy-accretion`."""
    parser = argparse.ArgumentParser(description="Run TinyTransformer A/B/C accretion tests.")
    parser.add_argument("--phase-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, or cuda")
    parser.add_argument("--variants", nargs="+", default=["baseline"], choices=list(VARIANTS))
    parser.add_argument("--compartments", type=int, default=0)
    parser.add_argument("--compartment-top-k", type=int, default=1)
    parser.add_argument("--compartment-mode", choices=["router", "dendritic"], default="router")
    parser.add_argument("--branch-repulsion-weight", type=float, default=0.0)
    parser.add_argument("--branch-load-balance-weight", type=float, default=0.0)
    parser.add_argument("--branch-inhibition-strength", type=float, default=0.5)
    parser.add_argument("--branch-inhibition-weight", type=float, default=0.0)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print grouped means instead of raw runs",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for `stt-toy-accretion`."""
    args = parse_args()
    if args.phase_steps < 1:
        raise ValueError("--phase-steps must be at least 1")
    if args.compartments < 0:
        raise ValueError("--compartments cannot be negative")
    if args.compartment_top_k < 1:
        raise ValueError("--compartment-top-k must be at least 1")
    if args.compartments and args.compartment_top_k > args.compartments:
        raise ValueError("--compartment-top-k cannot exceed --compartments")
    if args.branch_repulsion_weight < 0.0:
        raise ValueError("--branch-repulsion-weight cannot be negative")
    if args.branch_load_balance_weight < 0.0:
        raise ValueError("--branch-load-balance-weight cannot be negative")
    if args.branch_inhibition_strength < 0.0:
        raise ValueError("--branch-inhibition-strength cannot be negative")
    if args.branch_inhibition_weight < 0.0:
        raise ValueError("--branch-inhibition-weight cannot be negative")
    results = run_toy_accretion(
        args.variants,
        phase_steps=args.phase_steps,
        seed=args.seed,
        device=args.device,
        compartments=args.compartments,
        compartment_top_k=args.compartment_top_k,
        compartment_mode=args.compartment_mode,
        branch_repulsion_weight=args.branch_repulsion_weight,
        branch_load_balance_weight=args.branch_load_balance_weight,
        branch_inhibition_strength=args.branch_inhibition_strength,
        branch_inhibition_weight=args.branch_inhibition_weight,
    )
    payload: object = summarize_toy_accretion(results) if args.summary else results
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
