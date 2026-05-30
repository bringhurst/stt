"""Command-line experiment runner for minimal STT regularizer comparisons."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn

from stt.data import SyntheticSequenceTask
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
class Variant:
    """Regularization weights for one experiment condition."""

    name: str
    diversity: float = 0.0
    repulsion: float = 0.0
    sparse: float = 0.0
    gossip: float = 0.0
    gossip_tau: float = 0.85
    gossip_k: int = 8
    max_gossip_vectors: int = 256


class ExperimentResult(TypedDict):
    """Serializable output metrics for one trained variant."""

    variant: str
    device: str
    train_task_loss: float
    eval_task_loss: float
    head_similarity: float
    effective_rank: float
    isotropy: float
    active_fraction: float
    compartments: int
    compartment_top_k: int
    compartment_mode: str
    branch_repulsion_weight: float
    branch_load_balance_weight: float
    branch_inhibition_strength: float
    branch_inhibition_weight: float
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


VARIANTS = {
    "baseline": Variant("baseline"),
    "diversity": Variant("diversity", diversity=0.05),
    "repulsion": Variant("repulsion", repulsion=0.05),
    "sparse": Variant("sparse", sparse=0.002),
    "combined": Variant("combined", diversity=0.03, repulsion=0.03, sparse=0.001),
}


def resolve_device(requested: str) -> str:
    """Resolve `auto` to the best available local PyTorch device.

    On Apple Silicon Macs, this selects `mps` so the experiment can use unified
    memory and the Metal backend. CPU remains available and is the most portable
    option for tests.
    """
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_variant(
    variant: Variant,
    steps: int = 80,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cpu",
    compartments: int = 0,
    compartment_top_k: int = 1,
    compartment_mode: str = "router",
    branch_repulsion_weight: float = 0.0,
    branch_load_balance_weight: float = 0.0,
    branch_inhibition_strength: float = 0.5,
    branch_inhibition_weight: float = 0.0,
) -> ExperimentResult:
    """Train one variant and return final task and geometry measurements."""
    device = resolve_device(device)
    torch.manual_seed(seed)
    task = SyntheticSequenceTask(seed=seed)
    model = TinyTransformer(
        vocab_size=task.vocab_size,
        seq_len=task.seq_len,
        compartments=compartments,
        compartment_top_k=compartment_top_k,
        compartment_mode=compartment_mode,  # type: ignore[arg-type]
        branch_inhibition_strength=branch_inhibition_strength,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    task_loss_value = 0.0
    for _ in range(steps):
        tokens, targets = task.batch(batch_size, device=device)
        output = model(tokens)
        task_loss = criterion(output.logits.reshape(-1, task.vocab_size), targets.reshape(-1))
        loss = task_loss
        loss = loss + variant.diversity * attention_diversity_loss(output.attention)
        loss = loss + variant.repulsion * representation_repulsion_loss(output.hidden)
        loss = loss + variant.sparse * sparse_activation_loss(output.hidden)
        if output.branch_outputs is not None:
            loss = loss + branch_repulsion_weight * branch_output_repulsion_loss(
                output.branch_outputs
            )
        if output.branch_gates is not None:
            loss = loss + branch_load_balance_weight * branch_load_balance_loss(
                output.branch_gates
            )
        if output.branch_outputs is not None and output.branch_gates is not None:
            loss = loss + branch_inhibition_weight * branch_inhibition_loss(
                output.branch_outputs,
                output.branch_gates,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        task_loss_value = float(task_loss.detach().cpu())

    eval_tokens, eval_targets = task.batch(batch_size, device=device)
    with torch.no_grad():
        output = model(eval_tokens)
        eval_loss = criterion(output.logits.reshape(-1, task.vocab_size), eval_targets.reshape(-1))
        branch_repulsion_value = (
            float(branch_output_repulsion_loss(output.branch_outputs).detach().cpu())
            if output.branch_outputs is not None
            else 0.0
        )
        branch_load_balance_value = (
            float(branch_load_balance_loss(output.branch_gates).detach().cpu())
            if output.branch_gates is not None
            else 0.0
        )
        branch_inhibition_value = (
            float(
                branch_inhibition_loss(output.branch_outputs, output.branch_gates)
                .detach()
                .cpu()
            )
            if output.branch_outputs is not None and output.branch_gates is not None
            else 0.0
        )

    return {
        "variant": variant.name,
        "device": device,
        "train_task_loss": task_loss_value,
        "eval_task_loss": float(eval_loss.detach().cpu()),
        "head_similarity": head_similarity(output.attention),
        "effective_rank": effective_rank(output.hidden),
        "isotropy": isotropy(output.hidden),
        "active_fraction": active_fraction(output.hidden),
        "compartments": compartments,
        "compartment_top_k": compartment_top_k,
        "compartment_mode": compartment_mode,
        "branch_repulsion_weight": branch_repulsion_weight,
        "branch_load_balance_weight": branch_load_balance_weight,
        "branch_inhibition_strength": branch_inhibition_strength,
        "branch_inhibition_weight": branch_inhibition_weight,
        "branch_entropy": branch_entropy(output.branch_gates)
        if output.branch_gates is not None
        else 0.0,
        "branch_active_fraction": branch_active_fraction(output.branch_gates)
        if output.branch_gates is not None
        else 0.0,
        "branch_usage_min": branch_usage_min(output.branch_gates)
        if output.branch_gates is not None
        else 0.0,
        "branch_usage_max": branch_usage_max(output.branch_gates)
        if output.branch_gates is not None
        else 0.0,
        "branch_usage_std": branch_usage_std(output.branch_gates)
        if output.branch_gates is not None
        else 0.0,
        "branch_score_entropy": branch_score_entropy(output.branch_scores)
        if output.branch_scores is not None
        else 0.0,
        "branch_inhibition_mean": branch_inhibition_mean(output.branch_inhibition)
        if output.branch_inhibition is not None
        else 0.0,
        "branch_repulsion_loss": branch_repulsion_value,
        "branch_load_balance_loss": branch_load_balance_value,
        "branch_inhibition_loss": branch_inhibition_value,
    }


def run_experiment(
    variant_names: list[str],
    steps: int,
    seed: int,
    device: str,
    compartments: int = 0,
    compartment_top_k: int = 1,
    compartment_mode: str = "router",
    branch_repulsion_weight: float = 0.0,
    branch_load_balance_weight: float = 0.0,
    branch_inhibition_strength: float = 0.5,
    branch_inhibition_weight: float = 0.0,
) -> list[ExperimentResult]:
    """Train each requested variant and collect comparable result dictionaries."""
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    return [
        train_variant(
            VARIANTS[name],
            steps=steps,
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
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run minimal Surface Tension Transformer experiments."
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, or cuda")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--compartments", type=int, default=0)
    parser.add_argument("--compartment-top-k", type=int, default=1)
    parser.add_argument("--compartment-mode", choices=["router", "dendritic"], default="router")
    parser.add_argument("--branch-repulsion-weight", type=float, default=0.0)
    parser.add_argument("--branch-load-balance-weight", type=float, default=0.0)
    parser.add_argument("--branch-inhibition-strength", type=float, default=0.5)
    parser.add_argument("--branch-inhibition-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for `stt-experiment`."""
    args = parse_args()
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
    results = run_experiment(
        args.variants,
        steps=args.steps,
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
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
