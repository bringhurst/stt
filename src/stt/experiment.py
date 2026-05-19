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
    representation_repulsion_loss,
    sparse_activation_loss,
)
from stt.metrics import active_fraction, effective_rank, head_similarity, isotropy
from stt.model import TinyTransformer


@dataclass(frozen=True)
class Variant:
    """Regularization weights for one experiment condition."""

    name: str
    diversity: float = 0.0
    repulsion: float = 0.0
    sparse: float = 0.0


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
) -> ExperimentResult:
    """Train one variant and return final task and geometry measurements."""
    device = resolve_device(device)
    torch.manual_seed(seed)
    task = SyntheticSequenceTask(seed=seed)
    model = TinyTransformer(vocab_size=task.vocab_size, seq_len=task.seq_len).to(device)
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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        task_loss_value = float(task_loss.detach().cpu())

    eval_tokens, eval_targets = task.batch(batch_size, device=device)
    with torch.no_grad():
        output = model(eval_tokens)
        eval_loss = criterion(output.logits.reshape(-1, task.vocab_size), eval_targets.reshape(-1))

    return {
        "variant": variant.name,
        "device": device,
        "train_task_loss": task_loss_value,
        "eval_task_loss": float(eval_loss.detach().cpu()),
        "head_similarity": head_similarity(output.attention),
        "effective_rank": effective_rank(output.hidden),
        "isotropy": isotropy(output.hidden),
        "active_fraction": active_fraction(output.hidden),
    }


def run_experiment(
    variant_names: list[str],
    steps: int,
    seed: int,
    device: str,
) -> list[ExperimentResult]:
    """Train each requested variant and collect comparable result dictionaries."""
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    return [
        train_variant(VARIANTS[name], steps=steps, seed=seed, device=device)
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
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for `stt-experiment`."""
    args = parse_args()
    results = run_experiment(args.variants, steps=args.steps, seed=args.seed, device=args.device)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
