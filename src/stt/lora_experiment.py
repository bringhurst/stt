"""LoRA fine-tuning experiments for Surface Tension Transformer regularizers.

This module applies the same STT geometry losses used by the toy Transformer to
pretrained causal language models through parameter-efficient LoRA adapters. The
default configuration is intentionally small enough to smoke-test on CPU, while
`--device auto` can use Apple Silicon MPS for larger local runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from stt.experiment import Variant, resolve_device
from stt.losses import (
    attention_diversity_loss,
    gossip_repulsion_loss,
    representation_repulsion_loss,
    sparse_activation_loss,
)
from stt.metrics import active_fraction, effective_rank, head_similarity, isotropy

DEFAULT_TEXTS = [
    "Surface tension stabilizes geometry while preserving local plasticity.",
    "Attention heads can specialize when redundant routing is weakly discouraged.",
    "Continual learning needs stable representations and controlled adaptation.",
    "Sparse competition can reduce semantic smearing in hidden activations.",
    "A useful experiment measures geometry before claiming better intelligence.",
    "Elastic manifolds bend under updates instead of collapsing into one direction.",
    "LoRA adapters let us test regularizers without updating the base model.",
    "Dream big, measure tiny, and keep the experiment reproducible.",
]

VARIANTS = {
    "baseline": Variant("baseline"),
    "diversity": Variant("diversity", diversity=0.02),
    "repulsion": Variant("repulsion", repulsion=0.02),
    "gossip": Variant("gossip", gossip=1.0),
    "sparse": Variant("sparse", sparse=0.0005),
    "combined": Variant("combined", diversity=0.01, repulsion=0.01, sparse=0.0002),
}


@dataclass(frozen=True)
class LoraSettings:
    """Configuration for one LoRA fine-tuning run.

    Args:
        model_name: Hugging Face model id or local model path.
        max_length: Tokenized sequence length used for training/evaluation.
        batch_size: Number of tokenized text samples per optimizer micro-step.
        eval_batches: Number of bounded holdout batches to average for evaluation.
        grad_accum: Number of micro-steps to accumulate before each optimizer step.
        learning_rate: AdamW learning rate for LoRA parameters.
        lora_rank: LoRA matrix rank.
        lora_alpha: LoRA scaling parameter.
        lora_dropout: Dropout applied inside LoRA adapters.
        target_modules: Module names to receive LoRA adapters.
    """

    model_name: str
    max_length: int = 128
    batch_size: int = 1
    eval_batches: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ()


class LoraExperimentResult(TypedDict):
    """Serializable metrics for one LoRA STT fine-tuning run."""

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
    train_lm_loss: float
    eval_lm_loss: float
    head_similarity: float
    effective_rank: float
    isotropy: float
    active_fraction: float
    eval_diversity_loss: float
    eval_repulsion_loss: float
    eval_sparse_loss: float
    eval_gossip_loss: float
    trainable_parameters: int
    total_parameters: int
    trainable_fraction: float


class RunRecord(TypedDict):
    """Persisted LoRA experiment record."""

    created_at: str
    config: dict[str, Any]
    git_status: str
    results: list[LoraExperimentResult]
    summary: dict[str, dict[str, float]]


def infer_target_modules(model_name: str) -> tuple[str, ...]:
    """Infer common LoRA target modules from the model name.

    Qwen/Llama-style models expose separate query/key/value/output projection
    modules. GPT-2-style models use a fused `c_attn` projection and `c_proj`.
    Users can override this with `--target-modules` for other architectures.
    """
    lowered = model_name.lower()
    if "gpt2" in lowered:
        return ("c_attn", "c_proj")
    return ("q_proj", "k_proj", "v_proj", "o_proj")


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer and ensure it has a padding token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    device: str,
) -> dict[str, Tensor]:
    """Tokenize text samples for causal-LM training."""
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_texts(path: str | None) -> list[str]:
    """Load non-empty corpus lines from a UTF-8 text file or return defaults."""
    if path is None:
        return DEFAULT_TEXTS
    texts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    texts = [line for line in texts if line]
    if len(texts) < 2:
        raise ValueError("text file must contain at least two non-empty lines")
    return texts


def split_corpus(corpus: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Return deterministic shuffled train/eval text splits for one seed."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(corpus), generator=generator).tolist()
    shuffled = [corpus[index] for index in indices]
    split = max(1, int(len(shuffled) * 0.75))
    train_texts = shuffled[:split]
    eval_texts = shuffled[split:] or shuffled[:1]
    return train_texts, eval_texts


def stack_attentions(attentions: tuple[Tensor, ...]) -> Tensor:
    """Stack Hugging Face attention outputs into STT loss shape.

    Hugging Face causal LMs return one tensor per layer, each shaped
    `(batch, heads, seq, seq)`. STT losses expect `(batch, layers, heads, seq,
    seq)`.
    """
    if not attentions:
        raise ValueError("model did not return attentions; try attn_implementation='eager'")
    return torch.stack(attentions, dim=1)


def last_hidden_state(hidden_states: tuple[Tensor, ...]) -> Tensor:
    """Return the final hidden-state tensor from a Hugging Face output."""
    if not hidden_states:
        raise ValueError("model did not return hidden states")
    return hidden_states[-1]


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def build_lora_model(settings: LoraSettings, device: str) -> PreTrainedModel:
    """Load a causal LM and attach LoRA adapters."""
    target_modules = settings.target_modules or infer_target_modules(settings.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        attn_implementation="eager",
        dtype=torch.float32,
    )
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=list(target_modules),
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model.to(device)


def batch_slice(encoded: dict[str, Tensor], start: int, batch_size: int) -> dict[str, Tensor]:
    """Return a wraparound batch from encoded tensors."""
    size = encoded["input_ids"].shape[0]
    indices = torch.arange(start, start + batch_size, device=encoded["input_ids"].device) % size
    return {name: value.index_select(0, indices) for name, value in encoded.items()}


def stt_components(
    output: Any,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return unweighted diversity, repulsion, sparse, and gossip components."""
    attention = stack_attentions(output.attentions)
    hidden = last_hidden_state(output.hidden_states)
    return (
        attention_diversity_loss(attention),
        representation_repulsion_loss(hidden),
        sparse_activation_loss(hidden),
        gossip_repulsion_loss(hidden, attention_mask=attention_mask),
    )


def stt_loss(output: Any, variant: Variant, attention_mask: Tensor | None = None) -> Tensor:
    """Compute weighted STT regularization loss from a model output."""
    hidden = last_hidden_state(output.hidden_states)
    diversity, repulsion, sparse, gossip = stt_components(output, attention_mask=attention_mask)
    if variant.gossip != 0.0:
        gossip = gossip_repulsion_loss(
            hidden,
            attention_mask=attention_mask,
            tau=variant.gossip_tau,
            k=variant.gossip_k,
            max_vectors=variant.max_gossip_vectors,
        )
    loss = hidden.new_zeros(())
    loss = loss + variant.diversity * diversity
    loss = loss + variant.repulsion * repulsion
    loss = loss + variant.sparse * sparse
    loss = loss + variant.gossip * gossip
    return loss


def evaluate_model(
    model: torch.nn.Module,
    encoded: dict[str, Tensor],
    batch_size: int,
    eval_batches: int,
    variant: Variant | None = None,
) -> dict[str, float]:
    """Average LM and geometry metrics over bounded evaluation batches."""
    values: dict[str, list[float]] = {
        "eval_lm_loss": [],
        "head_similarity": [],
        "effective_rank": [],
        "isotropy": [],
        "active_fraction": [],
        "eval_diversity_loss": [],
        "eval_repulsion_loss": [],
        "eval_sparse_loss": [],
        "eval_gossip_loss": [],
    }
    for batch_index in range(max(1, eval_batches)):
        batch = batch_slice(encoded, batch_index * batch_size, batch_size)
        output = model(**batch, output_attentions=True, output_hidden_states=True)
        attention = stack_attentions(output.attentions)
        hidden = last_hidden_state(output.hidden_states)
        diversity_loss, repulsion_loss, sparse_loss, gossip_loss = stt_components(
            output,
            attention_mask=batch.get("attention_mask"),
        )
        if variant is not None:
            gossip_loss = gossip_repulsion_loss(
                hidden,
                attention_mask=batch.get("attention_mask"),
                tau=variant.gossip_tau,
                k=variant.gossip_k,
                max_vectors=variant.max_gossip_vectors,
            )
        values["eval_lm_loss"].append(float(output.loss.detach().cpu()))
        values["head_similarity"].append(head_similarity(attention))
        values["effective_rank"].append(effective_rank(hidden))
        values["isotropy"].append(isotropy(hidden))
        values["active_fraction"].append(active_fraction(hidden))
        values["eval_diversity_loss"].append(float(diversity_loss.detach().cpu()))
        values["eval_repulsion_loss"].append(float(repulsion_loss.detach().cpu()))
        values["eval_sparse_loss"].append(float(sparse_loss.detach().cpu()))
        values["eval_gossip_loss"].append(float(gossip_loss.detach().cpu()))
    return {name: statistics.fmean(metric_values) for name, metric_values in values.items()}


def train_lora_variant(
    variant: Variant,
    settings: LoraSettings,
    steps: int,
    seed: int,
    device: str,
    texts: list[str] | None = None,
) -> LoraExperimentResult:
    """Fine-tune one LoRA variant and return LM plus geometry metrics."""
    resolved_device = resolve_device(device)
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(settings.model_name)
    corpus = texts or DEFAULT_TEXTS
    train_texts, eval_texts = split_corpus(corpus, seed)

    model = build_lora_model(settings, resolved_device)
    trainable, total = parameter_counts(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings.learning_rate,
    )
    train_encoded = encode_texts(tokenizer, train_texts, settings.max_length, resolved_device)
    eval_sample_count = max(settings.batch_size, settings.batch_size * settings.eval_batches)
    eval_encoded = encode_texts(
        tokenizer,
        eval_texts[:eval_sample_count],
        settings.max_length,
        resolved_device,
    )

    model.train()
    train_lm_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        batch = batch_slice(train_encoded, step * settings.batch_size, settings.batch_size)
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

    model.eval()
    with torch.no_grad():
        eval_metrics = evaluate_model(
            model,
            eval_encoded,
            batch_size=settings.batch_size,
            eval_batches=settings.eval_batches,
            variant=variant,
        )

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
        "train_lm_loss": train_lm_loss,
        "eval_lm_loss": eval_metrics["eval_lm_loss"],
        "head_similarity": eval_metrics["head_similarity"],
        "effective_rank": eval_metrics["effective_rank"],
        "isotropy": eval_metrics["isotropy"],
        "active_fraction": eval_metrics["active_fraction"],
        "eval_diversity_loss": eval_metrics["eval_diversity_loss"],
        "eval_repulsion_loss": eval_metrics["eval_repulsion_loss"],
        "eval_sparse_loss": eval_metrics["eval_sparse_loss"],
        "eval_gossip_loss": eval_metrics["eval_gossip_loss"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }


def run_lora_experiment(
    variants: list[Variant],
    settings: LoraSettings,
    steps: int,
    seeds: list[int],
    device: str,
    texts: list[str] | None = None,
) -> list[LoraExperimentResult]:
    """Run multiple LoRA STT variants and seeds with shared settings."""
    results = []
    for seed in seeds:
        for variant in variants:
            results.append(
                train_lora_variant(
                    variant,
                    settings=settings,
                    steps=steps,
                    seed=seed,
                    device=device,
                    texts=texts,
                )
            )
    return results


def variant_with_overrides(
    name: str,
    diversity: float | None,
    repulsion: float | None,
    sparse: float | None,
    gossip: float | None = None,
    gossip_tau: float | None = None,
    gossip_k: int | None = None,
    max_gossip_vectors: int | None = None,
) -> Variant:
    """Return a named variant with optional CLI-provided regularizer weights."""
    base = VARIANTS[name]
    if name == "baseline":
        return base
    accepts_diversity = name in {"diversity", "combined"}
    accepts_repulsion = name in {"repulsion", "combined"}
    accepts_sparse = name in {"sparse", "combined"}
    accepts_gossip = name in {"gossip", "combined"}
    return Variant(
        name=base.name,
        diversity=base.diversity if diversity is None or not accepts_diversity else diversity,
        repulsion=base.repulsion if repulsion is None or not accepts_repulsion else repulsion,
        sparse=base.sparse if sparse is None or not accepts_sparse else sparse,
        gossip=base.gossip if gossip is None or not accepts_gossip else gossip,
        gossip_tau=base.gossip_tau if gossip_tau is None or not accepts_gossip else gossip_tau,
        gossip_k=base.gossip_k if gossip_k is None or not accepts_gossip else gossip_k,
        max_gossip_vectors=(
            base.max_gossip_vectors
            if max_gossip_vectors is None or not accepts_gossip
            else max_gossip_vectors
        ),
    )


def variant_accepts_sweep(name: str, sweep_name: str) -> bool:
    """Return whether a swept parameter should expand a variant."""
    if name == "combined":
        return True
    if sweep_name in {"gossip", "gossip_tau", "gossip_k"}:
        return name == "gossip"
    return name == sweep_name


def parse_sweep(sweep: str | None) -> tuple[str, list[float]] | None:
    """Parse a sweep spec such as `repulsion=0,0.1,1.0`."""
    if sweep is None:
        return None
    if "=" not in sweep:
        raise ValueError("--sweep must look like name=value,value")
    name, raw_values = sweep.split("=", 1)
    if name not in {"diversity", "repulsion", "sparse", "gossip", "gossip_tau", "gossip_k"}:
        raise ValueError(
            "--sweep name must be diversity, repulsion, sparse, gossip, gossip_tau, or gossip_k"
        )
    values = [float(value) for value in raw_values.split(",") if value]
    if not values:
        raise ValueError("--sweep must include at least one value")
    return name, values


def build_variants(
    names: list[str],
    diversity: float | None,
    repulsion: float | None,
    sparse: float | None,
    gossip: float | None = None,
    gossip_tau: float | None = None,
    gossip_k: int | None = None,
    max_gossip_vectors: int | None = None,
    sweep: str | None = None,
) -> list[Variant]:
    """Build concrete variants from names, overrides, and optional sweep."""
    unknown = sorted(set(names) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    parsed_sweep = parse_sweep(sweep)
    variants = []
    for name in names:
        base = variant_with_overrides(
            name,
            diversity=diversity,
            repulsion=repulsion,
            sparse=sparse,
            gossip=gossip,
            gossip_tau=gossip_tau,
            gossip_k=gossip_k,
            max_gossip_vectors=max_gossip_vectors,
        )
        if parsed_sweep is None or name == "baseline":
            variants.append(base)
            continue
        sweep_name, values = parsed_sweep
        if not variant_accepts_sweep(name, sweep_name):
            variants.append(base)
            continue
        for value in values:
            variants.append(
                Variant(
                    name=f"{name}_{sweep_name}_{value:g}",
                    diversity=value if sweep_name == "diversity" else base.diversity,
                    repulsion=value if sweep_name == "repulsion" else base.repulsion,
                    sparse=value if sweep_name == "sparse" else base.sparse,
                    gossip=value if sweep_name == "gossip" else base.gossip,
                    gossip_tau=value if sweep_name == "gossip_tau" else base.gossip_tau,
                    gossip_k=int(value) if sweep_name == "gossip_k" else base.gossip_k,
                    max_gossip_vectors=base.max_gossip_vectors,
                )
            )
    return variants


def summarize_results(results: list[LoraExperimentResult]) -> dict[str, dict[str, float]]:
    """Aggregate mean and sample standard deviation per variant and metric."""
    metric_getters = {
        "eval_lm_loss": lambda result: result["eval_lm_loss"],
        "head_similarity": lambda result: result["head_similarity"],
        "effective_rank": lambda result: result["effective_rank"],
        "isotropy": lambda result: result["isotropy"],
        "active_fraction": lambda result: result["active_fraction"],
        "eval_diversity_loss": lambda result: result["eval_diversity_loss"],
        "eval_repulsion_loss": lambda result: result["eval_repulsion_loss"],
        "eval_sparse_loss": lambda result: result["eval_sparse_loss"],
        "eval_gossip_loss": lambda result: result["eval_gossip_loss"],
    }
    variants = sorted({result["variant"] for result in results})
    summary: dict[str, dict[str, float]] = {}
    for variant in variants:
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


def git_status() -> str:
    """Return concise git status for run provenance, or an empty string."""
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return completed.stdout.strip()


def write_run_record(record: Any, output_dir: str) -> Path:
    """Write a run record to a timestamped directory and return its path."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"
    record["config"]["result_path"] = str(result_path)
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return result_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `stt-lora`."""
    parser = argparse.ArgumentParser(
        description="Run LoRA Surface Tension Transformer experiments."
    )
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, or cuda")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
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
    parser.add_argument("--sweep", default=None, help="Dose sweep like repulsion=0,0.1,1.0")
    parser.add_argument(
        "--text-file",
        default=None,
        help="UTF-8 text file with one sample per line",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for persisted run records")
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "combined"],
        choices=list(VARIANTS),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for LoRA STT experiments."""
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
    seeds = args.seeds or [args.seed]
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
    texts = load_texts(args.text_file)
    results = run_lora_experiment(
        variants,
        settings=settings,
        steps=args.steps,
        seeds=seeds,
        device=args.device,
        texts=texts,
    )
    record: RunRecord = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "model": args.model,
            "device": args.device,
            "steps": args.steps,
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
            "text_file": args.text_file,
            "sweep": args.sweep,
            "gossip_weight": args.gossip_weight,
            "gossip_tau": args.gossip_tau,
            "gossip_k": args.gossip_k,
            "max_gossip_vectors": args.max_gossip_vectors,
        },
        "git_status": git_status(),
        "results": results,
        "summary": summarize_results(results),
    }
    if args.output_dir is not None:
        write_run_record(record, args.output_dir)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
