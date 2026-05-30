"""Representation geometry metrics used by STT experiments."""

from __future__ import annotations

import torch
from torch import Tensor

from stt.losses import attention_diversity_loss


@torch.no_grad()
def head_similarity(attention: Tensor) -> float:
    """Return mean pairwise attention-head cosine similarity.

    Lower values indicate less redundant heads under this specific metric.
    """
    return float(attention_diversity_loss(attention).detach().cpu())


@torch.no_grad()
def effective_rank(hidden: Tensor, eps: float = 1e-8) -> float:
    """Estimate usable representation dimensionality from singular values.

    Effective rank is `exp(entropy(normalized_singular_values))`. It is high
    when variance is spread across many dimensions and low when hidden states
    collapse into a narrow subspace.
    """
    vectors = hidden.detach().float().cpu().reshape(-1, hidden.shape[-1])
    vectors = vectors - vectors.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(vectors)
    probabilities = singular_values / singular_values.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return float(torch.exp(entropy).detach().cpu())


@torch.no_grad()
def isotropy(hidden: Tensor, eps: float = 1e-8) -> float:
    """Return a simple anisotropy score from singular values.

    Values closer to 1 mean the singular spectrum is flatter. Larger values mean
    one or a few dominant directions carry disproportionate variance.
    """
    vectors = hidden.detach().float().cpu().reshape(-1, hidden.shape[-1])
    vectors = vectors - vectors.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(vectors)
    return float((singular_values.max() / singular_values.mean().clamp_min(eps)).detach().cpu())


@torch.no_grad()
def active_fraction(hidden: Tensor, threshold: float = 0.05) -> float:
    """Return the fraction of hidden activations above a magnitude threshold."""
    return float((hidden.abs() > threshold).float().mean().detach().cpu())


def branch_usage(branch_gates: Tensor) -> Tensor:
    """Return mean compartment usage from gates shaped `(..., branches)`."""
    if branch_gates.ndim < 2:
        raise ValueError("branch_gates must have shape (..., branches)")
    return branch_gates.detach().float().mean(dim=tuple(range(branch_gates.ndim - 1))).cpu()


@torch.no_grad()
def branch_entropy(branch_gates: Tensor, eps: float = 1e-8) -> float:
    """Return normalized entropy of average compartment usage."""
    usage = branch_usage(branch_gates)
    if usage.numel() < 2:
        return 0.0
    probabilities = usage / usage.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return float((entropy / torch.log(torch.tensor(float(usage.numel())))).detach().cpu())


@torch.no_grad()
def branch_active_fraction(branch_gates: Tensor, threshold: float = 1e-6) -> float:
    """Return the fraction of branch gates active per token/layer."""
    if branch_gates.ndim < 2:
        raise ValueError("branch_gates must have shape (..., branches)")
    return float((branch_gates.detach().abs() > threshold).float().mean().cpu())


@torch.no_grad()
def branch_usage_min(branch_gates: Tensor) -> float:
    """Return minimum mean compartment usage."""
    return float(branch_usage(branch_gates).min().cpu())


@torch.no_grad()
def branch_usage_max(branch_gates: Tensor) -> float:
    """Return maximum mean compartment usage."""
    return float(branch_usage(branch_gates).max().cpu())


@torch.no_grad()
def branch_usage_std(branch_gates: Tensor) -> float:
    """Return standard deviation of mean compartment usage."""
    usage = branch_usage(branch_gates)
    return float(usage.std(unbiased=False).cpu())


@torch.no_grad()
def branch_score_entropy(branch_scores: Tensor, eps: float = 1e-8) -> float:
    """Return normalized entropy of pre-top-k branch score probabilities."""
    if branch_scores.ndim < 2:
        raise ValueError("branch_scores must have shape (..., branches)")
    probabilities = torch.softmax(branch_scores.detach().float(), dim=-1)
    branches = probabilities.shape[-1]
    if branches < 2:
        return 0.0
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    normalized = entropy / torch.log(torch.tensor(float(branches), device=entropy.device))
    return float(normalized.mean().cpu())


@torch.no_grad()
def branch_inhibition_mean(branch_inhibition: Tensor) -> float:
    """Return mean branch inhibition magnitude before top-k selection."""
    if branch_inhibition.ndim < 2:
        raise ValueError("branch_inhibition must have shape (..., branches)")
    return float(branch_inhibition.detach().float().mean().cpu())
