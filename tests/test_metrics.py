import torch

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
    isotropy,
)


def test_effective_rank_is_positive() -> None:
    hidden = torch.randn(4, 5, 6)

    assert effective_rank(hidden) > 0.0


def test_isotropy_is_at_least_one_for_nonzero_hidden() -> None:
    hidden = torch.randn(4, 5, 6)

    assert isotropy(hidden) >= 1.0


def test_active_fraction_counts_thresholded_values() -> None:
    hidden = torch.tensor([[[0.0, 0.1, -0.2, 0.01]]])

    assert active_fraction(hidden, threshold=0.05) == 0.5


def test_branch_metrics_detect_balanced_usage() -> None:
    collapsed = torch.zeros(1, 1, 8, 4)
    collapsed[..., 0] = 1.0
    balanced = torch.nn.functional.one_hot(torch.arange(8) % 4, num_classes=4).float()
    balanced = balanced.reshape(1, 1, 8, 4)

    assert branch_entropy(balanced) > branch_entropy(collapsed)
    assert branch_active_fraction(balanced) == 0.25
    assert branch_usage_min(balanced) == 0.25
    assert branch_usage_max(balanced) == 0.25
    assert branch_usage_std(balanced) == 0.0


def test_branch_score_entropy_detects_collapsed_scores() -> None:
    collapsed = torch.zeros(1, 1, 8, 4)
    collapsed[..., 0] = 10.0
    balanced = torch.zeros(1, 1, 8, 4)

    assert branch_score_entropy(balanced) > branch_score_entropy(collapsed)


def test_branch_inhibition_mean_reports_average_magnitude() -> None:
    inhibition = torch.tensor([[[0.0, 0.5, 1.0, 1.5]]])

    assert branch_inhibition_mean(inhibition) == 0.75
