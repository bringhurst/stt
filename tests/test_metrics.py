import torch

from stt.metrics import active_fraction, effective_rank, isotropy


def test_effective_rank_is_positive() -> None:
    hidden = torch.randn(4, 5, 6)

    assert effective_rank(hidden) > 0.0


def test_isotropy_is_at_least_one_for_nonzero_hidden() -> None:
    hidden = torch.randn(4, 5, 6)

    assert isotropy(hidden) >= 1.0


def test_active_fraction_counts_thresholded_values() -> None:
    hidden = torch.tensor([[[0.0, 0.1, -0.2, 0.01]]])

    assert active_fraction(hidden, threshold=0.05) == 0.5
