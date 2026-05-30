import pytest
import torch

from stt.model import CompartmentalizedFeedForward, DendriticFeedForward, TinyTransformer


def test_compartmentalized_feed_forward_routes_top_one() -> None:
    module = CompartmentalizedFeedForward(dim=8, branches=4, top_k=1)
    x = torch.randn(2, 3, 8)

    output = module(x)

    assert output.hidden.shape == (2, 3, 8)
    assert output.gates.shape == (2, 3, 4)
    assert output.branch_outputs.shape == (2, 3, 4, 8)
    assert torch.allclose(output.gates.sum(dim=-1), torch.ones(2, 3))
    assert torch.equal((output.gates > 0).sum(dim=-1), torch.ones(2, 3, dtype=torch.long))


def test_dendritic_feed_forward_routes_top_one_without_central_router() -> None:
    module = DendriticFeedForward(dim=8, branches=4, top_k=1, inhibition_strength=0.5)
    x = torch.randn(2, 3, 8)

    output = module(x)

    assert not hasattr(module, "router")
    assert output.hidden.shape == (2, 3, 8)
    assert output.gates.shape == (2, 3, 4)
    assert output.branch_outputs.shape == (2, 3, 4, 8)
    assert output.branch_scores.shape == (2, 3, 4)
    assert output.branch_inhibition.shape == (2, 3, 4)
    assert torch.allclose(output.gates.sum(dim=-1), torch.ones(2, 3))
    assert torch.equal((output.gates > 0).sum(dim=-1), torch.ones(2, 3, dtype=torch.long))


def test_dendritic_feed_forward_rejects_negative_inhibition() -> None:
    with pytest.raises(ValueError, match="inhibition_strength"):
        DendriticFeedForward(dim=8, branches=4, inhibition_strength=-0.1)


def test_tiny_transformer_records_compartment_internals() -> None:
    model = TinyTransformer(
        vocab_size=16,
        seq_len=5,
        dim=12,
        heads=3,
        layers=2,
        compartments=4,
        compartment_top_k=1,
    )
    tokens = torch.randint(0, 16, (2, 5))

    output = model(tokens)

    assert output.logits.shape == (2, 5, 16)
    assert output.attention.shape == (2, 2, 3, 5, 5)
    assert output.branch_gates is not None
    assert output.branch_outputs is not None
    assert output.branch_scores is not None
    assert output.branch_inhibition is not None
    assert output.branch_gates.shape == (2, 2, 5, 4)
    assert output.branch_outputs.shape == (2, 2, 5, 4, 12)
    assert output.branch_scores.shape == (2, 2, 5, 4)
    assert output.branch_inhibition.shape == (2, 2, 5, 4)


def test_tiny_transformer_records_dendritic_internals() -> None:
    model = TinyTransformer(
        vocab_size=16,
        seq_len=5,
        dim=12,
        heads=3,
        layers=2,
        compartments=4,
        compartment_top_k=1,
        compartment_mode="dendritic",
        branch_inhibition_strength=0.5,
    )
    tokens = torch.randint(0, 16, (2, 5))

    output = model(tokens)

    assert output.branch_gates is not None
    assert output.branch_outputs is not None
    assert output.branch_scores is not None
    assert output.branch_inhibition is not None
    assert output.branch_gates.shape == (2, 2, 5, 4)
    assert output.branch_outputs.shape == (2, 2, 5, 4, 12)
    assert output.branch_scores.shape == (2, 2, 5, 4)
    assert output.branch_inhibition.shape == (2, 2, 5, 4)


def test_tiny_transformer_default_has_no_compartment_internals() -> None:
    model = TinyTransformer(vocab_size=16, seq_len=5, dim=12, heads=3, layers=1)
    tokens = torch.randint(0, 16, (2, 5))

    output = model(tokens)

    assert output.branch_gates is None
    assert output.branch_outputs is None
    assert output.branch_scores is None
    assert output.branch_inhibition is None
