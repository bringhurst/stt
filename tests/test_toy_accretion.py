import torch

from stt.toy_accretion import (
    MarkedSequenceTask,
    condition_name,
    run_toy_accretion,
    summarize_toy_accretion,
)


def test_marked_sequence_task_uses_marker_and_target_offset() -> None:
    task_a = MarkedSequenceTask("a", marker_token=0, target_offset=0, seed=0)
    task_b = MarkedSequenceTask("b", marker_token=1, target_offset=1, seed=0)

    tokens_a, targets_a = task_a.batch(4, eval_seed=123)
    tokens_b, targets_b = task_b.batch(4, eval_seed=123)

    assert torch.equal(tokens_a[:, 1:], tokens_b[:, 1:])
    assert torch.equal(tokens_a[:, 0], torch.zeros(4, dtype=torch.long))
    assert torch.equal(tokens_b[:, 0], torch.ones(4, dtype=torch.long))
    shifted_targets_a = (targets_a - task_a.marker_count + 1) % task_a.content_vocab_size

    assert torch.equal(shifted_targets_a, targets_b - task_b.marker_count)


def test_condition_name_labels_dense_and_compartment_modes() -> None:
    assert condition_name(0, 1, "router") == "dense"
    assert condition_name(4, 1, "router") == "router_top1"
    assert condition_name(4, 2, "dendritic") == "dendritic_top2"


def test_toy_accretion_dense_smoke_run() -> None:
    results = run_toy_accretion(["baseline"], phase_steps=1, seed=0, device="cpu")

    result = results[0]
    assert result["condition"] == "dense"
    assert result["eval_a_after_c"] > 0.0
    assert result["learning_a"] != 0.0
    assert result["branch_entropy"] == 0.0


def test_toy_accretion_dendritic_smoke_run() -> None:
    results = run_toy_accretion(
        ["baseline"],
        phase_steps=1,
        seed=0,
        device="cpu",
        compartments=4,
        compartment_top_k=2,
        compartment_mode="dendritic",
        branch_load_balance_weight=0.05,
        branch_inhibition_weight=0.01,
    )

    result = results[0]
    assert result["condition"] == "dendritic_top2"
    assert result["branch_active_fraction"] == 0.5
    assert 0.0 <= result["branch_entropy"] <= 1.0
    assert result["branch_inhibition_mean"] >= 0.0


def test_summarize_toy_accretion_groups_by_condition_and_variant() -> None:
    results = run_toy_accretion(["baseline"], phase_steps=1, seed=0, device="cpu")

    summary = summarize_toy_accretion(results)

    assert summary["dense:baseline"]["count"] == 1.0
    assert summary["dense:baseline"]["eval_a_after_c_mean"] == results[0]["eval_a_after_c"]
