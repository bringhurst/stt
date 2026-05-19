from stt.analyze import (
    analyze_continual_record,
    analyze_record,
    is_continual_record,
    paired_continual_deltas,
)


def test_analyze_record_reports_baseline_relative_deltas() -> None:
    record = {
        "summary": {
            "baseline": {
                "eval_lm_loss_mean": 10.0,
                "head_similarity_mean": 1.0,
                "effective_rank_mean": 10.0,
                "isotropy_mean": 2.0,
                "active_fraction_mean": 1.0,
            },
            "repulsion": {
                "eval_lm_loss_mean": 10.5,
                "head_similarity_mean": 0.95,
                "effective_rank_mean": 12.0,
                "isotropy_mean": 1.7,
                "active_fraction_mean": 0.99,
            },
        }
    }

    lines = analyze_record(record, max_loss_delta=10.0, min_geometry_delta=10.0)

    assert lines[0].startswith("baseline=baseline")
    assert any("repulsion effective_rank" in line and "yes" in line for line in lines)
    assert any("repulsion isotropy" in line and "yes" in line for line in lines)


def test_analyze_continual_record_reports_forgetting_and_learning() -> None:
    record = {
        "summary": {
            "baseline": {
                "forgetting_a_mean": 0.2,
                "backward_transfer_a_mean": 0.2,
                "learning_b_mean": 1.0,
                "eval_b_after_b_mean": 2.0,
                "retention_ratio_mean": 0.9,
            },
            "repulsion": {
                "forgetting_a_mean": 0.1,
                "backward_transfer_a_mean": 0.1,
                "learning_b_mean": 0.95,
                "eval_b_after_b_mean": 1.9,
                "retention_ratio_mean": 0.95,
            },
        }
    }

    lines = analyze_continual_record(record, max_learning_b_delta=10.0)

    assert is_continual_record(record)
    assert lines[0].startswith("baseline=baseline")
    assert any("repulsion backward_transfer_a" in line and "yes" in line for line in lines)
    assert any("repulsion learning_b" in line and "yes" in line for line in lines)


def test_analyze_continual_handles_negative_backward_transfer() -> None:
    record = {
        "summary": {
            "baseline": {
                "forgetting_a_mean": -0.04,
                "backward_transfer_a_mean": -0.04,
                "learning_b_mean": 1.0,
                "eval_b_after_b_mean": 2.0,
                "retention_ratio_mean": 1.01,
            },
            "repulsion": {
                "forgetting_a_mean": -0.06,
                "backward_transfer_a_mean": -0.06,
                "learning_b_mean": 0.95,
                "eval_b_after_b_mean": 1.9,
                "retention_ratio_mean": 1.02,
            },
        }
    }

    lines = analyze_continual_record(record, max_learning_b_delta=10.0)

    assert any("repulsion backward_transfer_a" in line and "yes" in line for line in lines)


def test_paired_continual_deltas_compare_same_seed_baseline() -> None:
    record = {
        "results": [
            {
                "variant": "baseline",
                "seed": 0,
                "backward_transfer_a": 0.3,
                "learning_b": 5.0,
                "eval_b_after_b": 0.5,
                "retention_ratio": 0.8,
            },
            {
                "variant": "repulsion",
                "seed": 0,
                "backward_transfer_a": 0.2,
                "learning_b": 4.9,
                "eval_b_after_b": 0.55,
                "retention_ratio": 0.85,
            },
        ]
    }

    lines = paired_continual_deltas(record)

    assert any("repulsion backward_transfer_a -0.1000" in line for line in lines)
    assert any("repulsion learning_b -0.1000" in line for line in lines)
