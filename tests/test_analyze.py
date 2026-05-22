from stt.analyze import (
    aggregate_accretion_predictors,
    aggregate_continual_records,
    aggregate_oracle_records,
    aggregate_routed_records,
    analyze_accretion_record,
    analyze_continual_record,
    analyze_oracle_record,
    analyze_record,
    analyze_routed_record,
    is_accretion_record,
    is_continual_record,
    is_oracle_record,
    is_routed_record,
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


def test_aggregate_continual_records_combines_runs_and_pairs_by_seed() -> None:
    first = {
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
                "variant": "gossip",
                "seed": 0,
                "backward_transfer_a": 0.2,
                "learning_b": 5.1,
                "eval_b_after_b": 0.4,
                "retention_ratio": 0.9,
            },
        ]
    }
    second = {
        "results": [
            {
                "variant": "baseline",
                "seed": 1,
                "backward_transfer_a": 0.5,
                "learning_b": 4.0,
                "eval_b_after_b": 0.6,
                "retention_ratio": 0.7,
            },
            {
                "variant": "gossip",
                "seed": 1,
                "backward_transfer_a": 0.4,
                "learning_b": 4.1,
                "eval_b_after_b": 0.55,
                "retention_ratio": 0.75,
            },
        ]
    }

    lines = aggregate_continual_records([first, second], max_learning_b_delta=10.0)

    assert lines[0] == "combined_continual_records=2 seeds=0,1"
    assert any("gossip backward_transfer_a 0.3000 -25.00% yes" in line for line in lines)
    assert any("gossip learning_b 4.6000 +2.22% yes" in line for line in lines)
    assert any("gossip backward_transfer_a -0.1000 [-0.1000,-0.1000]" in line for line in lines)


def test_analyze_accretion_record_reports_accretion_and_interference() -> None:
    record = {
        "summary": {
            "baseline": {
                "accretion_a_after_b_mean": 0.1,
                "interference_a_after_c_mean": 0.2,
                "interference_b_after_c_mean": 0.3,
                "learning_b_mean": 1.0,
                "learning_c_mean": 1.0,
                "retention_a_after_c_mean": 0.8,
                "retention_b_after_c_mean": 0.9,
            },
            "gossip": {
                "accretion_a_after_b_mean": 0.2,
                "interference_a_after_c_mean": 0.1,
                "interference_b_after_c_mean": 0.2,
                "learning_b_mean": 1.0,
                "learning_c_mean": 0.99,
                "retention_a_after_c_mean": 0.9,
                "retention_b_after_c_mean": 0.95,
            },
        },
        "results": [
            {
                "variant": "baseline",
                "seed": 0,
                "accretion_a_after_b": 0.1,
                "interference_a_after_c": 0.2,
                "interference_b_after_c": 0.3,
                "learning_b": 1.0,
                "learning_c": 1.0,
                "retention_a_after_c": 0.8,
                "retention_b_after_c": 0.9,
            },
            {
                "variant": "gossip",
                "seed": 0,
                "accretion_a_after_b": 0.2,
                "interference_a_after_c": 0.1,
                "interference_b_after_c": 0.2,
                "learning_b": 1.0,
                "learning_c": 0.99,
                "retention_a_after_c": 0.9,
                "retention_b_after_c": 0.95,
            },
        ],
    }

    lines = analyze_accretion_record(record, max_learning_c_delta=2.0)

    assert is_accretion_record(record)
    assert any("gossip accretion_a_after_b" in line and "yes" in line for line in lines)
    assert any("gossip interference_a_after_c" in line and "yes" in line for line in lines)
    assert any("gossip accretion_a_after_b +0.1000" in line for line in lines)


def test_aggregate_accretion_predictors_correlates_paired_deltas() -> None:
    first = {
        "config": {"task_b_file": "data/accretion_task_b_related.txt"},
        "summary": {"baseline": {"accretion_a_after_b_mean": 0.1}},
        "results": [
            {
                "variant": "baseline",
                "seed": 0,
                "lora_cosine_a_b_mean": 0.10,
                "accretion_a_after_b": 0.10,
                "retention_a_after_c": 0.70,
                "learning_b": 1.00,
                "learning_c": 1.00,
            },
            {
                "variant": "gossip",
                "seed": 0,
                "lora_cosine_a_b_mean": 0.12,
                "accretion_a_after_b": 0.12,
                "retention_a_after_c": 0.72,
                "learning_b": 0.99,
                "learning_c": 1.05,
            },
            {
                "variant": "baseline",
                "seed": 1,
                "lora_cosine_a_b_mean": 0.20,
                "accretion_a_after_b": 0.10,
                "retention_a_after_c": 0.70,
                "learning_b": 1.00,
                "learning_c": 1.00,
            },
            {
                "variant": "gossip",
                "seed": 1,
                "lora_cosine_a_b_mean": 0.26,
                "accretion_a_after_b": 0.16,
                "retention_a_after_c": 0.76,
                "learning_b": 1.01,
                "learning_c": 1.10,
            },
        ],
    }
    second = {
        "config": {"task_b_file": "data/accretion_task_b_rehearsal.txt"},
        "summary": {"baseline": {"accretion_a_after_b_mean": 0.1}},
        "results": [
            {
                "variant": "baseline",
                "seed": 2,
                "lora_cosine_a_b_mean": 0.10,
                "accretion_a_after_b": 0.10,
                "retention_a_after_c": 0.70,
                "learning_b": 1.00,
                "learning_c": 1.00,
            },
            {
                "variant": "gossip",
                "seed": 2,
                "lora_cosine_a_b_mean": 0.20,
                "accretion_a_after_b": 0.20,
                "retention_a_after_c": 0.80,
                "learning_b": 1.03,
                "learning_c": 1.15,
            },
        ],
    }

    lines = aggregate_accretion_predictors([first, second])

    assert lines[0].startswith("combined_accretion_records=2")
    assert "accretion_task_b_related" in lines[0]
    assert any("all accretion_a_after_b 3 +1.0000 +1.0000" in line for line in lines)
    assert any("variant:gossip retention_a_after_c 3 +1.0000 +1.0000" in line for line in lines)
    assert any(
        "loo_without:accretion_task_b_rehearsal accretion_a_after_b" in line
        for line in lines
    )
    assert any("centered:condition_variant accretion_a_after_b" in line for line in lines)


def test_analyze_oracle_record_reports_methods_and_win_counts() -> None:
    record = {
        "config": {"task_b_file": "data/accretion_task_b_related.txt"},
        "summary": {
            "gossip": {
                "count": 3.0,
                "sequential_accretion_a_after_b_mean": 0.1,
                "sequential_interference_a_after_c_mean": 0.4,
                "sequential_interference_b_after_c_mean": 0.5,
                "sequential_learning_b_mean": 2.0,
                "sequential_learning_c_mean": 1.0,
                "sequential_eval_c_mean": 0.2,
                "fixed_accretion_a_mean": 0.2,
                "fixed_interference_a_mean": 0.01,
                "fixed_interference_b_mean": 0.02,
                "fixed_learning_b_mean": 1.9,
                "fixed_learning_c_mean": 1.5,
                "fixed_eval_c_mean": 0.6,
                "fixed_accretion_win_count": 3.0,
                "fixed_interference_a_win_count": 3.0,
                "fixed_interference_b_win_count": 3.0,
                "fixed_learning_c_preserved_count": 3.0,
                "oracle_accretion_a_mean": 0.25,
                "oracle_interference_a_mean": -0.01,
                "oracle_interference_b_mean": 0.0,
                "oracle_learning_b_mean": 1.8,
                "oracle_learning_c_mean": 1.4,
                "oracle_eval_c_mean": 0.7,
                "oracle_accretion_win_count": 3.0,
                "oracle_interference_a_win_count": 3.0,
                "oracle_interference_b_win_count": 3.0,
                "oracle_learning_c_preserved_count": 2.0,
            }
        },
    }

    lines = analyze_oracle_record(record)

    assert is_oracle_record(record)
    assert lines[0] == "oracle_record condition=accretion_task_b_related variants=gossip"
    assert "gossip fixed learning_c +1.5000" in lines
    assert (
        "oracle accretion=3/3 a_interference=3/3 b_interference=3/3 "
        "c_learning_preserved=2/3"
    ) in lines


def test_aggregate_oracle_records_reports_conditions() -> None:
    summary = {
        "gossip": {
            "count": 1.0,
            "sequential_accretion_a_after_b_mean": 0.1,
            "sequential_interference_a_after_c_mean": 0.4,
            "sequential_interference_b_after_c_mean": 0.5,
            "sequential_learning_b_mean": 2.0,
            "sequential_learning_c_mean": 1.0,
            "sequential_eval_c_mean": 0.2,
            "fixed_accretion_a_mean": 0.2,
            "fixed_interference_a_mean": 0.01,
            "fixed_interference_b_mean": 0.02,
            "fixed_learning_b_mean": 1.9,
            "fixed_learning_c_mean": 1.5,
            "fixed_eval_c_mean": 0.6,
            "oracle_accretion_a_mean": 0.25,
            "oracle_interference_a_mean": -0.01,
            "oracle_interference_b_mean": 0.0,
            "oracle_learning_b_mean": 1.8,
            "oracle_learning_c_mean": 1.4,
            "oracle_eval_c_mean": 0.7,
        }
    }
    first = {
        "config": {"task_b_file": "data/accretion_task_b_related.txt"},
        "summary": summary,
    }
    second = {
        "config": {"task_b_file": "data/accretion_task_b_rehearsal.txt"},
        "summary": summary,
    }

    lines = aggregate_oracle_records([first, second])

    assert lines[0] == "combined_oracle_records=2"
    assert any(line.startswith("accretion_task_b_related gossip fixed") for line in lines)
    assert any(line.startswith("accretion_task_b_rehearsal gossip oracle") for line in lines)


def test_analyze_routed_record_reports_methods_and_win_counts() -> None:
    record = {
        "config": {"task_b_file": "data/accretion_task_b_related.txt"},
        "summary": {
            "gossip": {
                "count": 3.0,
                "sequential_accretion_a_mean": 0.1,
                "sequential_interference_a_mean": 0.4,
                "sequential_interference_b_mean": 0.5,
                "sequential_learning_b_mean": 2.0,
                "sequential_learning_c_mean": 1.0,
                "sequential_eval_c_mean": 0.2,
                "routed_accretion_a_mean": 0.2,
                "routed_interference_a_mean": 0.01,
                "routed_interference_b_mean": 0.02,
                "routed_learning_b_mean": 1.9,
                "routed_learning_c_mean": 1.5,
                "routed_eval_c_mean": 0.6,
                "frontier_score_mean": 1.2,
                "route_b_scale_mean": 0.9,
                "route_c_scale_mean": 0.25,
                "routed_accretion_win_count": 2.0,
                "routed_interference_a_win_count": 3.0,
                "routed_interference_b_win_count": 3.0,
                "routed_learning_c_preserved_count": 3.0,
            }
        },
    }

    lines = analyze_routed_record(record)

    assert is_routed_record(record)
    assert lines[0] == "routed_record condition=accretion_task_b_related variants=gossip"
    assert "gossip routed learning_c +1.5000" in lines
    assert (
        "routed accretion=2/3 a_interference=3/3 b_interference=3/3 "
        "c_learning_preserved=3/3"
    ) in lines
    assert "gossip score=+1.2000 b=0.9 c=0.25" in lines


def test_aggregate_routed_records_reports_conditions() -> None:
    summary = {
        "gossip": {
            "count": 1.0,
            "sequential_accretion_a_mean": 0.1,
            "sequential_interference_a_mean": 0.4,
            "sequential_interference_b_mean": 0.5,
            "sequential_learning_b_mean": 2.0,
            "sequential_learning_c_mean": 1.0,
            "sequential_eval_c_mean": 0.2,
            "routed_accretion_a_mean": 0.2,
            "routed_interference_a_mean": 0.01,
            "routed_interference_b_mean": 0.02,
            "routed_learning_b_mean": 1.9,
            "routed_learning_c_mean": 1.5,
            "routed_eval_c_mean": 0.6,
            "frontier_score_mean": 1.2,
            "route_b_scale_mean": 0.9,
            "route_c_scale_mean": 0.25,
        }
    }
    first = {
        "config": {"task_b_file": "data/accretion_task_b_related.txt"},
        "summary": summary,
    }
    second = {
        "config": {"task_b_file": "data/accretion_task_b_rehearsal.txt"},
        "summary": summary,
    }

    lines = aggregate_routed_records([first, second])

    assert lines[0] == "combined_routed_records=2"
    assert any(line.startswith("accretion_task_b_related gossip routed") for line in lines)
    assert any(line.startswith("accretion_task_b_rehearsal gossip sequential") for line in lines)
    assert any("best_by_frontier" in line for line in lines)
