import argparse
from typing import cast

import pytest
import torch

from stt.lora_experiment import LoraSettings
from stt.memory_bank import (
    MemoryBank,
    MemoryBankResult,
    MemoryDelta,
    aggregate_domain_metrics,
    build_distilled_route_selector_from_scores,
    build_micro_probe_text,
    compose_memory_route_state,
    distilled_route_choices,
    evaluate_micro_probe_domain,
    expected_route_for_domain,
    generate_residual_routes,
    merge_route_exprs,
    parse_route_expr,
    probe_select_paths_from_args,
    probes_from_args,
    route_choice_from_scored,
    route_expr_string,
    route_optimality_audit,
    route_optimality_from_scores,
    route_variant_suffix,
    summarize_memory_bank,
)


def test_parse_route_expr_compiles_named_delta_scales() -> None:
    route = parse_route_expr("A+B+0.9C+0.4*D")

    assert route.expression == "A+B+0.9C+0.4D"
    assert route.scales == {"B": 1.0, "C": 0.9, "D": 0.4}
    assert route_expr_string(route) == "A+B+0.9C+0.4D"
    assert route_variant_suffix(route) == "A_B_0p9C_0p4D"


def test_merge_route_exprs_keeps_candidates_before_audit_routes() -> None:
    routes = [parse_route_expr("A+B"), parse_route_expr("A+C")]
    audit_routes = [parse_route_expr("A+0.5B+0.5C"), parse_route_expr("A+B")]

    merged = merge_route_exprs(routes, audit_routes)

    assert [route.expression for route in merged] == [
        "A+B",
        "A+C",
        "A+0.5B+0.5C",
    ]


def test_probes_from_args_normalizes_expected_routes() -> None:
    args = argparse.Namespace(
        probe_files=["data/memory_probe_gamma_boundary.txt"],
        probe_select_files=None,
        probe_names=None,
        probe_routes=["A+1.0*C"],
    )

    probes = probes_from_args(args, stable_phase="A")

    assert probes[0].name == "memory_probe_gamma_boundary"
    assert probes[0].expected_route == "A+C"


def test_probe_select_paths_from_args_requires_alignment() -> None:
    args = argparse.Namespace(
        probe_files=["data/memory_probe_gamma_boundary.txt"],
        probe_select_files=["data/memory_probe_select_gamma_boundary.txt", "extra.txt"],
        probe_names=None,
        probe_routes=["A+C"],
    )
    probes = probes_from_args(args, stable_phase="A")

    with pytest.raises(ValueError, match="probe-select-files"):
        probe_select_paths_from_args(args, probes)


def test_probe_select_paths_from_args_returns_aligned_files() -> None:
    args = argparse.Namespace(
        probe_files=["data/memory_probe_gamma_boundary.txt"],
        probe_select_files=["data/memory_probe_select_gamma_boundary.txt"],
        probe_names=["gamma_boundary"],
        probe_routes=["A+C"],
    )
    probes = probes_from_args(args, stable_phase="A")

    paths = probe_select_paths_from_args(args, probes)

    assert paths == ["data/memory_probe_select_gamma_boundary.txt"]


def test_generate_residual_routes_full_mode_expands_grid() -> None:
    base = parse_route_expr("A+0.9B+0.4C")

    routes = generate_residual_routes(
        base,
        phases=["B", "C"],
        grid=[-0.2, 0.0, 0.2],
        mode="full",
    )

    assert len(routes) == 9
    assert routes[0].base_route_expr == "A+0.9B+0.4C"
    assert "A+0.7B+0.2C" in {candidate.route.expression for candidate in routes}
    assert "A+1.1B+0.6C" in {candidate.route.expression for candidate in routes}


def test_generate_residual_routes_axis_mode_deduplicates_base() -> None:
    base = parse_route_expr("A+0.9B+0.4C")

    routes = generate_residual_routes(
        base,
        phases=["B", "C"],
        grid=[-0.2, 0.0, 0.2],
        mode="axis",
    )

    assert len(routes) == 5
    assert [candidate.route.expression for candidate in routes].count("A+0.9B+0.4C") == 1


def test_generate_residual_routes_clamps_scales() -> None:
    base = parse_route_expr("A+0.1B+0.4C")

    routes = generate_residual_routes(
        base,
        phases=["B"],
        grid=[-0.4, 0.0, 2.0],
        mode="full",
        min_scale=0.0,
        max_scale=1.0,
    )

    assert [candidate.route.expression for candidate in routes] == [
        "A+0.4C",
        "A+0.1B+0.4C",
        "A+B+0.4C",
    ]
    assert routes[0].residual["B"] == -0.1


def test_route_choice_marks_low_margin_as_abstained() -> None:
    choice = route_choice_from_scored(
        [("A+C", 1.0), ("A+B", 1.01)],
        ambiguity_margin=0.02,
    )

    assert choice.selected_route == "uncertain"
    assert choice.eval_route == "A+C"
    assert choice.ambiguous
    assert choice.abstained


def test_route_optimality_audit_reports_selected_and_expected_gaps(monkeypatch) -> None:
    calls = []

    def fake_score_route_states(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        if index == 0:
            return [("A+B", 1.0), ("A+C", 1.4), ("A", 1.8)]
        return [("A+C", 2.0), ("A+B", 2.5), ("A", 3.0)]

    monkeypatch.setattr("stt.memory_bank.score_route_states", fake_score_route_states)
    encoded = {
        "input_ids": torch.tensor([[1], [2]]),
        "attention_mask": torch.tensor([[1], [1]]),
        "labels": torch.tensor([[1], [2]]),
    }

    audit = route_optimality_audit(
        model=torch.nn.Linear(1, 1),
        route_states={"A": {}, "A+B": {}, "A+C": {}},
        encoded=encoded,
        settings=LoraSettings(model_name="tiny", batch_size=1, eval_batches=1),
        expected_route="A+B",
        selected_eval_routes=["A+C"],
    )

    assert audit["best_route_counts"] == {"A+B": 1, "A+C": 1}
    assert audit["most_best_route"] == "A+C"
    assert audit["best_eval_loss"] == 1.5
    assert round(float(audit["selected_loss_gap"]), 4) == 0.2
    assert round(float(audit["expected_loss_gap"]), 4) == 0.25
    assert audit["optimal_route_count"] == 1
    assert audit["optimal_route_rate"] == 0.5


def test_route_optimality_from_scores_reports_top_k_and_abstention_metrics() -> None:
    audit = route_optimality_from_scores(
        [
            [("A+B", 1.0), ("A+C", 1.01), ("A", 1.3)],
            [("A+C", 2.0), ("A+B", 2.2), ("A", 2.5)],
        ],
        expected_route="A+B",
        selected_eval_routes=["A+C", "A+C"],
        selected_routes=["uncertain", "A+C"],
        top_k=2,
        ambiguity_margin=0.02,
    )

    assert audit["top_k_route_counts"] == {"A+B": 2, "A+C": 2}
    assert audit["selected_top_k_rate"] == 1.0
    assert audit["expected_top_k_rate"] == 1.0
    assert round(float(audit["top_k_loss_gap"]), 4) == 0.105
    assert round(float(audit["top_k_boundary_margin"]), 4) == 0.295
    assert audit["low_margin_count"] == 1
    assert audit["low_margin_rate"] == 0.5
    assert audit["ambiguity_abstention_rate"] == 1.0
    assert audit["false_confident_route_rate"] == 0.5


def test_distilled_route_selector_predicts_without_default_abstention() -> None:
    select_encoded = {
        "input_ids": torch.tensor(
            [
                [10, 101, 0],
                [10, 102, 0],
                [20, 201, 0],
                [20, 202, 0],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
            ]
        ),
        "labels": torch.tensor(
            [
                [10, 101, -100],
                [10, 102, -100],
                [20, 201, -100],
                [20, 202, -100],
            ]
        ),
    }
    scored_examples = [
        [("A+B", 1.0), ("A+C", 2.0)],
        [("A+B", 1.0), ("A+C", 2.0)],
        [("A+C", 1.0), ("A+B", 2.0)],
        [("A+C", 1.0), ("A+B", 2.0)],
    ]
    selector = build_distilled_route_selector_from_scores(select_encoded, scored_examples)
    report_encoded = {
        "input_ids": torch.tensor([[10, 999, 0], [20, 999, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
        "labels": torch.tensor([[10, 999, -100], [20, 999, -100]]),
    }

    choices = distilled_route_choices(selector, report_encoded)

    assert [choice.selected_route for choice in choices] == ["A+B", "A+C"]
    assert not any(choice.abstained for choice in choices)


def test_distilled_route_selector_can_abstain_with_explicit_margin() -> None:
    encoded = {
        "input_ids": torch.tensor([[10, 0], [20, 0], [10, 20]]),
        "attention_mask": torch.tensor([[1, 0], [1, 0], [1, 1]]),
        "labels": torch.tensor([[10, -100], [20, -100], [10, 20]]),
    }
    scored_examples = [
        [("A+B", 1.0), ("A+C", 2.0)],
        [("A+C", 1.0), ("A+B", 2.0)],
        [("A+B", 1.0), ("A+C", 1.1)],
    ]
    selector = build_distilled_route_selector_from_scores(
        encoded,
        scored_examples,
        selector_margin=2.0,
    )

    choices = distilled_route_choices(selector, encoded)

    assert {choice.selected_route for choice in choices} == {"uncertain"}
    assert all(choice.abstained for choice in choices)


def test_distilled_knn_selector_uses_nearest_oracle_examples() -> None:
    select_encoded = {
        "input_ids": torch.tensor([[10, 101, 0], [20, 201, 0], [30, 301, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0]]),
        "labels": torch.tensor([[10, 101, -100], [20, 201, -100], [30, 301, -100]]),
    }
    scored_examples = [
        [("A+B", 1.0), ("A+C", 2.0)],
        [("A+C", 1.0), ("A+B", 2.0)],
        [("A+E", 1.0), ("A+B", 2.0)],
    ]
    selector = build_distilled_route_selector_from_scores(
        select_encoded,
        scored_examples,
        method="knn",
        knn_k=1,
    )
    report_encoded = {
        "input_ids": torch.tensor([[20, 999, 0], [30, 999, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]]),
        "labels": torch.tensor([[20, 999, -100], [30, 999, -100]]),
    }

    choices = distilled_route_choices(selector, report_encoded)

    assert [choice.selected_route for choice in choices] == ["A+C", "A+E"]
    assert not any(choice.abstained for choice in choices)


def test_build_micro_probe_text_strips_answer_suffix() -> None:
    probe = build_micro_probe_text(
        "Gamma boundary probe answer should say SQLite WAL and forbid Redis.",
        prefix_words=4,
        template="Route selector probe: {prefix}.",
    )

    assert probe == "Route selector probe: Gamma boundary probe."
    assert "SQLite" not in probe
    assert "Redis" not in probe


def test_build_micro_probe_text_strips_expected_route_suffix() -> None:
    probe = build_micro_probe_text(
        "Gamma boundary probe expected semantic route is A+C.",
        prefix_words=5,
        template="{prefix}",
    )

    assert probe == "Gamma boundary probe"
    assert "A+C" not in probe


def test_build_micro_probe_text_can_add_scope_answer_without_answer_leakage() -> None:
    probe = build_micro_probe_text(
        "Gamma boundary probe answer should say SQLite WAL and forbid Redis.",
        prefix_words=4,
        template="Question: {prefix}. Answer scope: {scope}.",
    )

    assert probe == "Question: Gamma boundary probe. Answer scope: Project Gamma."
    assert "SQLite" not in probe
    assert "Redis" not in probe


def test_build_micro_probe_text_marks_ambiguous_scope() -> None:
    probe = build_micro_probe_text(
        "Ambiguous scope probe asks which project cache should be used.",
        prefix_words=4,
        template="Answer scope: {scope}.",
    )

    assert probe == "Answer scope: scope unclear."


def test_evaluate_micro_probe_domain_selects_on_prefix_and_evals_report(
    monkeypatch,
) -> None:
    micro_encoded = {
        "input_ids": torch.tensor([[1], [2]]),
        "attention_mask": torch.tensor([[1], [1]]),
        "labels": torch.tensor([[1], [2]]),
    }
    report_encoded = {
        "input_ids": torch.tensor([[10], [20]]),
        "attention_mask": torch.tensor([[1], [1]]),
        "labels": torch.tensor([[10], [20]]),
    }
    calls = []

    def fake_route_scores_for_examples(*args, **kwargs):
        calls.append("select")
        assert args[2] is micro_encoded
        return [
            [("A+C", 1.0), ("A+B", 2.0)],
            [("A+B", 1.0), ("A+C", 2.0)],
        ]

    def fake_selected_route_eval_loss(*args, **kwargs):
        calls.append("eval")
        assert args[2] is report_encoded
        assert args[4] == ["A+C", "A+B"]
        return 3.5

    monkeypatch.setattr(
        "stt.memory_bank.route_scores_for_examples",
        fake_route_scores_for_examples,
    )
    monkeypatch.setattr(
        "stt.memory_bank.selected_route_eval_loss",
        fake_selected_route_eval_loss,
    )

    result = evaluate_micro_probe_domain(
        model=torch.nn.Linear(1, 1),
        route_states={"A+B": {}, "A+C": {}},
        micro_probe_encoded=micro_encoded,
        report_encoded=report_encoded,
        settings=LoraSettings(model_name="tiny", batch_size=1, eval_batches=1),
        expected_route="A+C",
        sequential_eval_loss=5.0,
        learned_eval_loss=2.0,
        initial_eval_loss=6.0,
        micro_probe_margin=0.0,
        ambiguity_margin=0.02,
        audit_scores=[
            [("A+C", 1.0), ("A+B", 2.0)],
            [("A+B", 1.0), ("A+C", 2.0)],
        ],
    )

    assert calls == ["select", "eval"]
    assert result["eval_loss"] == 3.5
    assert result["selected_route_counts"] == {"A+C": 1, "A+B": 1}
    assert result["route_accuracy"] == 0.5
    assert result["selected_loss_gap"] == 0.0


def test_compose_memory_route_state_adds_scaled_deltas() -> None:
    bank = MemoryBank(
        stable_phase="A",
        stable_adapter={"w": torch.tensor([1.0]), "b": torch.tensor([2.0])},
        deltas={
            "B": MemoryDelta(
                name="B",
                from_phase="A",
                to_phase="B",
                tensors={"w": torch.tensor([10.0]), "b": torch.tensor([20.0])},
                metadata={},
            ),
            "C": MemoryDelta(
                name="C",
                from_phase="B",
                to_phase="C",
                tensors={"w": torch.tensor([100.0]), "b": torch.tensor([200.0])},
                metadata={},
            ),
        },
    )

    composed = compose_memory_route_state(bank, parse_route_expr("A+0.5B+0.25C"))

    assert composed["w"].item() == 31.0
    assert composed["b"].item() == 62.0


def test_expected_route_prefers_related_b_for_a_and_direct_conflict_routes() -> None:
    routes = ["A", "A+B", "A+C", "A+D"]

    assert expected_route_for_domain("A", ["A", "B", "C", "D"], routes) == "A+B"
    assert expected_route_for_domain("B", ["A", "B", "C", "D"], routes) == "A+B"
    assert expected_route_for_domain("C", ["A", "B", "C", "D"], routes) == "A+C"
    assert expected_route_for_domain("D", ["A", "B", "C", "D"], routes) == "A+D"


def test_aggregate_domain_metrics_weights_by_selection_count() -> None:
    per_domain = {
        "A": {
            "selected_route_counts": {"A+B": 2},
            "expected_route": "A+B",
            "most_selected_route": "A+B",
            "selection_count": 2,
            "route_accuracy": 1.0,
            "ambiguous_count": 0,
            "ambiguous_rate": 0.0,
            "eval_loss": 1.0,
            "sequential_eval_loss": 1.5,
            "learned_eval_loss": 0.8,
            "initial_eval_loss": 2.0,
            "loss_delta_vs_sequential": 0.5,
            "learning_retained": 0.8,
            "interference": 0.2,
        },
        "C": {
            "selected_route_counts": {"A+C": 1},
            "expected_route": "A+C",
            "most_selected_route": "A+C",
            "selection_count": 1,
            "route_accuracy": 1.0,
            "ambiguous_count": 0,
            "ambiguous_rate": 0.0,
            "eval_loss": 0.7,
            "sequential_eval_loss": 0.8,
            "learned_eval_loss": 0.6,
            "initial_eval_loss": 1.2,
            "loss_delta_vs_sequential": 0.1,
            "learning_retained": 0.83,
            "interference": 0.1,
        },
    }

    metrics = aggregate_domain_metrics(per_domain)

    assert round(metrics["contextual_eval_loss"], 4) == 0.9
    assert round(metrics["sequential_eval_loss"], 4) == 1.2667
    assert round(metrics["frontier_score"], 4) == 0.3667


def test_summarize_memory_bank_reports_domains_and_wins() -> None:
    result = cast(MemoryBankResult, {
        "variant": "gossip_contextual_memory_bank_oracle",
        "heldout_report": True,
        "contextual_eval_loss": 1.0,
        "sequential_eval_loss": 1.5,
        "loss_delta_vs_sequential": 0.5,
        "mean_learning_retained": 0.9,
        "mean_interference": 0.1,
        "route_accuracy": 1.0,
        "ambiguous_rate": 0.0,
        "frontier_score": 0.5,
        "per_domain": {
            "A": {
                "selected_route_counts": {"A+B": 2},
                "expected_route": "A+B",
                "most_selected_route": "A+B",
                "selection_count": 2,
                "route_accuracy": 1.0,
                "ambiguous_count": 0,
                "ambiguous_rate": 0.0,
                "eval_loss": 1.0,
                "sequential_eval_loss": 1.5,
                "learned_eval_loss": 0.8,
                "initial_eval_loss": 2.0,
                "loss_delta_vs_sequential": 0.5,
                "learning_retained": 0.83,
                "interference": 0.2,
            }
        },
    })

    summary = summarize_memory_bank([result])

    values = summary["gossip_contextual_memory_bank_oracle"]
    assert values["contextual_win_count"] == 1.0
    assert values["route_accuracy_mean"] == 1.0
    assert values["domain_A_eval_loss_mean"] == 1.0
    assert values["domain_A_learning_retained_mean"] == 0.83


def test_summarize_memory_bank_reports_probe_metrics() -> None:
    result = cast(MemoryBankResult, {
        "variant": "gossip_contextual_memory_bank_calibration",
        "heldout_report": True,
        "contextual_eval_loss": 1.0,
        "sequential_eval_loss": 1.5,
        "loss_delta_vs_sequential": 0.5,
        "mean_learning_retained": 0.9,
        "mean_interference": 0.1,
        "route_accuracy": 1.0,
        "ambiguous_rate": 0.0,
        "optimal_route_rate": 0.8,
        "selected_loss_gap": 0.2,
        "expected_loss_gap": 0.3,
        "probe_eval_loss": 0.7,
        "probe_route_accuracy": 1.0,
        "probe_ambiguous_rate": 0.0,
        "probe_optimal_route_rate": 1.0,
        "probe_selected_loss_gap": 0.0,
        "probe_expected_loss_gap": 0.0,
        "frontier_score": 0.5,
        "per_domain": {},
        "per_probe": {
            "gamma_boundary": {
                "selected_route_counts": {"A+C": 2},
                "expected_route": "A+C",
                "most_selected_route": "A+C",
                "selection_count": 2,
                "route_accuracy": 1.0,
                "ambiguous_count": 0,
                "ambiguous_rate": 0.0,
                "eval_loss": 0.7,
                "sequential_eval_loss": 1.0,
                "learned_eval_loss": 0.7,
                "initial_eval_loss": 1.2,
                "loss_delta_vs_sequential": 0.3,
                "learning_retained": 1.0,
                "interference": 0.0,
                "optimal_route_rate": 1.0,
                "selected_loss_gap": 0.0,
                "expected_loss_gap": 0.0,
            }
        },
    })

    summary = summarize_memory_bank([result])

    values = summary["gossip_contextual_memory_bank_calibration"]
    assert values["probe_eval_loss_mean"] == 0.7
    assert values["probe_route_accuracy_mean"] == 1.0
    assert values["probe_gamma_boundary_eval_loss_mean"] == 0.7
