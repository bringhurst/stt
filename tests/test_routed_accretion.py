from typing import cast

from stt.routed_accretion import RoutedAccretionResult, summarize_routed_accretion


def test_summarize_routed_accretion_reports_wins() -> None:
    result = cast(RoutedAccretionResult, {
        "variant": "gossip",
        "route_b_scale": 0.9,
        "route_c_scale": 0.25,
        "sequential_eval_a": 1.4,
        "sequential_eval_b": 1.5,
        "sequential_eval_c": 0.2,
        "routed_eval_a": 1.0,
        "routed_eval_b": 1.0,
        "routed_eval_c": 0.6,
        "sequential_accretion_a": 0.1,
        "sequential_interference_a": 0.4,
        "sequential_interference_b": 0.5,
        "sequential_learning_b": 2.0,
        "sequential_learning_c": 1.0,
        "sequential_retention_a": 0.7,
        "sequential_retention_b": 0.8,
        "routed_accretion_a": 0.2,
        "routed_interference_a": 0.01,
        "routed_interference_b": 0.02,
        "routed_learning_b": 1.9,
        "routed_learning_c": 1.5,
        "routed_retention_a": 0.9,
        "routed_retention_b": 0.95,
    })

    summary = summarize_routed_accretion([result])

    assert summary["gossip"]["routed_accretion_a_mean"] == 0.2
    assert summary["gossip"]["routed_accretion_win_count"] == 1.0
    assert summary["gossip"]["routed_interference_a_win_count"] == 1.0
    assert summary["gossip"]["routed_learning_c_preserved_count"] == 1.0
