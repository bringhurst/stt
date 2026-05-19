from typing import cast

from stt.accretion import AccretionResult, summarize_accretion
from stt.accretion_data import generate_lines, group_for, task_a_line, task_c_line


def test_accretion_data_is_deterministic_and_conflicting() -> None:
    first = generate_lines(num_entities=16, seed=0)
    second = generate_lines(num_entities=16, seed=0)

    assert first == second
    assert set(first) == {
        "accretion_task_a.txt",
        "accretion_task_b_related.txt",
        "accretion_task_c_conflict.txt",
        "accretion_task_n_neutral.txt",
    }
    assert "Entity-" in first["accretion_task_a.txt"][0]
    assert "Entity-" in first["accretion_task_b_related.txt"][0]
    assert "Entity-" in first["accretion_task_c_conflict.txt"][0]
    assert "Artifact-" in first["accretion_task_n_neutral.txt"][0]

    index = 0
    assert group_for(index) != group_for(index, offset=3)
    assert task_a_line(index) != task_c_line(index)


def test_summarize_accretion_metric_signs() -> None:
    result = cast(AccretionResult, {
        "variant": "baseline",
        "eval_a_after_a": 1.0,
        "eval_a_after_b": 0.9,
        "eval_a_after_c": 1.1,
        "eval_b_after_b": 0.8,
        "eval_b_after_c": 1.0,
        "eval_c_after_c": 0.7,
        "learning_a": 2.0,
        "learning_b": 1.5,
        "learning_c": 1.2,
        "accretion_a_after_b": 0.1,
        "backward_transfer_a_after_b": -0.1,
        "interference_a_after_c": 0.2,
        "interference_b_after_c": 0.2,
        "retention_a_after_b": 1.0 / 0.9,
        "retention_a_after_c": 1.0 / 1.1,
        "retention_b_after_c": 0.8 / 1.0,
    })

    summary = summarize_accretion([result])

    assert summary["baseline"]["accretion_a_after_b_mean"] == 0.1
    assert summary["baseline"]["backward_transfer_a_after_b_mean"] == -0.1
    assert summary["baseline"]["interference_a_after_c_mean"] == 0.2
