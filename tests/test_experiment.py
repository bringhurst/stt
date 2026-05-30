from stt.experiment import run_experiment


def test_experiment_smoke_run() -> None:
    results = run_experiment(["baseline", "combined"], steps=2, seed=0, device="cpu")

    assert [result["variant"] for result in results] == ["baseline", "combined"]
    assert all(result["eval_task_loss"] > 0 for result in results)
    assert all(result["effective_rank"] > 0 for result in results)


def test_experiment_compartment_smoke_run() -> None:
    results = run_experiment(
        ["baseline"],
        steps=2,
        seed=0,
        device="cpu",
        compartments=4,
        compartment_top_k=1,
        branch_repulsion_weight=0.01,
        branch_load_balance_weight=0.01,
    )

    result = results[0]
    assert result["compartments"] == 4
    assert result["compartment_top_k"] == 1
    assert result["branch_active_fraction"] == 0.25
    assert 0.0 <= result["branch_entropy"] <= 1.0
    assert 0.0 <= result["branch_score_entropy"] <= 1.0
    assert result["branch_inhibition_mean"] == 0.0
    assert result["branch_repulsion_loss"] != 0.0


def test_experiment_dendritic_smoke_run() -> None:
    results = run_experiment(
        ["baseline"],
        steps=2,
        seed=0,
        device="cpu",
        compartments=4,
        compartment_top_k=1,
        compartment_mode="dendritic",
        branch_repulsion_weight=0.01,
        branch_load_balance_weight=0.01,
        branch_inhibition_strength=0.5,
        branch_inhibition_weight=0.01,
    )

    result = results[0]
    assert result["compartment_mode"] == "dendritic"
    assert result["branch_inhibition_strength"] == 0.5
    assert result["branch_inhibition_weight"] == 0.01
    assert result["branch_active_fraction"] == 0.25
    assert 0.0 <= result["branch_entropy"] <= 1.0
    assert 0.0 <= result["branch_score_entropy"] <= 1.0
    assert result["branch_inhibition_mean"] >= 0.0
