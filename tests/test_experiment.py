from stt.experiment import run_experiment


def test_experiment_smoke_run() -> None:
    results = run_experiment(["baseline", "combined"], steps=2, seed=0, device="cpu")

    assert [result["variant"] for result in results] == ["baseline", "combined"]
    assert all(result["eval_task_loss"] > 0 for result in results)
    assert all(result["effective_rank"] > 0 for result in results)
