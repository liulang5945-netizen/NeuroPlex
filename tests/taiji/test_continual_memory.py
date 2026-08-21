from scripts.training.verify_taiji_t5_continual_memory import run_benchmark


def test_twenty_one_shot_associations_retain_early_memories() -> None:
    report = run_benchmark(association_count=20, seed=20260821)

    assert report["status"] == "pass", report
    assert report["metrics"]["first_quarter_retention"] >= 0.70
    assert report["metrics"]["overall_retention"] >= 0.80
    assert report["metrics"]["memory_lesion_retention"] <= 0.30
    assert report["checks"]["slow_parameters_unchanged"] is True
