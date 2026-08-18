from scripts.verify_population_baseline import DEFAULT_SEED, run_baseline


def test_population_baseline_is_deterministic_and_sparse() -> None:
    report = run_baseline(seed=DEFAULT_SEED, include_api=False)

    assert report["quality_scope"] == "synthetic_probe_only"
    assert report["checks"]["deterministic"] is True
    assert report["checks"]["sparse_router_engaged"] is True
    assert report["checks"]["sparse_activation_reduced"] is True
    assert report["checks"]["cortex_roundtrip_ok"] is True
    assert report["status"] == "pass"
