import torch

from neuroplex.resonance import ResonanceEnsemble, ResonanceField
from scripts.verify_population_baseline import (
    DEFAULT_SEED,
    _make_inputs,
    _make_population,
    run_baseline,
)


def test_population_baseline_is_deterministic_and_sparse() -> None:
    report = run_baseline(seed=DEFAULT_SEED, include_api=False)

    assert report["quality_scope"] == "synthetic_probe_only"
    assert report["checks"]["deterministic"] is True
    assert report["checks"]["sparse_router_engaged"] is True
    assert report["checks"]["sparse_activation_reduced"] is True
    assert report["checks"]["cortex_roundtrip_ok"] is True
    assert report["status"] == "pass"


def test_forward_train_without_targets_is_a_valid_probe_path() -> None:
    population = _make_population(DEFAULT_SEED)
    embeddings, _ = _make_inputs(DEFAULT_SEED)
    ensemble = ResonanceEnsemble(
        population,
        ResonanceField(dim=16),
        max_rounds=1,
    )

    with torch.inference_mode():
        result = ensemble.forward_train(
            shared_embeddings=embeddings,
            n_rounds=1,
            fusion_mode="soft",
        )

    assert torch.isfinite(result["fused_logits"]).all()
    assert torch.isfinite(result["contrastive_loss"])
