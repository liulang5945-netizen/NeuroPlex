import torch

from taiji.resonance import (
    ResonanceEnsemble,
    ResonanceField,
    ResonanceNeuron,
    get_domain_neuron_config,
)


def test_ensemble_passes_side_signals():
    cfg = get_domain_neuron_config("zh", spec="compact")
    n0 = ResonanceNeuron(cfg)
    n1 = ResonanceNeuron(cfg)
    n0.establish_side_channel("n1", n1, "excite")
    n1.establish_side_channel("n0", n0, "excite")
    neurons = {"n0": n0, "n1": n1}
    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)

    B, T = 1, 5
    torch.manual_seed(0)
    emb = {
        "n0": torch.randn(B, T, 512),
        "n1": torch.randn(B, T, 512),
    }
    result = ensemble.forward(
        neuron_embeddings=emb, return_logits=True, fusion_mode="soft"
    )
    assert "neuron_logits" in result or "weighted_logits" in result


def test_side_signals_affect_round2():
    cfg = get_domain_neuron_config("zh", spec="compact")
    n0 = ResonanceNeuron(cfg)
    n1 = ResonanceNeuron(cfg)
    n0.establish_side_channel("n1", n1, "excite")
    n1.establish_side_channel("n0", n0, "excite")
    neurons = {"n0": n0, "n1": n1}
    field = ResonanceField(dim=cfg.field_dim)

    B, T = 1, 5
    torch.manual_seed(0)
    emb = {
        "n0": torch.randn(B, T, 512),
        "n1": torch.randn(B, T, 512),
    }

    ensemble1 = ResonanceEnsemble(neurons, field, max_rounds=1)
    result1 = ensemble1.forward(neuron_embeddings=emb, return_logits=True, fusion_mode="soft")

    ensemble2 = ResonanceEnsemble(neurons, field, max_rounds=2)
    result2 = ensemble2.forward(neuron_embeddings=emb, return_logits=True, fusion_mode="soft")

    # max_rounds=2 应该产生与 max_rounds=1 不同的 logits
    logits1 = result1["weighted_logits"] if "weighted_logits" in result1 else result1["neuron_logits"]["n0"]
    logits2 = result2["weighted_logits"] if "weighted_logits" in result2 else result2["neuron_logits"]["n0"]
    assert not torch.allclose(logits1, logits2, atol=1e-5)
