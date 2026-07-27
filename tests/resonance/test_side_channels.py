import torch

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config


def test_establish_side_channel():
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg)
    peer_cfg = get_domain_neuron_config("zh", spec="compact")
    peer = ResonanceNeuron(peer_cfg)
    neuron.establish_side_channel("peer_0", peer, channel_type="excite")
    assert "peer_0" in neuron.excite_channels
    channel = neuron.excite_channels["peer_0"]
    assert channel.weight.shape == (cfg.hidden_size, peer.config.field_dim)


def test_establish_inhibit_channel():
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg)
    peer_cfg = get_domain_neuron_config("zh", spec="compact")
    peer = ResonanceNeuron(peer_cfg)
    neuron.establish_side_channel("peer_0", peer, channel_type="inhibit")
    assert "peer_0" in neuron.inhibit_channels


def test_forward_with_side_signals():
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg)
    peer_cfg = get_domain_neuron_config("zh", spec="compact")
    peer = ResonanceNeuron(peer_cfg)
    neuron.establish_side_channel("peer_0", peer, channel_type="excite")

    B, T = 2, 10
    x = torch.randn(B, T, cfg.embed_dim)
    side_signals = {
        "peer_0": torch.randn(B, peer.config.field_dim),
    }
    out = neuron.forward(x, side_signals=side_signals, return_logits=False)
    assert "hidden" in out
    assert out["hidden"].shape == (B, T, cfg.hidden_size)


def test_forward_without_side_signals():
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg)
    B, T = 2, 10
    x = torch.randn(B, T, cfg.embed_dim)
    out = neuron.forward(x, return_logits=False)
    assert "hidden" in out
    assert out["hidden"].shape == (B, T, cfg.hidden_size)
