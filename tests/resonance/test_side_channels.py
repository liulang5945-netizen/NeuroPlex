import torch

from neuroplex.resonance import ResonanceNeuron, get_domain_neuron_config


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


def test_forward_with_side_signals_changes_output():
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg)
    peer_cfg = get_domain_neuron_config("zh", spec="compact")
    peer = ResonanceNeuron(peer_cfg)
    neuron.establish_side_channel("peer_0", peer, channel_type="excite")

    B, T = 2, 10
    embed_dim = 512
    torch.manual_seed(0)
    x = torch.randn(B, T, embed_dim)
    side_signals = {
        "peer_0": torch.randn(B, peer.config.field_dim),
    }
    out_with = neuron.forward(x, side_signals=side_signals, return_logits=False)
    out_without = neuron.forward(x, side_signals=None, return_logits=False)
    assert "field_vector" in out_with
    assert "field_vector" in out_without
    # side_signals 必须改变输出（否则说明没接线）
    assert not torch.allclose(out_with["field_vector"], out_without["field_vector"], atol=1e-6)


def test_forward_without_side_signals():
    cfg = get_domain_neuron_config("zh", spec="compact")
    neuron = ResonanceNeuron(cfg)
    B, T = 2, 10
    embed_dim = 512
    x = torch.randn(B, T, embed_dim)
    out = neuron.forward(x, return_logits=False)
    assert "field_vector" in out
