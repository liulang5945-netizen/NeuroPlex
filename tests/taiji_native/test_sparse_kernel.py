import torch

from taiji import SparseSynapses


def _dense_view(synapses: SparseSynapses) -> torch.Tensor:
    dense = torch.zeros(synapses.out_features, synapses.in_features)
    posts = torch.arange(synapses.out_features).unsqueeze(1).expand_as(
        synapses.pre_index
    )
    dense[posts, synapses.pre_index] = synapses.edge_weight
    return dense


def test_edge_indexed_kernel_matches_dense_reference_for_all_operators() -> None:
    generator = torch.Generator().manual_seed(31)
    synapses = SparseSynapses(
        out_features=7,
        in_features=11,
        fan_in=4,
        generator=generator,
        init_scale=0.45,
        max_weight_norm=2.5,
    )
    dense_before = _dense_view(synapses)
    presynaptic = torch.linspace(-0.8, 0.9, 11)
    error = torch.linspace(-0.4, 0.5, 7)

    assert torch.allclose(
        synapses.forward(presynaptic), dense_before @ presynaptic, atol=1e-6
    )
    assert torch.allclose(
        synapses.backproject(error), dense_before.T @ error, atol=1e-6
    )

    learning_rate = 0.07
    weight_decay = 1e-3
    mask = dense_before != 0
    scale = max(1.0, float((presynaptic != 0).sum().item()) ** 0.5)
    # Decay is eligibility gated: an edge decays only when its presynaptic
    # partner is silent in this plasticity event.  The dense reference must
    # reproduce that per-contact gate, not a global multiplier.
    presynaptic_by_edge = presynaptic[synapses.pre_index]
    silent_by_edge = (presynaptic_by_edge == 0).to(dense_before.dtype)
    silent_dense = torch.zeros_like(dense_before)
    posts = torch.arange(synapses.out_features).unsqueeze(1).expand_as(
        synapses.pre_index
    )
    silent_dense[posts, synapses.pre_index] = silent_by_edge
    expected = dense_before * (1.0 - weight_decay * silent_dense)
    expected.add_(learning_rate * torch.outer(error, presynaptic) / scale * mask)
    expected.mul_(mask)
    norms = expected.norm(dim=1, keepdim=True).clamp_min(1e-8)
    expected.mul_(torch.clamp(2.5 / norms, max=1.0))

    synapses.local_update(
        error,
        presynaptic,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    assert torch.allclose(_dense_view(synapses), expected, atol=1e-6)


def test_local_update_decay_touches_only_presynaptically_silent_edges() -> None:
    generator = torch.Generator().manual_seed(57)
    synapses = SparseSynapses(
        out_features=6,
        in_features=10,
        fan_in=4,
        generator=generator,
        init_scale=0.45,
        max_weight_norm=2.5,
    )
    before = synapses.edge_weight.clone()
    presynaptic = torch.zeros(10)
    presynaptic[0] = 1.0  # exactly one lit partner

    # Zero error isolates the decay term.
    synapses.local_update(
        torch.zeros(6),
        presynaptic,
        learning_rate=0.1,
        weight_decay=0.5,
    )

    lit = presynaptic[synapses.pre_index] != 0
    silent = ~lit
    assert torch.equal(synapses.edge_weight[lit], before[lit]), (
        "被本次事件点亮的接触不得被衰减抽走"
    )
    assert torch.allclose(
        synapses.edge_weight[silent], 0.5 * before[silent], atol=1e-6
    ), "沉默接触必须按衰减率放松"


def test_sparse_kernel_stores_only_existing_edges_and_roundtrips() -> None:
    generator = torch.Generator().manual_seed(43)
    synapses = SparseSynapses(
        out_features=9,
        in_features=13,
        fan_in=5,
        generator=generator,
        init_scale=0.45,
        max_weight_norm=2.5,
    )

    assert synapses.edge_weight.shape == (9, 5)
    assert synapses.pre_index.shape == synapses.edge_weight.shape
    assert synapses.pre_index.dtype == torch.int32
    assert not any(
        isinstance(value, torch.Tensor) and value.shape == (9, 13)
        for value in vars(synapses).values()
    )

    payload = synapses.to_payload()
    assert payload["storage"] == "fixed-fan-in-v1"
    assert "mask" not in payload
    assert "weight" not in payload
    assert "post_index" not in payload

    restored = SparseSynapses(
        out_features=9,
        in_features=13,
        fan_in=5,
        generator=torch.Generator().manual_seed(43),
        init_scale=0.45,
        max_weight_norm=2.5,
    )
    restored.load_payload(payload)
    assert torch.equal(restored.pre_index, synapses.pre_index)
    assert torch.equal(restored.edge_weight, synapses.edge_weight)
