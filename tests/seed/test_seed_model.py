from __future__ import annotations

import torch

from seed import Seed, SeedConfig
from taiji import Taiji, TaijiConfig


def _config() -> SeedConfig:
    return SeedConfig(
        taiji=TaijiConfig(
            alphabet_size=257,
            boundary_symbol=256,
            region_sizes=(12, 8),
            synapse_fan_in=4,
            motor_fan_in=6,
            memory_units=16,
            memory_fan_in=4,
            memory_readout_fan_in=6,
            memory_meta_dim=6,
            memory_iterations=2,
            memory_time_dim=4,
            memory_episode_dim=4,
            lateral_fan_in=4,
            seed=41,
        )
    )


def test_seed_owns_a_native_taiji_substrate() -> None:
    model = Seed(_config(), episode_id="seed-boundary")

    assert isinstance(model.substrate, Taiji)
    assert model.config.taiji == model.substrate.config
    assert model.parameter_count() == model.substrate.parameter_count()


def test_seed_checkpoint_wraps_taiji_and_continues_exactly() -> None:
    model = Seed(_config(), episode_id="seed-checkpoint")
    for symbol in (1, 2, 3, 1):
        model.observe(symbol, learn=True)

    checkpoint = model.checkpoint()
    restored = Seed.from_checkpoint(checkpoint)
    expected = model.observe(2, learn=True)
    actual = restored.observe(2, learn=True)

    assert checkpoint["format"] == "seed-native-v1"
    assert checkpoint["substrate"]["format"] == Taiji.CHECKPOINT_FORMAT
    assert expected.predicted_symbol == actual.predicted_symbol
    assert torch.equal(expected.probabilities, actual.probabilities)
    for left, right in zip(
        model.substrate.parameter_tensors(),
        restored.substrate.parameter_tensors(),
    ):
        assert torch.equal(left, right)


def test_seed_rejects_a_bare_taiji_checkpoint() -> None:
    substrate = Taiji(_config().taiji)

    try:
        Seed.from_checkpoint(substrate.checkpoint())
    except ValueError as error:
        assert "Seed checkpoint" in str(error)
    else:
        raise AssertionError("Seed must not erase the model/substrate boundary")
