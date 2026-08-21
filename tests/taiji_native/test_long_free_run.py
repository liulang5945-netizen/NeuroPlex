import torch

from taiji import Taiji, TaijiConfig


def test_learned_cycle_remains_exact_and_bounded_for_128_free_steps() -> None:
    model = Taiji(TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=7,
    ))
    # N9 is an explicitly cyclic stream. A terminal boundary would teach the
    # fourth ``d`` to stop, which contradicts the free-running target.
    model.learn_bytes(b"abcd" * 4, epochs=200, include_boundary=False)

    generated = model.generate(b"a", 128)
    state = model.snapshot()

    assert generated == b"bcda" * 32
    assert set(generated) == set(b"abcd")
    assert all(
        float(region.membrane.norm()) <= model.config.max_membrane_norm + 1e-6
        and float(region.trace.norm()) <= model.config.max_trace_norm + 1e-6
        and bool(torch.isfinite(region.membrane).all())
        and bool(torch.isfinite(region.trace).all())
        for region in state.regions
    )
