import torch

from taiji import Taiji, TaijiConfig


CUES = tuple(ord(value) for value in "ABCD")
ACTIONS = (ord("0"), ord("1"))
OUTCOMES = (ord("+"), ord("-"))


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_context_dim=32,
        memory_iterations=3,
        seed=23,
    )


def _store_episodes(model: Taiji) -> None:
    for index, cue in enumerate(CUES):
        model.reset_dynamics(episode_id=f"store-{index}")
        model.observe(256, learn=False, learn_motor=False)
        model.observe(cue, learn=False, learn_motor=False)
        action = ACTIONS[index % len(ACTIONS)]
        assert model.act((action,), sample=False).action_symbol == action
        model.settle_action(1.0, learn=False, learn_memory=True)
        model.observe(
            OUTCOMES[index % len(OUTCOMES)],
            learn=False,
            learn_motor=False,
        )


def _thresholds(model: Taiji) -> tuple[torch.Tensor, ...]:
    return tuple(region.threshold.detach().clone() for region in model._state.regions)


def _baselines(model: Taiji) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().clone() for value in model.fabric.trace_baselines)


def test_replay_reads_the_homeostatic_set_point_without_writing_it() -> None:
    """Replay must not let homeostasis integrate over its own degenerate input.

    Consolidation drives one symbol for sixteen ticks with no waking traffic to
    balance it, so a unit the engram drives gains ``rate * (1 - target)`` every
    tick while a silent one sheds only ``rate * target``.  That 7:1 ratchet acts
    on exactly the units carrying the memory: measured, it drove the set point to
    twenty-one times ``threshold_base`` and, because ``activity`` subtracts the
    threshold straight off the drive, collapsed the write basis to 1/22 of the
    basis the probe reproduces.  The bout then wrote almost nothing while
    churning one decoder row through 118 rewires.

    The set point is therefore read during replay and never written.  This test
    pins that directly rather than pinning a downstream accuracy number, because
    the defect it guards against is silent: the bout still runs, still accepts
    engrams, and still reports plausible statistics while learning nothing.
    """

    model = Taiji(_config())
    _store_episodes(model)
    before = _thresholds(model)
    baselines_before = _baselines(model)
    consolidation_before = tuple(
        decoder.edge_weight.clone()
        for decoder in model.fabric.consolidation_decoders
    )

    summary = model.consolidate(cycles=24, learn=True)
    assert summary.accepted > 0

    after = _thresholds(model)
    for index, (start, end) in enumerate(zip(before, after)):
        assert torch.equal(start, end), f"region {index} set point drifted during sleep"
    for index, (start, end) in enumerate(
        zip(baselines_before, _baselines(model))
    ):
        assert torch.equal(start, end), f"region {index} baseline drifted during sleep"
    assert any(
        not torch.equal(start, decoder.edge_weight)
        for start, decoder in zip(
            consolidation_before, model.fabric.consolidation_decoders
        )
    ), "sleep did not write the slow consolidation pathway"


def test_waking_homeostasis_still_adapts() -> None:
    """The freeze is scoped to replay, not a global disable.

    Homeostasis is the mechanism that keeps regions near ``target_activity``
    under varied input, and waking input is exactly the varied traffic it is
    meant to integrate over.  Without this test, satisfying the freeze above by
    simply deleting the update would look like a pass.
    """

    model = Taiji(_config())
    before = _thresholds(model)
    for symbol in b"abcdabcd":
        model.observe(symbol, learn=True)
    after = _thresholds(model)

    assert any(
        not torch.equal(start, end) for start, end in zip(before, after)
    ), "waking observation left every set point untouched"


def test_consolidation_rewiring_terminates() -> None:
    """Structural turnover must converge, not churn.

    A row rewires only while it is mispredicting, so a reproducible write basis
    lets it capture its partners and stop.  Under the collapsed basis the
    ``captured`` measure was computed on a near-null trace and became arbitrary,
    so rows crossed the capture target repeatedly and never settled.

    The invariant is saturation, not a rate.  Measured across bout lengths of
    24/48/96/192 cycles the fixed substrate rewires 12 contacts and then stops --
    the same 12 at every length -- while the pre-fix substrate climbs 8/23/43/81,
    roughly linear in cycles.  A per-replay rate bound would not separate the two
    regimes at short bouts; quadrupling the bout and demanding the work not
    follow does.
    """

    short = Taiji(_config())
    _store_episodes(short)
    short_summary = short.consolidate(cycles=48, learn=True)

    long = Taiji(_config())
    _store_episodes(long)
    long_summary = long.consolidate(cycles=192, learn=True)

    assert short_summary.accepted > 0
    assert long_summary.accepted > 3 * short_summary.accepted
    assert short_summary.structural_events > 0, "no rewiring happened at all"
    assert long_summary.structural_events <= 1.5 * short_summary.structural_events


def test_consolidation_leaves_the_field_untouched() -> None:
    """A self generated pattern must not be able to reinforce itself.

    Replay's only channel into the fabric is the episodic feedback gain waking
    observation already uses.  If sleep wrote back into the field, the bout would
    be training on its own output and the lesion controls in the M6 benchmark
    would lose their meaning.
    """

    model = Taiji(_config())
    _store_episodes(model)
    before = tuple(
        tensor.detach().clone() for tensor in model.memory.parameter_tensors()
    )

    model.consolidate(cycles=16, learn=True)

    after = model.memory.parameter_tensors()
    assert len(before) == len(after)
    for index, (start, end) in enumerate(zip(before, after)):
        assert torch.equal(start, end), f"sleep modified memory tensor {index}"
