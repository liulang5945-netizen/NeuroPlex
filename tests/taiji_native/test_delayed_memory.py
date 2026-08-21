from taiji import Taiji, TaijiConfig


DATA = b"a1234xbc1234xd" * 4
PROBE = ord("x")


def _intervene(model: Taiji, mode: str, index: int) -> None:
    if mode == "all":
        model.reset_dynamics(episode_id=f"n8-all-{index}")
        return
    for region in model._state.regions:
        if mode == "no_trace":
            region.trace.zero_()
        elif mode == "trace_only":
            region.membrane.zero_()
            region.activity.zero_()
            region.prediction.zero_()
            region.error.zero_()
            region.threshold.fill_(model.config.threshold_base)
            region.inhibition = 0.0
    model._state.motor_context.zero_()


def _probe_accuracy(model: Taiji, mode: str) -> float:
    model.reset_dynamics(episode_id=f"n8-{mode}")
    sequence = (model.config.boundary_symbol, *DATA, model.config.boundary_symbol)
    hits = []
    for index, symbol in enumerate(sequence[:-1]):
        if symbol == PROBE and mode != "full":
            _intervene(model, mode, index)
        step = model.observe(symbol, learn=False)
        if symbol == PROBE:
            hits.append(step.predicted_symbol == sequence[index + 1])
    return sum(hits) / len(hits)


def test_slow_trace_is_necessary_and_sufficient_after_shared_distractors() -> None:
    model = Taiji(TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=7,
    ))
    model.learn_bytes(DATA, epochs=200)
    learned = model.checkpoint()

    full = _probe_accuracy(Taiji.from_checkpoint(learned), "full")
    without_trace = _probe_accuracy(Taiji.from_checkpoint(learned), "no_trace")
    trace_only = _probe_accuracy(Taiji.from_checkpoint(learned), "trace_only")
    without_state = _probe_accuracy(Taiji.from_checkpoint(learned), "all")

    assert full >= 0.75
    assert full - without_trace >= 0.20
    assert trace_only >= 0.75
    assert trace_only - without_state >= 0.20
