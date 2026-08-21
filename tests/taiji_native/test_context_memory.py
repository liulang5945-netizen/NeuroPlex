from taiji import Taiji, TaijiConfig


def _ambiguous_accuracy(model: Taiji, data: bytes, *, lesion: str) -> float:
    model.reset_dynamics(episode_id="n7-eval")
    sequence = (model.config.boundary_symbol, *data, model.config.boundary_symbol)
    correct = 0
    count = 0
    for index, symbol in enumerate(sequence[:-1]):
        if lesion == "all":
            model.reset_dynamics(episode_id=f"n7-lesion-{index}")
        elif lesion == "trace":
            for region in model._state.regions:
                region.trace.zero_()
            model._state.motor_context.zero_()
        step = model.observe(symbol, learn=False)
        if symbol == ord("x"):
            count += 1
            correct += int(step.predicted_symbol == sequence[index + 1])
    return correct / count


def test_temporal_state_solves_ambiguous_second_order_successors() -> None:
    data = b"axbcxd" * 4
    model = Taiji(TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=7,
    ))
    model.learn_bytes(data, epochs=200)

    full = _ambiguous_accuracy(model, data, lesion="none")
    lesioned = Taiji.from_checkpoint(model.checkpoint())
    without_dynamic_state = _ambiguous_accuracy(lesioned, data, lesion="all")

    assert full >= 0.75
    assert full - without_dynamic_state >= 0.20
