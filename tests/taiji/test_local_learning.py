import torch

from neuroplex.taiji import EventKind, EventMode, TaijiConfig, TaijiRuntime


def _config() -> TaijiConfig:
    return TaijiConfig(
        event_dim=8,
        state_dim=12,
        field_dim=16,
        dendritic_branches=3,
        fast_memory_slots=4,
        cell_ids=("cell_a", "cell_b", "cell_c"),
        active_budget=2,
        seed=23,
    )


def _cue_value() -> torch.Tensor:
    return torch.tensor([1.0, -0.5, 0.25, 0.75, -1.0, 0.5, -0.25, 0.125])


def _target_value() -> torch.Tensor:
    return torch.tensor([0.7, -0.7, 0.7, -0.7, 0.7, -0.7, 0.7, -0.7])


def test_one_shot_association_reduces_error_without_changing_slow_weights() -> None:
    seed_runtime = TaijiRuntime(_config(), episode_id="association")
    initial = seed_runtime.checkpoint()
    baseline = TaijiRuntime.from_checkpoint(initial)
    learner = TaijiRuntime.from_checkpoint(initial)
    target = _target_value()

    baseline_cue = baseline.make_event(
        _cue_value(), kind=EventKind.SENSORY, source="cue_sensor"
    )
    baseline_result = baseline.step([baseline_cue])
    baseline_error = torch.mean((baseline_result.output - target) ** 2)

    weights_before = {
        name: value.detach().clone() for name, value in learner.state_dict().items()
    }
    memory_before = learner.snapshot()
    cue = learner.make_event(
        _cue_value(), kind=EventKind.SENSORY, source="cue_sensor"
    )
    outcome = learner.make_event(
        target,
        kind=EventKind.SENSORY,
        source="environment",
        mode=EventMode.REAL,
        delay=1,
    )
    learned = learner.learn_association(cue, outcome, reward=1.0)

    assert learned.active_cell_ids == baseline_result.active_cell_ids
    assert set(learned.written_slots) == set(learned.active_cell_ids)
    for name, before in weights_before.items():
        assert torch.equal(before, learner.state_dict()[name]), name

    memory_after = learner.snapshot()
    inactive = set(_config().cell_ids) - set(learned.active_cell_ids)
    for cell_id in learned.active_cell_ids:
        assert float(memory_after.cells[cell_id].memory_usage.sum()) > 0.0
    for cell_id in inactive:
        assert torch.equal(
            memory_before.cells[cell_id].memory_keys,
            memory_after.cells[cell_id].memory_keys,
        )
        assert torch.equal(
            memory_before.cells[cell_id].memory_values,
            memory_after.cells[cell_id].memory_values,
        )
        assert torch.equal(
            memory_before.cells[cell_id].memory_usage,
            memory_after.cells[cell_id].memory_usage,
        )

    learner.reset_dynamics(preserve_fast_memory=True)
    probe = learner.make_event(
        _cue_value(), kind=EventKind.SENSORY, source="cue_sensor"
    )
    recalled = learner.step([probe])
    recalled_error = torch.mean((recalled.output - target) ** 2)

    assert recalled_error <= baseline_error * 0.70
    assert all(
        recalled.memory_confidences[cell_id] > 0.99
        for cell_id in learned.active_cell_ids
    )

    learner.reset()
    erased_probe = learner.make_event(
        _cue_value(), kind=EventKind.SENSORY, source="cue_sensor"
    )
    erased = learner.step([erased_probe])
    assert torch.allclose(erased.output, baseline_result.output)


def test_fast_association_survives_complete_checkpoint_roundtrip() -> None:
    runtime = TaijiRuntime(_config(), episode_id="memory-checkpoint")
    cue = runtime.make_event(_cue_value(), source="cue_sensor")
    outcome = runtime.make_event(
        _target_value(), source="environment", delay=1, mode=EventMode.REAL
    )
    runtime.learn_association(cue, outcome)
    runtime.reset_dynamics(preserve_fast_memory=True)
    restored = TaijiRuntime.from_checkpoint(runtime.checkpoint())

    event_a = runtime.make_event(_cue_value(), source="cue_sensor")
    event_b = restored.make_event(_cue_value(), source="cue_sensor")
    result_a = runtime.step([event_a])
    result_b = restored.step([event_b])

    assert result_a.active_cell_ids == result_b.active_cell_ids
    assert torch.equal(result_a.output, result_b.output)
    assert result_a.memory_confidences == result_b.memory_confidences


def test_imagined_outcome_cannot_be_committed_as_real_experience() -> None:
    runtime = TaijiRuntime(_config())
    before = runtime.checkpoint()
    cue = runtime.make_event(_cue_value(), source="cue_sensor")
    imagined = runtime.make_event(
        _target_value(),
        source="world_model",
        mode=EventMode.IMAGINED,
        delay=1,
    )

    try:
        runtime.learn_association(cue, imagined)
    except ValueError as error:
        assert "imagined" in str(error)
    else:
        raise AssertionError("imagined outcome was accepted as real experience")

    after = runtime.checkpoint()
    for name, tensor in before["module_state"].items():
        assert torch.equal(tensor, after["module_state"][name])
    assert before["taiji_state"]["tick"] == after["taiji_state"]["tick"] == 0
