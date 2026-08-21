import ast
import io
from pathlib import Path

import torch

from neuroplex.taiji import EventKind, TaijiConfig, TaijiField, TaijiRuntime
from neuroplex.taiji.state import FieldWrite, TaijiState


def _config(**overrides) -> TaijiConfig:
    values = TaijiConfig(
        event_dim=8,
        state_dim=12,
        field_dim=16,
        dendritic_branches=3,
        fast_memory_slots=4,
        cell_ids=("cell_a", "cell_b", "cell_c"),
        active_budget=2,
        seed=17,
    ).to_dict()
    values.update(overrides)
    return TaijiConfig.from_dict(values)


def _sensory_event(runtime: TaijiRuntime, scale: float = 1.0):
    value = torch.linspace(-1.0, 1.0, runtime.config.event_dim) * scale
    return runtime.make_event(value, kind=EventKind.SENSORY, source="test_sensor")


def _assert_state_close(left: TaijiState, right: TaijiState) -> None:
    assert left.version == right.version
    assert left.tick == right.tick
    assert left.episode_id == right.episode_id
    assert set(left.cells) == set(right.cells)
    for name in ("fast", "working", "context", "inhibit"):
        assert torch.allclose(getattr(left.field, name), getattr(right.field, name))
    assert torch.allclose(left.last_output, right.last_output)
    for cell_id in left.cells:
        a = left.cells[cell_id]
        b = right.cells[cell_id]
        for name in (
            "dendrites",
            "apical",
            "soma",
            "prediction",
            "error",
            "phase",
            "eligibility",
            "memory_keys",
            "memory_values",
            "memory_usage",
        ):
            assert torch.allclose(getattr(a, name), getattr(b, name))
        assert a.energy == b.energy
        assert a.threshold == b.threshold
        assert a.refractory == b.refractory
    assert set(left.pending_events) == set(right.pending_events)
    for tick in left.pending_events:
        left_events = left.pending_events[tick]
        right_events = right.pending_events[tick]
        assert len(left_events) == len(right_events)
        for a, b in zip(left_events, right_events):
            assert a.tick == b.tick
            assert a.source == b.source
            assert a.target == b.target
            assert a.kind == b.kind
            assert a.mode == b.mode
            assert torch.allclose(a.value, b.value)


def test_core_imports_do_not_depend_on_legacy_sequence_layers() -> None:
    package_dir = Path(__file__).resolve().parents[2] / "neuroplex" / "taiji"
    forbidden_prefixes = ("transformers", "neuroplex.layers")
    imported_modules = set()
    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
    assert not any(
        module.startswith(forbidden_prefixes) for module in imported_modules
    ), imported_modules


def test_field_persists_decays_and_only_explicit_reset_clears_it() -> None:
    config = _config()
    field = TaijiField(config)
    initial = field.initial_state()
    write = FieldWrite(
        source="cell_a",
        excite=torch.ones(config.field_dim),
        inhibit=torch.zeros(config.field_dim),
        scale=0.25,
    )

    written = field.advance(initial, [write])
    decayed = field.advance(written, [])
    written_norm = float(field.effective(written).norm())
    decayed_norm = float(field.effective(decayed).norm())

    assert written_norm > 0.0
    assert 0.0 < decayed_norm < written_norm
    cleared = field.reset()
    assert float(field.effective(cleared).norm()) == 0.0


def test_experience_changes_future_state_until_an_explicit_reset() -> None:
    config = _config()
    experienced = TaijiRuntime(config)
    first = experienced.step([_sensory_event(experienced, scale=2.0)])
    assert 0 < len(first.active_cell_ids) <= config.active_budget
    experienced.step([])
    experienced_readout = experienced.readout()

    fresh = TaijiRuntime(config)
    fresh.step([])
    fresh_readout = fresh.readout()
    assert float((experienced_readout - fresh_readout).norm()) > 1e-4

    experienced.reset()
    assert torch.allclose(experienced.readout(), TaijiRuntime(config).readout())


def test_two_phase_commit_is_independent_of_cell_evaluation_order() -> None:
    config = _config()
    forward_runtime = TaijiRuntime(config)
    reverse_runtime = TaijiRuntime.from_checkpoint(forward_runtime.checkpoint())
    event = _sensory_event(forward_runtime, scale=1.5)

    forward = forward_runtime.step(
        [event], evaluation_order=("cell_a", "cell_b", "cell_c")
    )
    reverse = reverse_runtime.step(
        [event], evaluation_order=("cell_c", "cell_b", "cell_a")
    )

    assert forward.active_cell_ids == reverse.active_cell_ids
    assert torch.allclose(forward.output, reverse.output)
    _assert_state_close(forward_runtime.snapshot(), reverse_runtime.snapshot())


def test_sparse_budget_energy_and_state_bounds_hold_over_time() -> None:
    config = _config()
    runtime = TaijiRuntime(config)
    result = runtime.step([_sensory_event(runtime, scale=3.0)])
    assert 0 < len(result.active_cell_ids) <= config.active_budget

    after_first = runtime.snapshot()
    for cell_id in result.active_cell_ids:
        assert after_first.cells[cell_id].energy < config.energy_capacity

    for _ in range(256):
        result = runtime.step([])
        assert len(result.active_cell_ids) <= config.active_budget
        state = runtime.snapshot()
        assert torch.isfinite(runtime.readout()).all()
        for component in (
            state.field.fast,
            state.field.working,
            state.field.context,
            state.field.inhibit,
        ):
            assert float(component.norm()) <= config.max_field_norm + 1e-5
        for cell_state in state.cells.values():
            assert float(cell_state.soma.norm()) <= config.max_state_norm + 1e-5
            assert 0.0 <= cell_state.energy <= config.energy_capacity


def test_checkpoint_roundtrip_preserves_the_next_tick_and_pending_events() -> None:
    config = _config()
    original = TaijiRuntime(config, episode_id="roundtrip")
    original.step([_sensory_event(original, scale=2.0)])
    assert original.snapshot().pending_events

    buffer = io.BytesIO()
    torch.save(original.checkpoint(), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=False)
    restored = TaijiRuntime.from_checkpoint(payload)
    _assert_state_close(original.snapshot(), restored.snapshot())

    next_value = torch.linspace(1.0, -1.0, config.event_dim)
    event_a = original.make_event(next_value, source="second_sensor")
    event_b = restored.make_event(next_value, source="second_sensor")
    result_a = original.step([event_a])
    result_b = restored.step([event_b])

    assert result_a.active_cell_ids == result_b.active_cell_ids
    assert torch.allclose(result_a.output, result_b.output)
    _assert_state_close(original.snapshot(), restored.snapshot())

