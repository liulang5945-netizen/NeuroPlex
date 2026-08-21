import ast
import importlib
from pathlib import Path

import torch

import neuroplex
from taiji import ByteSensor, Taiji, TaijiConfig


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(32, 24),
        synapse_fan_in=8,
        motor_fan_in=16,
        seed=19,
    )


def test_taiji_owns_an_independent_top_level_namespace() -> None:
    native = importlib.import_module("taiji")

    assert native is not neuroplex
    assert Path(native.__file__).resolve().parent.name == "taiji"
    assert native.Taiji.__module__ == "taiji.model"


def test_native_core_has_no_legacy_or_sequence_model_dependency() -> None:
    package = Path(__file__).resolve().parents[2] / "taiji"
    imported = set()
    forbidden_calls = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Attribute):
                forbidden_calls.add(node.attr)

    assert not any(
        module.startswith(("neuroplex", "transformers"))
        for module in imported
    ), imported
    assert not {
        "backward",
        "topk",
        "MultiheadAttention",
        "TransformerEncoder",
    } & forbidden_calls


def test_raw_bytes_are_receptors_not_tokenizer_ids() -> None:
    sensor = ByteSensor(_config())

    for symbol in (0, 65, 255, sensor.config.boundary_symbol):
        encoded = sensor.encode(symbol)
        assert encoded.shape == (257,)
        assert float(encoded.sum()) == 1.0
        assert float(encoded[symbol]) == 1.0


def test_history_is_causal_and_only_explicit_reset_removes_it() -> None:
    seed = Taiji(_config())
    experienced = Taiji.from_checkpoint(seed.checkpoint())
    fresh = Taiji.from_checkpoint(seed.checkpoint())

    experienced.observe(97, learn=False)
    experienced.observe(98, learn=False)
    with_history = experienced.observe(99, learn=False).probabilities
    without_history = fresh.observe(99, learn=False).probabilities
    assert not torch.allclose(with_history, without_history)

    experienced.reset_dynamics(episode_id="reset")
    reset_probe = experienced.observe(99, learn=False).probabilities
    fresh.reset_dynamics(episode_id="reset")
    fresh_probe = fresh.observe(99, learn=False).probabilities
    assert torch.allclose(reset_probe, fresh_probe)


def test_learning_is_local_masked_and_has_no_autograd_parameters() -> None:
    model = Taiji(_config())
    before = [tensor.clone() for tensor in model.parameter_tensors()]
    for symbol in (256, 97, 98, 99, 97, 98, 99, 256):
        model.observe(symbol, learn=True)
    after = model.parameter_tensors()

    assert any(not torch.equal(left, right) for left, right in zip(before, after))
    assert all(tensor.requires_grad is False for tensor in after)
    synapses = (
        *model.fabric.decoders,
        *model.fabric.transitions,
        model.motor.synapses,
        model.memory.cue_encoder,
        model.memory.action_encoder,
        model.memory.outcome_encoder,
        model.memory.time_encoder,
        model.memory.episode_encoder,
        model.memory.provenance_encoder,
        model.memory.association,
        model.memory.action_readout,
        model.memory.outcome_readout,
        model.memory.reward_readout,
        model.memory.familiarity_readout,
        model.memory.cortical_readout,
        model.memory.time_readout,
        model.memory.episode_readout,
        model.memory.provenance_readout,
    )
    for projection in synapses:
        posts = torch.arange(projection.out_features).unsqueeze(1).expand_as(
            projection.pre_index.cpu()
        )
        edge_keys = (
            posts * projection.in_features
            + projection.pre_index.cpu()
        )
        assert projection.pre_index.shape == (
            projection.out_features,
            projection.row_fan_in,
        )
        assert projection.row_fan_in <= projection.fan_in
        assert torch.unique(edge_keys).numel() == projection.edge_count
        assert torch.isfinite(projection.edge_weight).all()


def test_motor_receptors_cover_every_cortical_coordinate_once() -> None:
    model = Taiji(_config())
    receptors = model.motor.receptors
    counts = torch.bincount(
        receptors.channel.cpu(), minlength=receptors.out_features
    )

    assert receptors.channel.numel() == model.config.cortical_context_dim
    assert int(counts.sum()) == model.config.cortical_context_dim
    assert int(counts.max() - counts.min()) <= 1
    motor = model.motor.synapses
    assert motor.row_fan_in == model.config.motor_context_dim
    for post in range(motor.out_features):
        assert torch.equal(
            torch.sort(motor.pre_index[post].cpu()).values,
            torch.arange(model.config.motor_context_dim, dtype=torch.int32),
        )


def test_checkpoint_preserves_learning_state_and_exact_next_step() -> None:
    original = Taiji(_config(), episode_id="roundtrip")
    for symbol in (256, 116, 97, 105, 106, 105):
        original.observe(symbol, learn=True)
    restored = Taiji.from_checkpoint(original.checkpoint())

    left = original.observe(33, learn=True)
    right = restored.observe(33, learn=True)

    assert left.predicted_symbol == right.predicted_symbol
    assert left.activity_rates == right.activity_rates
    assert torch.equal(left.probabilities, right.probabilities)
    for a, b in zip(original.parameter_tensors(), restored.parameter_tensors()):
        assert torch.equal(a, b)
