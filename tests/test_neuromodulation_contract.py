"""Regression tests for neutral neuromodulation defaults."""

import pytest

from neuroplex.resonance.neuro_modulation import NeuromodulatorState


def test_default_dopamine_keeps_ffn_at_training_baseline():
    state = NeuromodulatorState()

    assert state.get_ffn_gain() == pytest.approx(1.0)
    state.dopamine = 0.0
    assert state.get_ffn_gain() == pytest.approx(0.5)
    state.dopamine = 1.0
    assert state.get_ffn_gain() == pytest.approx(1.5)
