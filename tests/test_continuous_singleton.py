"""Regression tests for singleton continuous routing."""

import pytest
import torch

from neuroplex.resonance.continuous import ContinuousResonance


def test_singleton_binding_stats_are_finite():
    continuous = ContinuousResonance()
    binding = torch.tensor([0.75])

    assert continuous.converged([binding, binding]) is True
    assert continuous.lock_degree(binding) == pytest.approx(0.75)
