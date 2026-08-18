"""NeuroPlex — bio-inspired multi-neuron resonance architecture.

Compatibility shim: old checkpoints reference `taiji.*` module paths (PyTorch
pickle stores the fully-qualified class path). Register `taiji` as an alias
for `neuroplex` so torch.load(weights_only=False) can resolve them.
"""
import sys as _sys

# Register top-level alias: `taiji` → `neuroplex`
if "taiji" not in _sys.modules:
    _sys.modules["taiji"] = _sys.modules[__name__]
