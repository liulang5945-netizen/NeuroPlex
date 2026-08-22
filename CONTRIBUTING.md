# Contributing to Taiji

Taiji is an experimental architecture project. Contributions should strengthen or falsify the native algorithm, not add biological names around a Transformer component.

## Setup

```bash
git clone https://github.com/<your-username>/Seed.git
cd Seed
python -m pip install -e ".[dev]"
python scripts/training/verify_taiji_native_v1.py
python -m pytest tests/taiji_native -q
```

Run the complete regression suite before submitting:

```bash
python -m pytest tests -q
```

## Native-core rules

- `seed/` may depend on the public `taiji` API, but not on `neuroplex` or `transformers`.
- `taiji/` must not import `seed`, `neuroplex`, `transformers`, attention implementations, tokenizers, or legacy checkpoints.
- Normal learning must not call `backward()` or a global optimizer.
- New synaptic updates must state which presynaptic trace, postsynaptic error, and broadcast signal are locally available.
- Every persistent state must define its update, decay, reset, checkpoint, and lesion behavior.
- Parameter-count claims must distinguish active masked edges from dense tensor storage and measured FLOPs.
- Capability claims need a deterministic benchmark and a causal lesion/control.
- Increasing model size or epochs is not an accepted response to a failed mechanism gate.

## Pull requests

1. Create a branch, normally `feat/<name>` or `fix/<name>`.
2. Add the smallest failing test or falsification benchmark first.
3. Implement one architecture change with matching equations and code documentation.
4. Run native tests, the benchmark, and the full regression suite.
5. Update `plans/active/TAIJI_SUBSTRATE_ARCHITECTURE.md` and the active implementation plan.

## Legacy boundary

The previous NeuroPlex Transformer runtime remains under `neuroplex/` for reproducibility. Changes there must be labeled Legacy and must not become a dependency of native Taiji. Historical `taiji.*` pickle paths use the scoped compatibility loader rather than a global module alias.

## License

Contributions are licensed under Apache License 2.0.
