# Taiji Code Wiki

This page maps the executable Native v3 algorithm to source code. The formal equations and ordering contract are in [TAIJI_SUBSTRATE_ARCHITECTURE.md](plans/active/TAIJI_SUBSTRATE_ARCHITECTURE.md).

## Runtime path

```text
Taiji.observe(symbol)
  ├─ settle previous ByteMotor prediction with the real symbol
  ├─ ByteSensor.encode(symbol)
  ├─ TaijiFabric.step(sensor_activity, previous_regions)
  │    ├─ decoder prediction and lower-level error
  │    ├─ reciprocal error backprojection
  │    ├─ recurrent next-state prediction
  │    ├─ delayed top-down prediction
  │    ├─ membrane / inhibition / threshold / trace update
  │    └─ masked local decoder and transition updates
  ├─ concatenate every region's fast activity and slow trace
  ├─ SparseReceptorBank folds every cortical coordinate into K shared channels
  ├─ ByteMotor.probabilities(context)
  └─ atomically install the next TaijiState
```

## Modules

### `taiji/config.py`

`TaijiConfig` is the complete architecture contract: alphabet, region sizes, fan-in, dynamics, homeostasis, learning rates, norm limits, motor temperature, and seed. A config is serialized inside every checkpoint.

### `taiji/sparse.py`

`SparseSynapses` owns a compressed fixed-fan-in topology. `pre_index[post, edge]` and `edge_weight[post, edge]` are the only synapse arrays; the postsynaptic row is implicit. Key methods:

- `forward(pre)` — postsynaptic evidence;
- `backproject(error)` — reciprocal bottom-up error;
- `local_update(error, trace)` — existing-edge-only error × eligibility update;
- `to_payload/load_payload` — exact topology and weight persistence.

Weights are regular tensors with `requires_grad=False`. There is no mask or absent-edge weight to allocate. Forward uses gather-sum, reciprocal projection uses scatter-add, and learning never constructs a dense outer product.

### `taiji/state.py`

`RegionState` contains membrane, activity, trace, prediction, error, adaptive threshold, and inhibitory-pool state. `TaijiState` owns every region plus motor context/probabilities and causal time. `TaijiStep` is the immutable observation result.

### `taiji/organs.py`

`ByteSensor` maps bytes 0–255 and boundary 256 to fixed one-hot receptor activity. `SparseReceptorBank` gives every cortical activity/trace coordinate exactly one fixed signed edge, balances those edges across `K` shared channels, and normalizes their joint evidence. `ByteMotor` is the only output organ; all 257 actions read the same channels, and it updates only after the next real symbol arrives.

### `taiji/fabric.py`

`TaijiFabric` constructs one reciprocal decoder and recurrent transition graph per region. `step()` is the canonical forward/learning algorithm; changing its operation order is an architecture change and requires a new state version.

### `taiji/model.py`

`Taiji` owns organs, fabric, state, RNG and persistence:

- `observe(symbol, learn=True)` — one causal tick;
- `learn_bytes(data, epochs)` — online local development;
- `score_bytes(data)` — no-side-effect teacher-forced evaluation;
- `generate(prompt, length)` — action feedback loop;
- `checkpoint()/restore()` — parameters + masks + state + RNG.

## Checkpoint format

Native checkpoints use `format = taiji-native-v3` and contain:

```text
config
fabric.decoders[].pre_index + edge_weight
fabric.transitions[].pre_index + edge_weight
motor.receptors.channel + motor.receptors.polarity
motor.synapses + motor.bias
state
rng_state
```

They never contain a NeuroPlex neuron, tokenizer, Transformer block, LM head, LoRA adapter, or teacher reference.

## Verification

- `tests/taiji_native/test_architecture_contract.py` checks independence, raw receptors, causal state, local masks, complete motor coverage and exact checkpoint continuation.
- `tests/taiji_native/test_sequence_learning.py` checks online learning and eight-step free generation.
- `tests/taiji_native/test_context_memory.py` checks history-dependent successors against a full dynamic-state lesion.
- `tests/taiji_native/test_delayed_memory.py` isolates slow trace after four shared distractors with necessary/sufficient lesions.
- `tests/taiji_native/test_long_free_run.py` checks 128 autonomous feedback steps on an explicitly non-terminal cycle.
- `scripts/training/verify_taiji_native_v3.py` produces the Native v3 machine-readable report.
- `scripts/training/verify_taiji_n7_context.py` measures full, first-order, trace-lesioned and all-state-lesioned context behavior.
- `scripts/training/verify_taiji_n8_delayed_trace.py` measures full, no-trace, trace-only and all-state delayed behavior.
- `scripts/training/verify_taiji_n9_long_free_run.py` records every free-running tick and all state bounds.
- `tests/taiji_native/test_sparse_kernel.py` compares fixed-fan-in forward, reciprocal projection and local update against a dense reference.
- `scripts/training/verify_taiji_n10_sparse_migration.py` reruns N5–N9 and compares behavior with committed v2 evidence.

## Legacy code

Everything under `neuroplex/` is the frozen Transformer population baseline. Its old interface notes remain in [INTERFACE_REFERENCE.md](INTERFACE_REFERENCE.md), which is not a Taiji API reference.
