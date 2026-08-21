# Taiji Code Wiki

This page maps the executable Native v1 algorithm to source code. The formal equations and ordering contract are in [TAIJI_SUBSTRATE_ARCHITECTURE.md](plans/active/TAIJI_SUBSTRATE_ARCHITECTURE.md).

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
  ├─ normalize concatenated region traces
  ├─ ByteMotor.probabilities(context)
  └─ atomically install the next TaijiState
```

## Modules

### `taiji/config.py`

`TaijiConfig` is the complete architecture contract: alphabet, region sizes, fan-in, dynamics, homeostasis, learning rates, norm limits, motor temperature, and seed. A config is serialized inside every checkpoint.

### `taiji/sparse.py`

`SparseSynapses` owns one immutable structural mask and one mutable weight tensor. Key methods:

- `forward(pre)` — postsynaptic evidence;
- `backproject(error)` — reciprocal bottom-up error;
- `local_update(error, trace)` — masked error × eligibility update;
- `to_payload/load_payload` — exact topology and weight persistence.

Weights are regular tensors with `requires_grad=False`. Masked-out values are forced to zero after every update.

### `taiji/state.py`

`RegionState` contains membrane, activity, trace, prediction, error, adaptive threshold, and inhibitory-pool state. `TaijiState` owns every region plus motor context/probabilities and causal time. `TaijiStep` is the immutable observation result.

### `taiji/organs.py`

`ByteSensor` maps bytes 0–255 and boundary 256 to fixed one-hot receptor activity. `ByteMotor` is the only output organ. It updates only after the next real symbol arrives.

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

Native checkpoints use `format = taiji-native-v1` and contain:

```text
config
fabric.decoders[]
fabric.transitions[]
motor.synapses + motor.bias
state
rng_state
```

They never contain a NeuroPlex neuron, tokenizer, Transformer block, LM head, LoRA adapter, or teacher reference.

## Verification

- `tests/taiji_native/test_architecture_contract.py` checks independence, raw receptors, causal state, local masks and exact checkpoint continuation.
- `tests/taiji_native/test_sequence_learning.py` checks online learning and short free generation.
- `scripts/training/verify_taiji_native_v1.py` produces the committed machine-readable report.

## Legacy code

Everything under `neuroplex/` is the frozen Transformer population baseline. Its old interface notes remain in [INTERFACE_REFERENCE.md](INTERFACE_REFERENCE.md), which is not a Taiji API reference.
