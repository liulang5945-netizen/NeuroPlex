# Taiji Code Wiki

This page maps the executable Native v5 algorithm to source code. **Scope: the top-level `taiji/` substrate only** — the frozen Legacy NeuroPlex Transformer baseline in `neuroplex/` is documented in [INTERFACE_REFERENCE.md](INTERFACE_REFERENCE.md). The formal equations and ordering contract are in [TAIJI_SUBSTRATE_ARCHITECTURE.md](plans/active/TAIJI_SUBSTRATE_ARCHITECTURE.md).

## Runtime path

```text
Taiji.observe(symbol)
  ├─ if an action was settled: bind cue/action/reward/current outcome in EpisodicField
  ├─ settle previous ByteMotor prediction with the real symbol
  ├─ ByteSensor.encode(symbol)
  ├─ TaijiFabric.step(sensor_activity, previous_regions, prior_memory_feedback)
  │    ├─ decoder prediction and lower-level error
  │    ├─ reciprocal error backprojection
  │    ├─ recurrent next-state prediction
  │    ├─ delayed top-down prediction
  │    ├─ membrane / inhibition / threshold / trace update
  │    └─ existing-edge local decoder and transition updates
  ├─ concatenate every region's fast activity and slow trace
  ├─ EpisodicField.recall(cortical_state)
  │    ├─ distributed cue code + recurrent pattern completion
  │    ├─ resonance-gated action/outcome/value/time/source readouts
  │    └─ install cortical feedback for the next causal tick
  ├─ SparseReceptorBank folds every cortical coordinate into K shared channels
  ├─ ByteMotor.probabilities(context + episodic action evidence)
  └─ atomically install the next TaijiState
```

Active interaction adds:

```text
Taiji.act(affordances)
  → store pending context / restricted policy / executed action
environment.step(action)
  → reward + next sensation
Taiji.settle_action(reward)
  → reward-baseline-modulated local motor update
  → install PendingExperience(cue, action, reward, tick, episode, provenance)
Taiji.observe(outcome, learn_motor=False)
  → consume PendingExperience and write one distributed engram
```

## Modules

### `taiji/config.py`

`TaijiConfig` is the complete architecture contract: alphabet, region and field sizes, fan-in, dynamics, homeostasis, cortical/episodic learning rates, norm limits, memory completion/readback gains, motor temperature, and seed. A config is serialized inside every checkpoint.

### `taiji/sparse.py`

`SparseSynapses` owns a compressed fixed-fan-in topology. `pre_index[post, edge]` and `edge_weight[post, edge]` are the only synapse arrays; the postsynaptic row is implicit. Key methods:

- `forward(pre)` — postsynaptic evidence;
- `backproject(error)` — reciprocal bottom-up error;
- `local_update(error, trace)` — existing-edge-only error × eligibility update;
- `to_payload/load_payload` — exact topology and weight persistence.

Weights are regular tensors with `requires_grad=False`. There is no mask or absent-edge weight to allocate. Forward uses gather-sum, reciprocal projection uses scatter-add, and learning never constructs a dense outer product.

### `taiji/state.py`

`RegionState` contains membrane, activity, trace, prediction, error, adaptive threshold, and inhibitory-pool state. `MemoryState` contains field activity/trace, threshold, inhibition, confidence and one-tick-delayed cortical feedback. `TaijiState` owns every region plus memory, motor context/probabilities, causal time, optional `PendingAction`, and optional `PendingExperience`. `MemoryRecall`, `TaijiStep`, `TaijiDecision` and `TaijiOutcome` are immutable public results.

### `taiji/organs.py`

`ByteSensor` maps bytes 0–255 and boundary 256 to fixed one-hot receptor activity. `SparseReceptorBank` gives every cortical activity/trace coordinate exactly one fixed signed edge. `ByteMotor` supports both next-symbol local settlement and three-factor reward learning with a running reward baseline.

### `taiji/environment.py`

`TaijiEnvironment` defines `reset()` and action-dependent `step()`. `EnvironmentOutcome` carries only sensation, scalar reward and terminal state—never a teacher action.

### `taiji/memory.py`

`EpisodicField` is the native field-memory algorithm. Fixed sparse encoders project cortical state, action, outcome, reward, sinusoidal tick, hashed episode signature and four-valued provenance into overlapping field units. A blank fixed-fan-in recurrent graph learns cue→event completion and autoassociation with the same `SparseSynapses.local_update()` operator used elsewhere. Novelty and absolute reward gate each write.

Readout uses a shared compressed field context. It reconstructs value-modulated action evidence, outcome distribution, reward, cortical state, time code, episode code and provenance. Learned familiarity and recurrent resonance jointly gate every effect; zeroing recurrent weights therefore leaves the same field width and readouts but removes behavior. No event object, key, value or slot is appended when `write_count` increases.

### `taiji/fabric.py`

`TaijiFabric` constructs one reciprocal decoder and recurrent transition graph per region. `step()` is the canonical forward/learning algorithm and adds the previous field recall to each region's fast/slow coordinates with `memory_feedback_gain`. This one-tick delay avoids an algebraic loop. Changing the operation order is an architecture change and requires a new state version.

### `taiji/model.py`

`Taiji` owns organs, fabric, state, RNG and persistence:

- `observe(symbol, learn=True, use_memory=True)` — one causal tick and optional causal memory lesion;
- `act(affordances)` / `settle_action(reward)` — active action transaction;
- `learn_bytes(data, epochs)` — online local development;
- `score_bytes(data)` — no-side-effect teacher-forced evaluation;
- `generate(prompt, length)` — action feedback loop;
- `checkpoint()/restore()` — topology + parameters + all fast/transaction state + RNG.

## Checkpoint format

Native checkpoints use `format = taiji-native-v5`, state version 4, and contain:

```text
config
fabric.decoders[].pre_index + edge_weight
fabric.transitions[].pre_index + edge_weight
motor.receptors.channel + motor.receptors.polarity
motor.synapses + motor.bias + reward baseline/update count
memory fixed encoders + recurrent/readout synapses + write count
state.regions + state.memory + pending action/experience
rng_state
```

They never contain a NeuroPlex neuron, tokenizer, Transformer block, LM head, LoRA adapter, or teacher reference.

## Verification

- `tests/taiji_native/test_architecture_contract.py` checks independence, raw receptors, causal state, existing-edge topology, complete motor coverage and exact checkpoint continuation.
- `tests/taiji_native/test_sequence_learning.py` checks online learning and eight-step free generation.
- `tests/taiji_native/test_context_memory.py` checks history-dependent successors against a full dynamic-state lesion.
- `tests/taiji_native/test_delayed_memory.py` isolates slow trace after four shared distractors with necessary/sufficient lesions.
- `tests/taiji_native/test_long_free_run.py` checks 128 autonomous feedback steps on an explicitly non-terminal cycle.
- `scripts/training/verify_taiji_native_v5.py` produces the Native v5 machine-readable report.
- `scripts/training/verify_taiji_n7_context.py` measures full, first-order, trace-lesioned and all-state-lesioned context behavior.
- `scripts/training/verify_taiji_n8_delayed_trace.py` measures full, no-trace, trace-only and all-state delayed behavior.
- `scripts/training/verify_taiji_n9_long_free_run.py` records every free-running tick and all state bounds.
- `tests/taiji_native/test_sparse_kernel.py` compares fixed-fan-in forward, reciprocal projection and local update against a dense reference.
- `scripts/training/verify_taiji_n10_sparse_migration.py` reruns N5–N9 and compares behavior with committed v2 evidence.
- `tests/taiji_native/test_active_environment.py` checks pending-action atomicity and reward-learning causality.
- `scripts/training/verify_taiji_n11_active_environment.py` compares learned, random and action-lesioned policies.
- `tests/taiji_native/test_episodic_field.py` checks transactional writes, one-shot cross-episode recall, recurrent/read lesions, metadata and cortical feedback.
- `scripts/training/verify_taiji_m5_episodic_field.py` compares full field, equal-width trace-only and recurrent-association lesions and emits the M5 report.

## Legacy code

Everything under `neuroplex/` is the frozen Transformer population baseline. Its old interface notes remain in [INTERFACE_REFERENCE.md](INTERFACE_REFERENCE.md), which is not a Taiji API reference.
