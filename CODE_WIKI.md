# NeuroPlex Code Wiki

> **Current architecture**: a sparsely routed population neural network. Independent neurons exchange partial beliefs through a resonance field and evolve through memory, plasticity, and lifecycle control.

This page describes the active code paths. Historical experiments and retired migration utilities stay in `plans/archive/` or are marked as compatibility code; they are not the runtime architecture.

## 1. Runtime map

```text
Input
  ↓
shared sensory alignment + domain tokenizers
  ↓
Cortex
  ├─ neuron population (independent Transformer neurons)
  ├─ ResonanceEnsemble (routing, rounds, fusion)
  ├─ ResonanceField (shared communication state)
  ├─ peer channels / cross-spec projectors
  └─ life interfaces (memory, sleep, plasticity, growth, pruning)
  ↓
domain-aware logits → generated output
```

The population is the unit of capability. `Cortex` is the orchestrator; it is not a replacement monolithic model.

The production loader assembles nine members by default: five dialogue neurons
(`zh_aug0_dialogue`, `zh_aug1_dialogue`, `zh_aug2_dialogue`, `zh_aug3_dialogue`,
`zh_std0_dialogue`) plus four general neurons (`code`, `en`, `math`, `zh`). The
dialogue members provide the default conversation path; the general members
provide domain routing and shared-space composition when their checkpoints are available.

## 2. Core modules

### `neuroplex/brain/cortex.py`

`Cortex` owns neuron loading, tokenizer wiring, generation, memory interfaces, and lifecycle integration. The main entry points are:

- `generate()` — autoregressive population inference;
- `think()` — inspectable resonance reasoning;
- `add_neuron()` / `remove_neuron()` / `isolate_neuron()` / `revive_neuron()` — population membership changes;
- `set_field_memory()` — connect experience memory to the resonance path.

### `neuroplex/resonance/neuron.py`

`ResonanceNeuron` is an independently trainable population member. Its internal Transformer can have its own hidden size, tokenizer vocabulary, field projection, neuron subtype, and local adapters.

The field interface is explicit:

```text
input embedding → Transformer body → field_write
field_state ───────────────────────→ field_read / conditioning
hidden state ──────────────────────→ domain lm_head
```

### `neuroplex/resonance/field.py`

`ResonanceField` is a shared communication medium, not a language model. It stores the current population state, normalizes contributions, exposes leave-one-out scoring, and provides the conditioning vector used in later rounds.

### `neuroplex/resonance/ensemble.py`

`ResonanceEnsemble` coordinates the population:

1. route or probe candidate neurons;
2. run an independent first response;
3. write field and peer signals;
4. apply confidence/quality/topology gates;
5. run later conditional rounds for the active subset;
6. fuse compatible logits and expose routing diagnostics.

Continuous resonance, phase dynamics, cross-spec projection, and instance-level routing are optional mechanisms on this same population path.

### `neuroplex/resonance/topology.py` and `tribal.py`

Topology and coactivation tracking provide local structure: neurons can form task-relevant groups, strengthen useful peer channels, and avoid an all-to-all dependency. An expert/relay neuron can be used as one node in the graph, but the graph does not require a central controller.

### `neuroplex/life/`

The life plane controls population change:

- `sleep_engine.py` — replay, consolidation, evaluation, and scheduled learning;
- `integrate_engine.py` — bring a new neuron through silent, plastic, validation, and commit/apoptosis stages;
- `evolution_engine.py` — detect capacity or domain gaps and request population growth;
- lifecycle trackers — maturity, survival, neurogenesis, and apoptosis.

## 3. Data and training paths

The active training flow is:

```text
domain data
  → independent neuron training
  → peer/channel and cross-spec coordination
  → population evaluation and routing calibration
  → sleep replay + memory consolidation
  → growth, specialization, isolation, or pruning
```

Recommended entry points:

- `scripts/training/finetune_neuron_dialogue.py` — specialize a neuron;
- `scripts/training/train_neurons_from_scratch.py` — create a neuron through independent training;
- `scripts/training/train_cross_domain_collab.py` — train cross-domain coordination;
- `scripts/training/train_hub_neuron.py` — train an optional relay/anchor member;
- `scripts/training/verify_*.py` — mechanism and product-path checks.

Legacy alignment code is a compatibility utility only. It is not required for creating the active population and is intentionally absent from the quick-start path.

## 4. Public API layers

- `api/app.py` — FastAPI application factory and router registration;
- `api/routes_neuroplex.py` — population status, generation, and core operations;
- `api/routes_life.py` — feed, sleep, memory, and lifecycle operations;
- `api/routes_chat.py` — streaming conversations;
- `frontend/` — Vue client;
- `desktop/` — PyQt client and packaging.

Use `neuroplex` imports in new code. The `taiji` module alias remains only so old serialized checkpoints can be loaded safely.

## 5. Verification

```bash
python -m pytest tests/ -q
```

The small regression suite covers dialogue-format contracts and resonance side-channel behavior. Mechanism-level checks live under `scripts/training/verify_*.py` and should be run when touching the corresponding module.

## 6. Design rules

1. Do not introduce a hidden central backbone into the population path.
2. Keep neuron-local failures local; routing must be able to isolate a weak member.
3. Treat field state as communication, not as a substitute for learned neuron parameters.
4. Add metrics for active members, field contribution, peer traffic, routing decisions, and lifecycle transitions.
5. New identity or training data must describe population growth, peer coordination, and experience consolidation.

## 7. Compatibility boundary

The repository still contains historical checkpoints, environment variables, and migration helpers using the former package name or legacy terminology. They may be used to load old artifacts, but new code and product messaging must use the population vocabulary documented above. Any removal of a compatibility path requires a checkpoint migration test first.
