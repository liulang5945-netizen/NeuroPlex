# Taiji / NeuroPlex Architecture Direction

> **Status**: active decision record · 2026-08-21
>
> `NeuroPlex` remains the population/product runtime during migration. `Taiji` is the new non-Transformer computational substrate that will become the cognitive member implementation if its falsification gates pass.

## Decision

Adopt a two-level architecture identity:

1. **Population topology** — sparse, field-coupled, independently persistent members remain the system-level direction.
2. **Cell substrate** — replace Transformer-based `ResonanceNeuron` as the target member implementation with native `TaijiCell` dynamics.

The existing architecture proved that population assembly, routing, field communication and lifecycle mechanisms can be implemented, but its member still recomputes a Transformer sequence on every call. Adding biological labels around that body does not create persistent cellular state or local online learning. Taiji therefore changes the state transition itself rather than adding another adapter.

```text
Current baseline
  NeuroPlex → ResonanceEnsemble → Transformer ResonanceNeuron × 9

Target
  NeuroPlex → TaijiPopulation
             ├── event sensors
             ├── persistent TaijiCell population
             ├── multi-timescale TaijiField
             ├── native memory / local plasticity / sparse scheduler
             └── motor population
```

The full state equations, tick order, learning rule and falsification gates are defined in [TAIJI_SUBSTRATE_ARCHITECTURE.md](TAIJI_SUBSTRATE_ARCHITECTURE.md).

## Candidate comparison

| Option | Persistent native state | Online local learning | Sparse population fit | Removes Transformer dependency | Decision |
|---|---:|---:|---:|---:|---|
| Transformer + more bio-inspired adapters | Low | Low | Medium | No | Stop extending as target substrate |
| Smaller Transformer neurons | Low | Low | High | No | Retain only as cost/quality baseline |
| Generic RNN/GRU members | Medium | Medium | Medium | Yes | Baseline against Taiji, not target identity |
| State-space sequence model members | Medium | Low/Medium | Medium | Mostly | Useful comparison; still sequence-model-first |
| MoE parameter shards | Low | Low | High | No | Routing reference only |
| **Taiji event/state/field cells** | **High by contract** | **High by contract** | **High** | **Yes** | **Adopt as falsifiable target** |

## Canonical terms

| Term | Meaning |
|---|---|
| `Taiji` | new non-Transformer computational substrate |
| `TaijiCell` | persistent, locally plastic cognitive member |
| `TaijiField` | persistent multi-timescale population state |
| `TaijiPopulation` | sparse event-driven cell network |
| `NeuroPlex` | current product and migration-time population runtime |
| `LegacyResonancePopulation` | existing 9 Transformer members, including all 5 dialogue members |
| `event gateway` | explicit boundary between sensors/motors, legacy runtime and Taiji |

Do not describe Taiji as a distilled model, a `1.5B` replacement, or a fixed `7.58M/10M` model. Those sizes may appear in historical experiments, not in the substrate identity.

## Compatibility and package boundary

The canonical implementation path is `neuroplex.taiji`. A new top-level Python package named `taiji` must not be introduced while `neuroplex/__init__.py` maps `taiji` to `neuroplex` for old PyTorch checkpoint loading.

The current 9 members remain untouched as a reproducible benchmark. Taiji will not write into the existing `ResonanceField` during Phase A–D because the two fields have different time, state and vector semantics. Interoperation, if earned, happens through typed events after independent evaluation.

## AGI claim boundary

The working hypothesis is that persistent state, native memory, local plasticity, sparse asynchronous population dynamics and a real perception–action loop are more appropriate AGI research primitives than a stateless Transformer member. These properties are necessary design targets, not proof of sufficiency. Environment design, objectives, developmental curriculum, grounding and safety remain separate unsolved problems.

## Current implementation slice

Taiji-0 Phase A is implemented in `neuroplex/taiji`: the six targeted state contracts pass and the full suite is 53/53 passing. The next slice is T4 one-shot local association in per-cell fast memory, with a strict no-optimizer/no-slow-weight/no-unrelated-cell-update contract. No production integration or language training is authorized. Existing PlayEngine and D1 work are paused baseline work, not deleted.
