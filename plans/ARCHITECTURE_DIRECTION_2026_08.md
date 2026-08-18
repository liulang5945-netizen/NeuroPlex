# NeuroPlex Architecture Direction

> **Status**: active decision record · 2026-08-18
>
> NeuroPlex is a **population neural network**: independent neurons cooperate through a shared resonance field, adaptive peer connections, routing, memory, and lifecycle control. The population—not a single large language model—is the unit of capability, scaling, and evolution.

## Decision

Adopt the **sparsely routed population resonance network** as the canonical architecture.

The runtime has four cooperating planes:

1. **Sensory plane** — shared input/token alignment makes different neurons able to receive the same experience without forcing them to share an internal representation.
2. **Neuron population** — domain-, role-, and subtype-specific Transformer neurons remain independently trainable and hot-swappable.
3. **Resonance plane** — field read/write, peer excitation/inhibition, cross-spec projection, and multi-round aggregation let active neurons exchange partial beliefs.
4. **Life plane** — routing, confidence/quality gates, field memory, sleep replay, synaptic plasticity, neurogenesis, maturation, and apoptosis control which neurons participate and how the population changes.

The optional expert/relay neuron is a communication aid, not a central model or a mandatory teacher. New neurons grow from data, field experience, and peer coordination. The old whole-model migration path is retained only where compatibility requires it and is not part of the product architecture.

## Candidate comparison

| Option | Core abstraction | Fit with current code | Independent growth | Runtime sparsity | Lifecycle support | Decision |
|---|---|---:|---:|---:|---:|---|
| Monolithic model + adapters | One backbone, many adapters | Low | Low | Medium | Low | Retire as primary direction |
| Dense neuron ensemble | Every neuron participates every round | Medium | High | Low | Medium | Useful baseline only |
| MoE-style experts | One router selects parameter shards | Medium | Medium | High | Low | Borrow routing ideas, not the identity |
| **Sparse population resonance** | Independent neurons + field + peer topology | **High** | **High** | **High** | **High** | **Adopt** |
| Hierarchical cortical graph | Clusters, relays, and local circuits | High in the long term | High | High | High | Future extension after population baseline |

### Why the adopted option wins

- It matches the implementation that already exists: `Cortex`, `ResonanceEnsemble`, `ResonanceField`, peer channels, instance routing, field memory, and lifecycle modules.
- Capability scales by adding, specializing, isolating, or retiring neurons instead of replacing a central model.
- The routing decision is inspectable at task and instance level; field contributions and quality gates can be measured.
- Training can be split into neuron learning, peer coordination, and population evaluation, which keeps failures local and makes hot-swapping practical.
- The design leaves room for hierarchical clusters without forcing a second central backbone today.

## Canonical terms

Use these terms in new documentation, APIs, logs, and training data:

| Prefer | Avoid as the primary framing |
|---|---|
| neuron population / population network | one big model / model size ladder |
| peer coordination / field alignment | teacher–student migration |
| experience replay / consolidation | recursive model distillation |
| neuron growth / specialization | whole-model upgrade |
| relay or anchor neuron (optional) | central teacher model |

Legacy teacher-alignment utilities may remain for checkpoint compatibility, but they must be clearly isolated from the active population path and must not appear in the product identity or quick-start flow.

## Migration boundary

This decision changes the public contract and the active narrative first:

- public README and code wiki describe the population network;
- active plans describe current population mechanisms and experiments;
- generated identity/developer data teaches the population model;
- stale local file URIs, old package paths, and contradictory license wording are removed;
- legacy upgrade/distillation entry points are either isolated as compatibility code or replaced by population-growth interfaces;
- historical audit files remain historical evidence, but are not linked as current architecture guidance.

## Next implementation slice

The naming migration and compatibility boundary are now complete enough for the public path. The next slice is a **minimal reproducible population baseline**: fixed evaluation inputs, a small repeatable neuron population, dense/sparse A/B comparison, route/field observability, and an API/Cortex smoke result that can be reproduced without private long-running checkpoints.

Only after that baseline is stable should the project spend compute on cross-domain peer coordination. The first collaboration experiment must be short, resumable, independently validated per domain, and guarded against anchor or domain regression. The 1.5B distillation path, monolithic upgrade path, and new biological mechanisms remain outside the active roadmap.
