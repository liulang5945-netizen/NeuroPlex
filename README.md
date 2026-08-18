# NeuroPlex — Population Neural Network with Resonance

> **A population of specialized neurons, coordinated through resonance.**
>
> NeuroPlex is a population neural network made of domain-specific neurons (24M–134M each). Neurons coordinate through a shared resonance field, peer connections, routing, memory, and lifecycle control. Capability grows by specializing and composing the population, not by replacing it with one larger model.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Why NeuroPlex?

| Centralized model assumption | NeuroPlex population design |
|---|---|
| One model must absorb every domain | Hot-swap one new neuron (24M), keep the rest stable |
| A single representation serves every task | Domain-specific neurons specialize (zh/en/code/math) |
| New knowledge risks global interference | Per-neuron protection, peer coordination, and local consolidation |
| Routing is hidden inside one forward pass | Resonance rounds expose field vectors, gates, quality, and active neurons |

## Core Idea

Four cooperating planes, each replaceable:

```
Level 0: Shared sensory alignment              ← I/O protocol layer
    ↓
Level 1: Neuron population (24M–134M each)   ← Independent Transformers
    ↓  field_write / field_read
Level 2: Resonance field (3072–4096 dim)     ← Shared communication medium
    ↓  routing / memory / lifecycle
Level 3: Population control plane             ← topology, plasticity, growth, pruning
```

**Round 1**: Each neuron reads the input independently, writes its `field_vector` to the shared field.
**Round 2–N**: Neurons read the updated field state, re-condition, and write again — until logits converge or confidence gate fires.
**Routing and aggregation**: Confidence, quality, topology, and instance-level signals select an active subset. The selected neurons exchange field state and peer signals, then aggregate domain logits into the next token.

## Quick Start

### Install

```bash
# From the repository root
python -m pip install -e ".[dev]"
```

### Verify the core (30 tests)

```bash
python -m pytest tests/ -q
# Expected: 30 passed
```

### Run the reproducible population baseline

```bash
python scripts/verify_population_baseline.py --output reports/population_baseline.json
```

This uses a fixed seed and a tiny synthetic population to compare individual,
dense, and sparse resonance, then checks route observability, checkpoint
serialization, Cortex assembly, and the API health endpoint. Its metrics are
marked `synthetic_probe_only`; they validate the population runtime path, not
trained language quality.

### Run the API

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` or check `http://127.0.0.1:8000/api/health`.

### Train your first domain neuron

```bash
# Domain SFT fine-tune (dialogue neuron)
python scripts/training/finetune_neuron_dialogue.py

# Cross-domain collaboration layer
python scripts/training/train_cross_domain_collab.py

# Hub neuron (EXPERT spec, general 256K vocab, from scratch)
python scripts/training/train_hub_neuron.py
```

### Use Cortex (the orchestrator)

```python
from neuroplex.loader import assemble_cortex

cortex, tokenizer, modules = assemble_cortex(
    neurons_dir="data/neurons",
    device="cpu",
    max_rounds=2,
)
result = cortex.generate("今天天气怎么样？")
print(result)
```

If `data/neurons` is empty, NeuroPlex starts with a limited random fallback
neuron. Add trained domain checkpoints for useful generation quality.

## Neuron Specifications

| Spec | Hidden | Layers | Params | Role |
|------|--------|--------|--------|------|
| `compact` | 512 | 6 | ~24M | Auxiliary / domain-specific |
| `standard` | 768 | 10 | ~59M | Main executor |
| `expert` | 1024 | 14 | ~134M | Decision-maker / hub |

## Project Structure

```
NeuroPlex/
├── neuroplex/
│   ├── resonance/          # ★ Resonance field engine (core)
│   │   ├── neuron.py       #   ResonanceNeuron (per-neuron forward + field I/O)
│   │   ├── ensemble.py      #   Multi-round orchestration
│   │   ├── field.py        #   Resonance field (shared communication medium)
│   │   ├── translator.py   #   Cross-vocab translator
│   │   └── config.py       #   Neuron spec config
│   ├── brain/cortex.py     #   Cortex (top-level orchestrator)
│   ├── domains/            #   Per-domain tokenizers (zh/en/code/math/general)
│   ├── life/               #   Bio-inspired lifecycle (sleep consolidation, etc.)
│   ├── safety/             #   Output safety + sandbox
│   ├── tools/              #   Tool-use system
│   └── loader.py           #   Assembly + checkpoint loading
├── api/                    #   FastAPI server
├── frontend/               #   Vue 3 web UI
├── desktop/                #   PyQt6 desktop app
├── scripts/training/       #   Training scripts + verify_*.py tests
├── tests/                  #   pytest regression suite
└── plans/                  #   Architecture documentation
```

## Key Findings

- **1+1 > 2 is conditional**: Resonance helps only when neurons are uncertain. A `ConfidenceGate` skips resonance when max_prob > 0.9 (avoiding field noise on confident predictions).
- **Weak neurons dilute strong ones**: `QualityFilter` excludes PPL > 100 neurons from resonance. Scale-layering outperforms equal-weight consensus by 2.6× on code domain.
- **Optional relay anchoring**: An expert neuron can serve as a cross-domain relay, while `cross_spec_projectors` align field vectors into a unified space. The relay is a population member, not a central teacher.
- **Global anchoring > single-domain anchoring**: Computing anchor loss on ALL non-hub domains every batch (not just the current batch's domain) yielded ×2.3 mean cosine improvement.

## Architecture Documentation

- [BIO_INSPIRED_ARCHITECTURE_PLAN.md](plans/BIO_INSPIRED_ARCHITECTURE_PLAN.md) — Full architecture plan
- [ARCHITECTURE_DIRECTION_2026_08.md](plans/ARCHITECTURE_DIRECTION_2026_08.md) — Architecture comparison and current decision
- [DESIGN_PRINCIPLES.md](plans/DESIGN_PRINCIPLES.md) — Design principles
- [CODE_WIKI.md](CODE_WIKI.md) — Code wiki

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

NeuroPlex draws inspiration from:
- **MoCo** (momentum contrast) — dynamic logit fusion
- **SMCS** (subspace clustering) — instance-level routing
- **KoPE** (Kronecker-position encoding) — field vector phase encoding
- **BioOSS** (biologically-inspired OSS) — p/o dual neuron models
- **Hub-and-spoke** neuroscience — associative cortex topology
