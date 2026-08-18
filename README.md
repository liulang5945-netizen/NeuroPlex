# NeuroPlex — Bio-inspired Multi-Neuron Resonance Architecture

> **Small neurons, together, match large models.**
>
> NeuroPlex replaces a single monolithic LLM with multiple domain-specific small neurons (24M–134M each) that collaborate through a shared resonance field. Instead of scaling one model, we scale the *collaboration* — each neuron stays independently trainable, hot-swappable, and CPU-affordable.

[![CI](https://github.com/NeuroPlex/NeuroPlex/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuroPlex/NeuroPlex/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Why NeuroPlex?

| Problem with monolithic LLM | NeuroPlex alternative |
|---|---|
| Retrain the whole 1.5B model to add one domain | Hot-swap one new neuron (24M), keep others frozen |
| GPU required for fine-tuning | Domain neurons trainable on CPU (~11s/step) |
| One model serves all domains equally | Domain-specific neurons specialize (zh/en/code/math) |
| Catastrophic forgetting on new data | Frozen body + LoRA per-neuron, zero cross-contamination |
| Black-box internal reasoning | Resonance rounds are inspectable (field vectors, gating, quality) |

## Core Idea

Three layers, each replaceable:

```
Level 0: Universal tokenizer (256K vocab)     ← I/O protocol layer
    ↓
Level 1: Domain neurons (24M–134M each)      ← Independent Transformers
    ↓  field_write / field_read
Level 2: Resonance field (3072–4096 dim)     ← Shared communication, not a model
```

**Round 1**: Each neuron reads the input independently, writes its `field_vector` to the shared field.
**Round 2–N**: Neurons read the updated field state, re-condition, and write again — until logits converge or confidence gate fires.
**Aggregation**: Cluster-dominant × scale-layering produces the final token logits.

## Quick Start

### Install

```bash
git clone https://github.com/NeuroPlex/NeuroPlex.git
cd NeuroPlex
pip install -e .
```

### Verify the core (16 tests)

```bash
python -m pytest tests/ -q
# Expected: 16 passed
```

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
from neuroplex.brain.cortex import Cortex

cortex = Cortex(neurons_dir="data/neurons")
result = cortex.generate("今天天气怎么样？")
print(result)
```

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
- **Hub-and-spoke anchoring**: A 1024-dim hub neuron serves as a cross-domain semantic anchor. Per-neuron `cross_spec_projectors` project domain field vectors into a unified space. Anchor loss aligns them; contrastive loss pulls same-meaning cross-domain pairs closer.
- **Global anchoring > single-domain anchoring**: Computing anchor loss on ALL non-hub domains every batch (not just the current batch's domain) yielded ×2.3 mean cosine improvement.

## Architecture Documentation

- [BIO_INSPIRED_ARCHITECTURE_PLAN.md](plans/BIO_INSPIRED_ARCHITECTURE_PLAN.md) — Full architecture plan
- [DESIGN_PRINCIPLES.md](plans/DESIGN_PRINCIPLES.md) — Design principles
- [TRAINING_REFERENCE.md](plans/TRAINING_REFERENCE.md) — Training reference
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
