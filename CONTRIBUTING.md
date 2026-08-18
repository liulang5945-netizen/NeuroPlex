# Contributing to NeuroPlex

Thank you for your interest in NeuroPlex! This document describes how to contribute.

## Getting Started

1. **Fork & Clone**
   ```bash
   git clone https://github.com/<your-username>/NeuroPlex.git
   cd NeuroPlex
   ```

2. **Install (development mode)**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify the core**
   ```bash
   python -m pytest tests/ -q
   # Expected: 16 passed
   ```

## Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes**
   - Code follows PEP 8 (enforced by `ruff`)
   - Functions use type hints
   - Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

3. **Run tests**
   ```bash
   python -m pytest tests/ -q
   ```

4. **Submit a Pull Request**
   - Describe what changed and why
   - Link to relevant issue(s)

## Areas for Contribution

- **New domain neurons** (e.g., science, legal, medical) — train and submit the checkpoint
- **Training improvements** — domain interleaving, LR scheduling, multi-epoch
- **Multi-language support** — add new tokenizers (Korean, Japanese, etc.)
- **Hardware acceleration** — CUDA/MPS support
- **Benchmark suite** — standard tasks for evaluating resonance benefits
- **Documentation** — tutorials, architecture deep-dives

## Architecture Notes

- **The population is the product**: Each domain or role neuron is an independently trainable Transformer. New capability should normally be added by specializing or adding neurons, not by growing a central backbone.
- **Resonance field is a communication medium**: Neurons write/read `field_vector`s, peer channels provide local coordination, and routing selects the active subset for each task.
- **Relay neurons are optional**: An expert neuron can provide cross-domain anchoring, but it remains one member of the population and must not become a hidden central teacher.
- **Lifecycle is part of the architecture**: memory replay, synaptic updates, maturation, neurogenesis, and apoptosis are first-class population mechanisms.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
