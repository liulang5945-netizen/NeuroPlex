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

- **Neurons are independent**: Each domain neuron is a standalone Transformer with its own tokenizer and vocab. You can add one without touching others.
- **Resonance field is not a model**: It's a shared communication medium (a tensor). Neurons write/read `field_vector`s to/from it. The field has no parameters of its own (except optional projection layers).
- **Hub neuron** (1024-dim, 14-layer, 134M params) serves as a cross-domain semantic anchor. It's the only neuron that spans all domains.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
