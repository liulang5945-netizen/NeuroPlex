# Taiji — Native Persistent Predictive Computing

Taiji is an experimental non-Transformer sequence architecture with its own input representation, persistent state transition, local learning rule, motor output, free-running generation loop, and checkpoint format.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

> Current status: executable research prototype, not an AGI claim and not a language-quality replacement yet.

## What Taiji replaces

Taiji does not wrap a Transformer in neuron terminology. The native path is:

```text
raw byte receptors
  → hierarchical reciprocal prediction errors
  → persistent recurrent region states
  → one byte motor organ
  → emitted action returns as the next sensation
```

| Transformer responsibility | Taiji Native v1 |
|---|---|
| tokenizer + learned embedding | 256 raw-byte receptors + boundary receptor |
| positional encoding | causal ticks and persistent state |
| self-attention | sparse reciprocal prediction and recurrent transitions |
| residual/FFN state | membrane integration, inhibition, adaptive thresholds, traces |
| KV cache | bounded dynamic state and learned transition synapses |
| global backpropagation | masked local prediction/state/motor deltas |
| LM head | one motor population |
| autoregressive decode | motor byte fed back through the same sensor |

The implementation imports neither `transformers` nor the legacy `neuroplex` runtime. PyTorch is used only as a tensor execution engine.

## Algorithm

For region `r`, the previous local trace predicts both the current lower-level activity and the region's own next activity:

```math
\hat y_t^{r-1}=D^r q_{t-1}^r, \qquad e_t^{r-1}=y_t^{r-1}-\hat y_t^{r-1}
```

```math
\hat a_t^r=T^r q_{t-1}^r
```

The region integrates bottom-up error, recurrent prediction, and delayed top-down context:

```math
u_t^r=Bound(\lambda_u u_{t-1}^r+\alpha_g(D^r)^Te_t^{r-1}+\alpha_T\hat a_t^r+\alpha_c c_t^r)
```

Activity is formed through an adaptive threshold and a local inhibitory pool. Learning is online and local:

```math
\Delta D^r=\eta_D e_t^{r-1}(q_{t-1}^r)^T
```

```math
\Delta T^r=\eta_T(a_t^r-T^rq_{t-1}^r)(q_{t-1}^r)^T
```

```math
\Delta M=\eta_M(onehot(b_t)-p_{t-1})c_{t-1}^T
```

Every update is restricted by a fixed fan-in mask. There is no attention matrix, context window, optimizer, `backward()`, teacher model, or distillation path.

The complete tensor shapes, update order, state contract, complexity, and code mapping are in [the architecture specification](plans/active/TAIJI_SUBSTRATE_ARCHITECTURE.md).

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/training/verify_taiji_native_v1.py
python -m pytest tests/taiji_native -q
```

Minimal API:

```python
from taiji import Taiji

model = Taiji()
model.learn_bytes(b"abcdabcdabcdabcd", epochs=200)

print(model.score_bytes(b"abcdabcdabcdabcd"))
print(model.generate(b"a", length=8))

checkpoint = model.checkpoint()
restored = Taiji.from_checkpoint(checkpoint)
```

## Reproducible Native v1 result

The committed verification uses two regions `[64, 48]`, seed `7`, and raw bytes:

| Metric | Result |
|---|---:|
| active sparse parameters | 19,521 |
| dense tensor storage | 54,961 |
| structural sparsity | 64.48% |
| byte-cycle accuracy | 0% → 76.47% |
| mean surprise | 5.5622 → 1.0484 |
| surprise reduction | 81.15% |
| short free generation | `a → bcdaccbd` (first four steps correct) |
| checkpoint exact-next-step | pass |

This is evidence that the algorithm can learn a tiny stream. It is not evidence of language understanding: generation currently drifts after four steps, and second-order context has not yet passed its lesion test.

Report: [reports/taiji_native_v1_20260821.json](reports/taiji_native_v1_20260821.json).

## Source layout

```text
taiji/
├── config.py    architecture and dynamics contract
├── sparse.py    fixed fan-in synapses and local updates
├── state.py     persistent region and whole-system state
├── organs.py    raw-byte sensor and byte motor
├── fabric.py    predictive recurrent tick
└── model.py     observe, learn, score, generate, checkpoint

tests/taiji_native/                 native architecture contracts
scripts/training/verify_taiji_native_v1.py
plans/active/TAIJI_SUBSTRATE_ARCHITECTURE.md
```

## Legacy NeuroPlex

`neuroplex/` contains the previous nine-member Transformer population, including all five dialogue members. It is retained only for reproducibility and future same-budget comparisons. It is not imported by Taiji.

Historical checkpoints that serialized `taiji.*` names are loaded through the scoped `neuroplex.legacy_checkpoint` compatibility utility; importing NeuroPlex no longer shadows the native `taiji` package.

Install legacy application dependencies only when reproducing that baseline:

```bash
python -m pip install -e ".[legacy]"
```

## Current falsification target

The next gate is N7: the same current byte must lead to different correct successors under different histories, and clearing the temporal trace must remove that advantage. The project does not scale parameters or download larger text data until that context mechanism is demonstrated.

## License

Apache License 2.0. See [LICENSE](LICENSE).
