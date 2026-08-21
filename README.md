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
  → balanced sparse cortical receptor bank
  → one byte motor organ
  → emitted action returns as the next sensation
```

| Transformer responsibility | Taiji Native v2 |
|---|---|
| tokenizer + learned embedding | 256 raw-byte receptors + boundary receptor |
| positional encoding | causal ticks and persistent state |
| self-attention | sparse reciprocal prediction and recurrent transitions |
| residual/FFN state | membrane integration, inhibition, adaptive thresholds, traces |
| KV cache | bounded dynamic state and learned transition synapses |
| global backpropagation | masked local prediction/state/motor deltas |
| LM head | all-state sparse receptor bank + one motor population |
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

The motor does not discard a random cortical subset. It concatenates every region's fast activity and slow trace into `s_t`, then a fixed balanced single-fan-out receptor map `H` folds every coordinate into `K` shared evidence channels:

```math
\tilde c_{t,k}=|G_k|^{-1/2}\sum_{j\in G_k}\sigma_j s_{t,j},
\qquad
c_t=\gamma_c\frac{\tilde c_t}{\lVert\tilde c_t\rVert_2+\epsilon}
```

Every cortical coordinate reaches exactly one receptor, and every action competes on the same `K` channels.

Every update is restricted by a fixed fan-in mask. There is no attention matrix, context window, optimizer, `backward()`, teacher model, or distillation path.

The complete tensor shapes, update order, state contract, complexity, and code mapping are in [the architecture specification](plans/active/TAIJI_SUBSTRATE_ARCHITECTURE.md).

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/training/verify_taiji_native_v2.py
python scripts/training/verify_taiji_n7_context.py
python scripts/training/verify_taiji_n8_delayed_trace.py
python scripts/training/verify_taiji_n9_long_free_run.py
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

## Reproducible Native v2 results

The committed verification uses two regions `[64, 48]`, seed `7`, and raw bytes:

| Metric | Result |
|---|---:|
| active sparse parameters | 19,521 |
| fixed receptor edges | 224 (one per cortical coordinate) |
| dense learned tensor storage | 38,513 |
| learned structural sparsity | 49.31% |
| byte-cycle accuracy | 0% → 94.12% |
| mean surprise | 5.4041 → 0.1090 |
| surprise reduction | 97.98% |
| free generation | `a → bcdabcda` (all eight steps correct) |
| checkpoint exact-next-step | pass |

On the N7 ambiguous stream, full Taiji predicts all eight history-dependent `x → b/d` successors correctly. A first-order model and a full dynamic-state lesion both score 50%. N7's trace-only lesion remains at 100%, showing that its immediate history can live in fast state.

N8 inserts the shared distractors `1234` between cue and probe. Full and trace-only states score 100%; removing trace or all dynamic state scores 50%. This establishes that slow trace is necessary and sufficient for this fixed-delay behavior, but it is not yet evidence of episodic or autobiographical memory.

N9 trains the same 16-byte cycle under an explicit non-terminal stream contract, then feeds back 128 motor actions with no teacher forcing. All 128 positions are exact, all four actions remain present, and membrane/trace/threshold bounds hold at every tick. A terminal boundary is deliberately excluded from this benchmark because teaching “stop after the fourth cycle” would contradict an infinite-cycle target.

Reports: [Native v2](reports/taiji_native_v2_20260821.json), [N7 context](reports/taiji_n7_context_20260821.json), [N8 delayed trace](reports/taiji_n8_delayed_trace_20260821.json), and [N9 free run](reports/taiji_n9_long_free_run_20260821.json).

## Source layout

```text
taiji/
├── config.py    architecture and dynamics contract
├── sparse.py    fixed fan-in synapses and local updates
├── state.py     persistent region and whole-system state
├── organs.py    raw-byte sensor, sparse receptor bank, and byte motor
├── fabric.py    predictive recurrent tick
└── model.py     observe, learn, score, generate, checkpoint

tests/taiji_native/                 native architecture contracts
scripts/training/verify_taiji_native_v2.py
scripts/training/verify_taiji_n7_context.py
scripts/training/verify_taiji_n8_delayed_trace.py
scripts/training/verify_taiji_n9_long_free_run.py
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

The next gate is N10: replace masked-dense region execution with a true edge-indexed sparse/event kernel while preserving exact topology, local updates, checkpoint continuation, and N5/N7/N8/N9 behavior.

## License

Apache License 2.0. See [LICENSE](LICENSE).
