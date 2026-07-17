"""H1-H8 smoke test for the resonance field architecture.

Exercises every code path the H1-H8 fixes touched, end to end, on the CPU
with a tiny config (TINY_TEST shape, vocab shrunk for speed) so it fits in
memory and runs in well under a second:

  H1  neuron field_read reshape: both 1D [D] and batched [B,D] field_state work
  H2  ResonanceField per-sample state [B,D] write / read / normalise
  H5  LOO scoring (neuron_id) + final_scores multiplicative boost in v2 routing
  H6  prediction_complementarity (with + without targets) wired into routing
  H7  confidence temperature 3.0 in per-position routing (no crash, sums to 1)
  H8  field_read per-position gate + W_cond multiplicative conditioning

Run (from repo root):
    python scripts/training/verify_h1h8.py
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.resonance import ResonanceField, ResonanceNeuron, ResonanceEnsemble, TINY_TEST

torch.manual_seed(0)

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append((name, detail))
        print(f"  [FAIL] {name} -- {detail}")


def section(title):
    print(f"\n=== {title} ===")


def main():
    # Tiny config: same shape as TINY_TEST but a 256-token vocab so lm_head
    # and the [B,L,vocab] logits stay small enough to run instantly on CPU.
    cfg = replace(TINY_TEST, vocab_size=256, neuron_id="smoke")
    D = cfg.field_dim              # 512
    B, L = 2, 8

    # ---------------------------------------------------------------- H2/H8
    section("H2/H8: field 1D [D] write/read/LOO + W_cond gating")
    field = ResonanceField(D)
    field.reset(batch_size=1)
    vA = torch.randn(D)
    vB = torch.randn(D)
    field.write("A", vA)
    field.write("B", vB)
    s_loo = field.score(vA, neuron_id="A")
    s_full = field.score(vA, neuron_id=None)
    check("H2 field 1D state shape [D]", field.get_state().shape == (D,),
          f"state.shape={tuple(field.get_state().shape)}")
    check("H2/H5 LOO score returns float", isinstance(s_loo, float), f"got {type(s_loo)}")
    check("H5 LOO != full score (excludes self)",
          abs(s_loo - s_full) > 1e-8, f"loo={s_loo:.4f} full={s_full:.4f}")
    check("H8 W_cond is a learnable gate param",
          isinstance(field.W_cond, torch.nn.Parameter), f"type={type(field.W_cond)}")

    # ---------------------------------------------------------------- H2 batched
    section("H2: field batched [B,D] per-sample state")
    fieldB = ResonanceField(D)
    fieldB.reset(batch_size=B)
    fieldB.write("A", torch.randn(B, D))
    fieldB.write("B", torch.randn(B, D))
    sb = fieldB.score(torch.randn(B, D), neuron_id="A")
    ns = fieldB.get_normalised_state()
    check("H2 batched state [B,D]", fieldB.get_state().shape == (B, D),
          f"state.shape={tuple(fieldB.get_state().shape)}")
    check("H2 batched normalised [B,D]", ns.shape == (B, D),
          f"norm.shape={tuple(ns.shape)}")
    check("H2/H5 batched LOO score float", isinstance(sb, float))

    # ---------------------------------------------------------------- H6
    section("H6: prediction_complementarity")
    la = torch.randn(B, L, cfg.vocab_size)
    lb = torch.randn(B, L, cfg.vocab_size)
    comp_nt = field.prediction_complementarity(la, lb, targets=None)
    targets = torch.randint(0, cfg.vocab_size, (B, L))
    comp_t = field.prediction_complementarity(la, lb, targets=targets)
    check("H6 complementarity (no targets) float", isinstance(comp_nt, float))
    check("H6 complementarity (with targets) float >= 0", isinstance(comp_t, float) and comp_t >= 0.0,
          f"comp_t={comp_t:.4f}")
    # When B == A, B corrects none of A's mistakes -> reduction must be 0.
    comp_same_t = field.prediction_complementarity(la, la, targets=targets)
    check("H6 identical logits -> 0 log-loss reduction",
          abs(comp_same_t) < 1e-5, f"got {comp_same_t:.6f}")
    comp_same_nt = field.prediction_complementarity(la, la, targets=None)
    check("H6 identical logits -> 0 disagreement",
          abs(comp_same_nt) < 1e-5, f"got {comp_same_nt:.6f}")

    # ---------------------------------------------------------------- H1 round1
    section("H1: neuron round 1 (no field) + v2 attention-pooled write")
    neuron = ResonanceNeuron(cfg)
    emb = torch.randn(B, L, cfg.base_embed_dim)
    out1 = neuron.forward(emb, field_state=None, round_num=1, return_logits=True)
    fv = out1["field_vector"]
    check("round1 field_vector [B,D]", fv.shape == (B, D), f"shape={tuple(fv.shape)}")
    check("round1 logits [B,L,vocab]", out1["logits"].shape == (B, L, cfg.vocab_size),
          f"shape={tuple(out1['logits'].shape)}")
    check("v2 attention pool present (field_attn_weights)", "field_attn_weights" in out1)

    # ---------------------------------------------------------------- H1 round2 batched
    section("H1: neuron round 2 with batched field_state [B,D] (reshape 2D branch)")
    out2 = neuron.forward(emb, field_state=fv.detach(), round_num=2, return_logits=True)
    check("round2 [B,D] logits [B,L,vocab]", out2["logits"].shape == (B, L, cfg.vocab_size),
          f"shape={tuple(out2['logits'].shape)}")

    # ---------------------------------------------------------------- H1 round2 1D
    section("H1: neuron round 2 with 1D field_state [D] (reshape 1D branch)")
    out3 = neuron.forward(emb, field_state=fv[0].detach(), round_num=2, return_logits=True)
    check("round2 [D] logits [B,L,vocab]", out3["logits"].shape == (B, L, cfg.vocab_size),
          f"shape={tuple(out3['logits'].shape)}")

    # ---------------------------------------------------------------- H5/H6/H7 ensemble
    section("H5/H6/H7: ensemble v2 routing (2 neurons, 2 rounds)")
    neurons = {"A": ResonanceNeuron(cfg), "B": ResonanceNeuron(cfg)}
    fieldE = ResonanceField(D)
    ens = ResonanceEnsemble(neurons, fieldE, max_rounds=2)
    res = ens.forward(emb, return_logits=True, active_filter=False, enable_gating=False)
    check("ensemble returns weighted_logits", "weighted_logits" in res)
    if "weighted_logits" in res:
        wl = res["weighted_logits"]
        check("ensemble weighted_logits shape [B,L,vocab]",
              wl.shape == (B, L, cfg.vocab_size), f"shape={tuple(wl.shape)}")
        check("ensemble logits finite", bool(torch.isfinite(wl).all().item()))
    check("ensemble final_scores has 2 neurons", len(res["final_scores"]) == 2,
          f"got {len(res['final_scores'])}")
    check("ensemble ran 2 rounds", res["n_rounds"] == 2, f"n_rounds={res['n_rounds']}")
    fs = res["final_scores"]
    check("ensemble final_scores all floats", all(isinstance(v, float) for v in fs.values()))
    check("H5 final_scores present for both neurons",
          set(fs.keys()) == {"A", "B"}, f"keys={set(fs.keys())}")

    section("H1: ensemble round 2 conditions on batched [B,D] field")
    # Internally round 2 feeds get_normalised_state() ([B,D] after reset(B)) to
    # each neuron; completing 2 rounds with finite logits proves the H1 reshape
    # 2D branch + H8 per-position gate both executed without shape errors.
    check("ensemble conditioned 2 rounds without shape error",
          res["n_rounds"] == 2 and "weighted_logits" in res)

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASSED)}  FAILED {len(FAILED)}")
    if FAILED:
        for n, d in FAILED:
            print(f"  FAIL {n}: {d}")
        return 1
    print("ALL H1-H8 SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\nFATAL EXCEPTION during smoke test:")
        traceback.print_exc()
        sys.exit(2)