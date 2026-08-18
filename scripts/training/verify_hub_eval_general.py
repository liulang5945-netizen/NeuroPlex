#!/usr/bin/env python3
"""hub 锚定效果评估（general 同款阵容口径，2026-08-16）。

修正 verify_hub_eval.py 的装配错位：该脚本装配 dialogue 变体 + dual 基座，
而 cross_domain_collab 训练用的是 general 基座（code/math/zh + hub）。
本脚本按训练同款阵容（general code/math/zh + hub, unified=3072）装配，
从 collab ckpt 注入 cross_spec 投影层，测 hub 锚点 cos——这才是有效口径。

对比三档：
- smoke 基线：无 collab 权重（随机投影层）
- w3.0：cross_domain_collab_verify_w3.ckpt.pt（原仅当前 batch 域锚定）
- global：cross_domain_collab_verify_global.ckpt.pt（全域锚定 ×3 频率）

用法：
    python -u scripts/training/verify_hub_eval_general.py
"""
from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from scripts.training.utils import load_general_tokenizer  # noqa: E402
import scripts.training.train_cross_domain_collab as tcdc  # noqa: E402
from scripts.training.verify_hub_collab_train import (  # noqa: E402
    build_ensemble_with_hub, make_batch,
)

SFT_DIR = "data/sft"
DOMAINS = ["code", "math", "zh"]
CKPTS = {
    "smoke (无 collab 权重)": None,
    "w3.0 (单域锚定)": "data/neurons/cross_domain_collab_verify_w3.ckpt.pt",
    "global (全域锚定)": "data/neurons/cross_domain_collab_verify_global.ckpt.pt",
    "full (正式全域锚定 17.5K步)": "data/neurons/cross_domain_collab_full.ckpt.pt",
}
FIELD_DIM = 3072  # 装配口径（训练 --unified-field-dim 3072 同款）


def inject_cross_spec(ensemble, ckpt_path: str) -> None:
    """从 collab ckpt 注入 cross_spec 投影层（锚定目标量所在）。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cs = ck.get("cross_spec_state") or ck.get("cross_spec") or {}
    forward = cs.get("forward", {})
    for nid, sd in forward.items():
        if nid in ensemble._cross_spec_projectors:
            ensemble._cross_spec_projectors[nid].load_state_dict(sd)


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("hub 锚定效果评估（general 同款阵容 code/math/zh + hub）", flush=True)
    print(f"field_dim={FIELD_DIM}，修正 verify_hub_eval 装配错位口径", flush=True)
    print("=" * 60, flush=True)

    neurons, shared_embeddings, ensemble, general_sp, _ = build_ensemble_with_hub(
        nids=DOMAINS, field_dim=FIELD_DIM)

    data = torch.load(os.path.join(SFT_DIR, "code_sft.pt"),
                      map_location="cpu", weights_only=False)
    texts = [d["full"] for d in data][:8]
    neuron_embeddings, _, _ = make_batch(texts, general_sp, shared_embeddings, seq_len=32)

    print("\n装配:", ", ".join(sorted(neurons.keys())), flush=True)
    print("cross_spec 投影层:", sorted(ensemble._cross_spec_projectors.keys()), flush=True)

    results = {}
    with torch.no_grad():
        for label, ckpt_path in CKPTS.items():
            if ckpt_path is not None:
                inject_cross_spec(ensemble, ckpt_path)
            cos_map = {}
            for nid in DOMAINS:
                v_d = neurons[nid].forward(neuron_embeddings[nid], round_num=1)["field_vector"]
                v_h = neurons["hub"].forward(neuron_embeddings["hub"], round_num=1)["field_vector"]
                if nid in ensemble._cross_spec_projectors:
                    v_d = ensemble._cross_spec_projectors[nid](v_d)
                if "hub" in ensemble._cross_spec_projectors:
                    v_h = ensemble._cross_spec_projectors["hub"](v_h)
                cos_map[nid] = float(F.cosine_similarity(v_d, v_h, dim=-1).mean().item())
            results[label] = cos_map
            mean = sum(cos_map.values()) / len(cos_map)
            detail = "  ".join(f"{k}={v:+.3f}" for k, v in cos_map.items())
            print(f"\n[{label}] hub 锚点 cos 均值 {mean:+.3f}", flush=True)
            print(f"    {detail}", flush=True)

    print("\n对比（全域锚定相对单域锚定的提升）：", flush=True)
    for nid in DOMAINS:
        s = results["smoke (无 collab 权重)"][nid]
        w3 = results["w3.0 (单域锚定)"][nid]
        g = results["global (全域锚定)"][nid]
        f = results["full (正式全域锚定 17.5K步)"][nid]
        print(f"  {nid}: smoke {s:+.3f} → w3.0 {w3:+.3f} → global {g:+.3f} → full {f:+.3f} "
              f"(Δw3 {w3 - s:+.3f}, Δglobal {g - w3:+.3f}, Δfull {f - g:+.3f})", flush=True)

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
