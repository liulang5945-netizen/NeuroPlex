"""C20 回合级质量监督验证（临时脚本，验证后清理）。

验证目标：
1. C20 训练后 quality_head 的回合级质量信号——不同域回合文本 → per-domain
   聚合 quality，best_q_domain 是否修正（math→math 等启发式误判场景）
2. 对比 C16 head（collab_v3_c16.ckpt.pt）vs C20 head（collab_v3_c20.ckpt.pt）
3. 完整 _executive_route 混合信号（EMA 预热后 quality 切换生效）

用法：
    python scripts/training/verify_c20_round_quality.py [--ckpt20 data/neurons/collab_v3_c20.ckpt.pt]
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.loader import assemble_cortex

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("dialogue", "你好，请介绍一下你自己"),
    ("en", "What is the capital of France?"),
]


def quality_probe(cortex, text):
    """直接 probe round1 quality_logits（不经 EMA），per-domain 聚合。"""
    general_ids = cortex._general_sp.encode(text)
    if not general_ids:
        general_ids = [0]
    ids_t = torch.tensor([general_ids], dtype=torch.long, device=cortex.device)
    nem = {nid: emb(ids_t) for nid, emb in cortex._neuron_shared_embeddings.items()}
    probe = cortex.think(active_nids=list(cortex.neurons.keys()), neuron_embeddings=nem)
    ql = probe.get("quality_logits")
    if ql is None:
        return None, {}
    nids = list(cortex.neurons.keys())
    assert len(ql) == len(nids), f"ql len {len(ql)} != nids {len(nids)}"
    per_domain = {}
    for i, nid in enumerate(nids):
        d = nid.split("_")[0]
        per_domain.setdefault(d, []).append(float(ql[i].detach()))
    scores = {d: sum(v) / len(v) for d, v in per_domain.items()}
    best = max(scores, key=scores.get) if scores else None
    return best, scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt20", default="data/neurons/collab_v3_c20.ckpt.pt")
    args = parser.parse_args()

    DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                    "zh_aug3_dialogue", "zh_std0_dialogue"]
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c16.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_general",
        neuron_ids=DIALOGUE_IDS,
    )
    print(f"[assemble_cortex] neurons: {list(cortex.neurons.keys())}")

    # 1. C16 head 基线
    print("\n=== C16 head 回合级 quality（基线）===")
    for tag, prompt in PROMPTS:
        best, scores = quality_probe(cortex, prompt)
        q_str = ", ".join(f"{d}={v:.2f}" for d, v in sorted(scores.items()))
        print(f"[{tag:<8}] best_q={best} | quality: {q_str}")

    # 2. 注入 C20 head
    print(f"\n=== 注入 C20 head（{args.ckpt20}）===")
    ck20 = torch.load(args.ckpt20, map_location="cpu", weights_only=False)
    hs = ck20.get("head_state", {})
    loaded = 0
    for nid, neuron in cortex.neurons.items():
        if nid in hs and getattr(neuron, "quality_head", None) is not None:
            neuron.quality_head.load_state_dict(hs[nid])
            loaded += 1
    print(f"  注入 {loaded}/{len(cortex.neurons)} head")

    # 3. C20 head 回合级 quality
    print("\n=== C20 head 回合级 quality ===")
    for tag, prompt in PROMPTS:
        best, scores = quality_probe(cortex, prompt)
        q_str = ", ".join(f"{d}={v:.2f}" for d, v in sorted(scores.items()))
        print(f"[{tag:<8}] best_q={best} | quality: {q_str}")

    # 4. 完整 _executive_route（EMA 预热后混合信号生效）
    # 预热用多样文本（单一文本会把 EMA mean 偏置到该文本，z-score 失真）
    print("\n=== _executive_route 混合信号（预热 30 次，多样文本）===")
    for k in range(30):
        cortex._executive_route(PROMPTS[k % len(PROMPTS)][1])
    for tag, prompt in PROMPTS:
        dom, conf, per_dom = cortex._executive_route(prompt)
        q_str = ", ".join(f"{d}={v:.2f}" for d, v in sorted(per_dom.items()))
        print(f"[{tag:<8}] → {dom} (conf={conf:.2f}) | quality z: {q_str}")

    # 5. C20 判定修正后的 executive 端到端生成（vs fusion）
    print("\n=== 生成对比（40 token, temp 0.9）===")
    for tag, prompt in PROMPTS:
        try:
            out_exec = cortex.generate(
                prompt, max_tokens=40, temperature=0.9, top_k=50,
                collab_mode="executive",
            )
            out_fusion = cortex.generate(
                prompt, max_tokens=40, temperature=0.9, top_k=50,
                collab_mode="fusion",
            )
            print(f"\n── [{tag}] {prompt}")
            print(f"  executive: {out_exec}")
            print(f"  fusion   : {out_fusion}")
        except Exception as e:
            print(f"\n── [{tag}] ERROR: {e}")


if __name__ == "__main__":
    main()
