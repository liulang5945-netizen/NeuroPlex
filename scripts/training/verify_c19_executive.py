"""C19 任务级路由冒烟验证（临时脚本，验证后清理）。

验证目标：
1. _executive_route 回合级判定（混合信号：启发式 + quality_head 回合级聚合）
2. executive 模式生成（dominant 域稳定生成，无 token 级竞争）vs fusion 模式
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("dialogue", "你好，请介绍一下你自己"),
    ("en", "What is the capital of France?"),
]


def main():
    DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                    "zh_aug3_dialogue", "zh_std0_dialogue"]
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c16.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_general",
        neuron_ids=DIALOGUE_IDS,  # 显式 9 阵容（排除 zh_general 旧产物干扰）
    )
    print(f"[assemble_cortex] neurons: {list(cortex.neurons.keys())}")

    # 1. 回合级判定
    print("\n=== C19 回合级判定（_executive_route）===")
    # 诊断：quality_head 注入 + probe
    print("[diag] quality_head 存在:", {
        nid: getattr(n, "quality_head", None) is not None for nid, n in cortex.neurons.items()
    })
    print("[diag] _neuron_shared_embeddings:", list(cortex._neuron_shared_embeddings.keys()) if cortex._neuron_shared_embeddings else "EMPTY")
    try:
        import torch
        gids = cortex._general_sp.encode(PROMPTS[0][1])
        ids_t = torch.tensor([gids], dtype=torch.long, device=cortex.device)
        nem = {nid: emb(ids_t) for nid, emb in cortex._neuron_shared_embeddings.items()}
        probe = cortex.think(active_nids=list(cortex.neurons.keys()), neuron_embeddings=nem)
        ql = probe.get("quality_logits")
        print("[diag] probe quality_logits:", ql)
        if ql is not None:
            print("[diag] nids:", list(cortex.neurons.keys()), "ql len:", len(ql))
        # 逐 neuron 检查 quality_logit
        for nid, n in cortex.neurons.items():
            try:
                r1 = n.forward(nem[nid], round_num=1, return_logits=False)
                print(f"  [per-neuron] {nid}: quality_logit={'present' if 'quality_logit' in r1 else 'MISSING'}, qh={'yes' if getattr(n,'quality_head',None) is not None else 'NO'}")
            except Exception as e:
                print(f"  [per-neuron] {nid}: ERROR {type(e).__name__}: {e}")
        print("[diag] probe final_scores:", {k: round(float(v), 3) for k, v in probe.get("final_scores", {}).items()})
    except Exception as e:
        import traceback; traceback.print_exc()

    for tag, prompt in PROMPTS:
        dom, conf, per_dom = cortex._executive_route(prompt)
        q_str = ", ".join(f"{d}={v:.2f}" for d, v in sorted(per_dom.items()))
        print(f"[{tag:<8}] → {dom} (conf={conf:.2f}) | quality: {q_str}")

    # 2. executive vs fusion 生成
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
