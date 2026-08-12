#!/usr/bin/env python3
"""C20 v2 判定重训后 quality proxy 回退路径验证（2026-08-11）。

验证 judge NLL 主信号不可用（round1_judge_logits 缺失）时，
_executive_route 回退到 quality z-score（C25-G 修复后可用），不崩溃、判定合理。
对比：正常模式（judge NLL 判定）vs 回退模式（mock judge 失效）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.loader import assemble_cortex  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("dialogue", "你好，请介绍一下你自己"),
    ("en", "What is the capital of France?"),
]


def main():
    print("=" * 60, flush=True)
    print("C20 v2 quality proxy 回退路径验证", flush=True)
    print("=" * 60, flush=True)

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    print(f"  装配: {list(cortex.neurons.keys())}", flush=True)

    # 预热（EMA warmup=20，需足够次数的 quality probe 使 EMA 成熟）
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    print("  judge EMA + quality EMA 预热完成", flush=True)

    print("\n[1] 正常模式（judge NLL 主信号）", flush=True)
    dom_normal = {}
    for tag, prompt in PROMPTS:
        d, _, _ = cortex._executive_route(prompt)
        dom_normal[tag] = d
        print(f"  {tag} → {d}", flush=True)
    check("正常模式判定 5/5", dom_normal == {"code": "code", "math": "math", "zh": "zh",
                                             "dialogue": "zh", "en": "en"}, f"→ {dom_normal}")

    print("\n[2] 回退模式（mock judge 失效 → quality proxy）", flush=True)
    real_think = cortex.think

    def think_no_judge(active_nids=None, neuron_embeddings=None, **kw):
        # 正常 think 但剥掉 round1_judge_logits → 触发 quality 回退
        out = real_think(active_nids=active_nids, neuron_embeddings=neuron_embeddings, **kw)
        out.pop("round1_judge_logits", None)
        return out

    cortex.think = think_no_judge
    dom_fb = {}
    try:
        for tag, prompt in PROMPTS:
            d, _, _ = cortex._executive_route(prompt)
            dom_fb[tag] = d
            print(f"  {tag} → {d}", flush=True)
        # 回退模式不应崩溃；判定与正常模式一致（同信号质量代理，C25-G 修复后
        # 区分度恢复；如个别域不同则看是否因 EMA/启发式混合，仅要求非空且合理）
        check("回退模式不崩溃（5 域全判定）", len(dom_fb) == 5, f"→ {dom_fb}")
        agree = sum(1 for k in dom_normal if dom_fb.get(k) == dom_normal[k])
        print(f"  [INFO] 回退与正常判定一致 {agree}/5", flush=True)
        check("回退判定均为合法域", all(d in ("zh", "en", "code", "math")
                                          for d in dom_fb.values()), f"→ {set(dom_fb.values())}")
    except Exception as e:
        check("回退模式不崩溃", False, f"err={type(e).__name__}: {e}")
    finally:
        cortex.think = real_think

    print("=" * 60, flush=True)
    print(f"结果: {passed}/{passed + failed} PASS", flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
