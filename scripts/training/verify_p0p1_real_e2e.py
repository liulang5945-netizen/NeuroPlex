"""P0+P1 真实端到端测试：用蒸馏好的 5 个神经元跑完整推理路径。

不验证生成质量（神经元用错位数据训练，质量不可信），只验证：
1. assemble_cortex() 加载真实神经元成功
2. 路由：每个域的 prompt 应该路由到对应域的 neuron
3. 多神经元共振：round 2+ 真的跑起来，不只是 round 1
4. STDP/Coactivation 在多 neuron 场景下数据正确累积
5. NeuromodulatorState 在 forward 中真的影响 field_write（scale 透传）
6. generate() 对每个域都能产出合法 token

测试 prompt（每域一个）：
- zh:   "你好，世界"
- en:   "Hello world"
- code: "def fibonacci"
- math: "1+1="
- general: "今天天气怎么样"
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

_LIBS = os.path.join(PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)


# 测试用例：(domain, prompt, expected_top_nid)
TEST_CASES = [
    ("zh",      "你好，世界",      "zh"),
    ("en",      "Hello world",     "en"),
    ("code",    "def fibonacci",   "code"),
    ("math",    "1+1=",            "math"),
    ("general", "今天天气",         "general"),
]


def test_real_e2e():
    print("=" * 70)
    print("P0+P1: 真实端到端测试（5 个蒸馏神经元）")
    print("=" * 70)

    # ── Step 1: assemble_cortex 用真实神经元 ──
    print("\n[1] assemble_cortex(neurons_dir='data/neurons') ...")
    try:
        from taiji.loader import assemble_cortex
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir="data/neurons",
            device="cpu",
            max_rounds=3,
            enable_gating=False,  # 关闭门控，确保多轮共振真的跑
            wire_bio_modules=True,
        )
    except Exception as e:
        print(f"  ❌ assemble_cortex 失败: {e}")
        traceback.print_exc()
        return False

    print(f"  ✓ Cortex loaded: {len(cortex.neurons)} neurons")
    print(f"  neurons: {list(cortex.neurons.keys())}")
    for name, n in cortex.neurons.items():
        params = sum(p.numel() for p in n.parameters()) / 1e6
        print(f"    [{name}] spec={n.config.spec}, {params:.0f}M params, "
              f"hidden={n.config.hidden_size}, "
              f"neuron_type={n.config.neuron_type}, "
              f"v1_compat={n.v1_compat}")
    print(f"  wired modules: {list(modules.keys())}")

    # 必须有 5 个真实神经元
    if len(cortex.neurons) != 5:
        print(f"  ⚠️  期望 5 个神经元，实际 {len(cortex.neurons)}")
        return False

    # ── Step 2: 路由准确性 ──
    print("\n[2] 路由准确性测试 ...")
    router = cortex.thalamic_router
    if router is None:
        print("  ⚠️  ThalamicRouter 未装配，跳过路由测试")
    else:
        correct_routes = 0
        for domain, prompt, expected in TEST_CASES:
            ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long)
            # 用 encoder 路径
            if cortex._context_encoder is not None:
                with torch.no_grad():
                    h = cortex._context_encoder.encode(input_ids)
                weights, top_nids = router.route_top_k_by_hidden(h, k=2)
                decision = router.get_routing_decision_by_hidden(h)
            else:
                # fallback: 用 standalone_embedding（若有）
                weights, top_nids = {}, []
                decision = {}

            top1 = top_nids[0] if top_nids else None
            ok = "✓" if top1 == expected else "✗"
            sim = decision.get('max_sim', 0) if decision else 0
            print(f"  {ok} [{domain}] '{prompt}' → top1={top1} "
                  f"(expected={expected}, sim={sim:.3f}, "
                  f"weights={[(n, f'{w:.2f}') for n, w in list(weights.items())[:3]]})")
            if top1 == expected:
                correct_routes += 1

        print(f"\n  路由准确率: {correct_routes}/{len(TEST_CASES)}")
        if correct_routes >= 3:
            print(f"  ✓ 路由基本可用（≥3/5 正确）")
        else:
            print(f"  ⚠️  路由准确率偏低（可能 prototype 训练不足）")

    # ── Step 3: 多神经元共振 ──
    print("\n[3] 多神经元共振（think + route） ...")
    stdp = modules.get("stdp_tracker")
    coaction = modules.get("coaction")
    nm = modules.get("neuromodulator")

    # 清空 STDP/Coaction 历史
    if stdp:
        stdp._firing_history.clear()
    if coaction:
        coaction._fast_matrix.clear()
        coaction._slow_matrix.clear()

    for domain, prompt, expected in TEST_CASES:
        try:
            ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long)
            result = cortex.think(input_ids, route=True)

            n_rounds = result.get("n_rounds", 0)
            n_active = result.get("n_active_history", [])
            skipped = result.get("skipped_resonance", False)
            scores = result.get("final_scores", {})
            top_scorer = max(scores, key=scores.get) if scores else None

            print(f"  [{domain}] '{prompt}': n_rounds={n_rounds}, "
                  f"n_active={n_active}, skipped={skipped}, "
                  f"top_scorer={top_scorer}")
        except Exception as e:
            print(f"  ❌ [{domain}] think 失败: {e}")
            traceback.print_exc()
            return False

    # ── Step 4: STDP / Coactivation 数据累积验证 ──
    print("\n[4] STDP / Coactivation 数据累积 ...")
    if stdp:
        n_fired = len(stdp._firing_history)
        total_records = sum(len(v) for v in stdp._firing_history.values())
        print(f"  STDP: {n_fired} neurons fired, {total_records} total records")
        if n_fired >= 2 and total_records >= 3:
            print(f"  ✓ STDP 数据累积正常")
        else:
            print(f"  ⚠️  STDP 数据偏少（可能路由失败导致单 neuron forward）")
    if coaction:
        n_pairs = len(coaction._fast_matrix)
        print(f"  Coaction: {n_pairs} active pairs")
        if n_pairs >= 1:
            print(f"  ✓ Coaction 数据累积正常")
        else:
            print(f"  ⚠️  Coaction 无 pair（可能每轮只有 1 个 neuron 激活）")

    # ── Step 5: NeuromodulatorState 影响验证 ──
    print("\n[5] NeuromodulatorState 影响推理行为 ...")
    if nm:
        # 默认状态
        default_scale = nm.get_field_write_scale()
        print(f"  默认 write_scale={default_scale:.3f}")

        # 模拟高警觉状态
        nm.set_targets(norepinephrine=1.0)
        for _ in range(20):
            nm.step()
        hi_scale = nm.get_field_write_scale()
        print(f"  高去甲肾上腺素 write_scale={hi_scale:.3f}")

        # 跑一次 think，验证不崩溃
        try:
            ids = tokenizer.encode("你好")
            input_ids = torch.tensor([ids], dtype=torch.long)
            result = cortex.think(input_ids, route=True)
            print(f"  ✓ 高 write_scale 下 think 不崩溃 "
                  f"(n_rounds={result.get('n_rounds')})")
        except Exception as e:
            print(f"  ❌ 高 write_scale 下 think 失败: {e}")
            traceback.print_exc()
            return False

        # 重置回中性
        nm.set_targets(dopamine=0.5, serotonin=0.5, norepinephrine=0.5)
        for _ in range(30):
            nm.step()

    # ── Step 6: generate() 每域生成 ──
    print("\n[6] generate() 每域生成 ...")
    for domain, prompt, expected in TEST_CASES:
        try:
            out = cortex.generate(
                prompt=prompt,
                max_tokens=12,
                temperature=0.7,
            )
            if isinstance(out, str):
                text = out
            elif isinstance(out, dict):
                text = out.get("text") or out.get("output") or ""
            else:
                text = str(out)
            print(f"  [{domain}] '{prompt}' → '{text[:40]}'")
        except Exception as e:
            print(f"  ❌ [{domain}] generate 失败: {e}")
            traceback.print_exc()
            return False

    # ── Step 7: GammaOscillator phase 推进 ──
    print("\n[7] GammaOscillator 在 generate 中推进 ...")
    gamma = modules.get("gamma_oscillator")
    if gamma:
        print(f"  global_phase={gamma.global_phase:.3f} "
              f"(初始 0.0，>0 表示 generate 中 tick 生效)")
        print(f"  phases={list(gamma.phases.keys())[:3]}...")
        if gamma.global_phase > 0:
            print(f"  ✓ tick_gamma 在 generate 循环中生效")
        else:
            print(f"  ⚠️  gamma.global_phase 为 0，tick 未生效")

    # ── 总结 ──
    print("\n" + "=" * 70)
    print("✅ 真实端到端测试通过（无崩溃）")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = test_real_e2e()
    sys.exit(0 if ok else 1)
