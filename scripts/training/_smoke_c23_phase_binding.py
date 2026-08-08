"""C23 相位同步本体化冒烟验证（_smoke_c23_phase_binding.py）。

验证目标：
1. GammaOscillator.pairwise_binding：同相群体 → +1，异相 → -1，混合正确
2. Kuramoto 闭环：共激活强的 neuron 对相位相互牵引 → 相位差缩小 → binding 上升
   （"共激活 → 相位同步 → 绑结"动态闭环）
3. binding 调制共振分：同相群体 scores 相对增强（(1 + bs·binding) 乘法调制）
4. 推理/训练接入存在性：ensemble.forward 与 forward_train 的 gamma 分支
   使用 pairwise_binding（源码级断言）

运行：python scripts/training/_smoke_c23_phase_binding.py
预期：6/6 PASS
"""
import math
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance.gamma_oscillator import GammaOscillator


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        raise SystemExit(1)


def test_1_same_phase_binding_positive():
    """同相群体 binding → +1。"""
    osc = GammaOscillator(binding_scale=0.3)
    osc.assign_phase("zh_1", 0.0)
    osc.assign_phase("zh_2", 0.0)
    osc.assign_phase("zh_3", 0.0)
    b = osc.pairwise_binding(["zh_1", "zh_2", "zh_3"])
    check("同相群体 binding≈+1", all(abs(b[n] - 1.0) < 1e-9 for n in b),
          f"binding={ {k: round(v, 3) for k, v in b.items()} }")


def test_2_opposite_phase_binding_negative():
    """异相群体（π）binding → -1。"""
    osc = GammaOscillator()
    osc.assign_phase("zh_1", 0.0)
    osc.assign_phase("en_1", math.pi)
    b = osc.pairwise_binding(["zh_1", "en_1"])
    check("异相群体 binding≈-1", abs(b["zh_1"] + 1.0) < 1e-9 and abs(b["en_1"] + 1.0) < 1e-9,
          f"binding={ {k: round(v, 3) for k, v in b.items()} }")


def test_3_mixed_domain_binding():
    """同相群体（3 个 zh）vs 反相（1 个 en，π）：zh 平均 binding 显著更高。"""
    osc = GammaOscillator()
    osc.assign_phase("zh_1", 0.0)
    osc.assign_phase("zh_2", 0.0)
    osc.assign_phase("zh_3", 0.0)
    osc.assign_phase("en_1", math.pi)  # 反相
    b = osc.pairwise_binding(["zh_1", "zh_2", "zh_3", "en_1"])
    # 同相 zh 群体平均 binding 高（与 2 个同相 zh 的 cos=1 拉高），en 被解绑
    zh_binding = (b["zh_1"] + b["zh_2"] + b["zh_3"]) / 3
    check("同相群体平均 binding 高于异相", zh_binding > b["en_1"],
          f"zh_avg={zh_binding:.3f} en_1={b['en_1']:.3f}")


def test_4_kuramoto_binding_loop():
    """Kuramoto 闭环：共激活强 → 相位牵引 → 同步 → binding 上升。"""
    osc = GammaOscillator(binding_scale=0.3)
    # 初始相位差较大（0 vs 2.0 rad）
    osc.assign_phase("a", 0.0)
    osc.assign_phase("b", 2.0)

    class FakeCoact:
        def get_coactivation(self, i, j):
            return 0.9  # 强共激活 → 强牵引

    b0 = osc.pairwise_binding(["a", "b"], coactivation=FakeCoact())
    for _ in range(60):  # 多轮 Kuramoto 牵引
        osc.kuramoto_step(coupling_strength=0.5, active_ids=["a", "b"], coactivation=FakeCoact())
    b1 = osc.pairwise_binding(["a", "b"], coactivation=FakeCoact())
    check("Kuramoto 牵引后 binding 上升", b1["a"] > b0["a"],
          f"b0={b0['a']:.3f} → b1={b1['a']:.3f}")


def test_5_binding_modulates_scores():
    """binding 调制共振分：同相群体 scores 相对增强，异相衰减。"""
    osc = GammaOscillator(binding_scale=0.5)
    osc.assign_phase_by_domain({"zh": ["zh_1", "zh_2"], "en": ["en_1"]})
    b = osc.pairwise_binding(["zh_1", "zh_2", "en_1"])
    scores = {"zh_1": 0.5, "zh_2": 0.5, "en_1": 0.5}  # 初始等分
    bs = osc.binding_scale
    modulated = {n: s * (1.0 + bs * b[n]) for n, s in scores.items()}
    # 同相 zh 群体增强；en（跨域相位）相对最弱
    check("同相群体增强、异相衰减",
          modulated["zh_1"] > scores["zh_1"] and modulated["zh_2"] > scores["zh_2"]
          and modulated["en_1"] < modulated["zh_1"],
          f"调制后 zh_1={modulated['zh_1']:.3f} zh_2={modulated['zh_2']:.3f} "
          f"en_1={modulated['en_1']:.3f}（bs={bs}）")


def test_6_wired_into_ensemble():
    """源码级断言：ensemble.forward / forward_train 已使用 pairwise_binding。"""
    import inspect
    from taiji.resonance import ensemble as en
    fwd_src = inspect.getsource(en.ResonanceEnsemble.forward)
    fwdtr_src = inspect.getsource(en.ResonanceEnsemble.forward_train)
    check("forward 已接入 pairwise_binding", "pairwise_binding" in fwd_src)
    check("forward_train 已接入 pairwise_binding", "pairwise_binding" in fwdtr_src)


if __name__ == "__main__":
    test_1_same_phase_binding_positive()
    test_2_opposite_phase_binding_negative()
    test_3_mixed_domain_binding()
    test_4_kuramoto_binding_loop()
    test_5_binding_modulates_scores()
    test_6_wired_into_ensemble()
    print("\nC23 冒烟 6/6 PASS")
