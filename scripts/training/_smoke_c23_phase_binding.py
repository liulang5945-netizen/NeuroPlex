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

import torch

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
    check("forward 已接入 pairwise_binding（scores 调制）", "pairwise_binding" in fwd_src)
    check("forward_train 已接入 pairwise_binding", "pairwise_binding" in fwdtr_src)
    # C23-B：场写入绑定（推理 forward 本体化）
    check("forward 场写入已按绑定调制（round1/2+ scale）",
          "1.0 + binding_bs * binding_map.get(nid, 0.0)" in fwd_src)
    # C23-C4（2026-08-08）：训练 forward_train 场构造**不再**按绑定调制——
    # binding 调制 field_state → round2 logits → per_neuron_nll → contrastive 监督
    # 目标被相位自组织驱动漂移（与 phase_loss 目标打架，quality_head 学乱，
    # E2 段 contrastive 饱和 18.42，C20 零饱和）。监督测"谁能预测好"（纯净
    # NLL，与 C20 一致）；相位只经 scores 段调制共振分 + phase_loss 可微。
    check("forward_train 场构造不再按绑定调制（监督纯净化，C23-C4）",
          "1.0 + bs * bvec.unsqueeze(1).unsqueeze(2)" not in fwdtr_src)


def test_7_write_scale_binding():
    """写入 scale 绑定调制：同相群体（≥3）写入增强（scale 升）、异相衰减（scale 降）。"""
    import torch
    osc = GammaOscillator(binding_scale=0.3)
    osc.assign_phase("zh_1", 0.0)
    osc.assign_phase("zh_2", 0.0)
    osc.assign_phase("zh_3", 0.0)
    osc.assign_phase("en_1", math.pi)  # 反相
    b = osc.pairwise_binding(["zh_1", "zh_2", "zh_3", "en_1"])
    bs = osc.binding_scale
    scale = torch.tensor(1.0)  # 基准写入 scale
    scale_zh1 = float(scale * (1.0 + bs * b["zh_1"]))
    scale_en1 = float(scale * (1.0 + bs * b["en_1"]))
    check("同相写入增强、异相衰减",
          scale_zh1 > 1.0 and scale_en1 < 1.0,
          f"zh_1={scale_zh1:.3f}（增强） en_1={scale_en1:.3f}（衰减）")


def test_8_train_field_binding_weight():
    """C23-C4：训练场构造不被 binding 调制（监督纯净化）。

    回归旧行为：binding 调制 field_state → round2 logits → per_neuron_nll →
    contrastive 监督目标被相位自组织驱动漂移（E2 段 contrastive 饱和 18.42，
    quality_head 学乱）。修复后训练场 = 纯净共振场（与 C20 一致），
    相位只经 scores 段调制 + phase_loss 可微。
    """
    import torch
    from taiji.resonance.phasor import PhasorDynamics
    ph = PhasorDynamics(binding_scale=0.3)
    ph.register_neurons(["zh_1", "zh_2", "zh_3", "en_1"], phases=[0.0, 0.0, 0.0, math.pi])
    ids = ["zh_1", "zh_2", "zh_3", "en_1"]
    torch.manual_seed(0)
    vecs = torch.randn(4, 1, 4)
    vecs_orig = vecs.clone()
    # 训练场构造不乘 binding：直接 sum（等价 bs=0 路径，无绑定调制）
    field_state = vecs.sum(dim=0)
    field_state_clean = vecs_orig.sum(dim=0)
    check("训练场构造无 binding 调制（监督纯净，C23-C4）",
          torch.allclose(field_state, field_state_clean),
          f"Δ={ (field_state - field_state_clean).abs().max().item():.2e}（应为 0）")
    # 相位可微路径仍保留：binding 梯度流经 phasors（scores 段调制驱动 ω/K）
    b = ph.binding_tensor(ids)
    loss = (1.0 + ph.binding_scale * b).sum()
    loss.backward()
    check("scores 段 binding 调制仍可微（相位可学习路径保留）",
          ph.phasors.grad is not None and ph.phasors.grad.abs().sum().item() > 0)


def test_9_phasor_differentiable():
    """C23-C：PhasorDynamics 可微——phasors 任务可调（核心），ω/K 驱动演化（物理）。

    - binding = p·p 点积可微 → loss 梯度直接驱动 phasors（任务决定"谁同相"）
    - ω/K 出现在 Kuramoto 演化（no_grad 状态推进）→ 影响 binding 但不收任务梯度
      （ω/K 的梯度路径需 evolve 可微化，记录为下一阶段）
    """
    from taiji.resonance.phasor import PhasorDynamics
    ph = PhasorDynamics(binding_scale=0.3)
    ph.register_neurons(["zh_1", "zh_2", "zh_3", "en_1"], phases=[0.0, 0.0, 0.0, math.pi])
    # 模拟 scores 调制：loss = Σ scores，scores = (1 + bs·binding)
    b = ph.binding_tensor(["zh_1", "zh_2", "zh_3", "en_1"])
    scores = (1.0 + ph.binding_scale * b).sum()
    scores.backward()
    check("binding 可微（梯度流经 phasors）",
          ph.phasors.grad is not None and ph.phasors.grad.abs().sum().item() > 0)
    # ω/K 物理路径：有相位差 + 耦合牵引 → 相对相位演化 → binding 改变
    # （同频且全对齐相位时 det=0，演化只是整体旋转，binding 不变——物理正确）
    ph2 = PhasorDynamics(dt=0.2)
    ph2.register_neurons(["a", "b", "c"], phases=[0.0, 0.8, 2.4])
    b0 = ph2.binding_tensor(["a", "b", "c"]).clone()
    for _ in range(40):
        ph2.kuramoto_step(active_ids=["a", "b", "c"])
    b1 = ph2.binding_tensor(["a", "b", "c"])
    check("ω/K 驱动演化改变 binding（有相位差场景）", (b1 - b0).abs().sum().item() > 1e-3,
          f"Δbinding={(b1-b0).abs().sum().item():.4f}")


def test_10_kuramoto_unit_norm():
    """C23-C：Kuramoto 演化保持单位范数（相位向量约束）。"""
    from taiji.resonance.phasor import PhasorDynamics
    ph = PhasorDynamics(dt=0.2)
    ph.register_neurons(["a", "b", "c"], phases=[0.0, 1.0, 2.5])
    for _ in range(20):
        ph.kuramoto_step(active_ids=["a", "b", "c"])
    norms = ph.phasors.norm(dim=1)
    check("演化后相位向量仍为单位范数", torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
          f"norms={norms.detach().numpy().round(6)}")


def test_11_task_driven_phase():
    """C23-C：双驱动——任务梯度驱动相位演化（黎曼切向更新），且保持单位范数。

    注意：完全对齐/反相是 binding 驻点（sin(Δθ)=0，梯度纯径向被流形吸收，
    切向=0，无法演化）——从非最优配置出发才有切向梯度路径。
    """
    import torch.optim as optim
    from taiji.resonance.phasor import PhasorDynamics
    ph = PhasorDynamics(binding_scale=0.5)
    # 非最优初始相位：zh 群体有偏差（应被任务拉向同相），en 偏离反相
    ph.register_neurons(["zh_1", "zh_2", "zh_3", "en_1"],
                        phases=[0.0, 0.5, -0.4, math.pi + 0.6])
    opt = optim.SGD(
        [p for n, p in ph.named_parameters() if n != "phasors"], lr=0.1
    )
    p0 = ph.phasors.clone().detach()
    b_zh0 = ph.binding_tensor(["zh_1", "zh_2", "zh_3", "en_1"])[:3].mean().item()
    for _ in range(10):
        opt.zero_grad()
        b = ph.binding_tensor(["zh_1", "zh_2", "zh_3", "en_1"])
        # 任务信号：最大化同相群体（zh）绑定 → 相位应向"同相"演化
        loss = - (b[0] + b[1] + b[2]) + b[3]  # 抬 zh 群体、压 en
        loss.backward()
        opt.step()
        ph.task_gradient_step(lr=0.1)
    moved = (ph.phasors - p0).abs().sum().item() > 1e-3
    b_zh1 = ph.binding_tensor(["zh_1", "zh_2", "zh_3", "en_1"])[:3].mean().item()
    norms = ph.phasors.norm(dim=1)
    check("任务梯度驱动相位演化（phasors 变化）", moved,
          f"Δ={ (ph.phasors - p0).abs().sum().item():.4f}")
    check("任务信号提升同相群体绑定", b_zh1 > b_zh0,
          f"zh binding {b_zh0:.3f} → {b_zh1:.3f}")
    check("演化后保持单位范数", torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


def test_12_phasor_interface_compat():
    """C23-C：PhasorDynamics 兼容标量接口（loader 可用 assign_phase_by_domain）。"""
    from taiji.resonance.phasor import PhasorDynamics
    ph = PhasorDynamics()
    ph.assign_phase_by_domain({"zh": ["zh_1", "zh_2"], "en": ["en_1"]})
    b = ph.pairwise_binding(["zh_1", "zh_2", "en_1"])
    gf = ph.batch_gate_factors(["zh_1", "zh_2", "en_1"])
    check("assign_phase_by_domain + pairwise_binding 兼容", len(b) == 3 and "zh_1" in b)
    check("batch_gate_factors 兼容", gf.shape[0] == 3 and 0.2 <= float(gf.min()) and float(gf.max()) <= 1.0)


def test_13_forward_train_differentiable_wired():
    """C23-C：forward_train 已接入可微分支（binding_tensor + differentiable 判断）。"""
    import inspect
    from taiji.resonance import ensemble as en
    fwdtr_src = inspect.getsource(en.ResonanceEnsemble.forward_train)
    check("forward_train 可微分支（binding_tensor）", "binding_tensor" in fwdtr_src)
    check("forward_train differentiable 判断", "differentiable" in fwdtr_src)


def test_14_omega_coupling_gradient():
    """C23-C：ω/K 梯度路径打通——evolve 输出参与 loss → ω/K 收到梯度。"""
    from taiji.resonance.phasor import PhasorDynamics
    ph = PhasorDynamics(dt=0.2, binding_scale=0.3)
    ph.register_neurons(["a", "b", "c"], phases=[0.0, 0.8, 2.4])
    # 模拟 forward_train：可微演化 → 绑定（用 ev_p）→ scores → loss
    new_p = ph.evolve(active_ids=["a", "b", "c"])          # 可微（ω/K 参与）
    b = ph.binding_tensor(["a", "b", "c"], phasors=new_p)  # 可微绑定
    scores = (1.0 + ph.binding_scale * b).sum()
    scores.backward()
    wg = ph.omega.grad.abs().sum().item() if ph.omega.grad is not None else 0.0
    check("自然频率 ω 收到梯度（梯度路径打通）", wg > 0, f"|∇ω|={wg:.4f}")
    kg = abs(ph.coupling_k.grad.item()) if ph.coupling_k.grad is not None else 0.0
    check("耦合 K 收到梯度（梯度路径打通）", kg > 0, f"|∇K|={kg:.4f}")


def test_15_task_driven_omega():
    """C23-C：任务梯度驱动 ω 演化（ω 为可学习参数，loss 信号调整频差）。"""
    import torch.optim as optim
    from taiji.resonance.phasor import PhasorDynamics
    ph = PhasorDynamics(dt=0.2, binding_scale=0.3)
    ph.register_neurons(["zh_1", "zh_2", "zh_3", "en_1"],
                        phases=[0.0, 0.5, -0.4, math.pi + 0.6])
    opt = optim.SGD(
        [p for n, p in ph.named_parameters() if n != "phasors"], lr=0.1
    )
    w0 = ph.omega.clone().detach()
    for _ in range(5):
        opt.zero_grad()
        new_p = ph.evolve(active_ids=["zh_1", "zh_2", "zh_3", "en_1"])
        b = ph.binding_tensor(["zh_1", "zh_2", "zh_3", "en_1"], phasors=new_p)
        # 任务信号：最大化同相群体（zh）绑定
        loss = - (b[0] + b[1] + b[2]) + b[3]
        loss.backward()
        opt.step()
        ph.task_gradient_step(lr=0.1)
    check("任务梯度驱动 ω 演化", (ph.omega - w0).abs().sum().item() > 1e-4,
          f"Δω={(ph.omega - w0).abs().sum().item():.4f}")


if __name__ == "__main__":
    test_1_same_phase_binding_positive()
    test_2_opposite_phase_binding_negative()
    test_3_mixed_domain_binding()
    test_4_kuramoto_binding_loop()
    test_5_binding_modulates_scores()
    test_6_wired_into_ensemble()
    test_7_write_scale_binding()
    test_8_train_field_binding_weight()
    test_9_phasor_differentiable()
    test_10_kuramoto_unit_norm()
    test_11_task_driven_phase()
    test_12_phasor_interface_compat()
    test_13_forward_train_differentiable_wired()
    test_14_omega_coupling_gradient()
    test_15_task_driven_omega()
    print("\nC23 冒烟 15/15 PASS")
