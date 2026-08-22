"""Taiji 架构级验证(含四项扩展)

验证项:
T1-T12: 基础功能(见各 docstring)
T13: 第 1 项 调质平面 — set_neuromodulator_targets 改变 gain 和输出
T14: 第 1 项 调质平面 — EMA 缓慢趋近目标(非突变)
T15: 第 2 项 STDP tracker — 记录发放、pair update(LTP/LTD)
T16: 第 2 项 coaction tracker — 共激活矩阵 EMA 更新
T17: 第 2 项 maturity tracker — 成熟度随激活累积
T18: 第 2 项 sleep_consolidate — 统计 STDP 更新
T19: 第 3 项 Phasor — 可微 Kuramoto 演化推进相位
T20: 第 3 项 Phasor — theta-gamma 嵌套(包络调制激活)
T21: 第 4 项 STE top-K — 前向 hard 选择、反向梯度穿透
T22: 第 4 项 hard top-K routing — 路由权重稀疏(只有 top-K 非零)
T23: 集成 — 四项机制同时启用,前向+反向不报错
"""
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")

from neuroplex.taiji_arch import (
    TaijiArchitecture, TaijiPopulationLayer,
    _NeuromodulatorState, _PhasorOscillator, _SparseRouter,
    _STDPTracker, _CoactivationTracker, _MaturityTracker,
    _ste_topk,
)


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise AssertionError(f"Test failed: {name}")


def test_t1_basic_forward():
    """T1: 基本前向能跑通。"""
    print("\n=== T1: 基本前向 ===")
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=3,
        max_seq_len=128,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    result = model(input_ids)
    check("T1a forward 不报错", "logits" in result)
    check("T1b logits 非空", result["logits"] is not None)
    check("T1c final_field 存在", "final_field" in result)
    print(f"  logits shape: {result['logits'].shape}")
    print(f"  final_field shape: {result['final_field'].shape}")


def test_t2_output_shape():
    """T2: 输出形状正确。"""
    print("\n=== T2: 输出形状 ===")
    B, L, V = 4, 32, 1000
    model = TaijiArchitecture(
        vocab_size=V, hidden_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=4,
        max_seq_len=128,
    )
    input_ids = torch.randint(0, V, (B, L))
    result = model(input_ids)
    check("T2a logits shape", result["logits"].shape == (B, L, V),
          f"got {result['logits'].shape}")
    check("T2b final_field shape", result["final_field"].shape == (B, 32),
          f"got {result['final_field'].shape}")


def test_t3_backward():
    """T3: 反向传播能计算梯度。"""
    print("\n=== T3: 反向传播 ===")
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=3,
        max_seq_len=128,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    target = torch.randint(0, 1000, (2, 16))
    result = model(input_ids)
    loss = F.cross_entropy(
        result["logits"].view(-1, 1000), target.view(-1)
    )
    loss.backward()
    # 检查关键参数有梯度
    has_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
            has_grad.append(name.split(".")[0])
    check("T3a loss 可计算", not torch.isnan(loss))
    check("T3b 有参数获得梯度", len(has_grad) > 0,
          f"{len(has_grad)} params with grad")
    check("T3c embedding 有梯度", "shared_embedding" in has_grad)
    check("T3d taiji_blocks 有梯度", "layers" in has_grad)
    check("T3e lm_head 有梯度", "lm_head" in has_grad or "shared_embedding" in has_grad)
    print(f"  loss = {loss.item():.4f}")
    print(f"  params with grad: {set(has_grad)}")


def test_t4_population_effect():
    """T4: 群体性验证(num_neurons>1 输出与单神经元不同)。"""
    print("\n=== T4: 群体性验证 ===")
    torch.manual_seed(42)
    # 单神经元
    model_single = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=1,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=1,
        max_seq_len=128,
    )
    # 多神经元(群体)
    torch.manual_seed(42)
    model_multi = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=1,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=4,
        max_seq_len=128,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    out_single = model_single(input_ids)["logits"]
    out_multi = model_multi(input_ids)["logits"]
    diff = (out_single - out_multi).abs().max().item()
    check("T4a 群体输出与单体不同", diff > 1e-4,
          f"max diff = {diff:.4f}")
    print(f"  单神经元 vs 4 神经元 max diff: {diff:.4f}")


def test_t5_cross_layer_field():
    """T5: 跨层场传导(field_history 非空且层间变化)。"""
    print("\n=== T5: 跨层场传导 ===")
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=3,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=3,
        max_seq_len=128, cross_layer_diffusion=0.5,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    result = model(input_ids, return_field_history=True)
    check("T5a field_history 存在", "field_history" in result)
    check("T5b field_history 长度 = num_layers",
          len(result["field_history"]) == 3)
    # 层间场状态应不同(跨层传导生效)
    f0 = result["field_history"][0]
    f1 = result["field_history"][1]
    f2 = result["field_history"][2]
    diff_01 = (f0 - f1).abs().max().item()
    diff_12 = (f1 - f2).abs().max().item()
    check("T5c 层0与层1场不同", diff_01 > 1e-6, f"diff = {diff_01:.6f}")
    check("T5d 层1与层2场不同", diff_12 > 1e-6, f"diff = {diff_12:.6f}")
    print(f"  层间场差异: 0→1={diff_01:.6f}, 1→2={diff_12:.6f}")


def test_t6_per_sample_routing():
    """T6: per-sample routing(不同样本路由权重不同)。"""
    print("\n=== T6: per-sample routing ===")
    P = 4
    router = _SparseRouter(hidden_size=64, num_neurons=P, hard_top_k=0)
    # 两个不同样本的 pooled 表征
    pooled = torch.randn(2, P, 64)
    weights = router(pooled)
    check("T6a weights shape", weights.shape == (2, P))
    check("T6b weights softmax(行和=1)",
          torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5))
    # 不同样本权重应不同
    diff = (weights[0] - weights[1]).abs().max().item()
    check("T6c 不同样本路由不同", diff > 1e-4, f"diff = {diff:.4f}")
    print(f"  样本0权重: {weights[0].tolist()}")
    print(f"  样本1权重: {weights[1].tolist()}")


def test_t7_phase_binding():
    """T7: 相位绑定激活(_PhasorOscillator 替代 _PhaseBinder)。"""
    print("\n=== T7: 相位绑定 ===")
    P = 5
    phasor = _PhasorOscillator(
        num_neurons=P, omega_init=math.pi / 4, coupling_init=0.05,
        act_temp=4.0, theta_omega=0.0,  # 关闭 theta 嵌套以隔离测试
    )
    phasors, activations, theta_env = phasor(evolve=False)
    check("T7a phasors shape", phasors.shape == (P, 2))
    check("T7b activations shape", activations.shape == (P,))
    check("T7c activations ∈ (0,1)",
          (activations > 0).all() and (activations < 1).all())
    check("T7d 激活非全同", (activations.std() > 1e-4).item(),
          f"std = {activations.std().item():.4f}")
    print(f"  phasors: {phasors.tolist()}")
    print(f"  activations: {activations.tolist()}")


def test_t8_degenerate_single_neuron():
    """T8: 退化行为(num_neurons=1 等价于单神经元模式)。"""
    print("\n=== T8: 单神经元退化 ===")
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=1,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=1,
        max_seq_len=128,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    result = model(input_ids)
    check("T8a 单神经元前向不报错", "logits" in result)
    check("T8b 输出形状正确",
          result["logits"].shape == (2, 16, 1000))
    # 单神经元时路由权重应恒为 1(只有一个选择)
    layer = model.layers[0]
    check("T8c 单神经元层数量=1", layer.num_neurons == 1)


def test_t9_batch_independent_field():
    """T9: 多 batch 独立场(不同样本场状态不污染)。"""
    print("\n=== T9: batch 独立场 ===")
    B = 3
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=3,
        max_seq_len=128,
    )
    # 三个完全不同的输入
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [100, 200, 300, 400, 500, 600, 700, 800],
        [999, 998, 997, 996, 995, 994, 993, 992],
    ])
    result = model(input_ids, return_field_history=True)
    final_field = result["final_field"]  # [B, D]
    check("T9a final_field batch 维度", final_field.shape[0] == B)
    # 三个样本的场应不同
    f0, f1, f2 = final_field[0], final_field[1], final_field[2]
    diff_01 = (f0 - f1).abs().max().item()
    diff_02 = (f0 - f2).abs().max().item()
    check("T9b 样本0与1场不同", diff_01 > 1e-6, f"diff = {diff_01:.6f}")
    check("T9c 样本0与2场不同", diff_02 > 1e-6, f"diff = {diff_02:.6f}")
    print(f"  样本间场差异: 0↔1={diff_01:.6f}, 0↔2={diff_02:.6f}")


def test_t10_causal_mask():
    """T10: causal mask 与场共振特性。

    Taiji 架构的设计选择:
    - yang 流(标准 attention)遵循 causal mask
    - yin 流(场共振)是全局反馈,不遵循 causal(生物合理性)
    - 跨 forward 场状态独立(每次 forward 重置)

    本测试验证:
    - T10a: 跨 forward 场状态独立(相同输入产生相同输出)
    - T10b: 场共振是全局反馈(修改后半部分影响前半部分,设计特性)
    """
    print("\n=== T10: causal mask 与场共振 ===")
    torch.manual_seed(42)
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=2,
        max_seq_len=128,
    )
    model.eval()
    L = 8
    input_ids = torch.randint(0, 1000, (1, L))

    # T10a: 跨 forward 场状态独立(相同输入两次 forward 应产生相同输出)
    with torch.no_grad():
        out1 = model(input_ids)["logits"]
        out2 = model(input_ids)["logits"]
    diff_repeat = (out1 - out2).abs().max().item()
    check("T10a 跨 forward 场独立(相同输入相同输出)",
          diff_repeat < 1e-5, f"diff = {diff_repeat:.6f}")

    # T10b: 场共振是全局反馈(设计特性,非 bug)
    # 修改后半部分 token,前半部分 logits 会变化(场共振全局影响)
    input_mod = input_ids.clone()
    input_mod[0, L // 2:] = torch.randint(0, 1000, (L - L // 2,))
    with torch.no_grad():
        out_mod = model(input_mod)["logits"]
    diff_before = (out1[0, :L // 2] - out_mod[0, :L // 2]).abs().max().item()
    diff_after = (out1[0, L // 2:] - out_mod[0, L // 2:]).abs().max().item()
    check("T10b 场共振全局反馈(前半部分受后半部分影响)",
          diff_before > 1e-4, f"diff = {diff_before:.6f}")
    check("T10c 后半部分变化", diff_after > 1e-4, f"diff = {diff_after:.6f}")
    print(f"  跨 forward 重复差异: {diff_repeat:.6f}(应≈0)")
    print(f"  场共振全局影响: 前半={diff_before:.4f}, 后半={diff_after:.4f}")


def test_t11_population_layer_standalone():
    """T11: TaijiPopulationLayer 独立前向。"""
    print("\n=== T11: 群体层独立前向 ===")
    layer = TaijiPopulationLayer(
        hidden_size=64, num_heads=4, num_kv_heads=2,
        intermediate_size=128, field_dim=32, num_neurons=3,
    )
    x = torch.randn(2, 16, 64)
    mask = torch.triu(
        torch.full((1, 1, 16, 16), float("-inf")), diagonal=1
    )
    out, field = layer(x, mask=mask)
    check("T11a 输出形状", out.shape == (2, 16, 64))
    check("T11b 场状态形状", field.shape == (2, 32))
    check("T11c 输出有限", torch.isfinite(out).all())
    check("T11d 场状态有限", torch.isfinite(field).all())


def test_t12_neuromodulator_interface():
    """T12: 调质接口(set_neuromodulator_targets + EMA 推进后影响输出)。"""
    print("\n=== T12: 调质接口 ===")
    torch.manual_seed(42)
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=1,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=2,
        max_seq_len=128,
    )
    model.eval()
    input_ids = torch.randint(0, 1000, (2, 16))

    # 基线(调质全中性 0.5)
    with torch.no_grad():
        out_baseline = model(input_ids)["logits"]

    # 推高 DA + NE(奖励+警觉),EMA 多步推进后增益改变
    model.set_neuromodulator_targets(dopamine=0.9, norepinephrine=0.9)
    for _ in range(20):  # EMA 收敛
        model.neuromodulator.step()
    with torch.no_grad():
        out_modulated = model(input_ids)["logits"]

    diff = (out_baseline - out_modulated).abs().max().item()
    check("T12a 调质改变输出", diff > 1e-4, f"diff = {diff:.4f}")
    # 验证 gain 值确实改变
    temp_gain = model.neuromodulator.get_temp_gain()
    ffn_gain = model.neuromodulator.get_ffn_gain()
    check("T12b temp_gain > 1(NE 高)", temp_gain > 1.0, f"temp_gain = {temp_gain:.3f}")
    check("T12c ffn_gain > 1(DA 高)", ffn_gain > 1.0, f"ffn_gain = {ffn_gain:.3f}")


# ─────────────────────────────────────────────────────────────────────
# 第 1 项:调质平面扩展测试
# ─────────────────────────────────────────────────────────────────────

def test_t13_neuromodulator_targets():
    """T13: set_neuromodulator_targets 改变 gain 和输出。"""
    print("\n=== T13: 调质目标驱动 gain ===")
    nm = _NeuromodulatorState(dopamine=0.5, serotonin=0.5,
                              norepinephrine=0.5, acetylcholine=0.5)
    # 中性 gain
    temp_0 = nm.get_temp_gain()
    ffn_0 = nm.get_ffn_gain()
    refr_0 = nm.get_refractory_multiplier()
    check("T13a 中性 temp_gain ≈ 1.0", abs(temp_0 - 1.0) < 0.2,
          f"temp_gain = {temp_0:.3f}")
    check("T13b 中性 ffn_gain ≈ 1.0", abs(ffn_0 - 1.0) < 0.1,
          f"ffn_gain = {ffn_0:.3f}")
    check("T13c 中性 refractory_mult ≈ 1.0", abs(refr_0 - 1.0) < 0.1,
          f"refr = {refr_0:.3f}")

    # 推高所有调质
    nm.set_targets(dopamine=1.0, serotonin=1.0,
                   norepinephrine=1.0, acetylcholine=1.0)
    for _ in range(50):
        nm.step()
    temp_1 = nm.get_temp_gain()
    ffn_1 = nm.get_ffn_gain()
    refr_1 = nm.get_refractory_multiplier()
    check("T13d 高 NE+ACh → temp_gain 显著升高", temp_1 > temp_0,
          f"{temp_0:.3f} → {temp_1:.3f}")
    check("T13e 高 DA → ffn_gain 升高", ffn_1 > ffn_0,
          f"{ffn_0:.3f} → {ffn_1:.3f}")
    check("T13f 高 5-HT → refractory_mult 升高", refr_1 > refr_0,
          f"{refr_0:.3f} → {refr_1:.3f}")


def test_t14_neuromodulator_ema():
    """T14: EMA 缓慢趋近目标(非突变)。"""
    print("\n=== T14: 调质 EMA 缓慢趋近 ===")
    nm = _NeuromodulatorState(dopamine=0.5, ema_alpha=0.1)
    nm.set_targets(dopamine=1.0)
    # 一步 EMA:0.5 → 0.5 + 0.1*(1.0-0.5) = 0.55
    nm.step()
    da_after_1 = float(nm.dopamine)
    check("T14a 一步 EMA 后 DA ≈ 0.55", abs(da_after_1 - 0.55) < 1e-3,
          f"DA = {da_after_1:.4f}")
    # 多步后趋近 1.0
    for _ in range(100):
        nm.step()
    da_final = float(nm.dopamine)
    check("T14b 多步后 DA 趋近 1.0", da_final > 0.95,
          f"DA = {da_final:.4f}")
    check("T14c DA 不超过 1.0", da_final <= 1.0 + 1e-6)


# ─────────────────────────────────────────────────────────────────────
# 第 2 项:生命周期平面测试
# ─────────────────────────────────────────────────────────────────────

def test_t15_stdp_tracker():
    """T15: STDP tracker 记录发放、pair update(LTP/LTD)。"""
    print("\n=== T15: STDP tracker ===")
    tracker = _STDPTracker(eta_plus=0.1, eta_minus=0.05,
                           tau_plus=2.0, tau_minus=2.0,
                           similarity_threshold=0.0)
    # 两个相似方向的 field_vector
    vec_a = torch.tensor([1.0, 0.0, 0.0])
    vec_b = torch.tensor([0.9, 0.1, 0.0])  # 与 a 相似
    vec_c = torch.tensor([-1.0, 0.0, 0.0])  # 与 a 反向

    # LTP:pre(0) 先于 post(1),方向相似 → Δw > 0
    delta_ltp = tracker.compute_pair_update(
        pre_idx=0, post_idx=1, pre_step=0, post_step=1,
        pre_vec=vec_a, post_vec=vec_b,
    )
    check("T15a LTP(pre 先于 post,方向相似)Δw > 0", delta_ltp > 0,
          f"Δw = {delta_ltp:.4f}")

    # LTD:pre(1) 后于 post(0),方向相似 → Δw < 0
    delta_ltd = tracker.compute_pair_update(
        pre_idx=0, post_idx=1, pre_step=1, post_step=0,
        pre_vec=vec_a, post_vec=vec_b,
    )
    check("T15b LTD(post 先于 pre,方向相似)Δw < 0", delta_ltd < 0,
          f"Δw = {delta_ltd:.4f}")

    # 反向相似度 < 0(阈值 0.0 时仍记录,但 sim 为负→Δw 为负)
    delta_anti = tracker.compute_pair_update(
        pre_idx=0, post_idx=1, pre_step=0, post_step=1,
        pre_vec=vec_a, post_vec=vec_c,
    )
    check("T15c 反向方向 sim < 0", delta_anti < 0,
          f"Δw = {delta_anti:.4f}")

    # record_firing + clear
    tracker.record_firing(0, step=0, field_vector=vec_a)
    tracker.record_firing(1, step=1, field_vector=vec_b)
    check("T15d record_firing 后历史非空",
          len(tracker._firing_history) == 2)
    tracker.clear()
    check("T15e clear 后历史为空",
          len(tracker._firing_history) == 0)


def test_t16_coaction_tracker():
    """T16: coaction tracker 共激活矩阵 EMA 更新。"""
    print("\n=== T16: coaction tracker ===")
    tracker = _CoactivationTracker(num_neurons=4, ema_alpha=0.3)
    # 初始矩阵全 0
    check("T16a 初始矩阵全 0",
          (tracker.coactivation_matrix == 0).all())

    # 第一轮:0,1,2 共激活
    tracker.update([0, 1, 2])
    m = tracker.coactivation_matrix
    check("T16b 0-1 共激活 > 0", float(m[0, 1]) > 0,
          f"m[0,1] = {float(m[0,1]):.4f}")
    check("T16c 0-3 共激活 = 0(未共激活)",
          float(m[0, 3]) == 0.0)
    check("T16d 对角线 = 0(自身不共激活)",
          float(m[0, 0]) == 0.0)

    # 第二轮:0,1 共激活(2 退场)→ 0-2 共激活应衰减
    m_before = float(tracker.coactivation_matrix[0, 2])
    tracker.update([0, 1])
    m_after = float(tracker.coactivation_matrix[0, 2])
    check("T16e 退场后共激活衰减", m_after < m_before,
          f"{m_before:.4f} → {m_after:.4f}")


def test_t17_maturity_tracker():
    """T17: maturity tracker 成熟度随激活累积。"""
    print("\n=== T17: maturity tracker ===")
    tracker = _MaturityTracker(num_neurons=3, maturation_rate=0.1,
                               initial_maturity=0.1)
    # 初始:幼稚态 0.1,共振权重 0.1+0.9*0.1=0.19
    w_0 = tracker.get_resonance_weight(0)
    check("T17a 初始共振权重 < 0.3(幼稚态)", w_0 < 0.3,
          f"w = {w_0:.3f}")

    # 多次激活后成熟度上升
    for _ in range(20):
        tracker.update([0])
    w_final = tracker.get_resonance_weight(0)
    check("T17b 多次激活后共振权重显著升高", w_final > w_0,
          f"{w_0:.3f} → {w_final:.3f}")
    check("T17c 成熟度趋近 1.0",
          float(tracker.maturity[0]) > 0.5,
          f"maturity = {float(tracker.maturity[0]):.3f}")

    # 未激活的神经元仍幼稚
    w_2 = tracker.get_resonance_weight(2)
    check("T17d 未激活神经元仍幼稚", w_2 < 0.3,
          f"w = {w_2:.3f}")


def test_t18_sleep_consolidate():
    """T18: sleep_consolidate 统计 STDP 更新。"""
    print("\n=== T18: 睡眠巩固 ===")
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=1,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=3,
        max_seq_len=128,
    )
    # 训练模式前向一次,产生发放历史
    model.train()
    input_ids = torch.randint(0, 1000, (2, 16))
    result = model(input_ids)
    # 睡眠巩固
    stats = model.sleep_consolidate(recent_fields=[result["final_field"]])
    check("T18a sleep_consolidate 返回统计", "pairs_updated" in stats)
    check("T18b pairs_updated >= 0", stats["pairs_updated"] >= 0)
    print(f"  pairs_updated: {stats['pairs_updated']}")
    print(f"  total_delta_w: {stats['total_delta_w']:.4f}")


# ─────────────────────────────────────────────────────────────────────
# 第 3 项:Phasor 模块测试
# ─────────────────────────────────────────────────────────────────────

def test_t19_phasor_kuramoto_evolve():
    """T19: 可微 Kuramoto 演化推进相位。"""
    print("\n=== T19: Kuramoto 演化 ===")
    P = 4
    phasor = _PhasorOscillator(
        num_neurons=P, omega_init=math.pi / 4, coupling_init=0.1,
        act_temp=4.0, theta_omega=0.0,  # 关闭 theta 隔离测试
        dt=0.2,
    )
    # 记录初始相位
    p_init = phasor.phasors.detach().clone()
    # 演化一步
    phasor.evolve = True
    p_new, activations, _ = phasor(evolve=True)
    # 相位应变化(演化推进)
    diff = (p_init - phasor.phasors.detach()).abs().max().item()
    check("T19a 演化后相位变化", diff > 1e-4,
          f"max diff = {diff:.6f}")
    check("T19b 演化后仍是单位向量",
          torch.allclose(phasor.phasors.norm(dim=-1), torch.ones(P), atol=1e-5))
    check("T19c activations ∈ (0,1)",
          (activations > 0).all() and (activations < 1).all())

    # 梯度可流通 omega/K(可微)
    phasor.train()
    p, activ, _ = phasor(evolve=False)
    loss = activ.sum()
    loss.backward()
    check("T19d omega 有梯度", phasor.omega.grad is not None)
    check("T19e coupling_k 有梯度", phasor.coupling_k.grad is not None)


def test_t20_phasor_theta_gamma_nesting():
    """T20: theta-gamma 嵌套(包络调制激活)。"""
    print("\n=== T20: theta-gamma 嵌套 ===")
    P = 4
    # 启用 theta 嵌套
    phasor = _PhasorOscillator(
        num_neurons=P, omega_init=math.pi / 4, coupling_init=0.05,
        act_temp=4.0,
        theta_omega=0.5, theta_amp=0.3,
        theta_init=0.0,
        dt=0.2,
    )
    # theta_phase = 0 → 包络 = 1 + 0.3 = 1.3(峰值)
    env_peak = phasor._theta_envelope()
    check("T20a theta_phase=0 → 包络峰值 > 1", env_peak > 1.0,
          f"env = {env_peak:.3f}")

    # 推进 theta 到 π(波谷)
    phasor.theta_phase.fill_(math.pi)
    env_trough = phasor._theta_envelope()
    check("T20b theta_phase=π → 包络波谷 < 1", env_trough < 1.0,
          f"env = {env_trough:.3f}")

    # theta_omega=0 时包络恒 1(向后兼容)
    phasor_no_theta = _PhasorOscillator(
        num_neurons=P, theta_omega=0.0, theta_amp=0.3,
    )
    env_no = phasor_no_theta._theta_envelope()
    check("T20c theta_omega=0 → 包络恒 1", abs(env_no - 1.0) < 1e-6,
          f"env = {env_no:.3f}")

    # 激活被 theta 包络调制
    phasor.theta_phase.fill_(0.0)
    _, activ_peak, _ = phasor(evolve=False)
    phasor.theta_phase.fill_(math.pi)
    _, activ_trough, _ = phasor(evolve=False)
    check("T20d 包络峰值时激活 > 包络波谷时激活",
          float(activ_peak.mean()) > float(activ_trough.mean()),
          f"peak={float(activ_peak.mean()):.4f} > trough={float(activ_trough.mean()):.4f}")


# ─────────────────────────────────────────────────────────────────────
# 第 4 项:hard top-K + STE 测试
# ─────────────────────────────────────────────────────────────────────

def test_t21_ste_topk():
    """T21: STE top-K 前向 hard、反向梯度穿透。"""
    print("\n=== T21: STE top-K ===")
    B, P, K = 2, 5, 2
    # 构造 soft weights(可微)
    soft = torch.randn(B, P, requires_grad=True)
    soft_w = F.softmax(soft, dim=-1)
    # STE top-K
    hard_w = _ste_topk(soft_w, K)
    # 前向:每行只有 K 个非零
    nonzero_per_row = (hard_w > 0).sum(dim=-1)
    check("T21a 前向 hard top-K(每行 K 个非零)",
          (nonzero_per_row == K).all(),
          f"nonzero = {nonzero_per_row.tolist()}")
    # 行和 = 1(归一化)
    check("T21b hard 权重行和=1",
          torch.allclose(hard_w.sum(dim=-1), torch.ones(B), atol=1e-5))

    # 反向:梯度穿透到 soft
    loss = hard_w.sum()
    loss.backward()
    check("T21c 反向梯度穿透到 soft", soft.grad is not None)
    check("T21d 梯度非零", soft.grad.abs().sum() > 0)


def test_t22_hard_top_k_routing():
    """T22: hard top-K routing 路由权重稀疏。"""
    print("\n=== T22: hard top-K routing ===")
    B, P, H, K = 3, 5, 64, 2
    router = _SparseRouter(hidden_size=H, num_neurons=P, hard_top_k=K)
    pooled = torch.randn(B, P, H)
    weights = router(pooled)
    # 每行只有 K 个非零
    nonzero_per_row = (weights > 0).sum(dim=-1)
    check("T22a hard top-K routing 每行 K 个非零",
          (nonzero_per_row == K).all(),
          f"nonzero = {nonzero_per_row.tolist()}")
    check("T22b 行和=1",
          torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5))

    # soft routing(hard_top_k=0)所有都非零
    router_soft = _SparseRouter(hidden_size=H, num_neurons=P, hard_top_k=0)
    weights_soft = router_soft(pooled)
    nonzero_soft = (weights_soft > 0).sum(dim=-1)
    check("T22c soft routing 所有 P 个非零",
          (nonzero_soft == P).all())


# ─────────────────────────────────────────────────────────────────────
# 集成测试
# ─────────────────────────────────────────────────────────────────────

def test_t23_full_integration():
    """T23: 四项机制同时启用,前向+反向不报错。"""
    print("\n=== T23: 四项机制集成 ===")
    torch.manual_seed(42)
    model = TaijiArchitecture(
        vocab_size=1000, hidden_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, intermediate_size=128,
        field_dim=32, num_neurons_per_layer=4,
        max_seq_len=128,
        # 启用所有机制
        omega_init=math.pi / 4, coupling_init=0.05,
        theta_omega=0.5, theta_amp=0.2,
        hard_top_k=2,  # hard top-K routing
        refractory_steps=2,
        stdp_strength=0.1,
        cross_layer_diffusion=0.3,
    )
    model.train()
    input_ids = torch.randint(0, 1000, (2, 16))

    # 设置调质目标
    model.set_neuromodulator_targets(
        dopamine=0.7, serotonin=0.5, norepinephrine=0.8, acetylcholine=0.6
    )

    # 前向
    result = model(input_ids, return_field_history=True)
    check("T23a 前向不报错", "logits" in result)
    check("T23b logits 形状正确",
          result["logits"].shape == (2, 16, 1000))
    check("T23c field_history 长度 = num_layers",
          len(result["field_history"]) == 2)
    check("T23d neuromodulator_state 存在",
          "neuromodulator_state" in result)

    # 反向
    target = torch.randint(0, 1000, (2, 16))
    loss = F.cross_entropy(
        result["logits"].view(-1, 1000), target.view(-1)
    )
    loss.backward()
    check("T23e 反向不报错", not torch.isnan(loss))

    # 检查关键参数有梯度
    has_omega_grad = model.layers[0].phasor.omega.grad is not None
    has_coupling_grad = model.layers[0].phasor.coupling_k.grad is not None
    has_router_grad = model.layers[0].router.router_proj.weight.grad is not None
    has_field_transfer_grad = model.field_transfer.weight.grad is not None
    check("T23f phasor.omega 有梯度", has_omega_grad)
    check("T23g phasor.coupling_k 有梯度", has_coupling_grad)
    check("T23h router 有梯度", has_router_grad)
    check("T23i field_transfer 有梯度", has_field_transfer_grad)

    # 睡眠巩固(STDP 应用)
    stats = model.sleep_consolidate(recent_fields=[result["final_field"]])
    check("T23j sleep_consolidate 不报错", "pairs_updated" in stats)

    print(f"  loss = {loss.item():.4f}")
    print(f"  neuromodulator: {result['neuromodulator_state']}")
    print(f"  sleep stats: {stats}")


def main():
    print("=" * 60)
    print("Taiji 架构级验证(含四项扩展)")
    print("=" * 60)

    tests = [
        # 基础(T1-T12)
        test_t1_basic_forward,
        test_t2_output_shape,
        test_t3_backward,
        test_t4_population_effect,
        test_t5_cross_layer_field,
        test_t6_per_sample_routing,
        test_t7_phase_binding,
        test_t8_degenerate_single_neuron,
        test_t9_batch_independent_field,
        test_t10_causal_mask,
        test_t11_population_layer_standalone,
        test_t12_neuromodulator_interface,
        # 第 1 项:调质平面
        test_t13_neuromodulator_targets,
        test_t14_neuromodulator_ema,
        # 第 2 项:生命周期
        test_t15_stdp_tracker,
        test_t16_coaction_tracker,
        test_t17_maturity_tracker,
        test_t18_sleep_consolidate,
        # 第 3 项:Phasor
        test_t19_phasor_kuramoto_evolve,
        test_t20_phasor_theta_gamma_nesting,
        # 第 4 项:hard top-K + STE
        test_t21_ste_topk,
        test_t22_hard_top_k_routing,
        # 集成
        test_t23_full_integration,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed}/{passed + failed} 测试通过")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
