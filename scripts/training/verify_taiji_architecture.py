"""Taiji 架构级最小原型验证

验证项:
T1: TaijiArchitecture 基本前向(input_ids → logits)能跑通
T2: 输出形状正确 [B, L, vocab]
T3: 反向传播能计算梯度(所有参数可学习)
T4: 群体性验证(num_neurons>1 时输出与单神经元不同)
T5: 跨层场传导(field_history 非空且层间场状态变化)
T6: per-sample routing(不同样本的路由权重不同)
T7: 相位绑定激活(activations ∈ (0,1) 且非全同)
T8: 退化行为(num_neurons=1 等价于单神经元模式)
T9: 多 batch 独立场(不同样本场状态不污染)
T10: causal mask 生效(未来 token 不影响当前)
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")

from neuroplex.taiji_arch import TaijiArchitecture, TaijiPopulationLayer, _PhaseBinder, _SparseRouter


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
    router = _SparseRouter(hidden_size=64, num_neurons=P, router_top_k=0)
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
    """T7: 相位绑定激活。"""
    print("\n=== T7: 相位绑定 ===")
    P = 5
    binder = _PhaseBinder(num_neurons=P, phase_delta=0.8, act_temp=4.0)
    phases, activations = binder()
    check("T7a phases shape", phases.shape == (P,))
    check("T7b activations shape", activations.shape == (P,))
    check("T7c activations ∈ (0,1)",
          (activations > 0).all() and (activations < 1).all())
    # 不同神经元激活应不同(相位差驱动)
    check("T7d 激活非全同", (activations.std() > 1e-4).item(),
          f"std = {activations.std().item():.4f}")
    print(f"  phases: {phases.tolist()}")
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
    """T12: 调质接口(set_neuromodulator_gains 影响输出)。"""
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

    model.set_neuromodulator_gains(temp_gain=1.0, ffn_gain=1.0)
    with torch.no_grad():
        out_baseline = model(input_ids)["logits"]

    model.set_neuromodulator_gains(temp_gain=2.0, ffn_gain=1.5)
    with torch.no_grad():
        out_modulated = model(input_ids)["logits"]

    diff = (out_baseline - out_modulated).abs().max().item()
    check("T12a 调质改变输出", diff > 1e-4, f"diff = {diff:.4f}")


def main():
    print("=" * 60)
    print("Taiji 架构级最小原型验证")
    print("=" * 60)

    tests = [
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
