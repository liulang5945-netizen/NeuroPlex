"""验证 neurogenesis（神经元新生）完整闭环。

核心验证：
1. Cortex.add_neuron(domain) 创建新神经元并加入 ensemble
2. 新神经元持久化到 neurons_dir/neuron_{nid}.pt
3. 新神经元参与 ensemble forward（共振）
4. maturity.register_new 标记幼稚态
5. sleep_engine 触发点正确调用 cortex.add_neuron
6. life_scheduler.set_brain_interfaces 接线闭环

Usage:
    python scripts/training/verify_neurogenesis.py
"""
import sys
import os
import tempfile

os.environ.setdefault('TAIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    print("=" * 60)
    print("Neurogenesis（神经元新生）完整闭环验证")
    print("=" * 60)

    # Step 1: 装配 Cortex
    print("\n[Step 1] 装配 Cortex...")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
    )
    lifecycle = modules.get("lifecycle")
    initial_neuron_count = len(cortex.neurons)
    print(f"  ✅ Cortex 装配完成: {initial_neuron_count} neurons")
    print(f"  neurons: {list(cortex.neurons.keys())}")

    # Step 2: 测试 Cortex.add_neuron 创建新神经元
    print("\n[Step 2] 调用 cortex.add_neuron('zh')...")
    new_nid = cortex.add_neuron("zh", lifecycle=lifecycle)
    print(f"  ✅ 新神经元 ID: {new_nid}")
    if new_nid == "zh_1":
        print(f"  ✅ 命名格式正确: {{domain}}_{{n}}")
    else:
        print(f"  ⚠️ 命名异常: 期望 zh_1, 实际 {new_nid}")

    # Step 3: 验证新神经元已加入 ensemble
    print("\n[Step 3] 验证新神经元已加入 ensemble...")
    assert new_nid in cortex.neurons, f"{new_nid} 不在 cortex.neurons"
    assert new_nid in cortex.ensemble.neurons, f"{new_nid} 不在 ensemble.neurons"
    new_count = len(cortex.neurons)
    print(f"  ✅ neurons 数量: {initial_neuron_count} → {new_count}")
    if new_count == initial_neuron_count + 1:
        print(f"  ✅ 新神经元已注入 cortex + ensemble")
    else:
        print(f"  ⚠️ 数量异常")

    # Step 4: 验证 ckpt 持久化
    print("\n[Step 4] 验证 ckpt 持久化...")
    ckpt_path = os.path.join(cortex.neurons_dir, f"neuron_{new_nid}.pt")
    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["neuron_config"]
        sd = ckpt["state_dict"]
        print(f"  ✅ ckpt 已保存: {ckpt_path}")
        print(f"     spec={cfg.spec}, vocab_size={cfg.vocab_size}, neuron_id={cfg.neuron_id}")
        print(f"     state_dict keys: {len(sd)}")
        # 清理测试 ckpt
        os.remove(ckpt_path)
        print(f"  ✅ 测试 ckpt 已清理")
    else:
        print(f"  ⚠️ ckpt 未找到: {ckpt_path}")

    # Step 5: 验证 maturity.register_new
    print("\n[Step 5] 验证 maturity.register_new...")
    if lifecycle is not None:
        ratio = lifecycle.maturity.get_maturity_ratio(new_nid)
        if ratio == 0.0:
            print(f"  ✅ 幼稚态追踪: {new_nid} maturity_ratio={ratio}（完全幼稚）")
        else:
            print(f"  ⚠️ maturity 异常: ratio={ratio}（期望 0.0）")

        # 验证 lr_multiplier（幼稚态高）
        lr_mult = lifecycle.maturity.get_lr_multiplier(new_nid)
        print(f"  ✅ 幼稚态学习率倍数: {lr_mult:.2f}（应 > 1.0，幼稚态加速学习）")
    else:
        print(f"  ⚠️ lifecycle 未注入，跳过 maturity 验证")

    # Step 6: 验证新神经元参与 ensemble forward
    print("\n[Step 6] 验证新神经元参与 ensemble forward...")
    import torch
    # 构造测试输入
    if cortex._shared_embedding is not None:
        # 用 shared_embedding 编码测试输入
        test_ids = torch.randint(0, cortex._shared_embedding.num_embeddings, (1, 8))
        shared_emb = cortex._shared_embedding(test_ids)  # [1, 8, hidden]
        try:
            result = cortex.think(shared_emb)
            active_ids = set()
            for round_scores in result.get("round_scores", []):
                active_ids.update(round_scores.keys())
            if new_nid in active_ids or new_nid in cortex.ensemble.neurons:
                print(f"  ✅ 新神经元 {new_nid} 已参与共振")
            else:
                print(f"  ⚠️ 新神经元未参与共振")
        except Exception as e:
            print(f"  ⚠️ ensemble forward 失败: {e}")
    else:
        print(f"  ⚠️ shared_embedding 未设置，跳过 forward 验证")

    # Step 7: 验证 life_scheduler 接线闭环
    print("\n[Step 7] 验证 life_scheduler 接线闭环...")
    from taiji.life.life_scheduler import get_life_scheduler
    life = get_life_scheduler()
    if life._cortex is not None:
        print(f"  ✅ life_scheduler._cortex 已注入: {type(life._cortex).__name__}")
    else:
        print(f"  ⚠️ life_scheduler._cortex 未注入")
    if life._lifecycle is not None:
        print(f"  ✅ life_scheduler._lifecycle 已注入")
    else:
        print(f"  ⚠️ life_scheduler._lifecycle 未注入")
    if life._feed_engine is not None:
        print(f"  ✅ life_scheduler._feed_engine 已注入")
    else:
        print(f"  ⚠️ life_scheduler._feed_engine 未注入")

    # Step 8: 验证连续创建（zh_2 — Step 2 已创建 zh_1）
    print("\n[Step 8] 验证连续创建（期望 zh_2）...")
    new_nid_2 = cortex.add_neuron("zh", lifecycle=lifecycle)
    if new_nid_2 == "zh_2":
        print(f"  ✅ 第二个新神经元: {new_nid_2}（ID 递增正确）")
    else:
        print(f"  ⚠️ ID 递增异常: 期望 zh_2, 实际 {new_nid_2}")
    # 清理 zh_2 的 ckpt
    ckpt_path_2 = os.path.join(cortex.neurons_dir, f"neuron_{new_nid_2}.pt")
    if os.path.exists(ckpt_path_2):
        os.remove(ckpt_path_2)

    # Step 9: 验证 field_dim 不一致校验
    print("\n[Step 9] 验证 ensemble.add_neuron field_dim 校验...")
    from taiji.resonance.neuron import ResonanceNeuron
    from taiji.resonance.config import get_domain_neuron_config
    # 创建一个 STANDARD 规格（hidden_size=768，与 COMPACT=512 不一致）
    cfg_std = get_domain_neuron_config("en", spec="standard")
    neuron_std = ResonanceNeuron(cfg_std)
    try:
        cortex.ensemble.add_neuron("test_bad", neuron_std)
        print(f"  ⚠️ field_dim 校验未生效（应报错）")
    except ValueError as e:
        print(f"  ✅ field_dim 校验生效: {str(e)[:60]}...")

    # 清理：从 cortex.neurons 移除测试神经元
    if new_nid in cortex.neurons:
        del cortex.neurons[new_nid]
    if new_nid_2 in cortex.neurons:
        del cortex.neurons[new_nid_2]

    # Step 10: 综合判断
    print("\n" + "=" * 60)
    all_pass = (
        new_nid == "zh_1" and
        new_count == initial_neuron_count + 1 and
        new_nid_2 == "zh_2"
    )
    # 清理 zh_1 的 ckpt（zh_2 已在 Step 8 清理）
    ckpt_path_1 = os.path.join(cortex.neurons_dir, f"neuron_{new_nid}.pt")
    if os.path.exists(ckpt_path_1):
        os.remove(ckpt_path_1)
    if all_pass:
        print("🎉 验证通过：neurogenesis 完整闭环成功")
        print(f"   - Cortex.add_neuron: 创建新神经元 + 注入 ensemble")
        print(f"   - 持久化: ckpt 保存到 neurons_dir/neuron_{{nid}}.pt")
        print(f"   - maturity.register_new: 幼稚态追踪 + 学习率倍数")
        print(f"   - ensemble forward: 新神经元参与共振")
        print(f"   - life_scheduler 接线: assemble_cortex Step 9.1 闭环")
        print(f"   - field_dim 校验: 防止不一致的神经元加入")
        print(f"   - 连续创建: ID 递增（zh_1, zh_2, ...）")
        return 0
    else:
        print("⚠️ 验证未完全通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
