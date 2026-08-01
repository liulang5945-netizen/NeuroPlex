"""S8 checkpoint round-trip smoke test.

验证 finetune_cross_spec.py 的 save_checkpoint / load_checkpoint / build_final_artifact
正确保存和恢复 body_state + shared_embedding_state + body_optimizer_state。

用 mock neuron 避免加载真实模型，聚焦 checkpoint 逻辑正确性。
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from scripts.training.finetune_cross_spec import (
    save_checkpoint, load_checkpoint, build_final_artifact,
)


class MockChannel(nn.Module):
    def __init__(self, src_dim, dst_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dst_dim, src_dim) * 0.01)


class MockNeuron(nn.Module):
    """模拟 ResonanceNeuron 的关键结构：layers / norm / lm_head / field_write / excite_channels / scale。"""

    def __init__(self, n_layers=4, hidden=32, field_dim=16, vocab=100):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.field_write = nn.Linear(hidden, field_dim, bias=False)
        # side_channels（用 ModuleDict 模拟）
        self.excite_channels = nn.ModuleDict()
        self.inhibit_channels = nn.ModuleDict()
        # scale 参数（0D）
        self.register_parameter("scale_global", nn.Parameter(torch.tensor(50.0)))
        # bias buffer
        self.register_buffer("bias_global", torch.zeros(1))

    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return super().named_buffers(*args, **kwargs)


class MockEnsemble:
    """模拟 ensemble 的 _cross_spec_projectors / _cross_spec_back_projectors。"""
    def __init__(self):
        self._cross_spec_projectors = nn.ModuleDict({
            "n0": nn.Linear(16, 32, bias=False),
            "n1": nn.Linear(16, 32, bias=False),
        })
        self._cross_spec_back_projectors = nn.ModuleDict({
            "n0": nn.Linear(32, 16, bias=False),
            "n1": nn.Linear(32, 16, bias=False),
        })


def main():
    print("=" * 60)
    print("S8 checkpoint round-trip smoke test")
    print("=" * 60)

    # 1. 构造 mock neurons + ensemble + shared_embeddings
    neurons = {"n0": MockNeuron(n_layers=4, hidden=32, field_dim=16, vocab=100),
               "n1": MockNeuron(n_layers=4, hidden=32, field_dim=16, vocab=100)}
    ensemble = MockEnsemble()
    shared_embeddings = {
        "n0": nn.Embedding(1000, 32),
        "n1": nn.Embedding(1000, 32),
    }

    # 2. 模拟 S8 冻结策略：解冻最后 2 层 + norm + lm_head + field_write
    print("\n[1] 模拟 S8 冻结策略 (unfreeze_layers=2)...")
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        # side_channels 可训练
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                p.requires_grad = True
        # 解冻最后 2 层 + norm + lm_head + field_write
        n_layers = len(neuron.layers)
        for i in range(n_layers - 2, n_layers):
            for p in neuron.layers[i].parameters():
                p.requires_grad = True
        for p in neuron.norm.parameters():
            p.requires_grad = True
        for p in neuron.lm_head.parameters():
            p.requires_grad = True
        for p in neuron.field_write.parameters():
            p.requires_grad = True

    # shared_embedding 可训练
    for emb in shared_embeddings.values():
        for p in emb.parameters():
            p.requires_grad = True

    # 统计
    body_count = sum(1 for n in neurons.values() for name, p in n.named_parameters()
                     if p.requires_grad and not any(name.startswith(pfx) for pfx in ["excite_", "inhibit_"])
                     and "scale_" not in name and "bias_" not in name)
    emb_count = sum(p.numel() for emb in shared_embeddings.values() for p in emb.parameters() if p.requires_grad)
    print(f"  body 可训练参数张量数: {body_count}, emb 可训练参数数: {emb_count}")
    assert body_count > 0, "body 参数应可训练"
    assert emb_count > 0, "emb 参数应可训练"

    # 3. 收集参数，构建优化器
    print("\n[2] 构建优化器...")
    body_params = []
    for nid, neuron in neurons.items():
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]):
                continue
            if "scale_" in name or "bias_" in name:
                continue
            body_params.append(p)
    emb_params = [p for emb in shared_embeddings.values() for p in emb.parameters() if p.requires_grad]

    # side_channels + 投影层 走 Muon (用 AdamW 代替 mock)
    side_params = []
    for nid, neuron in neurons.items():
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                if p.requires_grad:
                    side_params.append(p)
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
            side_params.append(p)
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
            side_params.append(p)

    optimizer = torch.optim.AdamW(side_params, lr=1e-3)
    body_optimizer = torch.optim.AdamW(body_params + emb_params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    body_scheduler = torch.optim.lr_scheduler.LambdaLR(body_optimizer, lambda s: 1.0)

    # 4. 模拟一步训练，改变 body 参数值
    print("\n[3] 模拟一步训练（改变 body 参数）...")
    # 记录训练前的 body 参数值
    body_before = {}
    for nid, neuron in neurons.items():
        for name, p in neuron.named_parameters():
            if p.requires_grad and not any(name.startswith(pfx) for pfx in ["excite_", "inhibit_"]):
                if "scale_" not in name and "bias_" not in name:
                    body_before[(nid, name)] = p.data.clone()

    # 模拟梯度 + step
    for p in body_params + emb_params:
        p.grad = torch.randn_like(p) * 0.01
    body_optimizer.step()
    body_scheduler.step()

    # 验证 body 参数确实变了
    changed = 0
    for nid, neuron in neurons.items():
        for name, p in neuron.named_parameters():
            if (nid, name) in body_before:
                if not torch.equal(body_before[(nid, name)], p.data):
                    changed += 1
    print(f"  body 参数改变数: {changed}/{len(body_before)}")
    assert changed > 0, "body 参数应被 optimizer.step() 改变"

    # 5. 保存 checkpoint
    print("\n[4] 保存 checkpoint...")
    tmpdir = tempfile.mkdtemp(prefix="s8_smoke_")
    ckpt_path = os.path.join(tmpdir, "test.ckpt.pt")
    final_path = os.path.join(tmpdir, "test.final.pt")

    save_checkpoint(ckpt_path, epoch=0, total_steps=1, optimizer=optimizer,
                    neurons=neurons, ensemble=ensemble, loss_history=[{"step": 1, "loss": 1.0}],
                    adamw_optimizer=None, scheduler=scheduler,
                    body_optimizer=body_optimizer, body_scheduler=body_scheduler,
                    shared_embeddings=shared_embeddings)
    print(f"  saved: {ckpt_path}")

    # 也保存 final artifact
    artifact = build_final_artifact(neurons, ensemble, shared_embeddings)
    torch.save(artifact, final_path)
    print(f"  saved: {final_path}")

    # 验证 ckpt 包含期望的 keys
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    expected_keys = ["side_channels_state", "scale_bias_state", "cross_spec_state",
                     "body_state", "body_optimizer_state", "body_scheduler_state",
                     "shared_embedding_state"]
    missing = [k for k in expected_keys if k not in ckpt]
    print(f"  checkpoint keys: {sorted(ckpt.keys())}")
    if missing:
        print(f"  MISSING: {missing}")
        sys.exit(1)
    else:
        print(f"  所有期望 keys 都存在 ✓")

    # 验证 final artifact 包含期望 keys
    artifact_loaded = torch.load(final_path, map_location="cpu", weights_only=False)
    expected_artifact_keys = ["side_channels", "cross_spec", "body_state", "shared_embedding_state"]
    missing_art = [k for k in expected_artifact_keys if k not in artifact_loaded]
    print(f"  artifact keys: {sorted(artifact_loaded.keys())}")
    if missing_art:
        print(f"  MISSING: {missing_art}")
        sys.exit(1)
    else:
        print(f"  artifact 所有期望 keys 都存在 ✓")

    # 6. 创建新 neurons + ensemble，加载 checkpoint，验证 body 参数恢复
    print("\n[5] 加载 checkpoint 验证 round-trip...")
    neurons2 = {"n0": MockNeuron(n_layers=4, hidden=32, field_dim=16, vocab=100),
                "n1": MockNeuron(n_layers=4, hidden=32, field_dim=16, vocab=100)}
    ensemble2 = MockEnsemble()
    shared_embeddings2 = {
        "n0": nn.Embedding(1000, 32),
        "n1": nn.Embedding(1000, 32),
    }
    # 镜像相同的 S8 冻结策略（保证 body_optimizer 参数组一致）
    for nid, neuron in neurons2.items():
        for p in neuron.parameters():
            p.requires_grad = False
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                p.requires_grad = True
        n_layers = len(neuron.layers)
        for i in range(n_layers - 2, n_layers):
            for p in neuron.layers[i].parameters():
                p.requires_grad = True
        for p in neuron.norm.parameters():
            p.requires_grad = True
        for p in neuron.lm_head.parameters():
            p.requires_grad = True
        for p in neuron.field_write.parameters():
            p.requires_grad = True
    for emb in shared_embeddings2.values():
        for p in emb.parameters():
            p.requires_grad = True

    # 新优化器（与原优化器相同结构）
    side_params2 = []
    for nid, neuron in neurons2.items():
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                if p.requires_grad:
                    side_params2.append(p)
    for proj in ensemble2._cross_spec_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
            side_params2.append(p)
    for proj in ensemble2._cross_spec_back_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
            side_params2.append(p)
    optimizer2 = torch.optim.AdamW(side_params2, lr=1e-3)

    body_params2 = []
    for nid, neuron in neurons2.items():
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]):
                continue
            if "scale_" in name or "bias_" in name:
                continue
            body_params2.append(p)
    emb_params2 = [p for emb in shared_embeddings2.values() for p in emb.parameters() if p.requires_grad]
    body_optimizer2 = torch.optim.AdamW(body_params2 + emb_params2, lr=1e-4)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda s: 1.0)
    body_scheduler2 = torch.optim.lr_scheduler.LambdaLR(body_optimizer2, lambda s: 1.0)

    epoch, steps, hist = load_checkpoint(
        ckpt_path, optimizer2, neurons2, ensemble2,
        adamw_optimizer=None, scheduler=scheduler2,
        body_optimizer=body_optimizer2, body_scheduler=body_scheduler2,
        shared_embeddings=shared_embeddings2,
    )
    print(f"  restored: epoch={epoch}, steps={steps}, hist={len(hist)}")

    # 7. 验证 body 参数完全匹配
    print("\n[6] 验证 body 参数 round-trip...")
    mismatches = 0
    for nid, neuron in neurons2.items():
        for name, p in neuron.named_parameters():
            if (nid, name) in body_before:
                # body_before 是训练前的；训练后变了；ckpt 保存的是训练后的
                # neurons2 加载后应等于 neurons 训练后的值
                original_trained = None
                for n2_id, n2 in neurons.items():
                    if n2_id == nid:
                        for n2_name, n2_p in n2.named_parameters():
                            if n2_name == name:
                                original_trained = n2_p.data
                                break
                if original_trained is not None and not torch.equal(original_trained, p.data):
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"  MISMATCH {nid}.{name}: orig={original_trained.flatten()[:3]} loaded={p.data.flatten()[:3]}")
    print(f"  body 参数不匹配数: {mismatches}")
    assert mismatches == 0, f"body 参数 round-trip 失败 ({mismatches} 不匹配)"

    # 8. 验证 shared_embedding round-trip
    print("\n[7] 验证 shared_embedding round-trip...")
    emb_mismatches = 0
    for nid in shared_embeddings:
        orig_emb = shared_embeddings[nid].weight.data
        loaded_emb = shared_embeddings2[nid].weight.data
        if not torch.equal(orig_emb, loaded_emb):
            emb_mismatches += 1
    print(f"  emb 不匹配数: {emb_mismatches}")
    assert emb_mismatches == 0, f"shared_embedding round-trip 失败"

    # 9. 验证 body_optimizer state 恢复
    print("\n[8] 验证 body_optimizer state 恢复...")
    assert len(body_optimizer2.state) > 0, "body_optimizer state 未恢复（应为每个参数有 exp_avg）"
    print(f"  body_optimizer state 参数数: {len(body_optimizer2.state)} ✓")

    # 清理
    import shutil
    shutil.rmtree(tmpdir)

    print("\n" + "=" * 60)
    print("✓ S8 checkpoint round-trip smoke test 全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
