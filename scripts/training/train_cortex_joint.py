"""整体训练态极——多神经元联合训练。

核心改变：从"单独训练每个神经元 → 拼装 → 测试协作"变为
"创建 N 个神经元 → 整体训练 → 协作在训练中学习"。

前向传播时所有神经元参与，共振场聚合（forward_train 全可微），
反向传播流经聚合权重 → 神经元学习如何写入场、如何协同输出。
专精化在训练中自然涌现（像 MoE）。

Usage:
    # 5 个 zh 神经元联合训练 3000 步
    python -u scripts/training/train_cortex_joint.py --n_neurons 5 --steps 3000

    # 10 个神经元
    python -u scripts/training/train_cortex_joint.py --n_neurons 10 --steps 5000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config,
)
from taiji.resonance.translator import batch_align_and_embed

# 复用 train_neuron.py 的数据/tokenizer 加载函数
from scripts.training.train_neuron import (
    load_domain_texts, load_domain_tokenizer, load_general_tokenizer,
    create_shared_embedding, OUTPUT_DIR, SHARED_EMBEDDING_PATH,
)

DOMAIN = "zh"
GENERAL_VOCAB_SIZE = 256000
SHARED_EMBED_DIM = 512


def train_cortex_joint(
    n_neurons: int = 5,
    domain: str = "zh",
    num_steps: int = 3000,
    batch_size: int = 4,
    lr: float = 5e-4,
    balance_lambda: float = 0.1,
    temperature: float = 1.0,
    device: str = "cpu",
    log_every: int = 50,
    max_texts: int = 8000,
    spec: str = "compact",
    freeze_embedding: bool = False,
    use_gamma: bool = False,
    fusion_mode: str = "residual",
    weight_decay: float = 0.1,
    warmup_steps: int = 200,
    dropout: float = 0.1,
):
    """联合训练 N 个神经元 + shared_embedding，端到端可微。

    所有神经元看所有数据，通过共振场聚合，专精化自然涌现。

    fusion_mode（方向③ 残差预测编码）：
      - "residual"（默认）：族长完整预测 + 其他神经元残差修正
      - "soft"：软加权融合（A/B 对照）

    正则化（社区规范 SmolLM/Chinchilla）：
      - weight_decay: AdamW 权重衰减，防止过拟合（默认 0.1）
      - warmup_steps: WSD 调度的 warmup 阶段步数（默认 200）
      - dropout: Transformer dropout 率，防止过拟合（默认 0.1）
    """
    print("=" * 70, flush=True)
    print(f"整体训练态极 — {n_neurons} 个 {domain}({spec}) 神经元联合训练", flush=True)
    print(f"  目标：协作在训练中学习，而非事后拼装", flush=True)
    print(f"  steps={num_steps} batch={batch_size} lr={lr} "
          f"balance_λ={balance_lambda} temp={temperature} max_texts={max_texts} spec={spec}", flush=True)
    print(f"  fusion_mode={fusion_mode} "
          f"({'族长+残差修正' if fusion_mode == 'residual' else '软加权融合'})", flush=True)
    print(f"  正则化: weight_decay={weight_decay} dropout={dropout} warmup={warmup_steps}", flush=True)
    print("=" * 70, flush=True)

    # ── 1. 加载数据 ──
    print(f"\n[1] 加载 {domain} 训练数据...", flush=True)
    texts = load_domain_texts(domain, max_texts=max_texts)
    print(f"  {len(texts)} 条文本", flush=True)

    # ── 2. 加载 tokenizers ──
    print(f"\n[2] 加载 tokenizers...", flush=True)
    domain_sp = load_domain_tokenizer(domain)
    general_sp = load_general_tokenizer()
    print(f"  domain vocab={domain_sp.vocab_size()}, general vocab={general_sp.vocab_size()}", flush=True)

    # ── 3. 创建 shared_embedding（可训练）──
    print(f"\n[3] 创建 shared_embedding...", flush=True)
    shared_embedding = create_shared_embedding(device=device)
    if freeze_embedding:
        emb_path = os.path.join(OUTPUT_DIR, "shared_embedding_joint.pt")
        if os.path.exists(emb_path):
            shared_embedding.load_state_dict(torch.load(emb_path, map_location=device))
            print(f"  已加载预训练 shared_embedding（冻结）", flush=True)
        else:
            print(f"  ⚠️ 预训练 embedding 不存在，使用随机初始化（冻结）", flush=True)
        shared_embedding.requires_grad_(False)
        shared_embedding.eval()

    # ── 4. 创建 N 个神经元（同域，随机初始化）──
    print(f"\n[4] 创建 {n_neurons} 个 {spec} 神经元...", flush=True)
    cfg = get_domain_neuron_config(domain, spec=spec)
    cfg.dropout = dropout  # 正则化：防止过拟合
    neurons: dict[str, ResonanceNeuron] = {}
    for i in range(n_neurons):
        nid = f"{domain}_j{i}"
        neuron = ResonanceNeuron(cfg).to(device)
        neuron.train()
        neurons[nid] = neuron
        n_params = sum(p.numel() for p in neuron.parameters())
        print(f"  [{nid}] {n_params/1e6:.0f}M params", flush=True)

    # ── 5. 创建 ensemble ──
    print(f"\n[5] 创建 ResonanceEnsemble...", flush=True)
    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)
    print(f"  field_dim={cfg.field_dim}, neurons={list(neurons.keys())}", flush=True)

    # 振荡同步（可选，方向②）
    gamma_oscillator = None
    if use_gamma:
        from taiji.resonance.gamma_oscillator import GammaOscillator
        gamma_oscillator = GammaOscillator()
        # 同域同相位（绑定成知觉单元），跨域不同相位（解绑）
        domain_to_nids = {domain: list(neurons.keys())}
        gamma_oscillator.assign_phase_by_domain(domain_to_nids)
        print(f"  振荡同步: 已启用 GammaOscillator，{len(neurons)} 个神经元同相位（同域绑定）", flush=True)

    # ── 6. 优化器：所有神经元参数 +（可选）shared_embedding ──
    # 社区规范：AdamW + weight_decay 防止过拟合（SmolLM 用 0.1）
    all_params = []
    for neuron in neurons.values():
        all_params.extend(neuron.parameters())
    if not freeze_embedding:
        all_params.extend(shared_embedding.parameters())

    optimizer = torch.optim.AdamW(
        all_params, lr=lr, weight_decay=weight_decay,
    )
    total_params = sum(p.numel() for p in all_params)
    print(f"\n[6] 优化器: AdamW(lr={lr}, weight_decay={weight_decay}), "
          f"{total_params/1e6:.0f}M total params", flush=True)

    # WSD 学习率调度（社区规范 SmolLM3）：
    # Warmup（线性升温）→ Stable（稳定）→ Decay（余弦衰减到最后 10% lr）
    # decay 在最后 20% 步数进行
    decay_start = max(warmup_steps + 1, int(num_steps * 0.8))
    def _wsd_lr(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps  # 线性升温
        elif step < decay_start:
            return 1.0  # 稳定
        else:
            # 余弦衰减到 10%
            progress = (step - decay_start) / max(1, num_steps - decay_start)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _wsd_lr)
    print(f"  WSD 调度: warmup={warmup_steps}步, stable到{decay_start}步, "
          f"decay到最后{decay_start}→{num_steps}步", flush=True)

    # ── 7. 训练循环 ──
    print(f"\n[7] 开始联合训练 ({num_steps} steps)...", flush=True)
    n_texts = len(texts)

    def _sample_batch():
        idx = torch.randint(0, n_texts, (batch_size,))
        return [texts[int(i)] for i in idx]

    total_loss = 0.0
    total_ce = 0.0
    total_balance = 0.0
    step, t_start = 0, time.time()
    best_loss = float("inf")
    best_step = 0
    best_states = None  # 所有神经元的 best state_dict
    recent_losses = []

    for _ in range(num_steps):
        batch_texts = _sample_batch()

        # 数据对齐：text → general_ids → shared_emb + domain targets
        shared_emb, targets, mask = batch_align_and_embed(
            batch_texts, domain_sp, general_sp, shared_embedding,
        )
        shared_emb = shared_emb.to(device)   # [B, L, 512]
        targets = targets.to(device)          # [B, L]
        mask = mask.to(device)                # [B, L]

        # 前向：所有神经元参与，共振场聚合（全可微）+ 振荡门控
        result = ensemble.forward_train(
            shared_emb, temperature=temperature,
            gamma_oscillator=gamma_oscillator,
            fusion_mode=fusion_mode,
        )
        fused_logits = result["fused_logits"]       # [B, L, V]
        balance_loss = result["balance_loss"]        # scalar
        weights = result["weights"]                  # [N] 或 [N-1]（residual 模式）

        # CE loss（next-token prediction on fused logits）
        shift_logits = fused_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()
        shift_targets = shift_targets.clone()
        shift_targets[~shift_mask] = -100

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        # 总 loss = CE + λ * 负载均衡（负熵）
        loss = ce_loss + balance_lambda * balance_loss

        # nan/inf 检查：跳过异常步（防止梯度爆炸污染统计累加器）
        # 8×standard 实验中 avg_ce=nan 即因此 bug 导致
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️ step {step+1}: nan/inf loss detected, skipping "
                  f"(consider lowering lr or adding warmup)", flush=True)
            step += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()  # WSD 学习率调度

        # 振荡同步：每步推进全局相位（模拟 Gamma 振荡周期）
        if gamma_oscillator is not None:
            gamma_oscillator.tick()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_balance += balance_loss.item()
        step += 1

        # 滑动窗口 best 追踪
        recent_losses.append(loss.item())
        if len(recent_losses) > 100:
            recent_losses.pop(0)
        if len(recent_losses) >= 50:
            recent_avg = sum(recent_losses) / len(recent_losses)
            if recent_avg < best_loss:
                best_loss = recent_avg
                best_step = step
                best_states = {
                    nid: {k: v.detach().clone() for k, v in neuron.state_dict().items()}
                    for nid, neuron in neurons.items()
                }

        if step % log_every == 0:
            avg_loss = total_loss / step
            avg_ce = total_ce / step
            ppl = math.exp(min(avg_ce, 20))
            elapsed = time.time() - t_start
            cur_lr = optimizer.param_groups[0]["lr"]
            # 残差模式：标注族长，权重只含其他神经元
            if fusion_mode == "residual" and "leader_idx" in result:
                leader_nid = list(neurons.keys())[result["leader_idx"]]
                other_nids = [n for n in neurons.keys() if n != leader_nid]
                w_str = ", ".join(f"{nid}:{w:.2f}" for nid, w in
                                  zip(other_nids, weights.tolist()))
                print(
                    f"  step {step}/{num_steps} "
                    f"loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                    f"bal={balance_loss.item():.4f} "
                    f"avg_ce={avg_ce:.4f} PPL={ppl:.1f} "
                    f"lr={cur_lr:.2e} "
                    f"leader={leader_nid} w=[{w_str}] "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
            else:
                w_str = ", ".join(f"{nid}:{w:.2f}" for nid, w in
                                  zip(neurons.keys(), weights.tolist()))
                print(
                    f"  step {step}/{num_steps} "
                    f"loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                    f"bal={balance_loss.item():.4f} "
                    f"avg_ce={avg_ce:.4f} PPL={ppl:.1f} "
                    f"lr={cur_lr:.2e} "
                    f"w=[{w_str}] "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    # ── 8. 训练结束，保存 ──
    avg_ce = total_ce / max(step, 1)
    ppl = math.exp(min(avg_ce, 20))
    elapsed = time.time() - t_start
    print(
        f"\n训练完成: {step} steps, avg_ce={avg_ce:.4f}, PPL={ppl:.1f}, "
        f"best_loss={best_loss:.4f}@step{best_step}, "
        f"time={elapsed:.0f}s ({elapsed/60:.1f}min)",
        flush=True,
    )

    # 保存 best 模型（或末步）
    save_states = best_states if best_states is not None else {
        nid: neuron.state_dict() for nid, neuron in neurons.items()
    }
    saved_label = "best" if best_states is not None else "final"

    print(f"\n保存 {n_neurons} 个神经元到 {OUTPUT_DIR}...", flush=True)
    for nid, state in save_states.items():
        save_path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
        torch.save({
            "neuron_config": cfg,
            "state_dict": state,
            "domain": domain,
            "result": {
                "final_loss": avg_ce,
                "final_ppl": ppl,
                "steps": step,
                "best_loss": best_loss,
                "best_step": best_step,
                "saved": saved_label,
                "joint_trained": True,
                "n_neurons": n_neurons,
                "spec": spec,
                "freeze_embedding": freeze_embedding,
                "fusion_mode": fusion_mode,
                "use_gamma": use_gamma,
            },
        }, save_path)
        print(f"  {save_path} ({saved_label})", flush=True)

    # 保存 shared_embedding（冻结时跳过，复用预训练的）
    if not freeze_embedding:
        emb_path = os.path.join(OUTPUT_DIR, "shared_embedding_joint.pt")
        torch.save(shared_embedding.state_dict(), emb_path)
        print(f"  {emb_path}", flush=True)
    else:
        print(f"  shared_embedding 已冻结，复用预训练版本", flush=True)

    print(f"\n✅ 联合训练完成！下一步运行评估:", flush=True)
    print(f"  python -u scripts/training/eval_joint.py --n_neurons {n_neurons} --spec {spec}", flush=True)

    return {"final_loss": avg_ce, "final_ppl": ppl, "steps": step, "best_loss": best_loss}


def main():
    parser = argparse.ArgumentParser(description="整体训练态极——多神经元联合训练")
    parser.add_argument("--n_neurons", type=int, default=5,
                        help="神经元数量（默认5，验证后可扩到10+）")
    parser.add_argument("--domain", default="zh", help="训练域（默认zh）")
    parser.add_argument("--steps", type=int, default=3000, help="训练步数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--balance_lambda", type=float, default=0.1,
                        help="负载均衡系数（防止神经元垄断）")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="软路由温度（低=更尖锐选择）")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--log_every", type=int, default=50, help="日志间隔步数")
    parser.add_argument("--max_texts", type=int, default=100000,
                        help="最大训练文本数（默认100000；数据量越大越防过拟合）")
    parser.add_argument("--spec", default="compact",
                        help="神经元规格 compact(36M)/standard(111M)/expert(253M)；10×standard≈1B")
    parser.add_argument("--freeze_embedding", action="store_true",
                        help="冻结 shared_embedding（省内存；复用预训练embedding）")
    parser.add_argument("--use_gamma", action="store_true",
                        help="启用振荡同步（GammaOscillator 相位门控，方向②）")
    parser.add_argument("--fusion_mode", default="residual",
                        choices=["residual", "soft"],
                        help="融合模式：residual(族长+残差修正,默认) | soft(软加权融合)")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="AdamW 权重衰减（默认0.1，SmolLM 规范）")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="WSD 调度 warmup 步数（默认200）")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout 率（默认0.1，防止过拟合）")
    args = parser.parse_args()

    train_cortex_joint(
        n_neurons=args.n_neurons,
        domain=args.domain,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        balance_lambda=args.balance_lambda,
        temperature=args.temperature,
        device=args.device,
        log_every=args.log_every,
        max_texts=args.max_texts,
        spec=args.spec,
        freeze_embedding=args.freeze_embedding,
        use_gamma=args.use_gamma,
        fusion_mode=args.fusion_mode,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
