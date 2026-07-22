"""SFT 训练脚本 v2：解决域漂移问题.

核心修复（v2）：
  1. 共享 W_base：所有 neuron 共享一个 nn.Linear(hidden, vocab) 基矩阵
     - 模拟"所有神经元共享基础语言能力"（架构本意）
     - 训练前注入到所有 neuron 的 lm_head_base
  2. 冻结 W_base：W_base 不进 optimizer，只训练 per-neuron delta_u/v
     - 避免 W_base 漂移导致所有 neuron 一起漂
  3. Domain regularizer：KL(output_token_dist || domain_token_dist)
     - domain_token_dist 从 SFT 数据统计得到（每域一个先验分布）
     - 让 zh neuron 输出的 token 分布偏向 zh 数据特征
     - 抑制 zh neuron 输出大量英文 token 这种域漂移

训练目标（每域）：
  total_loss = LM_loss + λ * KL_div(output_dist, domain_dist)
    - LM_loss: next-token cross_entropy（只对 response 部分）
    - KL_div: 让 neuron 输出 token 分布与域先验对齐

数据契约（来自 download_sft_data.py）：
  data/sft/sft_tokenized.pt
    dict[domain] -> {
      "input_ids":      LongTensor[N, 256],
      "labels":         LongTensor[N, 256],
      "response_mask":  LongTensor[N, 256],
    }

使用方式：
  python scripts/training/sft_train_neurons_v2.py --domain zh --steps 500
  python scripts/training/sft_train_neurons_v2.py --all --steps 1000
"""
from __future__ import annotations

import os
import sys
import math
import argparse
import time
from collections import Counter

# sentencepiece + 项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_libs"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from taiji.loader import create_cortex
from taiji.resonance import SharedContextEncoder


# ── 配置 ──
NEURONS_DIR = "data/neurons"
ENCODER_PATH = "data/distill/shared_context_encoder.pt"
SFT_DATA_PATH = "data/sft/sft_tokenized.pt"
W_BASE_PATH = "data/neurons/_shared_lm_head_base.pt"  # 持久化 W_base

BATCH_SIZE = 2
SEQ_LEN = 256
LR = 5e-5
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 25
SAVE_INTERVAL = 200
# v2: domain regularizer 权重
KL_WEIGHT = 0.5
# v2: domain 先验 smoothing
PRIOR_SMOOTHING = 1e-4


def compute_domain_prior(input_ids: torch.Tensor, vocab_size: int,
                         smoothing: float = PRIOR_SMOOTHING) -> torch.Tensor:
    """从 SFT 数据统计该域 token 频率分布.

    Returns:
        prior: [vocab_size] 概率分布（已归一化 + smoothing）
    """
    # 统计所有 token 出现次数（包括 prompt 和 response）
    counts = torch.bincount(input_ids.flatten(), minlength=vocab_size).float()
    # smoothing 避免零概率
    counts = counts + smoothing
    prior = counts / counts.sum()
    return prior


def build_shared_w_base(hidden_size: int, vocab_size: int, device: str,
                         neurons: dict = None) -> nn.Linear:
    """构建或加载共享 W_base（所有 neuron 共享的语言基础）.

    v2.0: 随机初始化 W_base（让 delta 学全部能力，避免与原 delta 重复）
    v2.1 实验证明：用 delta 平均初始化 W_base 反而让 W_base 与 delta 重复
                  (logits = W_base + delta ≈ 2*delta 早期)，破坏蒸馏后能力
    v2.0 随机 W_base 效果更好：LM 42.90→23.32 vs v2.1 50.62→30.80

    Args:
        hidden_size: hidden dim
        vocab_size: vocab size
        device: 计算设备
        neurons: （保留接口，v2.0 不使用）
    """
    if os.path.exists(W_BASE_PATH):
        print(f"  [W_base] Loading existing from {W_BASE_PATH}")
        w_base = nn.Linear(hidden_size, vocab_size, bias=False)
        state = torch.load(W_BASE_PATH, map_location=device, weights_only=False)
        if isinstance(state, dict) and "weight" in state:
            w_base.load_state_dict(state)
        else:
            w_base.weight.data = state
        w_base.to(device)
        w_base.eval()
        for p in w_base.parameters():
            p.requires_grad = False
        print(f"  [W_base] Loaded + frozen (shape={w_base.weight.shape})")
        return w_base

    # v2.0: 随机初始化（让 delta 学全部能力）
    print(f"  [W_base] Creating new random base (hidden={hidden_size}, vocab={vocab_size})")
    w_base = nn.Linear(hidden_size, vocab_size, bias=False)
    nn.init.normal_(w_base.weight, std=0.02)
    w_base.to(device)
    w_base.eval()
    for p in w_base.parameters():
        p.requires_grad = False
    os.makedirs(os.path.dirname(W_BASE_PATH), exist_ok=True)
    torch.save(w_base.state_dict(), W_BASE_PATH)
    print(f"  [W_base] Saved to {W_BASE_PATH} (frozen, random init)")
    return w_base


def inject_shared_w_base(cortex, w_base: nn.Linear):
    """把 W_base 注入到所有 neuron."""
    n_injected = 0
    for nid, neuron in cortex.neurons.items():
        if hasattr(neuron, "set_shared_lm_head"):
            try:
                neuron.set_shared_lm_head(w_base)
                n_injected += 1
            except RuntimeError as e:
                # 传统模式（lm_head_rank=0）不支持
                pass
    print(f"  [W_base] Injected into {n_injected}/{len(cortex.neurons)} neurons")
    return n_injected


def load_cortex_and_encoder(device: str):
    """加载 Cortex + SharedContextEncoder + 注入 W_base."""
    print("[1] Loading Cortex + neurons...")
    cortex, tokenizer = create_cortex(
        neurons_dir=NEURONS_DIR,
        device=device,
        max_rounds=2,
        enable_gating=False,
    )
    print(f"  Loaded {len(cortex.neurons)} neurons: {list(cortex.neurons.keys())}")

    # v2: 注入共享 W_base
    print("[2] Building shared W_base...")
    # H2 修复：校验所有 neuron hidden_size 一致后再构建 W_base，
    # 避免混合 spec 时 W_base shape 与部分 neuron 不匹配被静默吞掉。
    hidden_sizes = {n.config.hidden_size for n in cortex.neurons.values()}
    if len(hidden_sizes) > 1:
        raise ValueError(
            f"Neurons have mixed hidden_size: {hidden_sizes}. "
            f"Cannot build a single shared W_base. Re-distill with consistent spec."
        )
    first_neuron = next(iter(cortex.neurons.values()))
    w_base = build_shared_w_base(first_neuron.config.hidden_size,
                                  first_neuron.config.vocab_size, device,
                                  neurons=cortex.neurons)
    inject_shared_w_base(cortex, w_base)

    print("[3] Loading SharedContextEncoder...")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")
    encoder = SharedContextEncoder.load(ENCODER_PATH, device=device)
    print(f"  Encoder loaded: hidden_dim={encoder.hidden_dim}, n_domains={encoder.n_domains}")
    return cortex, tokenizer, encoder, w_base


def load_sft_data(domains: list, device: str) -> dict:
    """加载 SFT tokenized 数据 + 计算每域 prior."""
    print(f"[4] Loading SFT data from {SFT_DATA_PATH}...")
    if not os.path.exists(SFT_DATA_PATH):
        raise FileNotFoundError(f"SFT data not found: {SFT_DATA_PATH}")
    all_data = torch.load(SFT_DATA_PATH, map_location="cpu", weights_only=False)
    out = {}
    for domain in domains:
        if domain not in all_data:
            print(f"  WARNING: domain '{domain}' not in SFT data, skip")
            continue
        d = all_data[domain]
        n = d["input_ids"].shape[0]
        if n == 0:
            print(f"  WARNING: domain '{domain}' has 0 samples, skip")
            continue
        print(f"  [{domain}] {n} samples, input_ids={d['input_ids'].shape}")
        # v2: 计算域先验分布
        prior = compute_domain_prior(d["input_ids"], vocab_size=256000).to(device)
        d["domain_prior"] = prior
        # 检查 prior 是否有强偏置
        top5 = torch.topk(prior, 5)
        print(f"    prior top5 tokens: ids={top5.indices.tolist()}, "
              f"probs={[f'{p:.4f}' for p in top5.values.tolist()]}")
        out[domain] = d
    return out


def get_trainable_params(neuron) -> list:
    """获取 neuron 的可训练参数（低秩残差，不含 W_base）."""
    params = []
    if hasattr(neuron, "lm_head_delta_u"):
        params.extend(p for p in neuron.lm_head_delta_u.parameters() if p.requires_grad)
    if hasattr(neuron, "lm_head_delta_v"):
        params.extend(p for p in neuron.lm_head_delta_v.parameters() if p.requires_grad)
    if not params and hasattr(neuron, "lm_head"):
        # 传统模式（fallback）
        params = [p for p in neuron.lm_head.parameters() if p.requires_grad]
    return params


def train_neuron_sft(
    neuron,
    domain: str,
    sft_data: dict,
    encoder: SharedContextEncoder,
    w_base: nn.Linear,
    device: str,
    max_steps: int,
) -> dict:
    """对一个 neuron 做 SFT 训练 v2（带 domain regularizer）.

    训练流程：
      1. encoder.encode(input_ids) → shared_emb（no_grad）
      2. neuron.forward(shared_emb, return_logits=True) → logits
      3. LM loss: next-token cross_entropy，只对 response 部分
      4. KL regularizer: KL(mean_softmax(logits) || domain_prior)
      5. total = LM + λ * KL → backward + clip + step
    """
    input_ids_all = sft_data["input_ids"].to(device)
    labels_all = sft_data["labels"].to(device)
    response_mask_all = sft_data["response_mask"].to(device)
    domain_prior = sft_data["domain_prior"].to(device)  # [vocab_size]

    N = input_ids_all.shape[0]
    if N == 0:
        return {"avg_loss": None, "n_steps": 0}

    params = get_trainable_params(neuron)
    if not params:
        print(f"  [{domain}] no trainable params, skip")
        return {"avg_loss": None, "n_steps": 0}

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    neuron.train()
    encoder.eval()
    w_base.eval()  # W_base 冻结

    # 初始 loss
    with torch.no_grad():
        idx = 0
        sample_ids = input_ids_all[idx:idx+1]
        sample_labels = labels_all[idx:idx+1]
        shared_emb = encoder.encode(sample_ids)
        out = neuron.forward(shared_emb, return_logits=True)
        logits = out.get("logits") if isinstance(out, dict) else out
        if logits is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = sample_labels[:, 1:].contiguous()
            init_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            ).item()
            # 初始 KL（输出分布 vs 域先验）
            mean_probs = F.softmax(logits.mean(dim=(0, 1)), dim=-1)  # [vocab]
            log_mean_probs = torch.log(mean_probs + 1e-10)
            init_kl = F.kl_div(
                log_mean_probs, domain_prior, reduction="sum"
            ).item()
        else:
            init_loss = float("nan")
            init_kl = float("nan")
    print(f"  [{domain}] init: lm_loss={init_loss:.4f}, kl={init_kl:.4f}")

    total_loss = 0.0
    total_lm = 0.0
    total_kl = 0.0
    step_count = 0
    loss_history = []

    t_start = time.time()

    for step in range(max_steps):
        idx = torch.randint(0, N, (BATCH_SIZE,))
        input_ids = input_ids_all[idx]              # [B, L]
        labels = labels_all[idx]                     # [B, L]
        response_mask = response_mask_all[idx]       # [B, L]

        # encoder 提供 shared_emb（无梯度）
        with torch.no_grad():
            shared_emb = encoder.encode(input_ids)

        # neuron forward
        output = neuron.forward(shared_emb, return_logits=True)
        logits = output.get("logits") if isinstance(output, dict) else output
        if logits is None:
            continue

        # LM loss（next-token）
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_resp_mask = response_mask[:, 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # v2: Domain regularizer（KL 散度）
        # 让 neuron 输出 token 分布接近该域训练数据统计
        # 用 response 部分的 logits 均值
        mask_expanded = shift_resp_mask.unsqueeze(-1).float()  # [B, L-1, 1]
        if mask_expanded.sum() > 0:
            masked_logits = (shift_logits * mask_expanded).sum(dim=(0, 1))  # [vocab]
            mask_count = mask_expanded.sum() + 1e-8
            mean_logits = masked_logits / mask_count
            mean_probs = F.softmax(mean_logits, dim=-1)  # [vocab]
            log_mean_probs = torch.log(mean_probs + 1e-10)
            # KL(prior || output) = Σ prior * log(prior / output)
            kl_loss = F.kl_div(
                log_mean_probs, domain_prior, reduction="sum"
            )
        else:
            kl_loss = torch.tensor(0.0, device=device)

        total_step_loss = lm_loss + KL_WEIGHT * kl_loss

        if torch.isnan(total_step_loss) or torch.isinf(total_step_loss):
            continue

        optimizer.zero_grad(set_to_none=True)
        total_step_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
        optimizer.step()

        final_lm = lm_loss.item()
        final_kl = kl_loss.item()
        total_loss += total_step_loss.item()
        total_lm += final_lm
        total_kl += final_kl
        loss_history.append(final_lm)
        step_count += 1

        if step_count % LOG_INTERVAL == 0:
            avg = total_loss / step_count
            avg_lm = total_lm / step_count
            avg_kl = total_kl / step_count
            recent_lm = sum(loss_history[-10:]) / max(len(loss_history[-10:]), 1)
            # response token accuracy
            with torch.no_grad():
                preds = shift_logits.argmax(dim=-1)
                mask_bool = shift_resp_mask.bool()
                if mask_bool.any():
                    correct = (preds == shift_labels).float() * mask_bool.float()
                    acc = correct.sum() / mask_bool.float().sum()
                else:
                    acc = torch.tensor(0.0)
            elapsed = time.time() - t_start
            print(f"  [{domain}] step {step_count}/{max_steps}: "
                  f"lm={final_lm:.4f} (avg {avg_lm:.4f}), "
                  f"kl={final_kl:.4f} (avg {avg_kl:.4f}), "
                  f"total_avg={avg:.4f}, resp_acc={acc.item():.3f}, "
                  f"t={elapsed:.0f}s")

        # 早停：LM loss 连续 15 步上升（v3 放宽，v2 的 6 步太敏感导致 en 只跑 281 步）
        # 并要求至少跑 300 步（min_epochs）才允许早停，避免过早退出
        if len(loss_history) >= 15 and step_count >= 300:
            recent = loss_history[-15:]
            if all(recent[j] > recent[j-1] for j in range(1, len(recent))):
                print(f"  [{domain}] early stop at step {step_count} (LM loss rising 15 steps)")
                break

    neuron.eval()

    if step_count == 0:
        return {"avg_loss": None, "n_steps": 0, "init_loss": init_loss}

    avg_lm_final = total_lm / step_count
    avg_kl_final = total_kl / step_count
    final_ppl = math.exp(avg_lm_final) if avg_lm_final < 20 else 999.0

    return {
        "avg_loss": total_loss / step_count,
        "avg_lm": avg_lm_final,
        "avg_kl": avg_kl_final,
        "init_loss": init_loss,
        "init_kl": init_kl,
        "final_ppl": final_ppl,
        "n_steps": step_count,
    }


def save_neuron(neuron, domain: str, output_dir: str):
    """保存训练后的 neuron（不保存 W_base，因为它是共享的）.

    C6 修复：键名与 distill_neurons.py 对齐，使用 "neuron_config"（而非 "config"），
    否则 cortex._load_neurons 读取时会 KeyError: 'neuron_config'。
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"neuron_{domain}.pt")
    save_dict = {
        "state_dict": neuron.state_dict(),
        "neuron_config": neuron.config,
        "domain": domain,
    }
    torch.save(save_dict, out_path)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default=None,
                        help="只训练指定域（不指定则全部）")
    parser.add_argument("--all", action="store_true",
                        help="训练所有域")
    parser.add_argument("--steps", type=int, default=500,
                        help="每域最大训练步数（默认 500）")
    parser.add_argument("--save", action="store_true",
                        help="训练后保存到 data/neurons/（覆盖原 neuron）")
    parser.add_argument("--backup", action="store_true", default=True,
                        help="保存前先备份原 neuron 到 data/neurons_backup_sft_v2/")
    parser.add_argument("--reset-w-base", action="store_true",
                        help="重建共享 W_base（默认复用）")
    args = parser.parse_args()

    device = "cpu"

    all_domains = ["zh", "en", "code", "math", "general"]
    if args.domain:
        domains = [args.domain]
    elif args.all:
        domains = all_domains
    else:
        print("ERROR: 必须指定 --domain <name> 或 --all")
        sys.exit(1)

    print("=" * 70)
    print(f"SFT Training v2（共享 W_base + Domain Regularizer）")
    print(f"  domains: {domains}")
    print(f"  steps:   {args.steps}")
    print(f"  batch:   {BATCH_SIZE}")
    print(f"  lr:      {LR}")
    print(f"  kl_w:    {KL_WEIGHT}")
    print(f"  save:    {args.save}")
    print("=" * 70)

    # v2: 可选重建 W_base
    if args.reset_w_base and os.path.exists(W_BASE_PATH):
        os.remove(W_BASE_PATH)
        print(f"  [W_base] Reset: removed {W_BASE_PATH}")

    cortex, tokenizer, encoder, w_base = load_cortex_and_encoder(device)
    sft_data_all = load_sft_data(domains, device)

    # 备份
    if args.save and args.backup:
        backup_dir = "data/neurons_backup_sft_v2"
        os.makedirs(backup_dir, exist_ok=True)
        import shutil
        for domain in domains:
            src = os.path.join(NEURONS_DIR, f"neuron_{domain}.pt")
            if os.path.exists(src):
                dst = os.path.join(backup_dir, f"neuron_{domain}.pt.bak")
                shutil.copy2(src, dst)
                print(f"  Backup: {src} -> {dst}")

    # 逐域训练
    results = {}
    for domain in domains:
        if domain not in sft_data_all:
            print(f"\n[Skip] {domain}: no SFT data")
            continue
        if domain not in cortex.neurons:
            print(f"\n[Skip] {domain}: no neuron in cortex")
            continue

        print(f"\n{'=' * 70}")
        print(f"[Train] domain={domain}")
        print(f"{'=' * 70}")

        neuron = cortex.neurons[domain]
        params_count = sum(p.numel() for p in get_trainable_params(neuron))
        print(f"  trainable params (delta only): {params_count:,}")
        print(f"  W_base frozen params: {sum(p.numel() for p in w_base.parameters() if not p.requires_grad):,}")

        stats = train_neuron_sft(
            neuron=neuron,
            domain=domain,
            sft_data=sft_data_all[domain],
            encoder=encoder,
            w_base=w_base,
            device=device,
            max_steps=args.steps,
        )
        results[domain] = stats

        print(f"\n  [{domain}] Result: {stats}")

        if args.save and stats.get("avg_loss") is not None:
            save_neuron(neuron, domain, NEURONS_DIR)

    # 总结
    print(f"\n{'=' * 70}\n[Summary v2]\n{'=' * 70}")
    for domain, s in results.items():
        if s.get("avg_loss") is None:
            print(f"  {domain}: SKIPPED")
            continue
        init_lm = s.get("init_loss", float("nan"))
        avg_lm = s.get("avg_lm", float("nan"))
        init_kl = s.get("init_kl", float("nan"))
        avg_kl = s.get("avg_kl", float("nan"))
        print(f"  {domain}: LM {init_lm:.4f}→{avg_lm:.4f} (Δ={init_lm-avg_lm:+.4f}), "
              f"KL {init_kl:.4f}→{avg_kl:.4f}, "
              f"ppl={s.get('final_ppl', 0):.1f}, steps={s['n_steps']}")


if __name__ == "__main__":
    main()
