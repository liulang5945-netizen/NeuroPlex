"""SFT 训练脚本：用 instruction-response 数据对每个域神经元的低秩残差做微调.

核心思想（方案 2）：
  当前神经元蒸馏自 1.5B base model，学到的是"续写"行为而非"回答"行为。
  本脚本用 SFT 数据（instruction + response）做 teacher-forcing 训练，
  只对 response 部分计算 LM loss，让神经元学"被问就回答"的对话能力。

数据契约（来自 download_sft_data.py）：
  data/sft/sft_tokenized.pt
    dict[domain] -> {
      "input_ids":      LongTensor[N, 256],   # prompt + response + EOS + pad
      "labels":         LongTensor[N, 256],   # -100(prompt/pad) + response_ids + EOS
      "response_mask":  LongTensor[N, 256],   # 0(prompt/pad) + 1(response+EOS)
    }

训练目标（每域）：
  - 训练 lm_head_delta_u/v（W_base 冻结，符合架构设计）
  - 用 response_mask 只对 response 部分 backward
  - batch_size=2, max_steps=1000（CPU）

使用方式：
  python scripts/training/sft_train_neurons.py --domain zh --steps 500
  python scripts/training/sft_train_neurons.py --all --steps 1000
"""
from __future__ import annotations

import os
import sys
import math
import argparse
import time

# sentencepiece + 项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_libs"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import torch.nn.functional as F
from taiji.loader import create_cortex
from taiji.resonance import SharedContextEncoder


# ── 配置 ──
NEURONS_DIR = "data/neurons"
ENCODER_PATH = "data/distill/shared_context_encoder.pt"
SFT_DATA_PATH = "data/sft/sft_tokenized.pt"

BATCH_SIZE = 2
SEQ_LEN = 256
LR = 5e-5
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 25
SAVE_INTERVAL = 200


def load_cortex_and_encoder(device: str):
    """加载 Cortex + SharedContextEncoder（用于提供 shared_embedding）."""
    print("[1] Loading Cortex + neurons...")
    cortex, tokenizer = create_cortex(
        neurons_dir=NEURONS_DIR,
        device=device,
        max_rounds=2,
        enable_gating=False,
    )
    print(f"  Loaded {len(cortex.neurons)} neurons: {list(cortex.neurons.keys())}")

    print("[2] Loading SharedContextEncoder...")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")
    encoder = SharedContextEncoder.load(ENCODER_PATH, device=device)
    print(f"  Encoder loaded: hidden_dim={encoder.hidden_dim}, n_domains={encoder.n_domains}")
    return cortex, tokenizer, encoder


def load_sft_data(domains: list) -> dict:
    """加载 SFT tokenized 数据."""
    print(f"[3] Loading SFT data from {SFT_DATA_PATH}...")
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
        print(f"  [{domain}] {n} samples, input_ids={d['input_ids'].shape}")
        out[domain] = d
    return out


def get_trainable_params(neuron) -> list:
    """获取 neuron 的可训练参数（低秩残差优先）."""
    params = []
    if hasattr(neuron, "lm_head_delta_u"):
        params.extend(p for p in neuron.lm_head_delta_u.parameters() if p.requires_grad)
    if hasattr(neuron, "lm_head_delta_v"):
        params.extend(p for p in neuron.lm_head_delta_v.parameters() if p.requires_grad)
    if not params and hasattr(neuron, "lm_head"):
        params = [p for p in neuron.lm_head.parameters() if p.requires_grad]
    return params


def train_neuron_sft(
    neuron,
    domain: str,
    sft_data: dict,
    encoder: SharedContextEncoder,
    device: str,
    max_steps: int,
) -> dict:
    """对一个 neuron 做 SFT 训练.

    训练流程：
      1. 用 encoder.encode(input_ids) 得到上下文感知的 shared_emb
      2. neuron.forward(shared_emb, return_logits=True) 得到 logits
      3. 用 SFT labels（只对 response 部分非 -100）算 next-token loss
      4. backward + clip + step
    """
    input_ids_all = sft_data["input_ids"].to(device)        # [N, L]
    labels_all = sft_data["labels"].to(device)              # [N, L] -100/真实
    response_mask_all = sft_data["response_mask"].to(device)  # [N, L] 0/1

    N = input_ids_all.shape[0]
    if N == 0:
        return {"avg_loss": None, "n_steps": 0}

    params = get_trainable_params(neuron)
    if not params:
        print(f"  [{domain}] no trainable params, skip")
        return {"avg_loss": None, "n_steps": 0}

    # 备份原参数（用于早停回滚）
    backup = {id(p): p.data.clone() for p in params}

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    neuron.train()
    encoder.eval()  # encoder 冻结，只训练 neuron

    # 计算初始 loss 作为基线
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
        else:
            init_loss = float("nan")
    print(f"  [{domain}] init loss: {init_loss:.4f}")

    total_loss = 0.0
    step_count = 0
    loss_history = []
    best_avg = float("inf")

    t_start = time.time()

    for step in range(max_steps):
        # 随机抽样 batch
        idx = torch.randint(0, N, (BATCH_SIZE,))
        input_ids = input_ids_all[idx]              # [B, L]
        labels = labels_all[idx]                     # [B, L]
        response_mask = response_mask_all[idx]       # [B, L]

        # 用 encoder 得到 shared_emb（无梯度）
        with torch.no_grad():
            shared_emb = encoder.encode(input_ids)  # [B, L, hidden_dim]

        # neuron forward
        output = neuron.forward(shared_emb, return_logits=True)
        logits = output.get("logits") if isinstance(output, dict) else output
        if logits is None:
            continue

        # SFT loss: next-token prediction，只对 response 部分计算
        shift_logits = logits[:, :-1, :].contiguous()           # [B, L-1, V]
        shift_labels = labels[:, 1:].contiguous()                # [B, L-1] -100/真实
        shift_resp_mask = response_mask[:, 1:].contiguous()      # [B, L-1] 0/1

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if loss.item() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
        optimizer.step()

        final_loss = loss.item()
        total_loss += final_loss
        loss_history.append(final_loss)
        step_count += 1

        if step_count % LOG_INTERVAL == 0:
            avg = total_loss / step_count
            recent = sum(loss_history[-10:]) / max(len(loss_history[-10:]), 1)
            # 计算 response token accuracy（粗略）
            with torch.no_grad():
                preds = shift_logits.argmax(dim=-1)  # [B, L-1]
                mask = shift_resp_mask.bool()
                if mask.any():
                    correct = (preds == shift_labels).float() * mask.float()
                    acc = correct.sum() / mask.float().sum()
                else:
                    acc = torch.tensor(0.0)
            elapsed = time.time() - t_start
            print(f"  [{domain}] step {step_count}/{max_steps}: "
                  f"loss={final_loss:.4f}, avg={avg:.4f}, recent={recent:.4f}, "
                  f"resp_acc={acc.item():.3f}, t={elapsed:.0f}s")

        # 早停：loss 连续 7 步上升
        if len(loss_history) >= 9:
            recent = loss_history[-5:]
            if all(recent[j] > recent[j-1] for j in range(1, len(recent))):
                print(f"  [{domain}] early stop at step {step_count} (loss rising)")
                break

    neuron.eval()

    if step_count == 0:
        return {"avg_loss": None, "n_steps": 0, "init_loss": init_loss}

    avg_loss = total_loss / step_count
    final_ppl = math.exp(avg_loss) if avg_loss < 20 else 999.0

    return {
        "avg_loss": avg_loss,
        "init_loss": init_loss,
        "final_ppl": final_ppl,
        "n_steps": step_count,
        "best_avg": min(best_avg, avg_loss),
    }


def save_neuron(neuron, domain: str, output_dir: str):
    """保存训练后的 neuron."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"neuron_{domain}.pt")
    torch.save({
        "state_dict": neuron.state_dict(),
        "config": neuron.config.__dict__ if hasattr(neuron.config, "__dict__") else dict(neuron.config),
        "neuron_id": domain,
    }, out_path)
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
                        help="保存前先备份原 neuron 到 data/neurons_backup_sft/")
    args = parser.parse_args()

    device = "cpu"

    # 确定要训练的域
    all_domains = ["zh", "en", "code", "math", "general"]
    if args.domain:
        domains = [args.domain]
    elif args.all:
        domains = all_domains
    else:
        print("ERROR: 必须指定 --domain <name> 或 --all")
        sys.exit(1)

    print("=" * 70)
    print(f"SFT Training（方案 2：脱离教师能力上限）")
    print(f"  domains: {domains}")
    print(f"  steps:   {args.steps}")
    print(f"  batch:   {BATCH_SIZE}")
    print(f"  lr:      {LR}")
    print(f"  save:    {args.save}")
    print("=" * 70)

    cortex, tokenizer, encoder = load_cortex_and_encoder(device)
    sft_data_all = load_sft_data(domains)

    # 备份
    if args.save and args.backup:
        backup_dir = "data/neurons_backup_sft"
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
        print(f"  trainable params: {params_count:,}")

        stats = train_neuron_sft(
            neuron=neuron,
            domain=domain,
            sft_data=sft_data_all[domain],
            encoder=encoder,
            device=device,
            max_steps=args.steps,
        )
        results[domain] = stats

        print(f"\n  [{domain}] Result: {stats}")

        if args.save and stats.get("avg_loss") is not None:
            save_neuron(neuron, domain, NEURONS_DIR)

    # 总结
    print(f"\n{'=' * 70}\n[Summary]\n{'=' * 70}")
    for domain, s in results.items():
        if s.get("avg_loss") is None:
            print(f"  {domain}: SKIPPED (no data/params)")
            continue
        init = s.get("init_loss", float("nan"))
        avg = s["avg_loss"]
        delta = init - avg
        print(f"  {domain}: init={init:.4f} → avg={avg:.4f} "
              f"(Δ={delta:+.4f}, ppl={s.get('final_ppl', 0):.1f}, steps={s['n_steps']})")


if __name__ == "__main__":
    main()
