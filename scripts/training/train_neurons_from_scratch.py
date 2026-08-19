"""P8-1: 从零独立训练神经元（无 W_base、纯 per-neuron 独立训练）。

P7 架构：每 neuron 有独立 embedding + 独立 Transformer body + 独立 lm_head。
域专用 tokenizer 控制 vocab 大小（10k-20k），独立 lm_head 仅 5-10M。

用法:
    # 单域训练
    python scripts/training/train_neurons_from_scratch.py --domain zh --steps 2000

    # 全域训练
    python scripts/training/train_neurons_from_scratch.py --all --steps 2000

    # 自定义 spec
    python scripts/training/train_neurons_from_scratch.py --domain en --spec expert --steps 5000 --lr 3e-5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neuroplex.resonance.config import get_domain_neuron_config, NeuronConfig, DOMAIN_VOCAB_SIZES
from neuroplex.resonance.neuron import ResonanceNeuron
from scripts.training.experiment_config import OUTPUT_DIR_STR as OUTPUT_DIR
from scripts.training.utils import create_shared_embedding

# ── 默认超参数 ──
BATCH_SIZE = 4
SEQ_LEN = 256
LR = 5e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
DEFAULT_STEPS = 2000
LOG_INTERVAL = 50
SAVE_INTERVAL = 500
EARLY_STOP_PATIENCE = 5
MIN_STEPS_BEFORE_STOP = 300

DOMAINS = ["zh", "en", "code", "math", "general"]
# general 复用 en tokenizer
GENERAL_TOKENIZER_DOMAIN = "en"

# OUTPUT_DIR 从 experiment_config 导入（见文件顶部 import）


def load_domain_data(domain: str, data_dir: str = "data/sft",
                     data_suffix: str = "") -> Optional[Dict[str, torch.Tensor]]:
    """加载 P8-2 产出的域 tokenized 数据。

    Args:
        domain: 域名（如 "code"）
        data_dir: 数据目录
        data_suffix: 文件名后缀（如 "mixed" 加载 p7_code_mixed_tokenized.pt，
            用于跨域混合数据训练，工作3）

    Returns:
        {"input_ids": [N, 256], "labels": [N, 256]} or None
    """
    suffix = f"_{data_suffix}" if data_suffix else ""
    path = os.path.join(data_dir, f"p7_{domain}{suffix}_tokenized.pt")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skip {domain}")
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def create_neuron(domain: str, spec: str = "compact", device: str = "cpu") -> ResonanceNeuron:
    """创建域专用 neuron（P7 独立 embedding + 独立 lm_head）。

    Args:
        domain: "zh"/"en"/"code"/"math"/"general"
        spec: "compact"/"standard"/"expert"/"foundation"
        device: 计算设备
    """
    # general 域特殊处理
    domain_key = "en" if domain == "general" else domain
    cfg = get_domain_neuron_config(domain_key, spec=spec)
    cfg.neuron_type = "excitatory"
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.train()
    n_params = sum(p.numel() for p in neuron.parameters())
    print(f"  [{domain}] {spec} neuron created: "
          f"vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, "
          f"layers={cfg.num_hidden_layers}, {n_params/1e6:.1f}M params "
          f"(lm_head={cfg.hidden_size * cfg.vocab_size / 1e6:.1f}M, "
          f"embed={cfg.vocab_size * cfg.base_embed_dim / 1e6:.1f}M)")
    return neuron


def train_one_domain(
    domain: str,
    neuron: ResonanceNeuron,
    data: Dict[str, torch.Tensor],
    steps: int,
    lr: float,
    batch_size: int,
    device: str,
    save_dir: str,
) -> Dict:
    """训练单个域的 neuron。

    Returns:
        {"domain": str, "final_loss": float, "steps_done": int, "saved": str}
    """
    input_ids = data["input_ids"].to(device)    # [N, 256]
    labels = data["labels"].to(device)           # [N, 256]
    n_samples = input_ids.shape[0]

    # 共享嵌入表（256K general vocab → 512-dim）
    shared_embedding = create_shared_embedding(device)

    # Optimizer: 训练全部参数（embedding + body + lm_head）
    optimizer = torch.optim.AdamW(neuron.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_loss = float("inf")
    best_step = 0
    loss_rising_count = 0
    total_loss = 0.0

    print(f"  [{domain}] Training {steps} steps, {n_samples} samples, "
          f"batch={batch_size}, lr={lr}")

    t0 = time.time()
    for step in range(1, steps + 1):
        # 随机采样 batch
        indices = torch.randint(0, n_samples, (batch_size,))
        batch_input_ids = input_ids[indices]   # [B, 256]
        batch_labels = labels[indices]          # [B, 256]

        # P7: 用 neuron 自带 embedding 编码
        shared_emb = shared_embedding(batch_input_ids)  # [B, 256, base_embed_dim]

        # Forward
        result = neuron.forward(shared_emb, return_logits=True)
        logits = result["logits"]  # [B, 256, domain_vocab]

        # CE loss（只在 response 位置计算，prompt=-100）
        shift_logits = logits[:, :-1, :].contiguous()    # [B, 255, vocab]
        shift_labels = batch_labels[:, 1:].contiguous()   # [B, 255]
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(neuron.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()

        # 日志
        if step % LOG_INTERVAL == 0:
            avg_loss = total_loss / LOG_INTERVAL
            elapsed = time.time() - t0
            # 计算 response 准确率（非 prompt 位置）
            with torch.no_grad():
                preds = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100
                acc = (preds[mask] == shift_labels[mask]).float().mean().item() if mask.any() else 0.0

            print(f"    step {step:5d}/{steps} | loss={avg_loss:.4f} | "
                  f"acc={acc:.3f} | elapsed={elapsed:.0f}s")
            total_loss = 0.0

        # 早停检查
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_step = step
            loss_rising_count = 0
        else:
            loss_rising_count += 1

        if step >= MIN_STEPS_BEFORE_STOP and loss_rising_count >= EARLY_STOP_PATIENCE:
            print(f"    Early stop at step {step}: loss rising for {EARLY_STOP_PATIENCE} consecutive steps")
            break

        # 保存 checkpoint
        if step % SAVE_INTERVAL == 0:
            _save_neuron(neuron, domain, step, save_dir)

    # 最终保存
    save_path = _save_neuron(neuron, domain, step, save_dir)
    elapsed = time.time() - t0

    print(f"  [{domain}] Done: {step} steps, best_loss={best_loss:.4f} "
          f"(step {best_step}), {elapsed:.0f}s total")

    return {
        "domain": domain,
        "final_loss": best_loss,
        "steps_done": step,
        "saved": save_path,
        "elapsed_sec": elapsed,
    }


def _save_neuron(neuron: ResonanceNeuron, domain: str, step: int, save_dir: str) -> str:
    """保存 neuron checkpoint。"""
    os.makedirs(save_dir, exist_ok=True)
    # 文件名：neuron_{domain}.pt（覆盖旧 ckpt）
    path = os.path.join(save_dir, f"neuron_{domain}.pt")
    ckpt = {
        "neuron_config": neuron.config,
        "state_dict": neuron.state_dict(),
        "step": step,
        "domain": domain,
    }
    torch.save(ckpt, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="P8-1: train neurons from scratch")
    parser.add_argument("--domain", type=str, default=None,
                        help="single domain to train (zh/en/code/math/general)")
    parser.add_argument("--all", action="store_true", help="train all 5 domains")
    parser.add_argument("--spec", type=str, default="compact",
                        help="neuron spec: compact/standard/expert/foundation")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="training steps per domain")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--device", type=str, default="cpu", help="compute device")
    parser.add_argument("--data-dir", type=str, default="data/sft",
                        help="P8-2 tokenized data directory")
    parser.add_argument("--data-suffix", type=str, default="",
                        help="数据文件后缀（如 mixed → p7_code_mixed_tokenized.pt，工作3 跨域混合数据）")
    parser.add_argument("--save-dir", type=str, default=OUTPUT_DIR,
                        help="output directory for neuron checkpoints")
    args = parser.parse_args()

    # 确定训练域
    if args.all:
        domains = DOMAINS
    elif args.domain:
        if args.domain not in DOMAINS:
            raise ValueError(f"Unknown domain: {args.domain}. Options: {DOMAINS}")
        domains = [args.domain]
    else:
        parser.error("Must specify --domain or --all")

    print(f"=== P8-1: Train neurons from scratch ===")
    print(f"Domains: {domains}")
    print(f"Spec: {args.spec}, Steps: {args.steps}, LR: {args.lr}, "
          f"Batch: {args.batch_size}, Device: {args.device}")
    print(f"Data: {args.data_dir}, Save: {args.save_dir}")
    print()

    results = []
    for domain in domains:
        print(f"--- {domain} ---")

        # 1. 加载域数据
        data = load_domain_data(domain, args.data_dir, args.data_suffix)
        if data is None:
            print(f"  [{domain}] SKIP: no data. Run tokenize_sft_p7.py first.")
            continue

        # 2. 创建 neuron
        neuron = create_neuron(domain, args.spec, args.device)

        # 3. 训练
        result = train_one_domain(
            domain=domain,
            neuron=neuron,
            data=data,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            save_dir=args.save_dir,
        )
        results.append(result)

    # 汇总
    print(f"\n=== Training Summary ===")
    for r in results:
        print(f"  {r['domain']}: loss={r['final_loss']:.4f}, "
              f"steps={r['steps_done']}, saved={r['saved']}")
    print(f"\nNext: python scripts/training/tokenize_sft_p7.py  (if not done)")
    print(f"      Then assemble and test: assemble_cortex() + cortex.generate()")


if __name__ == "__main__":
    main()
