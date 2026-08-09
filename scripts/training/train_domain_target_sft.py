"""C24: 域目标空间 SFT 微调 4 个 general neuron（词库多词表补全，2026-08-09）。

背景（C21 遗留）：
- 4 个 general neuron（code/math/zh/en）生成能力弱：zh 回显 / math/en 碎片 / code 简短
- 根因 1：foundation_v1_general 在 general 256K 统一空间训练（英文主导，域表达弱）
- 根因 2：foundation 为全文本续写训练，无 SFT QA 能力（不会"回答问题"）
- 修复路径（同 dialogue 修复，C21 已验证）：域目标空间——general 输入 + 域词表目标 +
  answer masking，让 neuron 在**自己的词表空间**表达域内容；生成时 leader 按词表空间
  decode（_generate_p7 多词表回填已支持，本脚本产出域头 ckpt 后 loader 自动走域头路径）

设计：
- 基座 = data/foundation_v1（已是域头版：code 12K/math 10K/zh 50K/en 16K，域空间 PPL 5.5）
- 数据 = data/sft/{domain}_sft.pt（3000 条，结构化 prompt/response 字段 → 精确 answer masking）
- 训练 = general 输入编码 + 域 tokenizer 目标（build_position_alignment 对齐）+ answer mask
- 配方 = lr 5e-4 + 冻结 shared_embedding（dialogue v3 成功配置）+ EOS 注入 + 周期保存回读验证
- 输出 = data/foundation_v1_sft/（neuron_{domain}.pt 含域头 + shared_embedding.pt，
  无 shared_lm_head.pt → loader 走 per-neuron 域头路径）

工程保障（用户规则：训练前检查 checkpoint 能正确保存）：
- 每个 checkpoint 保存后立即回读重算 val PPL（verify_checkpoint），防坏产物
- --smoke 模式：极小步数快速验证链路（保存/回读/收敛）

Usage:
    python -u scripts/training/train_domain_target_sft.py --smoke
    python -u scripts/training/train_domain_target_sft.py --domains code,math,zh,en --epochs 2
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from taiji.resonance import ResonanceNeuron
from taiji.resonance.translator import build_position_alignment
from scripts.training.utils import load_general_tokenizer
from scripts.training.train_cross_domain_collab import load_tokenizer_for_vocab

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
# C24 双头架构（2026-08-09）：基座 = foundation_v1_general（body 保留 general 256K 判定
# 能力——C20 判定 5/5 依赖的空间可比性），训练时同时优化域头（生成）+ judge_lm_head
# （general 空间保留，判定信号可比）。之前单域头版（foundation_v1 基座）判定退化根因：
# foundation_v1 body 在 general 256K 空间无 NLL 对角 → C20 head 失配 → native NLL 不可比。
BASE_DIR = os.path.join(PROJECT_ROOT, "data", "foundation_v1_general")  # 双头基座（general 判定能力保留）
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "foundation_v1_dual")      # C24v2 产物（双头 neuron）
SFT_DIR = os.path.join(PROJECT_ROOT, "data", "sft")
SEQ_LEN = 192
BATCH_SIZE = 8
LR = 5e-4
WEIGHT_DECAY = 0.1
EMBED_DIM = 512
DOMAINS = ["code", "math", "zh", "en"]
# 域词表尺寸（build_domain_tokenizers.py 配置；zh 为对话时代升级后的 50K）
DOMAIN_VOCAB = {"code": 12000, "math": 10000, "zh": 50000, "en": 16000}
# general 空间保留 loss 权重（防 body 漂移破坏判定空间；1.0 = 与域 SFT 同等）
GENERAL_LOSS_WEIGHT = 0.5


# ======================== 数据 ========================


def load_sft(domain: str) -> List[dict]:
    path = os.path.join(SFT_DIR, f"{domain}_sft.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  [{domain}] {len(data)} 条 SFT", flush=True)
    return data


def build_sample(text: str, prompt: str, domain_sp, general_sp) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """构造单条训练样本：general 输入 ids + 域目标 ids + answer 起始位置。

    answer 起点 = prompt+"\n" 在 general token 序列中的边界（结构化定位，
    不依赖文本 marker——code/en prompt 常含换行，marker 定位不可靠）。
    """
    g_ids, d_targets = build_position_alignment(text, domain_sp, general_sp)
    p_ids = general_sp.encode(prompt + "\n")
    # 边界定位：g_ids 与 prompt 编码的前缀匹配长度（sentencepiece BPE 确定性，无上下文依赖）
    k = 0
    n = min(len(p_ids), len(g_ids))
    while k < n and int(g_ids[k]) == int(p_ids[k]):
        k += 1
    return g_ids, d_targets, k


def build_batch(samples, domain_sp, general_sp, shared_emb) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """批量对齐：general 输入 embedding + 域目标 + 注意力 mask + answer(SFT) mask + general ids。

    返回 (emb, targets, attn_mask, sft_mask, general_ids)；targets 非 answer 位置为 -100。
    general_ids 供 C24 双头 general 空间保留 loss（next-token 预测）使用。
    """
    try:
        general_eos = general_sp.eos_id()
    except Exception:
        general_eos = 1
    try:
        domain_eos = domain_sp.eos_id()
    except Exception:
        domain_eos = 1

    rows = []
    for text, prompt in samples:
        g_ids, d_targets, ans_start = build_sample(text, prompt, domain_sp, general_sp)
        # 追加 EOS（answer 末尾停止信号）
        g_ids = torch.cat([g_ids, torch.tensor([general_eos], dtype=torch.long)])
        d_targets = torch.cat([d_targets, torch.tensor([domain_eos], dtype=torch.long)])
        # 截断（保留末尾 EOS）
        if len(g_ids) > SEQ_LEN:
            g_ids = torch.cat([g_ids[: SEQ_LEN - 1], torch.tensor([general_eos], dtype=torch.long)])
            d_targets = torch.cat([d_targets[: SEQ_LEN - 1], torch.tensor([domain_eos], dtype=torch.long)])
        rows.append((g_ids, d_targets, ans_start))

    max_len = max(len(r[0]) for r in rows)
    B = len(rows)
    padded_ids = torch.zeros(B, max_len, dtype=torch.long)
    padded_targets = torch.full((B, max_len), -100, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.bool)
    sft = torch.zeros(B, max_len, dtype=torch.bool)
    for b, (g_ids, d_targets, ans_start) in enumerate(rows):
        L = len(g_ids)
        padded_ids[b, :L] = g_ids
        padded_targets[b, :L] = d_targets
        attn[b, :L] = True
        sft[b, ans_start:L] = True  # answer + EOS 计入 loss
    emb = shared_emb(padded_ids)  # [B, L, 512]
    return emb, padded_targets, attn, sft, padded_ids


# ======================== 模型 ========================


def load_base_neuron(domain: str, device: str):
    """加载双头基座：foundation_v1_general body + general 256K 共享头。

    C24 双头架构（2026-08-09）：
    - lm_head = general 256K 共享头 → 作为 judge_lm_head（判定空间，可比）
    - 新建域头 lm_head（生成空间，训练目标）
    """
    path = os.path.join(BASE_DIR, f"neuron_{domain}.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["neuron_config"]
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg).to(device)
    # 跳过 ckpt 中的 lm_head（general 256K 共享头单独从 shared_lm_head.pt 加载）
    sd = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("lm_head")}
    neuron.load_state_dict(sd, strict=False)
    # general 256K 共享头 → judge_lm_head（判定空间，C20 信号链）
    shared_head_path = os.path.join(BASE_DIR, "shared_lm_head.pt")
    if os.path.exists(shared_head_path):
        sh = torch.load(shared_head_path, map_location=device, weights_only=False)
        judge_head = torch.nn.Linear(cfg.hidden_size, 256000, bias=False).to(device)
        judge_head.weight.data.copy_(sh["weight"])
        # C24 双头：judge_lm_head 冻结——它是共享 general 判定头（C20 信号链），
        # 不应随单域 SFT 漂移；其梯度仍经 body 传播（gen_loss 约束 body 不破坏
        # general 判定空间，head 本身保持不变 → 判定空间稳定）。
        for p in judge_head.parameters():
            p.requires_grad = False
        neuron.judge_lm_head = judge_head
    # 新建域头（生成空间，训练目标）
    neuron.lm_head = torch.nn.Linear(cfg.hidden_size, DOMAIN_VOCAB[domain], bias=False).to(device)
    torch.nn.init.normal_(neuron.lm_head.weight, std=cfg.hidden_size ** -0.5)
    return neuron, cfg


def load_shared_embedding(device: str) -> torch.nn.Embedding:
    """共享 general 256K embedding（C24 冻结，保留全局 token 语义）。"""
    path = os.path.join(BASE_DIR, "shared_embedding.pt")
    emb = torch.nn.Embedding(256000, EMBED_DIM)
    emb.weight.data.copy_(torch.load(path, map_location=device, weights_only=False))
    emb.to(device)
    return emb

# ======================== 保存 + 回读验证（用户规则） ========================


def verify_checkpoint(domain: str, eval_pairs: List[Tuple[str, str]],
                      domain_sp, general_sp, n_check: int = 8) -> float:
    """保存后立即回读：重算 answer-masked val PPL，防坏 checkpoint。"""
    ckpt = torch.load(os.path.join(OUT_DIR, f"neuron_{domain}.pt"), map_location="cpu", weights_only=False)
    cfg = ckpt["neuron_config"]
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    # 回读 judge_lm_head（双头判定空间）
    jh = ckpt.get("judge_lm_head_state")
    if jh is not None:
        judge_head = torch.nn.Linear(cfg.hidden_size, 256000, bias=False)
        judge_head.weight.data.copy_(jh)
        neuron.judge_lm_head = judge_head
    emb = torch.nn.Embedding(256000, EMBED_DIM)
    emb.weight.data.copy_(torch.load(os.path.join(OUT_DIR, "shared_embedding.pt"), map_location="cpu"))
    neuron.eval()
    total_loss, total_tok = 0.0, 0
    n_s = min(BATCH_SIZE, len(eval_pairs))
    with torch.no_grad():
        for _ in range(n_check):
            samples = random.sample(eval_pairs, n_s)
            e, y, m, sm, g_ids = build_batch(samples, domain_sp, general_sp, emb)
            r = neuron.forward(e, return_logits=True)
            lg = r["logits"][:, :-1, :].contiguous()
            tgt = y[:, 1:].clone().contiguous()
            am = m[:, 1:].contiguous()
            sm_ = sm[:, 1:].contiguous()
            tgt[~(am & sm_)] = -100
            nt = (am & sm_).sum().item()
            l = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.view(-1), ignore_index=-100, reduction="sum")
            total_loss += l.item()
            total_tok += max(nt, 1)
    avg = total_loss / total_tok
    ppl = math.exp(min(avg, 20))
    print(f"  [verify] {domain} 回读 answer PPL={ppl:.1f}", flush=True)
    return avg


def save_checkpoint(domain: str, neuron, shared_emb, step: int, ppl: Optional[float], loss_history: list):
    os.makedirs(OUT_DIR, exist_ok=True)
    judge_head_state = None
    if getattr(neuron, "judge_lm_head", None) is not None:
        judge_head_state = neuron.judge_lm_head.weight.data.clone().cpu()
    ckpt = {
        "neuron_config": neuron.config,
        "state_dict": neuron.state_dict(),
        "shared_embedding_state": {"weight": shared_emb.weight.data.clone()},
        "judge_lm_head_state": judge_head_state,  # C24 双头：general 256K 判定头
        "domain": domain,
        "step": step,
        "result": {"best_ppl": ppl, "best_step": step, "steps": step},
        "loss_history": loss_history,
        "c24_domain_sft": True,  # C24 标记：生成时输入需补 "\n"（训练 answer 起点在 prompt+"\n" 之后）
        "c24_dual_head": True,   # C24v2 标记：双头（域生成 + general 判定）
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(ckpt, os.path.join(OUT_DIR, f"neuron_{domain}.pt"))
    torch.save(shared_emb.weight.data.clone(), os.path.join(OUT_DIR, "shared_embedding.pt"))


# ======================== 训练 ========================


def train_domain(domain: str, samples: List[dict], args, device: str):
    print(f"\n{'=' * 60}\n[域] {domain} 域目标空间 SFT\n{'=' * 60}", flush=True)
    domain_sp = load_tokenizer_for_vocab(domain, DOMAIN_VOCAB[domain])
    general_sp = load_general_tokenizer()

    neuron, cfg = load_base_neuron(domain, device)
    shared_emb = load_shared_embedding(device)
    for p in shared_emb.parameters():
        p.requires_grad = False  # 冻结 embedding（dialogue v3 成功配置）
    n_params = sum(p.numel() for p in neuron.parameters())
    print(f"  参数 {n_params/1e6:.1f}M, vocab={cfg.vocab_size}, lm_head 维度={neuron.lm_head.out_features if neuron.lm_head is not None else '?'}", flush=True)

    # 训练/评估拆分（seed 在 shuffle 之前——对比公平性，C23-C4 教训）
    random.seed(args.seed)
    idx = list(range(len(samples)))
    random.shuffle(idx)
    n_eval = max(2, int(len(samples) * 0.05))
    train_idx, eval_idx = idx[n_eval:], idx[:n_eval]
    train_pairs = [(samples[i]["full"], samples[i]["prompt"]) for i in train_idx]
    eval_pairs = [(samples[i]["full"], samples[i]["prompt"]) for i in eval_idx]
    print(f"  train={len(train_pairs)} eval={len(eval_pairs)}", flush=True)

    optimizer = torch.optim.AdamW(neuron.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    warmup = args.warmup_steps

    steps_per_epoch = max(1, len(train_pairs) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"  steps/epoch={steps_per_epoch} total={total_steps} lr={args.lr}", flush=True)

    step = 0
    loss_history: list = []
    t0 = time.time()

    def lr_at(s: int) -> float:
        if s < warmup:
            return args.lr * (s + 1) / max(warmup, 1)
        return args.lr

    def do_eval(s: int) -> float:
        neuron.eval()
        total_loss, total_tok = 0.0, 0
        n_s = min(args.batch_size, len(eval_pairs))
        with torch.no_grad():
            for _ in range(3):
                es = random.sample(eval_pairs, n_s)
                e, y, m, sm, g_ids = build_batch(es, domain_sp, general_sp, shared_emb)
                r = neuron.forward(e, return_logits=True)
                lg = r["logits"][:, :-1, :].contiguous()
                tgt = y[:, 1:].clone().contiguous()
                am = m[:, 1:].contiguous()
                sm_ = sm[:, 1:].contiguous()
                tgt[~(am & sm_)] = -100
                nt = (am & sm_).sum().item()
                l = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.view(-1), ignore_index=-100, reduction="sum")
                total_loss += l.item()
                total_tok += max(nt, 1)
        neuron.train()
        return total_loss / max(total_tok, 1)

    save_checkpoint(domain, neuron, shared_emb, 0, None, loss_history)
    ppl0 = math.exp(min(do_eval(0), 20))
    print(f"  [step 0] init answer PPL={ppl0:.1f}", flush=True)
    best_ppl, best_step = ppl0, 0  # best 存 PPL（非 loss，勿再 exp）
    # 冒烟：保存即回读验证（用户规则——训练前确认 checkpoint 保存正确）
    verify_checkpoint(domain, eval_pairs, domain_sp, general_sp, n_check=2)

    # C24 双头：general 判定空间对角自检（训练前后对比，确认 body 未漂移出判定空间）
    def judge_diag() -> Dict[str, float]:
        """用 judge_lm_head（general 256K）算 4 类回合 NLL，检查对角保留。"""
        probe = [
            ("code", "Write a Python function to compute the Fibonacci sequence"),
            ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
            ("zh", "写一个 Python 函数计算斐波那契数列"),
            ("en", "What is the capital of France?"),
        ]
        neuron.eval()
        out: Dict[str, float] = {}
        with torch.no_grad():
            for tag, text in probe:
                ids = torch.tensor([general_sp.encode(text)], dtype=torch.long, device=device)
                e = shared_emb(ids)
                r = neuron.forward(e, return_judge_logits=True)
                if "judge_logits" not in r:
                    continue
                jg = r["judge_logits"][:, :-1, :].contiguous()
                tgt = ids[:, 1:].contiguous()
                out[tag] = float(F.cross_entropy(jg.view(-1, jg.size(-1)), tgt.view(-1), reduction="mean"))
        neuron.train()
        return out

    diag0 = judge_diag()
    if diag0:
        s = " ".join(f"{k}={v:.1f}" for k, v in diag0.items())
        print(f"  [judge] 训练前 general 判定 NLL（{domain} neuron）: {s}", flush=True)

    for ep in range(args.epochs):
        random.shuffle(train_pairs)
        for bi in range(steps_per_epoch):
            step += 1
            batch = train_pairs[bi * args.batch_size:(bi + 1) * args.batch_size]
            if len(batch) < 2:
                continue
            e, y, m, sm, g_ids = build_batch(batch, domain_sp, general_sp, shared_emb)
            # C24 双头：一次 forward 同时取域头 logits + judge logits（避免重复 forward）
            r = neuron.forward(e, return_logits=True, return_judge_logits=True)
            lg = r["logits"][:, :-1, :].contiguous()
            tgt = y[:, 1:].clone().contiguous()
            am = m[:, 1:].contiguous()
            sm_ = sm[:, 1:].contiguous()
            tgt[~(am & sm_)] = -100
            nt = (am & sm_).sum().item()
            if nt == 0:
                continue
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.view(-1), ignore_index=-100)
            # C24 双头：general 空间保留 loss（next-token 预测，judge_lm_head 输出）
            # 目标 = 域 SFT 生成能力 + general 判定空间保留（防 body 漂移破坏 C20 判定信号）
            gen_loss = torch.tensor(0.0, device=device)
            if getattr(neuron, "judge_lm_head", None) is not None and "judge_logits" in r:
                jg = r["judge_logits"][:, :-1, :].contiguous()
                gt = g_ids[:, 1:].clone().contiguous()  # general next-token 目标
                gm = m[:, 1:].contiguous()
                gt[~gm] = -100
                gen_loss = F.cross_entropy(
                    jg.view(-1, jg.size(-1)), gt.view(-1), ignore_index=-100,
                )
            loss = loss + GENERAL_LOSS_WEIGHT * gen_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(neuron.parameters(), 1.0)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.step()
            loss_history.append({"step": step, "loss": loss.item()})

            if step % args.log_every == 0 or step == total_steps:
                avg_loss = sum(x["loss"] for x in loss_history[-args.log_every:]) / min(args.log_every, len(loss_history))
                ppl = math.exp(min(avg_loss, 20))
                el = time.time() - t0
                print(f"  [step {step}/{total_steps}] loss={avg_loss:.3f} PPL={ppl:.1f} "
                      f"(gen_loss={gen_loss.item():.3f}) "
                      f"({el:.0f}s, {el/step*1000:.0f}ms/step)", flush=True)

            if step % args.save_every == 0 or step == total_steps:
                vavg = do_eval(step)
                vppl = math.exp(min(vavg, 20))
                print(f"  [eval step {step}] answer PPL={vppl:.1f} (best={best_ppl:.1f})", flush=True)
                save_checkpoint(domain, neuron, shared_emb, step, vppl if vppl < 20 else None, loss_history)
                verify_checkpoint(domain, eval_pairs, domain_sp, general_sp, n_check=2)
                if vppl < best_ppl:
                    best_ppl, best_step = vppl, step

    diagN = judge_diag()
    if diagN:
        s = " ".join(f"{k}={v:.1f}" for k, v in diagN.items())
        print(f"  [judge] 训练后 general 判定 NLL（{domain} neuron）: {s}", flush=True)

    save_checkpoint(domain, neuron, shared_emb, step, best_ppl, loss_history)
    verify_checkpoint(domain, eval_pairs, domain_sp, general_sp, n_check=4)
    print(f"  [done] {domain} best answer PPL={best_ppl:.1f} @ step {best_step}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="冒烟：1 epoch × 20 条数据，验证链路")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"domain_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"log: {log_path}", flush=True)

    torch.set_num_threads(6)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.smoke:
        args.epochs = 1
        args.log_every = 5
        args.save_every = 10

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    print("=" * 60)
    print("C24 域目标空间 SFT（词库多词表补全）")
    print(f"domains={domains} epochs={args.epochs} batch={args.batch_size} lr={args.lr}")
    print(f"base={BASE_DIR} out={OUT_DIR}")
    print("=" * 60, flush=True)

    for d in domains:
        samples = load_sft(d)
        if args.smoke:
            samples = samples[:20]
        train_domain(d, samples, args, args.device)

    print("\n✅ C24 全部域训练完成", flush=True)


if __name__ == "__main__":
    main()
