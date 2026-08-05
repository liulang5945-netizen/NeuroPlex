"""跨域协作层联合训练（缺口 M 消费方）。

在 ensemble 中同时加载多个不同 vocab 的域 neuron（code/math/zh），
用各自域的 SFT 数据轮转训练协作层（side_channels + 跨规格投影层 + Sparse Router）。
每个 batch 的目标域 = 该 batch 数据的域，forward_train(target_domain=域) 通过
词库转译矩阵把各 neuron logits 投影到目标域空间再融合。

数据形态：
- 每个域用自己的 SFT 数据（data/sft/{domain}_sft.pt 的 full 文本）
- 输入统一 general 空间编码（batch_align_and_embed），目标用域 tokenizer 编码
- 域轮转：每 epoch 依次遍历各域数据（batch 级单目标域，与 forward_train 语义一致）

词库可编辑层（AlignmentRules）：
- --rules-path 加载人工规则 JSON，新增特殊神经元时补充专业术语映射
- 规则增删（version 变化）→ 词库转译矩阵缓存自动失效重建

Usage:
    python -u scripts/training/train_cross_domain_collab.py \
        --neuron-dir data/verify_v3 --domains code,math,zh \
        --epochs 2 --lr 1e-3 --batch-size 4 \
        --rules-path taiji/domains/alignment_rules.json
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble, get_domain_neuron_config,
)
from taiji.resonance.geometry import NeuronGeometry
from taiji.resonance.topology import (
    build_topology, establish_topology_channels, topology_detail,
)
from taiji.resonance.translator import (
    TokenizerHub, AlignmentRules, batch_align_and_embed,
)
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    create_shared_embedding, build_muon_adamw_optimizers, make_wsd_scheduler,
    OUTPUT_DIR, DOMAIN_TOKENIZER_DIR,
)
from scripts.training.experiment_config import SFT_ANSWER_MARKER

DEVICE = "cpu"

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)


class TeeLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.fp = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, msg: str):
        sys.__stdout__.write(msg)
        self.fp.write(msg)

    def flush(self):
        sys.__stdout__.flush()
        self.fp.flush()

    def close(self):
        self.fp.close()


def load_neuron(nid: str, neuron_dir: str, device: str) -> ResonanceNeuron:
    """加载单个域 neuron（兼容 verify_v3 与训练产物格式）。"""
    path = os.path.join(neuron_dir, f"neuron_{nid}.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
        cfg = ckpt["neuron_config"]
    else:
        cfg = get_domain_neuron_config(nid, spec="compact")
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    result = ckpt.get("result", {})
    print(f"  [{nid}] vocab={cfg.vocab_size}, spec={cfg.spec}, "
          f"best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron


def load_shared_embedding(neuron_dir: str, device: str) -> nn.Embedding:
    """加载共享 embedding（支持 Tensor 权重或 state_dict，兼容 verify_v3）。"""
    emb = create_shared_embedding(device)
    emb_path = os.path.join(neuron_dir, "shared_embedding.pt")
    if os.path.exists(emb_path):
        w = torch.load(emb_path, map_location=device, weights_only=False)
        if isinstance(w, torch.Tensor):
            assert w.shape == emb.weight.shape, f"shared_embedding 形状不匹配: {w.shape}"
            emb.weight.data.copy_(w)
            print(f"  [shared_embedding] 从 {emb_path} 加载 Tensor 权重", flush=True)
        elif isinstance(w, dict) and "weight" in w:
            emb.load_state_dict(w)
            print(f"  [shared_embedding] 从 {emb_path} 加载 state_dict", flush=True)
    return emb


def load_tokenizer_for_vocab(domain: str, vocab_size: int):
    """加载与 neuron vocab 匹配的域 tokenizer（防御 vocab 不匹配）。

    neuron lm_head vocab 可能小于标准域 tokenizer（如 zh neuron 20K vs 标准
    sp_zh.model 50K），此时尝试 {domain}_v{k}k.model 变体，保证 token id 空间
    与 lm_head 一致（否则词库转译矩阵与 logits 形状错位）。
    """
    sp = load_domain_tokenizer(domain)
    if sp.GetPieceSize() == vocab_size:
        return sp
    k = vocab_size // 1000
    variant = os.path.join(DOMAIN_TOKENIZER_DIR, domain, f"sp_{domain}_v{k}k.model")
    if os.path.exists(variant):
        import sentencepiece as spm
        sp2 = spm.SentencePieceProcessor(model_file=variant)
        print(f"  [tokenizer] {domain}: 标准 vocab={sp.GetPieceSize()} ≠ neuron "
              f"vocab={vocab_size}，使用 {variant} (vocab={sp2.GetPieceSize()})", flush=True)
        return sp2
    print(f"  [tokenizer] ⚠️ {domain}: 标准 vocab={sp.GetPieceSize()} ≠ neuron "
          f"vocab={vocab_size}，未找到变体 {os.path.basename(variant)}，"
          f"继续用标准 tokenizer（logits 尾部可能无映射）", flush=True)
    return sp


def load_sft_texts(data_dir: str, domain: str, max_texts: int) -> List[str]:
    """加载域 SFT 数据（取 full 字段完整文本）。"""
    path = os.path.join(data_dir, f"{domain}_sft.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"SFT 数据不存在: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    texts = [d["full"] for d in data]
    if max_texts > 0:
        texts = texts[:max_texts]
    print(f"  [{domain}] SFT 数据 {len(texts)} 条 ({path})", flush=True)
    return texts


def save_checkpoint(path, epoch, total_steps, neurons, ensemble,
                    muon_optimizer, adamw_optimizer, body_optimizer,
                    loss_history):
    """保存协作层 checkpoint（side_channels + scale/bias + body + 投影层 + Router）。"""
    side_state = {}
    scale_bias_state = {}
    body_state = {}
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
        sb = {}
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                sb[name] = p.data.clone()
        for name, buf in neuron.named_buffers():
            if "bias_" in name:
                sb[name] = buf.clone()
        scale_bias_state[nid] = sb
        bp = {}
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]):
                continue
            if "scale_" in name or "bias_" in name:
                continue
            bp[name] = p.data.clone()
        if bp:
            body_state[nid] = bp

    ckpt = {
        "epoch": epoch,
        "total_steps": total_steps,
        "side_channels_state": side_state,
        "scale_bias_state": scale_bias_state,
        "body_state": body_state,
        "cross_spec_state": {
            "forward": {nid: p.state_dict() for nid, p in ensemble._cross_spec_projectors.items()},
            "backward": {nid: p.state_dict() for nid, p in ensemble._cross_spec_back_projectors.items()},
        },
        "loss_history": loss_history,
        "saved_at": datetime.now().isoformat(),
    }
    if ensemble.sparse_router is not None:
        ckpt["sparse_router_state"] = ensemble.sparse_router.state_dict()
    if muon_optimizer is not None:
        ckpt["muon_optimizer_state"] = muon_optimizer.state_dict()
    if adamw_optimizer is not None:
        ckpt["adamw_optimizer_state"] = adamw_optimizer.state_dict()
    if body_optimizer is not None:
        ckpt["body_optimizer_state"] = body_optimizer.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def main():
    parser = __import__("argparse").ArgumentParser(description="跨域协作层联合训练")
    parser.add_argument("--neuron-dir", default=os.path.join(OUTPUT_DIR),
                        help="neuron 目录（含 neuron_{domain}.pt + shared_embedding.pt）")
    parser.add_argument("--domains", default="code,math,zh",
                        help="参与协作的域（逗号分隔）")
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sft"), help="域 SFT 数据目录")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-texts-per-domain", type=int, default=0,
                        help="每域最大样本数（0=全部）")
    parser.add_argument("--unfreeze_layers", type=int, default=2,
                        help="S8: 解冻最后 N 层 transformer + norm + lm_head + field_write")
    parser.add_argument("--body_lr_ratio", type=float, default=0.1,
                        help="S8: body 参数学习率比例（相对 args.lr）")
    parser.add_argument("--topology", default="hybrid",
                        choices=["full", "knn", "hub_spoke", "hybrid"])
    parser.add_argument("--topology_k", type=int, default=3)
    parser.add_argument("--use_sparse_router", action="store_true",
                        help="§4.0c: 启用 Probe-based Sparse Router（自适应激活）")
    parser.add_argument("--sparse_router_top_k", type=int, default=3)
    parser.add_argument("--sparse_router_warmup_steps", type=int, default=2000)
    parser.add_argument("--rules-path", default=None,
                        help="AlignmentRules 词库规则 JSON（可编辑层，可选）")
    parser.add_argument("--save-name", default="cross_domain_collab",
                        help="checkpoint 文件名前缀")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    assert len(domains) >= 2, "跨域协作至少需要 2 个域"

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR, f"train_cross_domain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 60, flush=True)
    print("跨域协作层联合训练（缺口 M 消费方）", flush=True)
    print(f"域: {domains}", flush=True)
    print(f"neuron 目录: {args.neuron_dir}", flush=True)
    print(f"日志: {log_path}", flush=True)
    print(f"参数: {vars(args)}", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载多域 neuron + shared_embedding
    print("\n[1] 加载神经元...", flush=True)
    neurons = {}
    shared_embeddings = {}
    for nid in domains:
        n = load_neuron(nid, args.neuron_dir, DEVICE)
        neurons[nid] = n
        shared_embeddings[nid] = load_shared_embedding(args.neuron_dir, DEVICE)

    # 2. TokenizerHub + 词库规则层
    print("\n[2] TokenizerHub + 词库可编辑层...", flush=True)
    hub = TokenizerHub()
    for dom in domains:
        hub.register_domain(dom, load_tokenizer_for_vocab(dom, neurons[dom].config.vocab_size))
    general_sp = load_general_tokenizer()
    hub.register_domain("general", general_sp)

    rules = None
    if args.rules_path:
        rules = AlignmentRules(args.rules_path)
        print(f"  [AlignmentRules] 加载 {args.rules_path} "
              f"({len(rules.overrides)} 域规则, version={rules.version})", flush=True)

    # 3. 建立 side_channels（拓扑）
    print(f"\n[3] 建立 side_channels (topology={args.topology})...", flush=True)
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode=args.topology, k=args.topology_k)
    print(f"  {topology_detail(topology, neurons)}", flush=True)
    establish_topology_channels(neurons, topology, geometry)

    # 4. 冻结核心参数，仅协作层可训练
    print(f"\n[4] 冻结核心参数 (unfreeze_layers={args.unfreeze_layers})...", flush=True)
    for neuron in neurons.values():
        for p in neuron.parameters():
            p.requires_grad = False
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                p.requires_grad = True
        if args.unfreeze_layers > 0:
            n_layers = len(neuron.layers)
            unfreeze_from = max(0, n_layers - args.unfreeze_layers)
            for i in range(unfreeze_from, n_layers):
                for p in neuron.layers[i].parameters():
                    p.requires_grad = True
            for p in neuron.norm.parameters():
                p.requires_grad = True
            if hasattr(neuron, 'lm_head') and neuron.lm_head is not None:
                for p in neuron.lm_head.parameters():
                    p.requires_grad = True
            for p in neuron.get_field_write_parameters():
                p.requires_grad = True
        neuron.train()

    # 5. 创建 ensemble（跨域 vocab，缺口 M 融合路径）
    print("\n[5] 创建 ensemble...", flush=True)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(
        neurons, field, max_rounds=2, geometry=geometry,
        use_sparse_router=args.use_sparse_router,
        sparse_router_top_k=args.sparse_router_top_k,
        sparse_router_warmup_steps=args.sparse_router_warmup_steps,
    )
    ensemble.set_tokenizer_hub(hub)
    if rules is not None:
        ensemble.set_alignment_rules(rules)
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True

    # 6. 优化器：Muon(2D 协作层) + AdamW(1D) + body 低 lr
    print("\n[6] 构建优化器...", flush=True)
    muon_params, adamw_params, body_params = [], [], []
    for neuron in neurons.values():
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                if p.requires_grad and p.ndim == 2:
                    muon_params.append(p)
                elif p.requires_grad:
                    adamw_params.append(p)
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                if p.requires_grad and p.ndim == 2:
                    muon_params.append(p)
                elif p.requires_grad:
                    adamw_params.append(p)
        for name, p in neuron.named_parameters():
            if p.requires_grad and ("scale_" in name and p.ndim == 0):
                adamw_params.append(p)
            if p.requires_grad and not any(
                name.startswith(prefix) for prefix in ["excite_", "inhibit_"]
            ) and "scale_" not in name and "bias_" not in name:
                body_params.append(p)
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            if p.requires_grad and p.ndim == 2:
                muon_params.append(p)
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            if p.requires_grad and p.ndim == 2:
                muon_params.append(p)
    if ensemble.sparse_router is not None:
        for p in ensemble.sparse_router.parameters():
            if p.requires_grad and p.ndim == 2:
                muon_params.append(p)
            elif p.requires_grad:
                adamw_params.append(p)

    muon_optimizer, adamw_optimizer = build_muon_adamw_optimizers(
        muon_params, adamw_params, args.lr)
    body_optimizer = None
    if body_params:
        body_optimizer = torch.optim.AdamW(body_params, lr=args.lr * args.body_lr_ratio,
                                           weight_decay=0.01)
    print(f"  可训练: muon(2D)={len(muon_params)}, adamw(1D)={len(adamw_params)}, "
          f"body={len(body_params)}", flush=True)

    # 7. 加载各域数据
    print("\n[7] 加载训练数据...", flush=True)
    domain_texts: Dict[str, List[str]] = {}
    for dom in domains:
        domain_texts[dom] = load_sft_texts(args.data_dir, dom, args.max_texts_per_domain)
    total_steps_per_epoch = sum(
        max(1, (len(t) - args.batch_size) // args.batch_size) for t in domain_texts.values()
    )

    # 8. 训练循环（域轮转，batch 级 target_domain）
    print("\n[8] 开始训练...", flush=True)
    random.seed(42)
    total_steps = 0
    loss_history: List[dict] = []
    ckpt_path = os.path.join(OUTPUT_DIR, f"{args.save_name}.ckpt.pt")
    field_warmup_steps = max(1, int(total_steps_per_epoch * args.epochs * 0.1))

    for epoch in range(args.epochs):
        epoch_start = time.time()
        for domain in domains:
            texts = domain_texts[domain]
            random.shuffle(texts)
            domain_sp = hub.get_tokenizer(domain)
            # zh 用中文 answer marker；其他域全文本 loss（数据无中文 marker）
            answer_marker = SFT_ANSWER_MARKER if domain == "zh" else None

            for i in range(0, len(texts) - args.batch_size, args.batch_size):
                batch_texts = texts[i:i + args.batch_size]
                neuron_embeddings = {}
                targets = None
                mask = None
                sft_mask = None
                for nid, emb in shared_embeddings.items():
                    out = batch_align_and_embed(
                        batch_texts, domain_sp, general_sp, emb,
                        answer_marker=answer_marker,
                        answer_marker_mode="last" if answer_marker else "first",
                    )
                    neuron_embeddings[nid] = out[0].to(DEVICE)
                    if targets is None:
                        targets = out[1].to(DEVICE)
                        mask = out[2].to(DEVICE)
                        if len(out) > 3:
                            sft_mask = out[3].to(DEVICE)

                if muon_optimizer is not None:
                    muon_optimizer.zero_grad()
                if adamw_optimizer is not None:
                    adamw_optimizer.zero_grad()
                if body_optimizer is not None:
                    body_optimizer.zero_grad()

                field_cond = total_steps >= field_warmup_steps
                result = ensemble.forward_train(
                    neuron_embeddings=neuron_embeddings,
                    n_rounds=2,
                    fusion_mode="soft",
                    targets=targets,
                    field_conditioning=field_cond,
                    step=total_steps,
                    target_domain=domain,  # 缺口 M: batch 目标域
                )

                fused_logits = result["fused_logits"]
                shift_logits = fused_logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                if sft_mask is not None:
                    shift_sft = sft_mask[:, 1:].contiguous()
                    shift_targets = shift_targets.clone()
                    shift_targets[~(shift_mask & shift_sft)] = -100
                else:
                    shift_targets = shift_targets.clone()
                    shift_targets[~shift_mask] = -100
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100, reduction="sum",
                )
                n_tokens = max((shift_mask & (shift_sft if sft_mask is not None else shift_mask)).sum().item(), 1)
                ce_loss = ce_loss / n_tokens

                total_loss = ce_loss + 0.01 * result["balance_loss"] + 0.05 * result["diversity_loss"]
                total_loss.backward()
                if muon_optimizer is not None:
                    muon_optimizer.step()
                if adamw_optimizer is not None:
                    adamw_optimizer.step()
                if body_optimizer is not None:
                    body_optimizer.step()

                total_steps += 1
                if total_steps % 50 == 0:
                    ppl = math.exp(min(ce_loss.item(), 20))
                    elapsed = time.time() - epoch_start
                    print(f"  E{epoch+1}/{args.epochs} [{domain}] step {total_steps}: "
                          f"loss={ce_loss.item():.4f} PPL={ppl:.1f} "
                          f"({i//args.batch_size}/{max(1,(len(texts)-args.batch_size)//args.batch_size)} "
                          f"ETA {(elapsed/max(i//args.batch_size,1))*( (len(texts)-args.batch_size)//args.batch_size - i//args.batch_size)/60:.1f}min)",
                          flush=True)
                    loss_history.append({
                        "step": total_steps, "epoch": epoch + 1, "domain": domain,
                        "loss": ce_loss.item(), "ppl": ppl,
                    })

                if total_steps % 500 == 0:
                    save_checkpoint(ckpt_path, epoch + 1, total_steps, neurons,
                                    ensemble, muon_optimizer, adamw_optimizer,
                                    body_optimizer, loss_history)
                    print(f"  [checkpoint] step {total_steps} 已保存", flush=True)

        # epoch 结束保存
        save_checkpoint(ckpt_path, epoch + 1, total_steps, neurons, ensemble,
                        muon_optimizer, adamw_optimizer, body_optimizer, loss_history)
        epoch_ppl = math.exp(min(loss_history[-1]["loss"] if loss_history else 0, 20))
        print(f"  [Epoch {epoch+1} 完成] PPL≈{epoch_ppl:.1f}, "
              f"耗时 {(time.time()-epoch_start)/60:.1f} min", flush=True)

    print("\n[9] 训练完成。", flush=True)
    print(f"  checkpoint: {ckpt_path}", flush=True)
    history_path = os.path.join(LOG_DIR, f"{args.save_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)
    print(f"  训练历史: {history_path} ({len(loss_history)} 条)", flush=True)
    print("运行 eval_dialogue.py 或 test_api_dialogue.py 查看跨域协作效果。", flush=True)


if __name__ == "__main__":
    main()
