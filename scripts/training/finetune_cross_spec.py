"""微调跨规格 side_channels + 投影层：4×compact + 1×standard 协作。

基于 finetune_side_channels.py，增加：
  1. 加载 zh_std0 (standard) + zh_aug0~3 (compact)
  2. ensemble 自动创建跨规格投影层（field_dim -> unified_dim）
  3. 将跨规格投影层加入可训练参数
  4. 保存跨规格投影层权重

冻结：neuron 核心参数 + shared_embedding
可训练：side_channels + scale + 跨规格投影层（正向 + 反向）

Usage:
    python -u scripts/training/finetune_cross_spec.py
    python -u scripts/training/finetune_cross_spec.py --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroplex.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config, NeuronGeometry,
)
from neuroplex.resonance.topology import (
    build_topology, establish_topology_channels, topology_detail,
)
from neuroplex.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, load_simple_zh_texts, create_shared_embedding,
    make_wsd_scheduler, build_muon_adamw_optimizers,
    load_dialogue_texts_multi,
)

from scripts.training.experiment_config import ENSEMBLE_DIALOGUE_IDS as NEURON_IDS, DEFAULT_DOMAIN as DOMAIN, SFT_ANSWER_MARKER
from scripts.training.data_augmentation import DialogueAugmenter

DEVICE = "cpu"

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
# 对话训练产物（独立于 simple_zh 训练产物）
CKPT_PATH = os.path.join(OUTPUT_DIR, "cross_spec_dialogue.ckpt.pt")
FINAL_PATH = os.path.join(OUTPUT_DIR, "cross_spec_dialogue.pt")


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


def save_checkpoint(path, epoch, total_steps, optimizer, neurons, ensemble,
                    loss_history, adamw_optimizer=None, scheduler=None,
                    body_optimizer=None, body_scheduler=None, shared_embeddings=None):
    """保存训练 checkpoint，含 side_channels + 跨规格投影层 + (S8) body + emb。"""
    side_state = {}
    scale_bias_state = {}
    body_state = {}  # S8: unfrozen neuron body params
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
        # S8: 保存 body 参数（非 side_channels，非 scale，requires_grad=True）
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

    # 保存跨规格投影层
    cross_spec_state = {
        "forward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_projectors.items()},
        "backward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_back_projectors.items()},
    }

    ckpt = {
        "epoch": epoch,
        "total_steps": total_steps,
        "optimizer_state": optimizer.state_dict(),
        "side_channels_state": side_state,
        "scale_bias_state": scale_bias_state,
        "cross_spec_state": cross_spec_state,
        "loss_history": loss_history,
        "saved_at": datetime.now().isoformat(),
    }
    # §4.0c: 保存 Sparse Router 状态
    if ensemble.sparse_router is not None:
        ckpt["sparse_router_state"] = ensemble.sparse_router.state_dict()
        # R3（REMEDIATION_PLAN 2026-08-14）：router 拓扑参数随产物保存，
        # 否则生产 loader 无法按训练同款 top_k 重建（此前状态保存但生产不加载）
        ckpt["sparse_router_config"] = {
            "top_k": ensemble.sparse_router.top_k,
            "warmup_steps": ensemble.sparse_router.warmup_steps,
        }
    # R1: 场门控权重随产物保存（W_cond 训练闭环）
    if hasattr(ensemble._field, "W_cond"):
        ckpt["field_w_cond"] = ensemble._field.W_cond.data.clone()
    if body_state:
        ckpt["body_state"] = body_state
    if adamw_optimizer is not None:
        ckpt["adamw_optimizer_state"] = adamw_optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if body_optimizer is not None:
        ckpt["body_optimizer_state"] = body_optimizer.state_dict()
    if body_scheduler is not None:
        ckpt["body_scheduler_state"] = body_scheduler.state_dict()
    # S8: 保存 shared_embedding（如果训练）
    if shared_embeddings is not None:
        emb_state = {}
        for nid, emb in shared_embeddings.items():
            if any(p.requires_grad for p in emb.parameters()):
                emb_state[nid] = {k: v.detach().clone() for k, v in emb.state_dict().items()}
        if emb_state:
            ckpt["shared_embedding_state"] = emb_state
    torch.save(ckpt, path)


def load_checkpoint(path, optimizer, neurons, ensemble, adamw_optimizer=None, scheduler=None,
                    body_optimizer=None, body_scheduler=None, shared_embeddings=None):
    """加载 checkpoint，恢复 side_channels + 跨规格投影层 + (S8) body + emb。"""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    side_state = ckpt["side_channels_state"]
    scale_bias_state = ckpt.get("scale_bias_state", {})
    body_state = ckpt.get("body_state", {})  # S8: 可能不存在（旧 ckpt）
    for nid, neuron in neurons.items():
        if nid not in side_state:
            continue
        for pid, ch_state in side_state[nid].get("excite", {}).items():
            if pid in neuron.excite_channels:
                neuron.excite_channels[pid].load_state_dict(ch_state)
        for pid, ch_state in side_state[nid].get("inhibit", {}).items():
            if pid in neuron.inhibit_channels:
                neuron.inhibit_channels[pid].load_state_dict(ch_state)
        if nid in scale_bias_state:
            sb = scale_bias_state[nid]
            for name, p in neuron.named_parameters():
                if name in sb and "scale_" in name:
                    p.data.copy_(sb[name])
            for name, buf in neuron.named_buffers():
                if name in sb and "bias_" in name:
                    buf.copy_(sb[name])
        # S8: 恢复 body 参数（T12: shape 不匹配时跳过——词表升级后 lm_head 维度变化）
        if nid in body_state:
            n_skip = 0
            for name, p in neuron.named_parameters():
                if name in body_state[nid]:
                    saved = body_state[nid][name]
                    if saved.shape == p.shape:
                        p.data.copy_(saved)
                    else:
                        n_skip += 1
            if n_skip:
                print(f"  [resume-skip] [{nid}] {n_skip} 个 body 参数形状不匹配"
                      f"（词表升级），保留当前值", flush=True)

    # 恢复跨规格投影层（T6: 兼容旧单层 Linear checkpoint）
    cross_spec_state = ckpt.get("cross_spec_state", {})
    for nid, sd in cross_spec_state.get("forward", {}).items():
        if nid in ensemble._cross_spec_projectors:
            proj = ensemble._cross_spec_projectors[nid]
            if "weight" in sd and "linear1.weight" not in sd:
                # T6 兼容: 旧格式 {"weight": tensor} → 加载到 linear1, linear2 保持零初始化
                proj.load_legacy_linear_state(sd["weight"])
            else:
                proj.load_state_dict(sd)
    for nid, sd in cross_spec_state.get("backward", {}).items():
        if nid in ensemble._cross_spec_back_projectors:
            proj = ensemble._cross_spec_back_projectors[nid]
            if "weight" in sd and "linear1.weight" not in sd:
                proj.load_legacy_linear_state(sd["weight"])
            else:
                proj.load_state_dict(sd)

    # §4.0c: 恢复 Sparse Router 状态
    if ensemble.sparse_router is not None and "sparse_router_state" in ckpt:
        try:
            ensemble.sparse_router.load_state_dict(ckpt["sparse_router_state"])
            print("  [resume] Sparse Router 状态已恢复", flush=True)
        except (RuntimeError, ValueError) as e:
            print(f"  [resume-warn] Sparse Router 状态恢复失败，重建: {e}", flush=True)

    # T12: 优化器/scheduler 状态恢复加防御——词表升级导致参数集变化时重建而非崩溃
    for name, opt in [("optimizer", optimizer), ("adamw_optimizer", adamw_optimizer),
                      ("body_optimizer", body_optimizer)]:
        if opt is not None and f"{name}_state" in ckpt:
            try:
                opt.load_state_dict(ckpt[f"{name}_state"])
            except (RuntimeError, ValueError) as e:
                print(f"  [resume-warn] {name} 状态恢复失败（参数集变化），重建: {e}", flush=True)
    for name, sch in [("scheduler", scheduler), ("body_scheduler", body_scheduler)]:
        if sch is not None and f"{name}_state" in ckpt:
            try:
                sch.load_state_dict(ckpt[f"{name}_state"])
            except (RuntimeError, ValueError) as e:
                print(f"  [resume-warn] {name} 状态恢复失败（优化器重建），重建: {e}", flush=True)
    # S8: 恢复 shared_embedding
    if shared_embeddings is not None and "shared_embedding_state" in ckpt:
        emb_state = ckpt["shared_embedding_state"]
        for nid, emb in shared_embeddings.items():
            if nid in emb_state:
                emb.load_state_dict(emb_state[nid])
    return ckpt["epoch"], ckpt["total_steps"], ckpt.get("loss_history", [])


def load_neuron_with_embedding(nid):
    """加载单个神经元及其 shared_embedding（支持混合规格）。"""
    path = os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # 优先使用 checkpoint 中的 neuron_config
    if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
        cfg = ckpt["neuron_config"]
    else:
        cfg = get_domain_neuron_config(DOMAIN, spec="compact")
    cfg.unified_field_dim = None

    neuron = ResonanceNeuron(cfg).to(DEVICE)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)

    shared_emb = create_shared_embedding(DEVICE)
    if "shared_embedding_state" in ckpt and ckpt["shared_embedding_state"] is not None:
        shared_emb.load_state_dict(ckpt["shared_embedding_state"])
    shared_emb.to(DEVICE)

    result = ckpt.get("result", {})
    print(f"  [{nid}] spec={cfg.spec}, best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron, shared_emb


def load_dialogue_texts(jsonl_path: str, max_texts: int = 10000) -> list:
    """加载对话训练数据（alpaca-zh SFT 格式）。

    每条格式: "问：{instruction}\n答：{output}"
    """
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            if len(texts) >= max_texts:
                break
    return texts


def build_final_artifact(neurons, ensemble, shared_embeddings) -> dict:
    """S8: 构建最终交付产物，含 side_channels + cross_spec + body + emb。

    下游 eval 脚本加载此产物后，应用全部微调结果（不再丢失 body/emb）。
    2026-08-06: 补充 scale_bias_state（此前缺失，评估时 scale/bias 回退训练前值）。
    """
    side_state = {}
    body_state = {}
    scale_bias_state = {}
    for nid, neuron in neurons.items():
        side_state[nid] = {
            "excite": {pid: ch.state_dict() for pid, ch in neuron.excite_channels.items()},
            "inhibit": {pid: ch.state_dict() for pid, ch in neuron.inhibit_channels.items()},
        }
        # S8: 保存 body 参数（非 side_channels，非 scale，requires_grad=True）
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
        # scale_bias（可训练 scale/bias 参数）
        sb = {}
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                sb[name] = p.data.clone()
        for name, buf in neuron.named_buffers():
            if "bias_" in name:
                sb[name] = buf.clone()
        if sb:
            scale_bias_state[nid] = sb
    cross_spec_state = {
        "forward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_projectors.items()},
        "backward": {nid: proj.state_dict() for nid, proj in ensemble._cross_spec_back_projectors.items()},
    }
    artifact = {"side_channels": side_state, "cross_spec": cross_spec_state}
    if body_state:
        artifact["body_state"] = body_state
    if scale_bias_state:
        artifact["scale_bias_state"] = scale_bias_state
    # S8: 保存 shared_embedding（如果训练）
    if shared_embeddings is not None:
        emb_state = {}
        for nid, emb in shared_embeddings.items():
            if any(p.requires_grad for p in emb.parameters()):
                emb_state[nid] = {k: v.detach().clone() for k, v in emb.state_dict().items()}
        if emb_state:
            artifact["shared_embedding_state"] = emb_state
    # §4.0c: 保存 Sparse Router 状态
    if ensemble.sparse_router is not None:
        artifact["sparse_router_state"] = ensemble.sparse_router.state_dict()
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_texts", type=int, default=88730,
                        help="最大加载条数（默认 88730=清洗后全量数据）")
    parser.add_argument("--max_answer_chars", type=int, default=150,
                        help="答案字符数上限（0=不筛选，150=只保留短答案，匹配生成 max_tokens=60）")
    parser.add_argument("--data", type=str, default="dialogue",
                        choices=["dialogue", "simple_zh"],
                        help="dialogue=alpaca-zh SFT, simple_zh=作文数据")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    parser.add_argument("--topology", default="hybrid",
                        choices=["full", "knn", "hub_spoke", "hybrid"],
                        help="S7: side_channels 拓扑模式 (default: hybrid)")
    parser.add_argument("--topology_k", type=int, default=3,
                        help="k-NN 拓扑的 k 值 (仅 knn 模式)")
    parser.add_argument("--unfreeze_layers", type=int, default=2,
                        help="S8: 解冻最后 N 层 transformer + norm + lm_head + field_write (0=全冻结)")
    parser.add_argument("--train_embedding", action="store_true",
                        help="S8: 训练 shared_embedding（默认冻结）")
    parser.add_argument("--body_lr_ratio", type=float, default=0.1,
                        help="S8: body 参数学习率比例 (相对 args.lr)")
    parser.add_argument("--field_warmup_ratio", type=float, default=0.1,
                        help="T9: field_conditioning warm-up 比例 (前 N% 步关闭场注入, 0=全程启用)")
    parser.add_argument("--augment", action="store_true",
                        help="T4: 启用数据增强（模板改写 + 多轮拼接）")
    parser.add_argument("--aug_rewrite_prob", type=float, default=0.5,
                        help="T4: 模板改写概率")
    parser.add_argument("--aug_multi_turn_prob", type=float, default=0.4,
                        help="T4: 多轮拼接概率")
    # ── §4.0c: Sparse Router ──
    parser.add_argument("--use_sparse_router", action="store_true",
                        help="§4.0c: 启用 Probe-based Sparse Router（自适应激活）")
    parser.add_argument("--sparse_router_top_k", type=int, default=3,
                        help="§4.0c: Sparse Router top-K 值（round 2+ 激活的神经元数）")
    parser.add_argument("--sparse_router_warmup_steps", type=int, default=2000,
                        help="§4.0c: Sparse Router warm-up 步数（前 N 步 K=全选）")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"finetune_cross_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 60, flush=True)
    print("微调跨规格 side_channels + 投影层（对话训练）", flush=True)
    print(f"神经元: {NEURON_IDS}", flush=True)
    print(f"日志: {log_path}", flush=True)
    print(f"参数: {vars(args)}", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载神经元
    print("\n[1] 加载神经元...", flush=True)
    neurons = {}
    shared_embeddings = {}
    for nid in NEURON_IDS:
        n, emb = load_neuron_with_embedding(nid)
        neurons[nid] = n
        shared_embeddings[nid] = emb

    # 2. 建立 side_channels（S7: 拓扑驱动替代全连接 mesh）
    print(f"\n[2] 建立 side_channels (topology={args.topology})...", flush=True)
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(
        neurons, geometry, mode=args.topology, k=args.topology_k,
    )
    print(f"  {topology_detail(topology, neurons)}", flush=True)
    stats = establish_topology_channels(neurons, topology, geometry)
    for nid, n_ch in stats.items():
        print(f"  [{nid}] {n_ch} excite channels", flush=True)

    # 3. 冻结核心参数，仅 side_channels + scale + (S8: 最后N层) 可训练
    print(f"\n[3] 冻结核心参数 (unfreeze_layers={args.unfreeze_layers})...", flush=True)
    for nid, neuron in neurons.items():
        for p in neuron.parameters():
            p.requires_grad = False
        # side_channels 可训练
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                p.requires_grad = True
        # scale 参数可训练
        for name, p in neuron.named_parameters():
            if "scale_" in name:
                p.requires_grad = True
        # S8: 解冻最后 N 层 transformer + norm + lm_head + field_write
        if args.unfreeze_layers > 0:
            n_layers = len(neuron.layers)
            unfreeze_from = max(0, n_layers - args.unfreeze_layers)
            for i in range(unfreeze_from, n_layers):
                for p in neuron.layers[i].parameters():
                    p.requires_grad = True
            # final norm
            for p in neuron.norm.parameters():
                p.requires_grad = True
            # lm_head（如果存在）
            if hasattr(neuron, 'lm_head') and neuron.lm_head is not None:
                for p in neuron.lm_head.parameters():
                    p.requires_grad = True
            # field_write（让场写入适配协作动态，C6 多头兼容）
            for p in neuron.get_field_write_parameters():
                p.requires_grad = True
            # R2（REMEDIATION_PLAN 2026-08-14）：field_read 解冻训练——
            # round2+ 场条件化读取路径此前恒为随机初始化（审计发现），
            # 解冻使其成为可学习路径（落入 body 低 lr，温柔更新）。
            for p in neuron.get_field_read_parameters():
                p.requires_grad = True
        neuron.train()

    # S8: shared_embedding 可选训练
    for emb in shared_embeddings.values():
        if args.train_embedding:
            for p in emb.parameters():
                p.requires_grad = True
            emb.train()
        else:
            for p in emb.parameters():
                p.requires_grad = False
            emb.eval()

    # 4. 创建 ensemble（用最大 field_dim，自动创建跨规格投影层）
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(
        neurons, field, max_rounds=2, geometry=geometry,
        # §4.0c: Sparse Router
        use_sparse_router=args.use_sparse_router,
        sparse_router_top_k=args.sparse_router_top_k,
        sparse_router_warmup_steps=args.sparse_router_warmup_steps,
    )
    print(f"\n  field.dim={max_field_dim}, 跨规格投影层: "
          f"{len(ensemble._cross_spec_projectors)} 正向 + "
          f"{len(ensemble._cross_spec_back_projectors)} 反向", flush=True)
    # R1（REMEDIATION_PLAN 2026-08-14）：场门控 W_cond 参与训练
    # （训练-推理评分口径统一后，W_cond 需要梯度才能成为可学习门控）
    ensemble._field.W_cond.requires_grad = True

    # 跨规格投影层设为可训练
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            p.requires_grad = True

    # 统计可训练参数
    trainable_side = 0
    trainable_body = 0
    trainable_emb = 0
    for nid, neuron in neurons.items():
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]) or "scale_" in name:
                trainable_side += p.numel()
            else:
                trainable_body += p.numel()
    for emb in shared_embeddings.values():
        trainable_emb += sum(p.numel() for p in emb.parameters() if p.requires_grad)
    trainable_proj = sum(
        sum(p.numel() for p in proj.parameters() if p.requires_grad)
        for proj in ensemble._cross_spec_projectors.values()
    ) + sum(
        sum(p.numel() for p in proj.parameters() if p.requires_grad)
        for proj in ensemble._cross_spec_back_projectors.values()
    )
    print(f"  可训练: side_channels={trainable_side:,}, 跨规格投影={trainable_proj:,}, "
          f"body={trainable_body:,}, emb={trainable_emb:,}", flush=True)

    # 5. 加载训练数据（S5: 多文件合并扩充）
    print("\n[4] 加载训练数据...", flush=True)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()
    if args.data == "dialogue":
        dialogue_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "simple_zh",
        )
        texts = load_dialogue_texts_multi(
            dialogue_dir,
            max_texts=args.max_texts,
            max_answer_chars=args.max_answer_chars,
        )
        print(f"  训练集(多文件合并对话): {len(texts)} 条对话 "
              f"(max_answer_chars={args.max_answer_chars})", flush=True)
    else:
        texts = load_simple_zh_texts(["simple_zh_texts.jsonl"], max_texts=args.max_texts)
        print(f"  训练集(simple_zh): {len(texts)} 条文本", flush=True)

    # T4: 数据增强器（在线模板改写 + 多轮拼接）
    augmenter = None
    if args.augment:
        augmenter = DialogueAugmenter(
            rewrite_prob=args.aug_rewrite_prob,
            multi_turn_prob=args.aug_multi_turn_prob,
        )
        augmenter.set_context_pool(texts)  # 多轮拼接的上下文池
        print(f"  [T4] 数据增强启用: 改写={args.aug_rewrite_prob}, "
              f"多轮拼接={args.aug_multi_turn_prob}", flush=True)

    # 6. 训练循环
    print("\n[5] 开始训练...", flush=True)
    # S8: 参数分离 — side_channels+投影层走 Muon/AdamW，body+emb 走低 lr AdamW
    muon_params = []   # 2D weight (side_channels + 跨规格投影层)
    adamw_params = []  # 1D bias/norm + 0D scale (side_channels only)
    body_params = []   # S8: unfrozen neuron body params
    for nid, neuron in neurons.items():
        for ch in neuron.excite_channels.values():
            for p in ch.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
        for ch in neuron.inhibit_channels.values():
            for p in ch.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
        # scale 参数（0D scalar，可学习）
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if "scale_" in name and p.ndim == 0:
                adamw_params.append(p)
        # S8: body 参数（非 side_channels，非 scale）
        for name, p in neuron.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in ["excite_", "inhibit_"]):
                continue
            if "scale_" in name or "bias_" in name:
                continue
            body_params.append(p)

    # 跨规格投影层（2D weight -> Muon）
    for proj in ensemble._cross_spec_projectors.values():
        for p in proj.parameters():
            if p.requires_grad and p.ndim == 2:
                muon_params.append(p)
    for proj in ensemble._cross_spec_back_projectors.values():
        for p in proj.parameters():
            if p.requires_grad and p.ndim == 2:
                muon_params.append(p)

    # §4.0c: Sparse Router 参数（2D weight -> Muon, 1D bias -> AdamW）
    if ensemble.sparse_router is not None:
        for p in ensemble.sparse_router.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        router_param_count = sum(p.numel() for p in ensemble.sparse_router.parameters())
        print(f"  §4.0c Sparse Router 参数: {router_param_count:,} (top_k={args.sparse_router_top_k})", flush=True)

    # R1: 场门控 W_cond（2D → Muon）
    if hasattr(ensemble._field, "W_cond") and ensemble._field.W_cond.requires_grad:
        muon_params.append(ensemble._field.W_cond)

    # S8: shared_embedding 参数（如果训练）
    emb_params = []
    if args.train_embedding:
        for emb in shared_embeddings.values():
            emb_params.extend([p for p in emb.parameters() if p.requires_grad])

    # Muon + AdamW 混合优化器（配置抽取到 utils.build_muon_adamw_optimizers）
    muon_lr = args.lr
    optimizer, adamw_optimizer = build_muon_adamw_optimizers(
        muon_params, adamw_params, lr=muon_lr,
    )
    print(f"  Muon 参数: {sum(p.numel() for p in muon_params):,} (2D weight, lr={muon_lr})", flush=True)
    if adamw_optimizer is not None:
        print(f"  AdamW 参数: {sum(p.numel() for p in adamw_params):,} (1D bias/scale, lr={muon_lr})", flush=True)
    else:
        print(f"  AdamW 参数: 0 (无 1D 参数)", flush=True)

    # S8: body + emb 优化器（低 lr 温柔微调，避免破坏预训练表示）
    body_optimizer = None
    body_scheduler = None
    all_body_params = body_params + emb_params
    if all_body_params:
        body_lr = args.lr * args.body_lr_ratio
        body_optimizer = torch.optim.AdamW(all_body_params, lr=body_lr, weight_decay=0.1)
        print(f"  Body+Emb 参数: {sum(p.numel() for p in all_body_params):,} (lr={body_lr}, ratio={args.body_lr_ratio})", flush=True)

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    total_est_steps = NUM_EPOCHS * ((len(texts) - BATCH_SIZE) // BATCH_SIZE)
    warmup_steps = 100
    decay_ratio = 0.8
    scheduler = make_wsd_scheduler(
        optimizer, num_steps=total_est_steps,
        warmup_steps=warmup_steps, decay_ratio=decay_ratio,
    )
    # S8: body scheduler（与主调度同步）
    if body_optimizer is not None:
        body_scheduler = make_wsd_scheduler(
            body_optimizer, num_steps=total_est_steps,
            warmup_steps=warmup_steps, decay_ratio=decay_ratio,
        )
    decay_start = max(warmup_steps + 1, int(total_est_steps * decay_ratio))
    print(f"  LR 调度: warmup={warmup_steps}步, decay 从 {decay_start}/{total_est_steps} 步开始", flush=True)

    LOG_EVERY = 50
    BIAS_UPDATE_EVERY = 50
    BIAS_UPDATE_RATE = 0.1

    total_steps = 0
    start_epoch = 0
    loss_history = []

    # T9: field_conditioning warm-up
    # 前 warm_up_steps 步关闭场注入（neuron 独立学习），之后启用（neuron 开始协作）
    # 原理：训练初期 field_state 是随机的，注入会引入噪声；warm-up 后 field_state 有意义再启用
    field_warmup_steps = int(total_est_steps * args.field_warmup_ratio)
    if args.field_warmup_ratio > 0:
        print(f"  T9: field_conditioning warm-up {field_warmup_steps} 步 "
              f"(ratio={args.field_warmup_ratio}, 总预估 {total_est_steps} 步)", flush=True)

    if args.resume and os.path.exists(CKPT_PATH):
        print(f"\n[resume] 加载 checkpoint: {CKPT_PATH}", flush=True)
        start_epoch, total_steps, loss_history = load_checkpoint(
            CKPT_PATH, optimizer, neurons, ensemble, adamw_optimizer, scheduler,
            body_optimizer, body_scheduler, shared_embeddings,
        )
        print(f"  已恢复: epoch={start_epoch}, total_steps={total_steps}, "
              f"loss_history={len(loss_history)} 条", flush=True)
        start_epoch = start_epoch + 1
    elif args.resume:
        print(f"\n[resume] 未找到 checkpoint ({CKPT_PATH})，从头开始", flush=True)

    import random
    random.seed(42)

    for epoch in range(start_epoch, NUM_EPOCHS):
        random.shuffle(texts)
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start_time = time.time()
        # T4: 每 epoch 重置增强种子（epoch 间变体不同，epoch 内可复现）
        if augmenter is not None:
            augmenter.set_epoch(epoch)

        for i in range(0, len(texts) - BATCH_SIZE, BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            # T4: 在线数据增强（模板改写 + 多轮拼接）
            if augmenter is not None:
                batch_texts = [augmenter.augment(t) for t in batch_texts]

            neuron_embeddings = {}
            targets = None
            mask = None
            sft_mask = None
            for nid, shared_emb in shared_embeddings.items():
                # S3: 传入 answer_marker，获取 sft_mask（只对 answer 部分计算 loss）
                emb_out, tgt, msk, sft = batch_align_and_embed(
                    batch_texts, domain_sp, general_sp, shared_emb,
                    answer_marker=SFT_ANSWER_MARKER,
                    answer_marker_mode="last",  # T4: 多轮精确 masking
                )
                neuron_embeddings[nid] = emb_out.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)
                    sft_mask = sft.to(DEVICE)

            optimizer.zero_grad()
            if adamw_optimizer is not None:
                adamw_optimizer.zero_grad()
            # S8: body_optimizer 也要 zero_grad（否则 body 参数梯度会无限累积）
            if body_optimizer is not None:
                body_optimizer.zero_grad()

            # S1: 改用 forward_train（全可微多轮共振路径）
            # 让 side_channels + 跨规格投影层 + field_state + 调质在训练中真正生效
            # C12: 传入 targets 启用 contrastive_loss（共振分与 NLL 排序对齐）
            # T9: warm-up 阶段关闭 field_conditioning（前 field_warmup_steps 步）
            field_cond = total_steps >= field_warmup_steps
            result = ensemble.forward_train(
                neuron_embeddings=neuron_embeddings,
                n_rounds=2,
                fusion_mode="soft",
                return_individual_logits=False,
                targets=targets,
                field_conditioning=field_cond,  # T9: warm-up 控制
                step=total_steps,  # §4.0c: Router warm-up
                target_domain=DOMAIN,  # 缺口 M: batch 目标域（对应 domain_sp）
            )

            fused_logits = result["fused_logits"]
            shift_logits = fused_logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            # S3: SFT answer masking — 只对 answer 部分计算 loss
            shift_sft_mask = sft_mask[:, 1:].contiguous()
            shift_targets = shift_targets.clone()
            shift_targets[~(shift_mask & shift_sft_mask)] = -100
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_tokens = (shift_mask & shift_sft_mask).sum().item()
            ce_loss = ce_loss / max(n_tokens, 1)

            # 多任务 loss：CE + 负载均衡 + 多样性 + 对比约束
            # balance_loss 鼓励神经元均衡贡献（防死通道）
            # diversity_loss 鼓励 field_vector 差异化（防退化相同）
            # C12 contrastive_loss 让共振分与 NLL 排序对齐（小神经元在擅长主题上获高权重）
            balance_loss = result["balance_loss"]
            diversity_loss = result["diversity_loss"]
            contrastive_loss = result.get("contrastive_loss", torch.tensor(0.0))
            # §4.0d: Router 对比约束（让 Router 学"谁擅长当前样本谁上"）
            router_contrastive_loss = result.get("router_contrastive_loss", torch.tensor(0.0))
            balance_weight = 0.01   # 弱约束，避免压制主任务
            diversity_weight = 0.05  # 弱约束，鼓励差异但不强制正交
            contrastive_weight = 0.1  # C12: 弱约束，让共振分学习公平性
            router_contrastive_weight = 0.1  # §4.0d: 同 C12 权重
            loss = (
                ce_loss
                + balance_weight * balance_loss
                + diversity_weight * diversity_loss
                + contrastive_weight * contrastive_loss
                + router_contrastive_weight * router_contrastive_loss
            )

            loss.backward()
            optimizer.step()
            if adamw_optimizer is not None:
                adamw_optimizer.step()
            # S8: body 参数低 lr 微调（之前漏调 step，body 永不更新且梯度无限累积）
            if body_optimizer is not None:
                body_optimizer.step()
            scheduler.step()
            if body_scheduler is not None:
                body_scheduler.step()

            epoch_loss += ce_loss.item() * n_tokens
            epoch_tokens += n_tokens
            total_steps += 1

            if total_steps % BIAS_UPDATE_EVERY == 0:
                for nid, neuron in neurons.items():
                    neuron.update_channel_bias(update_rate=BIAS_UPDATE_RATE)

            if total_steps % LOG_EVERY == 0:
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                ppl = math.exp(min(avg_loss, 20))
                elapsed = time.time() - epoch_start_time
                steps_done = (i + BATCH_SIZE) / BATCH_SIZE
                steps_total = (len(texts) - BATCH_SIZE) / BATCH_SIZE
                eta = elapsed / max(steps_done, 1) * (steps_total - steps_done)
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} step {total_steps}: "
                      f"loss={avg_loss:.4f} PPL={ppl:.1f} "
                      f"[{steps_done:.0f}/{steps_total:.0f} ETA {eta/60:.1f}min]",
                      flush=True)
                loss_history.append({
                    "step": total_steps,
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "ppl": ppl,
                    "tokens": epoch_tokens,
                })

            if total_steps % 500 == 0:
                save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons,
                                ensemble, loss_history, adamw_optimizer, scheduler,
                                body_optimizer, body_scheduler, shared_embeddings)
                print(f"  [中途 checkpoint] step {total_steps} 已保存", flush=True)

        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_epoch_loss, 20))
        epoch_elapsed = time.time() - epoch_start_time
        print(f"  [Epoch {epoch+1} 完成] avg_loss={avg_epoch_loss:.4f} PPL={ppl:.1f} "
              f"耗时 {epoch_elapsed/60:.1f} min", flush=True)

        save_checkpoint(CKPT_PATH, epoch, total_steps, optimizer, neurons,
                        ensemble, loss_history, adamw_optimizer, scheduler,
                        body_optimizer, body_scheduler, shared_embeddings)
        print(f"  [checkpoint 已保存] {CKPT_PATH}", flush=True)

        # 保存最终产物（含 S8 body + emb，下游 eval 直接加载）
        torch.save(build_final_artifact(neurons, ensemble, shared_embeddings), FINAL_PATH)
        print(f"  [final 已保存] {FINAL_PATH}", flush=True)

        recent = loss_history[-5:]
        if len(recent) >= 2:
            first_ppl = recent[0]["ppl"]
            last_ppl = recent[-1]["ppl"]
            delta = last_ppl - first_ppl
            print(f"  [趋势] 最近 5 点 PPL: {first_ppl:.1f} -> {last_ppl:.1f} "
                  f"(Δ={delta:+.1f}, {'下降' if delta < 0 else '上升/停滞'})", flush=True)

    print("\n[6] 训练完成，最终保存...", flush=True)
    torch.save(build_final_artifact(neurons, ensemble, shared_embeddings), FINAL_PATH)
    print(f"  已保存: {FINAL_PATH}", flush=True)

    history_path = os.path.join(LOG_DIR, "finetune_cross_spec_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)
    print(f"  训练历史: {history_path} ({len(loss_history)} 条记录)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("微调完成。运行 eval_dialogue.py 查看交流效果。", flush=True)
    print("=" * 60, flush=True)

    logger.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
