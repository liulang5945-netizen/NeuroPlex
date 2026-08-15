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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
可复现配方（2026-08-07 记录——后续训练以这里为准，勿翻日志）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v1（基线，collab_v1_mixed.ckpt.pt，540 step ≈ 1.7h CPU）：
    python -u scripts/training/train_cross_domain_collab.py \
        --neuron-dir data/foundation_v1_general \
        --domains code,math,zh,en --target-space general \
        --max-texts-per-domain 100 --batch-size 2 --seq-len 64 \
        --dialogue-ids zh_aug0_dialogue,zh_aug1_dialogue,zh_aug2_dialogue,zh_aug3_dialogue,zh_std0_dialogue \
        --dialogue-max-texts 150 --dialogue-data-dir data/simple_zh \
        --save-name collab_v1_mixed --epochs 2

v2（域判别路由 loss，collab_v2_routing，2026-08-07）：
    同 v1 参数 + --routing-loss-weight 0.5 --save-name collab_v2_routing
    routing_loss = -log_softmax(scores/0.15)[domain_idx]（约束本 batch 域 neuron 共振分最高，
    修 scores 无域判别 → trust 校准失效 → 负 EMERGE；仅 4 general 域生效）
    ⚠️ 半成品（270/540 步被杀）：LOO cosine 场耦合梯度泄漏 80% + en 主导场方向 → 评估负 EMERGE 未消除

v3（C13 per-neuron 域判别 head，collab_v3_c13.ckpt.pt，2026-08-07 完整 540 步）：
    同 v1 参数 + --routing-loss-weight 0.5 --save-name collab_v3_c13
    domain_score_head: Linear(hidden→1)，round 1 独立前向（无场注入）→ 梯度只流向自身。
    替代 LOO cosine：泄漏从 125.8→8.37，CE 梯度增强。评估：code -43.1→-17.7%、zh -40.3→-19.0%、
    en 转正 +0.7%，但 math -28.2%/zh -19.0% 仍负 → diag_route_errors 定位：判别器只学到
    "表面语言"（英文 math 题判给 en）+ en 偏置梯度消失（温度 0.15 过尖锐）

v4（C14 MLP 判别器 + 温度校准，collab_v3_c14，2026-08-08）：
    同 v1 参数 + --routing-loss-weight 1.0 --routing-loss-temp 1.0 --save-name collab_v3_c14
    ① domain_score_head 升级为 MLP(hidden*2→128→1) + GELU + mean/max 双 pooling（学到"域语义"
    而非"表面语言"）；② routing_loss 温度 0.15→1.0（非赢家梯度不再消失，压平 en 系统性偏置）
    ⚠️ 结果：zh 修好（-6.9%）、code 略降（-11.6%）、math 恶化（-60.2%）；c14b（判别器 lr 1e-3）
    logit 膨胀到 8+ 全判给 en——域标签判别方案三次迭代（C13/C14/C14b）判定失败：
    判别任务不对称（math/code 是英文→en 覆盖）+ 判别器输入各自 round1 表征不可比 + softmax CE 无尺度约束

v5（C15 预测质量路由，collab_v3_c15，2026-08-08，D 方案）：
    同 v1 参数 + --contrastive-weight 0.5 --save-name collab_v3_c15
    domain_score_head 废弃 → quality_head（MLP 结构不变）：不预测"属于哪个域"，
    预测"我对当前样本的预测质量"。监督 = contrastive_loss（quality softmax 对齐
    softmax(-NLL/0.5)，NLL 是客观预测质量，训练时可得；推理时 quality_logit 直接可用）。
    替代域标签判别（判别任务不对称）+ LOO cosine（梯度泄漏/强 neuron 主导）+ 尺度游戏。
    所有域 batch 生效（含 dialogue）。quality_head 走 adamw 主 lr（快收敛）。

注意：
- --neuron-dir 必须是 general 基座目录（data/foundation_v1_general），默认 data/neurons 是旧的
- max_texts_per_domain/dialogue-max-texts 不设时按全部数据（4×3000+2000 → ~14000 步 ≈ 40h，勿踩）
- 评估：verify_collab_mixed.py（内部 CKPT_PATH 或改路径）
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
    load_dialogue_texts_multi,
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


def load_neuron(nid: str, neuron_dir: str, device: str,
                shared_lm_head: Optional[nn.Linear] = None) -> ResonanceNeuron:
    """加载单个域 neuron（兼容 verify_v3 与训练产物格式）。

    shared_lm_head：统一输出空间（general 基座）时传入共享 general 256K head——
    基座 ckpt 已剥离 lm_head（_strip_shared_head），必须注入共享 head 才能
    在 general 256K 空间输出 logits（否则用域 vocab head 算 256K 目标会崩溃）。
    """
    path = os.path.join(neuron_dir, f"neuron_{nid}.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "neuron_config" in ckpt and ckpt["neuron_config"] is not None:
        cfg = ckpt["neuron_config"]
    else:
        cfg = get_domain_neuron_config(nid, spec="compact")
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg, shared_lm_head=shared_lm_head).to(device)
    neuron.load_state_dict(ckpt["state_dict"], strict=False)
    result = ckpt.get("result", {})
    print(f"  [{nid}] vocab={cfg.vocab_size}, spec={cfg.spec}, "
          f"best_val_ppl={result.get('best_val_ppl', '?')}", flush=True)
    return neuron


def load_shared_lm_head(neuron_dir: str, hidden_size: int, device: str) -> Optional[nn.Linear]:
    """统一输出空间：加载共享 general 256K lm_head（shared_lm_head.pt）。

    general 基座（--target-space general）产物：neuron ckpt 已剥离 131M head，
    head 独立存 shared_lm_head.pt。文件不存在时返回 None（域基座，向后兼容）。
    """
    path = os.path.join(neuron_dir, "shared_lm_head.pt")
    if not os.path.exists(path):
        return None
    head = nn.Linear(hidden_size, 256000, bias=False)
    head.weight.data.copy_(
        torch.load(path, map_location=device, weights_only=False)["weight"])
    print(f"  [shared_lm_head] 从 {path} 加载 general 256K head", flush=True)
    return head


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


def load_hub_neuron(hub_path: str, device: str) -> Tuple[ResonanceNeuron, nn.Embedding]:
    """加载 hub neuron（缺口 L：联合皮层）进协作阵容。

    hub 与 general 基座同 256K 空间协作：保留自带 general lm_head（1024→256000，
    与 domains 路径注入 shared head 不同——注入会 shape 冲突丢 head），因此不走
    词库转译；输入 embedding 用全局 general 表（train_hub_neuron.py 已随 ckpt
    保存同表副本 shared_embedding.pt）。body 由 train_hub_neuron.py 独立训练，
    协作层训练仅训 side_channels 等（LoRA 模式冻结 body）。
    """
    ck = torch.load(hub_path, map_location=device, weights_only=False)
    cfg = ck["neuron_config"]
    cfg.unified_field_dim = None
    neuron = ResonanceNeuron(cfg).to(device)  # 自带 general 256K lm_head
    neuron.load_state_dict(ck["state_dict"], strict=False)
    emb = create_shared_embedding(device)
    hub_emb_path = os.path.join(os.path.dirname(hub_path), "shared_embedding.pt")
    if os.path.exists(hub_emb_path):
        w = torch.load(hub_emb_path, map_location=device, weights_only=False)
        if isinstance(w, torch.Tensor):
            assert w.shape == emb.weight.shape, f"hub embedding 形状不匹配: {w.shape}"
            emb.weight.data.copy_(w)
        elif isinstance(w, dict) and "weight" in w:
            emb.weight.data.copy_(w["weight"])
    print(f"  [hub] vocab={cfg.vocab_size}, spec={cfg.spec}, "
          f"field_dim={cfg.field_dim}, 同 general 空间协作（保留自身 256K head）", flush=True)
    return neuron, emb


def compute_hub_anchor_loss(
    ensemble, neurons: Dict[str, ResonanceNeuron],
    neuron_embeddings: Dict[str, torch.Tensor],
    nid: str,
) -> torch.Tensor:
    """hub 锚定 loss（缺口 L 阶段 3 第三部分，决策 4B→4C 渐进第一步）。

    hub 是跨域语义锚点（field=4096 即统一维度源头），约束各域 neuron 的
    field_vector（经 cross_spec_projector 投影到统一空间）向 hub field_vector
    靠拢（cosine 最大化）——hub 的 field 成为跨域共享语义锚点，而非 CE 副产物。

    域 neuron 与 hub body 均冻结（LoRA 模式，双阶段职责分离），本 loss 梯度
    只流 cross_spec_projectors[nid]（协作层参数）——个体生成能力零破坏。

    注意：输入必须 detach()——neuron_embeddings 来自共享 embedding 表输出
    （带 EmbeddingBackward grad_fn），若与主 loss 的 forward_train 共享同一
    输入张量，主 loss backward 会释放该 grad_fn 的 saved tensors，锚定 loss
    再 backward 将报 "backward a second time"。detach 后锚定 loss 图独立
    （embedding 表冻结，本 loss 只需投影层梯度，detach 语义正确）。
    """
    hub = neurons["hub"]
    v_d = neurons[nid].forward(neuron_embeddings[nid].detach(), round_num=1)["field_vector"]
    v_hub = hub.forward(neuron_embeddings["hub"].detach(), round_num=1)["field_vector"]
    if nid in ensemble._cross_spec_projectors:
        v_d = ensemble._cross_spec_projectors[nid](v_d)
    # 统一空间非 hub 原生维度（如装配口径 3072）时 hub 也经投影参与锚定
    if "hub" in ensemble._cross_spec_projectors:
        v_hub = ensemble._cross_spec_projectors["hub"](v_hub)
    return 1.0 - F.cosine_similarity(v_d, v_hub, dim=-1).mean()


PAIRS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cross_domain_pairs.jsonl",
)


def load_pairs_texts(max_pairs: int = 0) -> List[Tuple[str, str]]:
    """加载跨域平行语料（zh↔code 同义对，阶段 1 产物 1629 对）。"""
    if not os.path.exists(PAIRS_PATH):
        return []
    out = []
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            out.append((p["zh"], p["code"]))
    if max_pairs > 0:
        out = out[:max_pairs]
    print(f"  [pairs] {len(out)} 对跨域平行语料（{PAIRS_PATH}）", flush=True)
    return out


def compute_hub_contrastive_loss(
    ensemble, neurons: Dict[str, ResonanceNeuron],
    shared_embeddings: Dict[str, torch.nn.Embedding],
    general_sp,
    zh_id: str, code_id: str,
    zh_texts: List[str], code_texts: List[str],
    tau: float = 0.1, max_seq_len: int = 64,
) -> torch.Tensor:
    """跨域对比 loss（缺口 L 阶段 3 第三部分·渐进第二步，决策 4C 最终形态）。

    用跨域平行语料（zh 指令 ↔ code 实现）约束**投影到统一空间（hub 空间，
    即 hub field 4096 维度）后的域 field**做双向 InfoNCE：同义跨域对（zh"函数"
    ↔ code"function"）在 hub 空间靠近，不同义对远离——hub 空间成为真正的
    跨域共享语义空间，对比学习显式对齐（SimCLR/CLIP 范式，非 CE 副产物）。

    与锚定 loss 互补：锚定把域 field 拉向 hub 方向（同一文本），对比让同义
    跨域文本的域 field 彼此靠近（跨文本对）。域 neuron body 冻结（LoRA 双阶段
    职责分离），梯度只流两个域的 cross_spec_projectors。输入 detach（同锚定
    loss 的 backward 二次图教训）。
    """
    zh_emb = batch_align_and_embed(
        zh_texts, general_sp, general_sp, shared_embeddings[zh_id],
        max_seq_len=max_seq_len)[0]
    code_emb = batch_align_and_embed(
        code_texts, general_sp, general_sp, shared_embeddings[code_id],
        max_seq_len=max_seq_len)[0]
    v_zh = neurons[zh_id].forward(zh_emb.detach(), round_num=1)["field_vector"]
    v_code = neurons[code_id].forward(code_emb.detach(), round_num=1)["field_vector"]
    if zh_id in ensemble._cross_spec_projectors:
        v_zh = ensemble._cross_spec_projectors[zh_id](v_zh)
    if code_id in ensemble._cross_spec_projectors:
        v_code = ensemble._cross_spec_projectors[code_id](v_code)
    sim = F.cosine_similarity(v_zh.unsqueeze(1), v_code.unsqueeze(0), dim=-1) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    # 双向 InfoNCE（zh→code + code→zh）
    return 0.5 * F.cross_entropy(sim, labels) + 0.5 * F.cross_entropy(sim.t(), labels)


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
    """保存协作层 checkpoint（side_channels + scale/bias + body + 投影层 + Router）。

    C16（2026-08-08）：body_state 不再包含 quality_head（判别器拆为独立 head_state，
    解决分解验证发现的"判别器耦合 body 无法归因"问题）；LoRA 模式下 body_state 为空
    （body 冻结未动），尾层增量单独存 lora_state。
    """
    side_state = {}
    scale_bias_state = {}
    body_state = {}
    head_state = {}
    lora_state = {}
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
            if name.startswith("quality_head") or "lora_adapters" in name:
                continue
            bp[name] = p.data.clone()
        if bp:
            body_state[nid] = bp
        # C16: quality_head（判别器）独立分量
        if getattr(neuron, "quality_head", None) is not None:
            head_state[nid] = neuron.quality_head.state_dict()
        # C16: LoRA 尾层增量（body 冻结时唯一被动的 body 侧参数）
        if getattr(neuron, "lora_enabled", False) and len(neuron.lora_adapters) > 0:
            lora_state[nid] = neuron.lora_adapters.state_dict()

    ckpt = {
        "epoch": epoch,
        "total_steps": total_steps,
        "side_channels_state": side_state,
        "scale_bias_state": scale_bias_state,
        "body_state": body_state,
        "head_state": head_state,
        "lora_state": lora_state,
        "cross_spec_state": {
            "forward": {nid: p.state_dict() for nid, p in ensemble._cross_spec_projectors.items()},
            "backward": {nid: p.state_dict() for nid, p in ensemble._cross_spec_back_projectors.items()},
        },
        "loss_history": loss_history,
        "saved_at": datetime.now().isoformat(),
    }
    if ensemble.sparse_router is not None:
        ckpt["sparse_router_state"] = ensemble.sparse_router.state_dict()
        # R3（REMEDIATION_PLAN 2026-08-14）：router 拓扑参数随产物保存，
        # 否则生产 loader 无法按训练同款 top_k 重建（此前状态保存但生产不加载）
        ckpt["sparse_router_config"] = {
            "top_k": ensemble.sparse_router.top_k,
            "warmup_steps": ensemble.sparse_router.warmup_steps,
        }
    # R1: 场门控权重随协作层产物保存（W_cond 训练闭环）
    if ensemble._field is not None and hasattr(ensemble._field, "W_cond"):
        ckpt["field_w_cond"] = ensemble._field.W_cond.data.clone()
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
    parser.add_argument("--dialogue-ids", default="",
                        help="混合阵容：旧对话 neuron id 列表（逗号分隔，如 "
                             "zh_aug0_dialogue,zh_aug1_dialogue,...）。这些 neuron "
                             "保留各自域输出头（zh 50K），经词库转译参与统一空间协作")
    parser.add_argument("--dialogue-dir", default=OUTPUT_DIR,
                        help="旧对话 neuron 目录（默认 data/neurons）")
    parser.add_argument("--dialogue-max-texts", type=int, default=10000,
                        help="对话数据最大条数（混合阵容时加入训练）")
    parser.add_argument("--dialogue-data-dir",
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "data", "simple_zh"),
                        help="对话数据目录（DIALOGUE_DATA_FILES 所在，默认 data/simple_zh）")
    parser.add_argument("--save-name", default="cross_domain_collab",
                        help="checkpoint 文件名前缀")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-space", default="domain",
                        choices=["domain", "general"],
                        help="训练目标空间：domain=各域 tokenizer（原口径）；"
                             "general=通用 256K 空间（各 neuron 投影到 general 融合，"
                             "支持跨域组合：en/zh 理解指令 + code 生成代码）")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="batch_align_and_embed 最大序列长度")
    parser.add_argument("--contrastive-weight", type=float, default=0.5,
                        help="预测质量对比约束权重（C15/D 方案，2026-08-08：quality_head 输出"
                             "对齐 per-neuron NLL 排序——谁能预测好当前文本谁上。替代 C13/C14 "
                             "域标签判别 routing_loss（判别任务不对称 + 输入不可比 + 尺度游戏，"
                             "三次失败）。所有域 batch 生效）")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="C16（2026-08-08）：尾层 LoRA 秩。>0 时 body 尾层（最后 2 层）"
                             "冻结并改用低秩增量 BA（B 初始 0 → 个体生成能力零破坏起点），"
                             "解决 C14b 分解验证发现的 'collab 训练 body 微调破坏生成' 根因；"
                             "=0 关闭（退回直接微调 body 尾层旧行为）")
    parser.add_argument("--hub-path", default=None,
                        help="缺口 L：hub neuron（联合皮层）ckpt 路径，如 "
                             "data/hub_neuron/neuron_hub.pt。加入协作阵容训练 hub-and-spoke "
                             "side_channels：hybrid 拓扑按容量自动选 hub 为 global-hub "
                             "（1024×14=14336 最大），hub 保留自带 general 256K lm_head "
                             "（同 general 目标空间，无需词库转译）；body 冻结走 LoRA 模式，"
                             "个体能力由 train_hub_neuron.py 独立训练")
    parser.add_argument("--hub-anchor-weight", type=float, default=0.0,
                        help="缺口 L 阶段 3 第三部分：hub 锚定 loss 权重。>0 时每个 batch "
                             "约束当前域 neuron field_vector（经 cross_spec 投影到统一空间）"
                             "对齐 hub field_vector（cosine 最大化，hub 为跨域语义锚点）。"
                             "梯度只流 cross_spec_projectors（域 neuron 与 hub body 冻结，"
                             "零破坏），需 --hub-path 开启。先锚定后叠加对比 loss（渐进）")
    parser.add_argument("--hub-contrastive-weight", type=float, default=0.0,
                        help="缺口 L 阶段 3 第三部分·渐进第二步：跨域对比 loss 权重。>0 时"
                             "每 batch 用跨域平行语料（data/cross_domain_pairs.jsonl，zh↔code "
                             "同义对 1629 对）做双向 InfoNCE——同义对在统一空间（hub 空间）"
                             "靠近、不同义对远离，hub 空间成为跨域共享语义空间。梯度只流"
                             "zh/code 两侧 cross_spec_projectors，需 --hub-path 且阵容含两侧")
    parser.add_argument("--hub-contrastive-zh", default="zh",
                        help="对比 loss 的 zh 侧域 neuron id")
    parser.add_argument("--hub-contrastive-code", default="code",
                        help="对比 loss 的 code 侧域 neuron id")
    parser.add_argument("--hub-contrastive-tau", type=float, default=0.1,
                        help="对比 loss InfoNCE 温度（越小越尖锐，0.1 为 CLIP 常用量级）")
    parser.add_argument("--unified-field-dim", type=int, default=0,
                        help="统一场维度（协作层训练口径）。0=auto 取阵容 max(field_dim)（含 hub "
                             "为 4096）；装配综合体口径为 3072（对话 neuron 主导，hub 经 "
                             "add_neuron 补投影 4096→3072）——传 3072 让训练与装配维度一致，"
                             "hub 的 cross_spec 投影（4096→3072）参与训练，产物可装配复用")
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
    # 统一输出空间（general 基座）：shared_lm_head.pt 存在时注入所有 neuron
    # （基座 ckpt 已剥离 lm_head，不注入会因 vocab 错位崩溃）
    shared_lm_head = load_shared_lm_head(args.neuron_dir, 512, DEVICE)
    neurons = {}
    shared_embeddings = {}
    for nid in domains:
        n = load_neuron(nid, args.neuron_dir, DEVICE, shared_lm_head=shared_lm_head)
        neurons[nid] = n
        shared_embeddings[nid] = load_shared_embedding(args.neuron_dir, DEVICE)

    # 混合阵容：旧对话 neuron（跨 vocab 协作核心能力）
    # 保留各自域输出头（zh 50K），输入用各自 home embedding（ckpt 内 shared_embedding_state），
    # 协作训练时经词库转译矩阵投影到统一目标空间（缺口 M 既有路径）
    dialogue_ids = [d.strip() for d in (args.dialogue_ids or "").split(",") if d.strip()]
    for nid in dialogue_ids:
        path = os.path.join(args.dialogue_dir, f"neuron_{nid}.pt")
        if not os.path.exists(path):
            print(f"  ⚠️ 对话 neuron 缺失，跳过: {path}", flush=True)
            continue
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ck["neuron_config"]
        cfg.unified_field_dim = None
        n = ResonanceNeuron(cfg).to(DEVICE)  # 不注入 shared head：保留域输出头（跨 vocab 协作）
        n.load_state_dict(ck["state_dict"], strict=False)
        neurons[nid] = n
        # home embedding：从 ckpt 的 shared_embedding_state 加载（该 neuron 训练时的表）
        emb = create_shared_embedding(DEVICE)
        ses = ck.get("shared_embedding_state", {})
        if isinstance(ses, dict) and "weight" in ses:
            emb.weight.data.copy_(ses["weight"])
        elif isinstance(ses, torch.Tensor):
            emb.weight.data.copy_(ses)
        shared_embeddings[nid] = emb
        print(f"  [{nid}] vocab={cfg.vocab_size}, spec={cfg.spec}, "
              f"home embedding 从 ckpt 加载（保留域输出头 → 词库转译协作）", flush=True)

    # hub neuron（缺口 L：联合皮层）：同 general 256K 空间协作，自带 lm_head，
    # 不走词库转译（与 domains 同空间）；body 由 train_hub_neuron.py 独立训练。
    if args.hub_path:
        assert os.path.exists(args.hub_path), f"hub ckpt 不存在: {args.hub_path}"
        n, emb = load_hub_neuron(args.hub_path, DEVICE)
        neurons["hub"] = n
        shared_embeddings["hub"] = emb

    print(f"  阵容: {len(neurons)} neuron "
          f"(域 {domains} + 对话 {dialogue_ids} + hub {'✓' if args.hub_path else '—'})", flush=True)

    # 2. TokenizerHub + 词库规则层
    print("\n[2] TokenizerHub + 词库可编辑层...", flush=True)
    hub = TokenizerHub()
    for dom in domains:
        hub.register_domain(dom, load_tokenizer_for_vocab(dom, neurons[dom].config.vocab_size))
    # 旧对话 neuron（zh 域空间）：注册 zh tokenizer 供 _get_neuron_tokenizer 前缀解析
    if dialogue_ids:
        hub.register_domain("zh", load_tokenizer_for_vocab("zh", neurons[dialogue_ids[0]].config.vocab_size))
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
    print(f"\n[4] 冻结核心参数 (unfreeze_layers={args.unfreeze_layers}, "
          f"lora_rank={args.lora_rank})...", flush=True)
    lora_mode = args.lora_rank > 0
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
        if lora_mode:
            # C16：body 全部冻结（含 lm_head/尾层/field_write），尾层改用 LoRA 低秩增量。
            # LoRA B 初始 0 → 个体生成能力零破坏起点（C14b 分解验证结论：body 微调是破坏源）。
            neuron.enable_lora(args.lora_rank, layers=None)  # 最后 2 层
            for p in neuron.lora_adapters.parameters():
                p.requires_grad = True
            # quality_head（判别器）独立解冻走主 lr（C15 + C16：拆为独立分量）
            if hasattr(neuron, 'quality_head') and neuron.quality_head is not None:
                for p in neuron.quality_head.parameters():
                    p.requires_grad = True
            # R2（REMEDIATION_PLAN 2026-08-14）：field_read 解冻训练——
            # round2+ 场条件化读取路径此前恒为随机初始化（审计发现）。
            # 此处解冻落入 body 低 lr（温柔更新，不破坏 round1 独立能力）。
            for p in neuron.get_field_read_parameters():
                p.requires_grad = True
        elif args.unfreeze_layers > 0:
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
            # R2（REMEDIATION_PLAN 2026-08-14）：field_read 解冻训练——
            # round2+ 场条件化读取路径此前恒为随机初始化（审计发现）。
            for p in neuron.get_field_read_parameters():
                p.requires_grad = True
            # C15: 预测质量 head 解冻（contrastive_loss 监督它对齐 per-neuron NLL 排序）
            if hasattr(neuron, 'quality_head') and neuron.quality_head is not None:
                for p in neuron.quality_head.parameters():
                    p.requires_grad = True
        neuron.train()

    # 5. 创建 ensemble（跨域 vocab，缺口 M 融合路径）
    # 统一场维度：默认取阵容 max(field_dim)（含 hub = 4096）；显式传装配口径
    # （--unified-field-dim 3072）时 hub 4096 也参与 cross_spec 投影到统一空间，
    # 训练产物与装配综合体（unified=3072）维度一致可复用（锚定/对比传导）。
    print("\n[5] 创建 ensemble...", flush=True)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=args.unified_field_dim or max_field_dim)
    print(f"  统一场维度: {field.dim} "
          f"({'装配口径' if args.unified_field_dim else '阵容 max'}={max_field_dim})", flush=True)
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
    # R1（REMEDIATION_PLAN 2026-08-14）：场门控 W_cond 参与训练
    # （训练-推理评分口径统一后，W_cond 需要梯度才能成为可学习门控）
    if ensemble._field is not None and hasattr(ensemble._field, "W_cond"):
        ensemble._field.W_cond.requires_grad = True

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
                # C15 fix（2026-08-08）：quality_head（预测质量头）不走 body 低 lr。
                # C14 教训：判别器 lr=1e-4（body_lr_ratio=0.1）下 MLP 欠拟合；lr=1e-3 下
                # softmax CE 无尺度约束 → logit 膨胀作弊（全判给 en）。D 方案用
                # contrastive_loss（KL 对齐 NLL 排序）监督 quality_head——KL 天然防膨胀，
                # quality_head 走主 lr（args.lr，adamw 优化器）快速收敛。
                if name.startswith("quality_head") or "lora_adapters" in name:
                    adamw_params.append(p)
                else:
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
    # R1: 场门控 W_cond（2D → Muon，与 cross_spec 投影同级）
    if (ensemble._field is not None and hasattr(ensemble._field, "W_cond")
            and ensemble._field.W_cond.requires_grad):
        muon_params.append(ensemble._field.W_cond)

    # 2026-08-07 fix：去重——共享 general lm_head 被多个 general neuron 共享，
    # named_parameters 会重复收集（body 学习率 ×N 效果，同参数每步更新 N 次）
    def _dedup(params):
        seen, out = set(), []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    muon_params = _dedup(muon_params)
    adamw_params = _dedup(adamw_params)
    body_params = _dedup(body_params)

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
    data_sources = list(domains)
    # 混合阵容：对话数据桶（旧 5 的对话能力来源，目标 general 编码统一训练）
    # 仅 general 目标空间支持（对话文本统一 general 编码，domain tokenizer 未注册）
    if dialogue_ids and args.target_space == "general":
        domain_texts["dialogue"] = load_dialogue_texts_multi(
            args.dialogue_data_dir, max_texts=args.dialogue_max_texts, max_answer_chars=150)
        data_sources.append("dialogue")
        print(f"  dialogue: {len(domain_texts['dialogue'])} 条对话（短答案 ≤150 字）", flush=True)
    # 阶段 3 第三部分·渐进第二步：跨域平行语料桶（zh↔code 同义对，对比 loss 数据源）
    pairs_texts: List[Tuple[str, str]] = []
    if args.hub_path and args.hub_contrastive_weight > 0:
        assert args.hub_contrastive_zh in neurons, \
            f"对比 loss zh 侧 {args.hub_contrastive_zh} 不在阵容: {list(neurons)}"
        assert args.hub_contrastive_code in neurons, \
            f"对比 loss code 侧 {args.hub_contrastive_code} 不在阵容: {list(neurons)}"
        pairs_texts = load_pairs_texts()
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
        for domain in data_sources:
            texts = domain_texts[domain]
            random.shuffle(texts)
            # general_mode 下 domain_sp 不使用（输入/目标都 general 编码）；dialogue 桶亦然
            domain_sp = hub.get_tokenizer(domain) if domain in domains else None
            # zh 用中文 answer marker；其他域全文本 loss（数据无中文 marker）
            answer_marker = SFT_ANSWER_MARKER if domain == "zh" else None
            # general 目标空间：跨域组合训练（en/zh 理解指令 + code 表达），全文本 loss
            general_mode = args.target_space == "general"
            if general_mode:
                answer_marker = None

            for i in range(0, len(texts) - args.batch_size, args.batch_size):
                batch_texts = texts[i:i + args.batch_size]
                neuron_embeddings = {}
                targets = None
                mask = None
                sft_mask = None
                for nid, emb in shared_embeddings.items():
                    if general_mode:
                        # 输入与目标都在 general 256K 空间（domain_sp == general_sp）
                        out = batch_align_and_embed(
                            batch_texts, general_sp, general_sp, emb,
                            max_seq_len=args.seq_len,
                        )
                    else:
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
                    target_domain="general" if general_mode else domain,  # 缺口 M: batch 目标域
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
                # C15（D 方案，2026-08-08）：预测质量对比约束——quality_head 输出对齐
                # per-neuron NLL 排序（"谁能预测好当前文本谁上"）。替代 C13/C14 域标签
                # 判别 routing_loss（判别任务不对称：math/code 是英文 → en 覆盖；判别器
                # 输入各自 round1 表征不可比；softmax CE 无尺度约束 → logit 膨胀作弊）。
                # contrastive_loss 由 ensemble.forward_train 计算（NLL 需 targets，训练时
                # 可得；推理时 quality_logit 直接可用）。所有域 batch 生效（含 dialogue）。
                total_loss = total_loss + args.contrastive_weight * result["contrastive_loss"]

                # 缺口 L 阶段 3 第三部分：hub 锚定 loss——约束各域 neuron field_vector
                # 经 cross_spec 投影对齐 hub field_vector（hub=跨域语义锚点）。
                # 2026-08-15：全域化——每 batch 对**全部非 hub 域 neuron** 计算
                # （原仅当前 batch 域，频率 1/3；极小验证实测 zh 0.088→0.128 权重
                # 效应 +45% 但信号仍弱，全域化 ×3 信号频率，机制级增强而非调权重）。
                # 梯度只流 cross_spec_projectors（域 neuron/hub body 冻结，零破坏）。
                anchor_loss = None
                if args.hub_path and args.hub_anchor_weight > 0:
                    anchor_total = torch.tensor(0.0, device=next(iter(neurons.values())).device)
                    n_anchor = 0
                    for anid in neurons:
                        if anid == "hub":
                            continue
                        anchor_total = anchor_total + compute_hub_anchor_loss(
                            ensemble, neurons, neuron_embeddings, anid)
                        n_anchor += 1
                    if n_anchor:
                        anchor_loss = anchor_total / n_anchor
                        total_loss = total_loss + args.hub_anchor_weight * anchor_loss

                # 阶段 3 第三部分·渐进第二步：跨域对比 loss——每 batch 采样平行语料对，
                # 同义对（zh↔code）在统一空间（hub 空间）双向 InfoNCE 靠近、不同义远离。
                # 梯度只流 zh/code 两侧 cross_spec_projectors（域 neuron body 冻结）。
                if (args.hub_path and args.hub_contrastive_weight > 0 and pairs_texts):
                    sample = random.sample(pairs_texts, min(8, len(pairs_texts)))
                    zh_texts = [p[0] for p in sample]
                    code_texts = [p[1] for p in sample]
                    contrastive_loss = compute_hub_contrastive_loss(
                        ensemble, neurons, shared_embeddings, general_sp,
                        args.hub_contrastive_zh, args.hub_contrastive_code,
                        zh_texts, code_texts, tau=args.hub_contrastive_tau,
                        max_seq_len=args.seq_len)
                    total_loss = total_loss + args.hub_contrastive_weight * contrastive_loss

                total_loss.backward()
                if muon_optimizer is not None:
                    muon_optimizer.step()
                if adamw_optimizer is not None:
                    adamw_optimizer.step()
                if body_optimizer is not None:
                    body_optimizer.step()

                total_steps += 1
                if total_steps % 10 == 0:
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
                        "contrastive_loss": float(result.get("contrastive_loss", 0.0)),  # C15/C16d 质量监督
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
