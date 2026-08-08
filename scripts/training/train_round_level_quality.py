"""C20 回合级质量监督训练（quality_head 回合级对齐，2026-08-08）。

与 C16d（token 级 NLL 排序监督，quality_head 未校准 → code 恒高 16.9 独占）的区别：
- **监督粒度升级**：per_neuron_nll 只对 answer（回复）部分计算——回合级真实生成质量。
  prompt 部分所有 neuron 都能续写（输入即答案，无区分度）；answer 才是
  "谁能生成好这个回复"的回合粒度信号。C16d 的全序列 NLL 被 prompt 稀释。
- **监督机制复用**：C16d 的 per-neuron EMA z-score + 绝对质量 gate（防 code 独占 /
  防转译 neuron 抢位），但作用于回合级 NLL。
- **训练目标聚焦**：只训 quality_head（body/LoRA/side_channels 全部冻结，
  C16 LoRA 保护原则延续——不破坏个体生成能力）。
- quality_head 输入 = round1 对整个回合文本（prompt+answer）的 quality_logit，
  监督 = KL(softmax(q/1.0) ‖ softmax(-nll_gated_z/0.5))——与 C19 推理路径一致
  （_executive_route 用同一 head 对回合文本聚合判定）。

数据形态：
- 域 SFT（data/sft/{domain}_sft.pt）：prompt = instruction(+input)，answer = response
- 对话（data/simple_zh）：text "问：...\n答：..."，用 SFT_ANSWER_MARKER 切分

冒烟配方（2026-08-08）：
    python -u scripts/training/train_round_level_quality.py \
        --neuron-dir data/foundation_v1_general \
        --domains code,math,zh,en --data-dir data/sft \
        --max-texts-per-domain 200 --dialogue-max-texts 300 \
        --batch-size 2 --seq-len 64 --epochs 2 \
        --dialogue-ids zh_aug0_dialogue,zh_aug1_dialogue,zh_aug2_dialogue,zh_aug3_dialogue,zh_std0_dialogue \
        --dialogue-dir data/neurons --dialogue-data-dir data/simple_zh \
        --init-collab data/neurons/collab_v3_c16.ckpt.pt \
        --save-name collab_v3_c20
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.resonance import ResonanceEnsemble, ResonanceField, ResonanceNeuron
from taiji.resonance.translator import TokenizerHub
from scripts.training.utils import (
    load_general_tokenizer, create_shared_embedding,
    load_dialogue_texts_multi,
    OUTPUT_DIR,
)
from scripts.training.experiment_config import SFT_ANSWER_MARKER
from scripts.training.train_cross_domain_collab import (
    load_neuron, load_shared_lm_head, load_shared_embedding,
    load_tokenizer_for_vocab, TeeLogger, LOG_DIR,
)

DEVICE = "cpu"


def build_rounds(
    domains: List[str],
    data_dir: str,
    max_texts_per_domain: int,
    dialogue_ids: List[str],
    dialogue_data_dir: str,
    dialogue_max_texts: int,
) -> List[Tuple[str, str, str]]:
    """加载域 SFT + 对话，构造 (prompt, answer, source) 回合样本。"""
    rounds: List[Tuple[str, str, str]] = []
    for dom in domains:
        path = os.path.join(data_dir, f"{dom}_sft.pt")
        if not os.path.exists(path):
            print(f"  ⚠️ {path} 缺失，跳过 {dom}", flush=True)
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        for d in data[:max_texts_per_domain] if max_texts_per_domain > 0 else data:
            prompt = (d.get("instruction") or "") + (d.get("input") or "")
            answer = d.get("response") or ""
            if not answer.strip():
                continue
            rounds.append((prompt, answer, dom))
        print(f"  [{dom}] SFT 回合 {len([1 for r in rounds if r[2] == dom])} 条", flush=True)
    if dialogue_ids:
        texts = load_dialogue_texts_multi(
            dialogue_data_dir, max_texts=dialogue_max_texts, max_answer_chars=150)
        n_dia = 0
        for text in texts:
            idx = text.find(SFT_ANSWER_MARKER)
            if idx == -1:
                continue
            prompt, answer = text[:idx], text[idx:]
            if not answer.strip():
                continue
            rounds.append((prompt, answer, "dialogue"))
            n_dia += 1
        print(f"  [dialogue] 对话回合 {n_dia} 条", flush=True)
    return rounds


def batch_rounds(
    batch_rounds_in: List[Tuple[str, str, str]],
    general_sp,
    shared_embeddings: Dict[str, torch.nn.Embedding],
    max_seq_len: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """回合文本 → general 空间 targets + answer_mask + per-neuron embeddings。

    targets: [B, L] general 空间 token ids（prompt+answer+EOS，pad=-100）
    answer_mask: [B, L] bool（answer+EOS 为 True）
    """
    eos = general_sp.eos_id() if hasattr(general_sp, "eos_id") else 1
    all_ids: List[torch.Tensor] = []
    all_am: List[torch.Tensor] = []
    for prompt, answer, _src in batch_rounds_in:
        p_ids = general_sp.encode(prompt)
        a_ids = general_sp.encode(answer)
        # 截断：优先保留 answer + EOS（prompt 侧截）
        if len(p_ids) + len(a_ids) + 1 > max_seq_len:
            keep_p = max(0, max_seq_len - len(a_ids) - 1)
            p_ids = p_ids[-keep_p:] if keep_p > 0 else []
        ids = p_ids + a_ids + [eos]
        am = torch.zeros(len(ids), dtype=torch.bool)
        am[len(p_ids):len(p_ids) + len(a_ids) + 1] = True  # answer + EOS
        all_ids.append(torch.tensor(ids, dtype=torch.long))
        all_am.append(am)

    B = len(all_ids)
    L = max(len(t) for t in all_ids)
    padded_ids = torch.full((B, L), -100, dtype=torch.long)
    padded_am = torch.zeros(B, L, dtype=torch.bool)
    for b in range(B):
        padded_ids[b, :len(all_ids[b])] = all_ids[b]
        padded_am[b, :len(all_am[b])] = all_am[b]

    neuron_embeddings = {}
    for nid, emb in shared_embeddings.items():
        neuron_embeddings[nid] = emb(padded_ids.clamp(min=0))  # pad(-100)→0 查表
    return neuron_embeddings, padded_ids, padded_am


def save_head_checkpoint(path, epoch, total_steps, neurons, loss_history, phasor=None):
    """保存 C20 产物：head_state（quality_head 分量，C18 注入格式兼容）。

    C23-C：--enable-phasor 时额外保存 phasor_state（PhasorDynamics 可微相位）。
    """
    head_state = {}
    for nid, neuron in neurons.items():
        if getattr(neuron, "quality_head", None) is not None:
            head_state[nid] = neuron.quality_head.state_dict()
    ckpt = {
        "epoch": epoch,
        "total_steps": total_steps,
        "head_state": head_state,
        "loss_history": loss_history,
        "saved_at": datetime.now().isoformat(),
        "c20_round_level": True,
    }
    if phasor is not None:
        ckpt["phasor_state"] = phasor.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="C20 回合级质量监督训练")
    parser.add_argument("--neuron-dir", default=os.path.join(OUTPUT_DIR))
    parser.add_argument("--domains", default="code,math,zh,en")
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "sft"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-texts-per-domain", type=int, default=200)
    parser.add_argument("--dialogue-ids", default="")
    parser.add_argument("--dialogue-dir", default=OUTPUT_DIR)
    parser.add_argument("--dialogue-max-texts", type=int, default=300)
    parser.add_argument("--dialogue-data-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "simple_zh"))
    parser.add_argument("--contrastive-weight", type=float, default=0.5,
                        help="回合级质量监督权重（KL 对齐 answer-only NLL z-score）")
    parser.add_argument("--init-collab", default=None,
                        help="C16 训练产物 collab_v3_c16.ckpt.pt：warm start quality_head")
    parser.add_argument("--save-name", default="collab_v3_c20")
    parser.add_argument("--device", default="cpu")
    # C23-C：可微相位动力学（PhasorDynamics）——显式启用
    parser.add_argument("--enable-phasor", action="store_true",
                        help="启用可微相位动力学（PhasorDynamics）：相位绑结端到端可学")
    parser.add_argument("--phasor-binding-scale", type=float, default=0.3,
                        help="相位绑定强度 β（scores/场写入 × (1+β·binding)）")
    parser.add_argument("--phasor-lr", type=float, default=1e-3,
                        help="相位切向演化学习率（task_gradient_step）")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    dialogue_ids = [d.strip() for d in (args.dialogue_ids or "").split(",") if d.strip()]

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"train_round_level_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = TeeLogger(log_path)

    print("=" * 60, flush=True)
    print("C20 回合级质量监督训练（quality_head 回合级对齐）", flush=True)
    print(f"域: {domains} + 对话 {dialogue_ids}", flush=True)
    print(f"参数: {vars(args)}", flush=True)
    print("=" * 60, flush=True)

    # 1. 加载 9 阵容（general 基座 4 域 + 对话 5）
    print("\n[1] 加载神经元...", flush=True)
    shared_lm_head = load_shared_lm_head(args.neuron_dir, 512, DEVICE)
    neurons = {}
    shared_embeddings = {}
    for nid in domains:
        n = load_neuron(nid, args.neuron_dir, DEVICE, shared_lm_head=shared_lm_head)
        neurons[nid] = n
        shared_embeddings[nid] = load_shared_embedding(args.neuron_dir, DEVICE)
    for nid in dialogue_ids:
        path = os.path.join(args.dialogue_dir, f"neuron_{nid}.pt")
        if not os.path.exists(path):
            print(f"  ⚠️ 对话 neuron 缺失，跳过: {path}", flush=True)
            continue
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = ck["neuron_config"]
        cfg.unified_field_dim = None
        n = ResonanceNeuron(cfg).to(DEVICE)
        n.load_state_dict(ck["state_dict"], strict=False)
        neurons[nid] = n
        emb = create_shared_embedding(DEVICE)
        ses = ck.get("shared_embedding_state", {})
        if isinstance(ses, dict) and "weight" in ses:
            emb.weight.data.copy_(ses["weight"])
        elif isinstance(ses, torch.Tensor):
            emb.weight.data.copy_(ses)
        shared_embeddings[nid] = emb
    print(f"  阵容: {list(neurons.keys())}", flush=True)

    # warm start：C16 collab 产物的 quality_head
    if args.init_collab and os.path.exists(args.init_collab):
        ck16 = torch.load(args.init_collab, map_location=DEVICE, weights_only=False)
        hs = ck16.get("head_state", {})
        loaded = 0
        for nid, neuron in neurons.items():
            if nid in hs and getattr(neuron, "quality_head", None) is not None:
                neuron.quality_head.load_state_dict(hs[nid])
                loaded += 1
        print(f"  [warm start] quality_head 从 {args.init_collab} 加载 {loaded}/{len(neurons)}", flush=True)
    else:
        print("  [warm start] 未提供 init-collab → quality_head 随机初始化", flush=True)

    # 2. TokenizerHub
    print("\n[2] TokenizerHub...", flush=True)
    hub = TokenizerHub()
    for dom in domains:
        hub.register_domain(dom, load_tokenizer_for_vocab(dom, neurons[dom].config.vocab_size))
    if dialogue_ids:
        hub.register_domain("zh", load_tokenizer_for_vocab("zh", neurons[dialogue_ids[0]].config.vocab_size))
    general_sp = load_general_tokenizer()
    hub.register_domain("general", general_sp)

    # 3. 冻结全部，只解冻 quality_head（C20 聚焦判定信号校准）
    print("\n[3] 冻结全部，只解冻 quality_head...", flush=True)
    for neuron in neurons.values():
        for p in neuron.parameters():
            p.requires_grad = False
        if getattr(neuron, "quality_head", None) is not None:
            for p in neuron.quality_head.parameters():
                p.requires_grad = True
        neuron.train()

    # C23-C：可微相位动力学（PhasorDynamics，显式启用）
    # 相位绑结从启发式调制升级为端到端可学信号：forward_train 的可微绑定
    # 参与计算图（梯度经 evolve 输出 → ω/K），每 step 后 task_gradient_step
    # 黎曼切向更新 phasors（任务驱动"谁同相"）。同域同相作为先验初始相位。
    phasor = None
    if args.enable_phasor:
        from taiji.resonance.phasor import PhasorDynamics
        phasor = PhasorDynamics(binding_scale=args.phasor_binding_scale)
        domain_to_nids = {}
        for nid in neurons.keys():
            d = nid.split("_")[0] if "_" in nid else nid
            domain_to_nids.setdefault(d, []).append(nid)
        phasor.assign_phase_by_domain(domain_to_nids)
        phasor.to(args.device)
        print(f"  [phasor] PhasorDynamics wired（{len(phasor.list_phases())} phases）", flush=True)

    qh_params = [p for n in neurons.values() if getattr(n, "quality_head", None) is not None
                 for p in n.quality_head.parameters() if p.requires_grad]
    ph_params = []
    if phasor is not None:
        # ω/K 进 optimizer（可学）；phasors 用 task_gradient_step 切向更新（不进 optimizer，
        # 普通 SGD 径向梯度会被单位归一化抹掉）
        ph_params = [p for n, p in phasor.named_parameters() if n != "phasors"]
    optimizer = torch.optim.AdamW(qh_params + ph_params, lr=args.lr, weight_decay=0.01)
    print(f"  quality_head 可训练参数: {len(qh_params)}", flush=True)
    print(f"  phasor 可训练参数（ω/K）: {len(ph_params)}", flush=True)

    # 4. ensemble（field_dim 用 max，跨规格投影）
    print("\n[4] 创建 ensemble...", flush=True)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(
        neurons, field, max_rounds=2, geometry=None,
        gamma_oscillator=phasor,  # C23-C：可微相位动力学（None = 标量行为不变）
    )
    ensemble.set_tokenizer_hub(hub)

    # 5. 数据（按源域分组：batch 内同域 → NLL 可比，C16d gate 才有意义。
    #    混合域 batch 会被低 NLL 域（code）拉低 batch 最优 → dialogue neuron
    #    （转译，NLL 基线巨大）被 gate 全排除，监督失效）
    print("\n[5] 加载回合数据...", flush=True)
    rounds = build_rounds(
        domains, args.data_dir, args.max_texts_per_domain,
        dialogue_ids, args.dialogue_data_dir, args.dialogue_max_texts)
    by_source: Dict[str, List[Tuple[str, str, str]]] = {}
    for r in rounds:
        by_source.setdefault(r[2], []).append(r)
    sources = list(by_source.keys())
    for src, lst in by_source.items():
        random.shuffle(lst)
        print(f"  [{src}] {len(lst)} 回合", flush=True)
    random.seed(42)
    total_steps_per_epoch = sum(
        max(1, (len(lst) - args.batch_size) // args.batch_size) for lst in by_source.values()
    )

    # 6. 训练循环（域轮转，batch 内同域）
    print("\n[6] 开始训练...", flush=True)
    total_steps = 0
    loss_history: List[dict] = []
    ckpt_path = os.path.join(OUTPUT_DIR, f"{args.save_name}.ckpt.pt")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        for source in sources:
            lst = by_source[source]
            for i in range(0, len(lst) - args.batch_size, args.batch_size):
                batch = lst[i:i + args.batch_size]
                neuron_embeddings, targets, answer_mask = batch_rounds(
                    batch, general_sp, shared_embeddings, args.seq_len)

                optimizer.zero_grad()
                result = ensemble.forward_train(
                    neuron_embeddings=neuron_embeddings,
                    n_rounds=2,
                    fusion_mode="soft",
                    targets=targets,
                    answer_mask=answer_mask,
                    field_conditioning=True,
                    step=total_steps,
                    target_domain="general",
                )
                total_loss = args.contrastive_weight * result["contrastive_loss"]
                total_loss.backward()
                optimizer.step()
                if phasor is not None:
                    # C23-C：相位黎曼切向演化（任务梯度驱动"谁同相"；ω/K 已由 optimizer 更新）
                    phasor.task_gradient_step(lr=args.phasor_lr)
                total_steps += 1

                if total_steps % 10 == 0:
                    nll = result.get("per_neuron_nll")
                    nll_str = "N/A"
                    if nll is not None:
                        nll_str = ", ".join(f"{nid}={float(v.detach()):.2f}"
                                            for nid, v in zip(neurons.keys(), nll))
                    elapsed = time.time() - epoch_start
                    print(f"  E{epoch+1}/{args.epochs} [{source}] step {total_steps}: "
                          f"contrastive={float(result['contrastive_loss'].detach()):.4f} "
                          f"per-neuron NLL: {nll_str} "
                          f"(ETA {(elapsed/max(total_steps,1))*(total_steps_per_epoch*args.epochs-total_steps)/60:.1f}min)",
                          flush=True)
                    loss_history.append({
                        "step": total_steps, "epoch": epoch + 1, "source": source,
                        "contrastive_loss": float(result["contrastive_loss"].detach()),
                    })

                if total_steps % 500 == 0:
                    save_head_checkpoint(ckpt_path, epoch + 1, total_steps, neurons, loss_history, phasor=phasor)
                    print(f"  [checkpoint] step {total_steps} 已保存", flush=True)

        save_head_checkpoint(ckpt_path, epoch + 1, total_steps, neurons, loss_history, phasor=phasor)
        print(f"  [Epoch {epoch+1} 完成] 耗时 {(time.time()-epoch_start)/60:.1f} min", flush=True)

    print("\n[7] 训练完成。", flush=True)
    print(f"  checkpoint: {ckpt_path}", flush=True)
    history_path = os.path.join(LOG_DIR, f"{args.save_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)
    print(f"  训练历史: {history_path}", flush=True)


if __name__ == "__main__":
    main()
