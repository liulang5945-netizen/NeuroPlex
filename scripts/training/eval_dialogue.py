"""评估态极综合体的交流能力。

用对话 prompt 测试综合体的多轮交流能力：
  1. 单轮对话 PPL（对话数据评估）
  2. 多轮对话生成质量（实际交流测试）
  3. 个体 vs 协作对比

Usage:
    python -u scripts/training/eval_dialogue.py
    python -u scripts/training/eval_dialogue.py --weights dialogue  # 用对话训练权重
    python -u scripts/training/eval_dialogue.py --weights cross_spec  # 用 simple_zh 训练权重
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.resonance import (
    ResonanceNeuron, ResonanceField, ResonanceEnsemble,
    get_domain_neuron_config, NeuronGeometry,
)
from taiji.resonance.topology import (
    build_topology, establish_topology_channels,
    infer_topology_from_state, topology_detail,
)
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import (
    load_domain_tokenizer, load_general_tokenizer,
    OUTPUT_DIR, load_simple_zh_texts,
)
from scripts.training.finetune_cross_spec import load_dialogue_texts, load_neuron_with_embedding
from scripts.training.experiment_config import (
    ENSEMBLE_DIALOGUE_IDS as NEURON_IDS, DEFAULT_DOMAIN as DOMAIN,
    SAMPLING_TEMPERATURE, SAMPLING_TOP_K, SAMPLING_REPETITION_PENALTY, SAMPLING_MAX_TOKENS,
    DIALOGUE_PROMPTS,
)

DEVICE = "cpu"


def load_neurons_and_weights(weights_type: str = "dialogue", topology_mode: str = "hybrid"):
    """加载神经元和指定类型的权重。

    S7: 优先从 checkpoint 自动推断拓扑（匹配训练时拓扑），
    checkpoint 不存在时回退到 topology_mode 参数。

    Args:
        weights_type: "dialogue"=对话训练权重, "cross_spec"=simple_zh训练权重, "none"=无权重
        topology_mode: 回退拓扑模式 (当无法从 checkpoint 推断时使用)
    """
    neurons = {}
    shared_embeddings = {}

    print("[1] 加载神经元...", flush=True)
    for nid in NEURON_IDS:
        n, emb = load_neuron_with_embedding(nid)
        neurons[nid] = n
        shared_embeddings[nid] = emb

    # 确定权重路径（用于推断训练时拓扑）
    if weights_type == "dialogue":
        weights_path = os.path.join(OUTPUT_DIR, "cross_spec_dialogue.pt")
    elif weights_type == "cross_spec":
        weights_path = os.path.join(OUTPUT_DIR, "cross_spec_finetuned.pt")
    else:
        weights_path = None

    # S7: 优先从 checkpoint 推断拓扑，回退到 topology_mode
    topology = None
    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    ckpt_data = None
    if weights_path and os.path.exists(weights_path):
        ckpt_data = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        side_state_peek = ckpt_data.get("side_channels", ckpt_data)
        if side_state_peek and isinstance(side_state_peek, dict):
            topology = infer_topology_from_state(side_state_peek)
            print(f"  [topology] 从 checkpoint 推断: {topology_detail(topology, neurons)}", flush=True)

    if topology is None:
        topology = build_topology(neurons, geometry, mode=topology_mode)
        print(f"  [topology] 回退到 {topology_mode}: {topology_detail(topology, neurons)}", flush=True)

    stats = establish_topology_channels(neurons, topology, geometry)
    for nid, n_ch in stats.items():
        print(f"  [{nid}] {n_ch} excite channels", flush=True)

    # 加载权重
    if ckpt_data is not None:
        side_state = ckpt_data.get("side_channels", ckpt_data)
        for nid, neuron in neurons.items():
            if nid in side_state:
                for pid, ch_state in side_state[nid].get("excite", {}).items():
                    if pid in neuron.excite_channels:
                        neuron.excite_channels[pid].load_state_dict(ch_state)
        print(f"  [side_channels] 已加载: {weights_path}", flush=True)

        # S8: 加载 body 参数（最后N层微调结果，缺失则跳过=旧 ckpt 兼容）
        body_state = ckpt_data.get("body_state", {})
        if body_state:
            for nid, neuron in neurons.items():
                if nid in body_state:
                    for name, p in neuron.named_parameters():
                        if name in body_state[nid]:
                            p.data.copy_(body_state[nid][name])
            print(f"  [body] 已加载 S8 body 微调结果: {weights_path}", flush=True)

        # S8: 加载 shared_embedding（如果训练过）
        emb_state = ckpt_data.get("shared_embedding_state", {})
        if emb_state:
            for nid, emb in shared_embeddings.items():
                if nid in emb_state:
                    emb.load_state_dict(emb_state[nid])
            print(f"  [shared_embedding] 已加载 S8 emb 微调结果: {weights_path}", flush=True)
    else:
        print(f"  [side_channels] 未找到权重 ({weights_type})，使用随机初始化", flush=True)

    return neurons, shared_embeddings


def load_cross_spec_weights(ensemble, weights_type: str = "dialogue"):
    """加载跨规格投影层权重。"""
    if weights_type == "dialogue":
        weights_path = os.path.join(OUTPUT_DIR, "cross_spec_dialogue.pt")
    elif weights_type == "cross_spec":
        weights_path = os.path.join(OUTPUT_DIR, "cross_spec_finetuned.pt")
    else:
        return

    if not os.path.exists(weights_path):
        return

    ckpt_data = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if not isinstance(ckpt_data, dict) or "cross_spec" not in ckpt_data:
        return

    cross_spec_state = ckpt_data["cross_spec"]
    for nid, sd in cross_spec_state.get("forward", {}).items():
        if nid in ensemble._cross_spec_projectors:
            ensemble._cross_spec_projectors[nid].load_state_dict(sd)
    for nid, sd in cross_spec_state.get("backward", {}).items():
        if nid in ensemble._cross_spec_back_projectors:
            ensemble._cross_spec_back_projectors[nid].load_state_dict(sd)
    print(f"  [cross_spec] 已加载跨规格投影层权重: {weights_path}", flush=True)


def eval_dialogue_ppl(neurons, shared_embeddings, domain_sp, general_sp,
                      weights_type: str = "dialogue", n_eval: int = 100):
    """对话 PPL 评估（alpaca-zh 评估集）。"""
    print("\n" + "=" * 70, flush=True)
    print(f"[对话 PPL 评估] 个体 vs 协作 (weights={weights_type})", flush=True)
    print("=" * 70, flush=True)

    # 加载对话评估数据
    dialogue_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "simple_zh", "alpaca_zh_sft.jsonl",
    )
    all_texts = load_dialogue_texts(dialogue_path, max_texts=n_eval * 3)
    texts = all_texts[-n_eval:] if len(all_texts) > n_eval else all_texts
    print(f"  评估集(alpaca-zh SFT): {len(texts)} 条对话", flush=True)

    # 创建 Ensemble
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)
    load_cross_spec_weights(ensemble, weights_type)

    # 个体 PPL
    individual_ppls = {}
    for nid, neuron in neurons.items():
        shared_emb = shared_embeddings[nid]
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for text in texts:
                shared_emb_out, targets, mask = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                shared_emb_out = shared_emb_out.to(DEVICE)
                targets = targets.to(DEVICE)
                mask = mask.to(DEVICE)
                result = neuron.forward(shared_emb_out, return_logits=True)
                logits = result["logits"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                shift_targets = shift_targets.clone()
                shift_targets[~shift_mask] = -100
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += shift_mask.sum().item()
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        individual_ppls[nid] = ppl
        print(f"  个体 [{nid}]: PPL={ppl:.1f} (loss={avg_loss:.4f})", flush=True)

    # 协作 PPL
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            neuron_embeddings = {}
            targets = None
            mask = None
            for nid, shared_emb in shared_embeddings.items():
                shared_emb_out, tgt, msk = batch_align_and_embed(
                    [text], domain_sp, general_sp, shared_emb,
                )
                neuron_embeddings[nid] = shared_emb_out.to(DEVICE)
                if targets is None:
                    targets = tgt.to(DEVICE)
                    mask = msk.to(DEVICE)

            result = ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode="soft",
            )

            if "weighted_logits" in result:
                fused_logits = result["weighted_logits"]
            else:
                best_nid = max(result.get("final_scores", {}), key=result["final_scores"].get, default=NEURON_IDS[0])
                fused_logits = neurons[best_nid].forward(
                    neuron_embeddings[best_nid], return_logits=True
                )["logits"]

            shift_logits = fused_logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            shift_targets_clone = shift_targets.clone()
            shift_targets_clone[~shift_mask] = -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets_clone.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    collab_ppl = math.exp(min(avg_loss, 20))
    print(f"  协作 [side]:   PPL={collab_ppl:.1f} (loss={avg_loss:.4f})", flush=True)

    if "final_weights" in result:
        weights_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(result["final_weights"].items(), key=lambda x: -x[1]))
        print(f"  融合权重: {weights_str}", flush=True)

    min_ind = min(individual_ppls.values())
    best_id = min(individual_ppls, key=individual_ppls.get)
    print(f"\n  最强个体: [{best_id}] PPL={min_ind:.1f}", flush=True)
    print(f"  协作 PPL: {collab_ppl:.1f}", flush=True)
    if collab_ppl < min_ind:
        imp = (min_ind - collab_ppl) / min_ind * 100
        print(f"  EMERGE 协作比最强个体好 {imp:.1f}%", flush=True)
    else:
        print(f"  NO_EMERGE 协作 PPL={collab_ppl:.1f} >= 最强个体 {min_ind:.1f}", flush=True)

    return individual_ppls, collab_ppl


def _generate_collab(neurons, shared_embeddings, ensemble, domain_sp, general_sp,
                     prompt, max_tokens=SAMPLING_MAX_TOKENS,
                     temperature=SAMPLING_TEMPERATURE, top_k=SAMPLING_TOP_K,
                     repetition_penalty=SAMPLING_REPETITION_PENALTY):
    """协作生成（模块级函数，供单轮和多轮评测共用）。

    关键修复：ensemble 输出的 weighted_logits 维度 = zh domain vocab_size，
    next_token 是 domain token ID，需转回 general token IDs 才能追加到输入，
    解码用 domain_sp（不是 general_sp）。
    """
    general_ids = general_sp.EncodeAsIds(prompt)
    if not general_ids:
        return "(empty)"

    ids = torch.tensor([general_ids], dtype=torch.long, device=DEVICE)
    generated_domain = []

    # domain tokenizer 的 EOS
    domain_eos_id = None
    if hasattr(domain_sp, 'eos_id'):
        eid = domain_sp.eos_id()
        if eid is not None and eid >= 0:
            domain_eos_id = int(eid)

    with torch.no_grad():
        for _ in range(max_tokens):
            # 每个神经元独立编码
            neuron_embeddings = {}
            for nid, shared_emb in shared_embeddings.items():
                neuron_embeddings[nid] = shared_emb(ids)

            result = ensemble.forward(
                neuron_embeddings=neuron_embeddings,
                return_logits=True,
                fusion_mode="soft",
            )

            if "weighted_logits" in result:
                logits = result["weighted_logits"][:, -1, :].float()
            else:
                best_nid = max(result.get("final_scores", {}), key=result["final_scores"].get, default=NEURON_IDS[0])
                logits = neurons[best_nid].forward(
                    neuron_embeddings[best_nid], return_logits=True
                )["logits"][:, -1, :].float()

            # Repetition penalty
            if generated_domain:
                for prev_token in set(generated_domain[-20:]):
                    if prev_token < logits.size(-1):
                        logits[0, prev_token] /= repetition_penalty

            # Temperature + top-k
            logits = logits / temperature
            if top_k > 0:
                cur_top_k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits[0], cur_top_k)
                threshold = topk_vals[-1]
                logits[0][logits[0] < threshold] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_domain_token = torch.multinomial(probs, num_samples=1).item()
            generated_domain.append(next_domain_token)

            # EOS 检测（domain tokenizer）
            if domain_eos_id is not None and next_domain_token == domain_eos_id:
                break

            # 关键修复：domain token ID → 文本 → general token IDs → 追加到输入
            piece_text = domain_sp.decode([next_domain_token])
            new_general_ids = general_sp.encode(piece_text)
            if not new_general_ids:
                new_general_ids = [general_sp.pad_id()]
            ids = torch.cat([ids, torch.tensor([new_general_ids], dtype=torch.long, device=DEVICE)], dim=1)

    # 用 domain tokenizer 解码
    text = domain_sp.DecodeIds(generated_domain)
    return text


# 多轮对话场景：每个场景是一系列追问，测试综合体维持上下文的能力
MULTI_TURN_SCENARIOS = [
    ["你好，请介绍一下自己",
     "你和小神经元是什么关系？",
     "那你能做什么？"],
    ["什么是人工智能？",
     "它和机器学习有什么区别？",
     "能举个例子吗？"],
    ["如何学习编程？",
     "应该先学哪门语言？",
     "有什么好的学习资源推荐吗？"],
]


def eval_conversation(neurons, shared_embeddings, domain_sp, general_sp,
                      weights_type: str = "dialogue"):
    """实际对话生成质量评估（单轮）。"""
    print("\n" + "=" * 70, flush=True)
    print(f"[对话生成质量评估] (weights={weights_type})", flush=True)
    print("=" * 70, flush=True)

    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)
    load_cross_spec_weights(ensemble, weights_type)

    for prompt in DIALOGUE_PROMPTS:
        print(f"\n  {prompt}", flush=True)
        response = _generate_collab(neurons, shared_embeddings, ensemble,
                                    domain_sp, general_sp, prompt)
        print(f"  协作回复: {response}", flush=True)
        print(f"  {'-' * 60}", flush=True)


def eval_multi_turn_conversation(neurons, shared_embeddings, domain_sp, general_sp,
                                 weights_type: str = "dialogue"):
    """多轮对话评测（缺口 H 修复）：测试综合体维持上下文的能力。

    每个场景包含多轮追问，综合体需要根据上下文生成连贯回复。
    输入格式与训练数据对齐：问：xxx\\n答：xxx\\n问：yyy\\n答：
    """
    print("\n" + "=" * 70, flush=True)
    print(f"[多轮对话评测] (weights={weights_type})", flush=True)
    print("=" * 70, flush=True)

    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=2)
    load_cross_spec_weights(ensemble, weights_type)

    for scenario_idx, questions in enumerate(MULTI_TURN_SCENARIOS, 1):
        print(f"\n  场景 {scenario_idx}:", flush=True)
        history = ""
        for turn_idx, question in enumerate(questions, 1):
            # 拼接对话历史 + 当前问题（与训练数据格式对齐）
            prompt = f"{history}问：{question}\n答："
            print(f"\n  [轮 {turn_idx}] 问：{question}", flush=True)
            response = _generate_collab(neurons, shared_embeddings, ensemble,
                                        domain_sp, general_sp, prompt,
                                        max_tokens=128)
            print(f"  [轮 {turn_idx}] 答：{response}", flush=True)
            # 将本轮对话加入历史
            history = f"{history}问：{question}\n答：{response}\n"
        print(f"  {'=' * 60}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="dialogue",
                        choices=["dialogue", "cross_spec", "none"],
                        help="dialogue=对话训练权重, cross_spec=simple_zh训练权重, none=无权重")
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--skip_ppl", action="store_true", help="跳过 PPL 评估")
    parser.add_argument("--skip_gen", action="store_true", help="跳过生成评估")
    parser.add_argument("--multi_turn", action="store_true", help="启用多轮对话评测")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    parser.add_argument("--topology", default="hybrid",
                        choices=["full", "knn", "hub_spoke", "hybrid"],
                        help="S7: 回退拓扑模式 (checkpoint 无拓扑信息时使用)")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    print("=" * 70, flush=True)
    print(f"态极综合体交流能力评估 (weights={args.weights})", flush=True)
    print("=" * 70, flush=True)

    neurons, shared_embeddings = load_neurons_and_weights(args.weights, args.topology)
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()

    if not args.skip_ppl:
        eval_dialogue_ppl(neurons, shared_embeddings, domain_sp, general_sp,
                          weights_type=args.weights, n_eval=args.n_eval)

    if not args.skip_gen:
        eval_conversation(neurons, shared_embeddings, domain_sp, general_sp,
                          weights_type=args.weights)

    if args.multi_turn:
        eval_multi_turn_conversation(neurons, shared_embeddings, domain_sp, general_sp,
                                     weights_type=args.weights)

    print("\n" + "=" * 70, flush=True)
    print("评估完成", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
