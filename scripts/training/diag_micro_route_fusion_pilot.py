"""隔离的 9+3 路由/融合信用分配试验。

本试验不写入生产 checkpoint，也不修改默认装配路径。流程是：

1. 重新在内存中构造此前验证过的三个 6.97M 专家；
2. 加入真实的 5 个对话 + 4 个通用神经元，形成临时 12 成员群体；
3. 冻结 12 个语言主体、field、跨规格投影和 shared embedding，只训练
   ``quality_head``（质量路由参数）；
4. 用真实跨词表投影 logits 构造可微 shadow fusion，优化 teacher-forcing
   answer NLL + quality replay 对比损失；
5. 用真实生产硬路由重新评估，并与训练前的 no-op 对照。

注意：生产 forward_train 的跨词表主路径最终是 per-position hard route，
其 argmax 本身不可反传。shadow fusion 仅是本试验的信用分配代理，生产推理
仍走原有硬路由，因此该试验可以回答“路由质量信号是否值得继续训练”，而不
把未经验证的融合行为写回架构。
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from neuroplex.core.model_loader import DEFAULT_NEURON_IDS
from neuroplex.loader import assemble_cortex
from neuroplex.resonance import ResonanceField
from neuroplex.resonance.dialogue_format import SFT_ANSWER_MARKER
from scripts.training.diag_micro_data_ab import (
    DEFAULT_HF_RATIO,
    MAX_SEQ_LEN,
    SEED,
    _load_pools,
    _load_shared_embedding,
    _select_hf_for_ratio,
    _train_condition,
)
from scripts.training.diag_micro_specialist_group import (
    SPECIALIST_ROLES,
    _make_config,
)
from scripts.training.diag_micro_population_canary import PROMPTS, _generate, _surface_metrics
from scripts.training.train_round_level_quality import batch_rounds
from scripts.training.utils import (
    load_dialogue_texts_multi,
    load_domain_tokenizer,
    load_general_tokenizer,
    split_train_eval,
)


BASE_POPULATION_SIZE = 9
ROUTE_TRAIN_SAMPLE_CAP = 96
ROUTE_EVAL_SAMPLE_CAP = 24
ROUTE_BATCH_SIZE = 1
ROUTE_SEQ_LEN = 64
DEFAULT_SPECIALIST_STEPS = 800
DEFAULT_ROUTE_STEPS = 80
DEFAULT_LR = 1e-3
DEFAULT_NLL_WEIGHT = 1.0
DEFAULT_CONTRASTIVE_WEIGHT = 0.25
ROUTE_TEMPERATURE = 1.0


def _rounds_from_texts(texts: Iterable[str]) -> List[Tuple[str, str, str]]:
    rounds: List[Tuple[str, str, str]] = []
    for text in texts:
        index = text.find(SFT_ANSWER_MARKER)
        if index < 0:
            continue
        prompt, answer = text[:index], text[index:]
        if prompt.strip() and answer.strip():
            rounds.append((prompt, answer, "dialogue"))
    return rounds


def _load_route_rounds(train_cap: int, eval_cap: int) -> tuple[list, list]:
    texts = load_dialogue_texts_multi(
        "data/simple_zh", max_texts=100_000, max_answer_chars=96
    )
    train_texts, eval_texts = split_train_eval(texts, eval_ratio=0.05, seed=42)
    return (
        _rounds_from_texts(train_texts[:train_cap]),
        _rounds_from_texts(eval_texts[:eval_cap]),
    )


def _masked_teacher_forcing_nll(
    logits: torch.Tensor, targets: torch.Tensor, answer_mask: torch.Tensor
) -> torch.Tensor:
    """计算 answer+EOS 的 general-space teacher-forcing NLL。"""

    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_targets = targets[:, 1:].contiguous()
    mask = answer_mask[:, 1:].bool() & shift_targets.ge(0)
    if not bool(mask.any()):
        mask = shift_targets.ge(0)
    safe_targets = shift_targets.masked_fill(~mask, -100)
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        safe_targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / mask.sum().clamp_min(1)


def _prepare_population(specialist_steps: int, eval_cap: int):
    """构造内存中的 9+3 群体，并保留共享嵌入引用。"""

    pools = _load_pools(eval_cap=eval_cap)
    shared = _load_shared_embedding()
    for parameter in shared.parameters():
        parameter.requires_grad = False

    domain_sp = load_domain_tokenizer("zh")
    general_sp = load_general_tokenizer()
    eval_sets = {
        "current_eval": pools["current_eval"],
        "hf_eval": pools["hf_eval"],
    }
    selected_hf_train = _select_hf_for_ratio(
        pools["current_train"], pools["hf_train"], DEFAULT_HF_RATIO
    )
    train_sets = {
        "current_only": pools["current_train"],
        "hf_only": pools["hf_train"],
        "current_plus_hf_10": pools["current_train"] + selected_hf_train,
    }

    specialist_reports = {}
    specialists = {}
    for role in SPECIALIST_ROLES:
        neuron_id = f"zh_micro_specialist_{role}"
        print(f"[specialist:{role}] {specialist_steps} steps", flush=True)
        report, neuron = _train_condition(
            role,
            train_sets[role],
            eval_sets,
            shared,
            domain_sp,
            general_sp,
            specialist_steps,
            return_neuron=True,
            neuron_config=_make_config(neuron_id),
        )
        specialist_reports[role] = report
        specialists[neuron_id] = neuron

    with contextlib.redirect_stdout(io.StringIO()):
        cortex, _, _ = assemble_cortex(
            neurons_dir="data/neurons",
            extra_neurons_dir="data/foundation_v1_dual",
            collab_name="collab_v3_c24v2.ckpt.pt",
            device="cpu",
            max_rounds=3,
            wire_bio_modules=False,
            neuron_ids=list(DEFAULT_NEURON_IDS),
        )

    base_ids = list(cortex.neurons.keys())
    expected_general = {"code", "en", "math", "zh"}
    actual_general = set(base_ids) - set(DEFAULT_NEURON_IDS)
    if (
        len(base_ids) != BASE_POPULATION_SIZE
        or set(DEFAULT_NEURON_IDS) - set(base_ids)
        or not expected_general.issubset(actual_general)
    ):
        raise RuntimeError(f"real production population mismatch: {base_ids}")

    # assemble_cortex 的默认 hub 为兼容旧路径把 ``general`` 暂时映射到
    # 16K en tokenizer；本试验的 teacher-forcing targets 来自真正的
    # 256K general tokenizer，必须在临时 hub 中覆盖这个别名。只影响当前
    # 内存试验，不改 loader 默认行为。
    hub = getattr(cortex.ensemble, "_tokenizer_hub", None)
    if hub is None:
        raise RuntimeError("temporary population has no tokenizer hub")
    hub.register_domain("general", general_sp)
    cortex.set_tokenizer_hub(hub)

    for neuron_id, neuron in specialists.items():
        neuron.eval()
        neuron.config.neuron_id = neuron_id
        cortex.ensemble.add_neuron(neuron_id, neuron)

    embeddings = dict(cortex._neuron_shared_embeddings or {})
    for neuron_id in specialists:
        embeddings[neuron_id] = shared
    cortex.set_neuron_shared_embeddings(embeddings)
    expanded_ids = list(cortex.ensemble.neurons.keys())
    if len(expanded_ids) != 12:
        raise RuntimeError(f"expected 12 temporary members, got {expanded_ids}")
    return (
        cortex,
        shared,
        general_sp,
        expanded_ids,
        specialist_reports,
        {
            "current_train": len(pools["current_train"]),
            "current_eval": len(pools["current_eval"]),
            "hf_train": len(pools["hf_train"]),
            "hf_train_used_for_specialists": len(selected_hf_train),
            "hf_eval": len(pools["hf_eval"]),
            "eval_cap": eval_cap,
        },
    )


def _freeze_to_quality_heads(cortex) -> list[torch.nn.Parameter]:
    """锁定语言主体与所有融合模块，只开放 quality_head。"""

    for neuron in cortex.ensemble.neurons.values():
        for parameter in neuron.parameters():
            parameter.requires_grad = False
        quality_head = getattr(neuron, "quality_head", None)
        if quality_head is None:
            raise RuntimeError(f"neuron {neuron.config.neuron_id} has no quality_head")
        for parameter in quality_head.parameters():
            parameter.requires_grad = True
        neuron.eval()

    for module in [
        cortex.ensemble._field,
        *getattr(cortex.ensemble, "_cross_spec_projectors", {}).values(),
        *getattr(cortex.ensemble, "_cross_spec_back_projectors", {}).values(),
    ]:
        if hasattr(module, "parameters"):
            for parameter in module.parameters():
                parameter.requires_grad = False

    trainable = [
        parameter
        for neuron in cortex.ensemble.neurons.values()
        for parameter in getattr(neuron, "quality_head").parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("no quality_head parameters are trainable")
    return trainable


def _forward_batch(cortex, rounds, general_sp, trust_override=None):
    embeddings = {
        nid: cortex._neuron_shared_embeddings[nid]
        for nid in cortex.ensemble.neurons
    }
    neuron_embeddings, targets, answer_mask = batch_rounds(
        rounds,
        general_sp,
        embeddings,
        ROUTE_SEQ_LEN,
    )
    result = cortex.ensemble.forward_train(
        neuron_embeddings=neuron_embeddings,
        n_rounds=2,
        fusion_mode="soft",
        # 生产 9 阵容中存在历史 hidden=768 对话 neuron，没有 general judge
        # head。这里不调用 ensemble 内置的混合 judge-NLL 分支，避免把 general
        # token id 错喂给该 neuron 的 native vocab；下面用统一的 projected
        # logits 自己计算 quality replay NLL。
        targets=None,
        answer_mask=None,
        field_conditioning=True,
        step=0,
        target_domain="general",
        trust_override=trust_override,
        return_individual_logits=True,
    )
    return result, targets, answer_mask


def _projected_logits(cortex, result) -> torch.Tensor:
    """取真实跨词表投影 logits，统一到 general 判定空间。"""

    ids = list(cortex.ensemble.neurons.keys())
    individual = result.get("individual_logits")
    if not individual:
        raise RuntimeError("forward_train did not return individual logits")
    return cortex.ensemble._project_logits_to_target(
        individual, ids, "general"
    )


def _quality_replay_terms(cortex, result, targets, answer_mask):
    """返回 shadow NLL 和 quality-head 对比 replay loss。

    这段逻辑刻意在试验侧实现，避开历史阵容混合 judge-head 的不一致分支；
    所有成员都在同一个 general projected-logit 空间被比较。
    """

    quality_logits = result.get("quality_logits")
    if quality_logits is None:
        raise RuntimeError("forward_train did not return quality logits")
    projected = _projected_logits(cortex, result)
    trust = F.softmax(quality_logits / ROUTE_TEMPERATURE, dim=0)
    fused = torch.einsum("n,nblv->blv", trust, projected)
    shadow_nll = _masked_teacher_forcing_nll(fused, targets, answer_mask)

    nlls = []
    for member_logits in projected:
        nlls.append(_masked_teacher_forcing_nll(
            member_logits, targets, answer_mask
        ))
    per_member_nll = torch.stack(nlls)
    ql_std = quality_logits.detach().std() + 1e-6
    actual = F.softmax(
        (quality_logits - quality_logits.detach().mean()) / ql_std,
        dim=0,
    )
    ideal = F.softmax(-per_member_nll / 0.5, dim=0)
    replay_loss = (
        actual * (actual.clamp_min(1e-8).log() - ideal.clamp_min(1e-8).log())
    ).sum()
    return projected, shadow_nll, replay_loss, per_member_nll


def _shadow_fusion_nll(cortex, result, targets, answer_mask) -> torch.Tensor:
    """兼容保留：返回 quality replay 的可微 shadow NLL。"""

    _projected, shadow_nll, _replay, _nlls = _quality_replay_terms(
        cortex, result, targets, answer_mask
    )
    return shadow_nll


def _route_snapshot(cortex, rounds, general_sp) -> dict:
    if not rounds:
        return {"samples": 0}
    hard_nlls = []
    shadow_nlls = []
    weight_sums = {nid: 0.0 for nid in cortex.ensemble.neurons}
    quality_sums = {nid: 0.0 for nid in cortex.ensemble.neurons}
    with torch.no_grad():
        for start in range(0, len(rounds), ROUTE_BATCH_SIZE):
            result, targets, answer_mask = _forward_batch(
                cortex, rounds[start:start + ROUTE_BATCH_SIZE], general_sp
            )
            _projected, shadow_nll, _replay, _nlls = _quality_replay_terms(
                cortex, result, targets, answer_mask
            )
            hard_nlls.append(
                float(_masked_teacher_forcing_nll(
                    result["fused_logits"], targets, answer_mask
                ).detach())
            )
            shadow_nlls.append(float(shadow_nll.detach()))
            weights = result.get("weights")
            quality = result.get("quality_logits")
            if weights is not None:
                for nid, value in zip(cortex.ensemble.neurons, weights):
                    weight_sums[nid] += float(value.detach())
            if quality is not None:
                for nid, value in zip(cortex.ensemble.neurons, quality):
                    quality_sums[nid] += float(value.detach())
            del result
    count = max(len(hard_nlls), 1)
    return {
        "samples": len(rounds),
        "hard_route_teacher_forcing_nll": round(sum(hard_nlls) / count, 6),
        "shadow_soft_route_teacher_forcing_nll": round(sum(shadow_nlls) / count, 6),
        "hard_route_ppl": round(math.exp(min(sum(hard_nlls) / count, 20)), 4),
        "shadow_soft_route_ppl": round(math.exp(min(sum(shadow_nlls) / count, 20)), 4),
        "hard_route_mean_weights": {
            nid: round(value / count, 6) for nid, value in weight_sums.items()
        },
        "quality_logits_mean": {
            nid: round(value / count, 6) for nid, value in quality_sums.items()
        },
    }


def _generation_snapshot(cortex, active_ids: list[str]) -> dict:
    result = {}
    for index, prompt in enumerate(PROMPTS[:2]):
        seed = SEED + index
        text = _generate(cortex, active_ids, prompt, seed)
        result[prompt] = {"text": text, "surface": _surface_metrics(text)}
    return result


def run(
    specialist_steps: int = DEFAULT_SPECIALIST_STEPS,
    route_steps: int = DEFAULT_ROUTE_STEPS,
    train_cap: int = ROUTE_TRAIN_SAMPLE_CAP,
    eval_cap: int = ROUTE_EVAL_SAMPLE_CAP,
    lr: float = DEFAULT_LR,
    nll_weight: float = DEFAULT_NLL_WEIGHT,
    contrastive_weight: float = DEFAULT_CONTRASTIVE_WEIGHT,
) -> dict:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    random.seed(SEED)
    torch.manual_seed(SEED)

    (
        cortex,
        shared,
        general_sp,
        expanded_ids,
        specialist_reports,
        data_info,
    ) = _prepare_population(specialist_steps, eval_cap)
    train_rounds, eval_rounds = _load_route_rounds(train_cap, eval_cap)
    train_rounds = train_rounds or eval_rounds
    if not train_rounds or not eval_rounds:
        raise RuntimeError("route pilot has no usable dialogue rounds")

    trainable = _freeze_to_quality_heads(cortex)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    print(
        f"[route] 12 members; trainable quality-head tensors={len(trainable)}; "
        f"steps={route_steps}",
        flush=True,
    )

    before = _route_snapshot(cortex, eval_rounds, general_sp)
    generation_before = _generation_snapshot(cortex, expanded_ids)

    generator = torch.Generator().manual_seed(SEED + 7)
    history = []
    for step in range(1, route_steps + 1):
        index = int(torch.randint(0, len(train_rounds), (1,), generator=generator))
        batch = [train_rounds[index]]
        result, targets, answer_mask = _forward_batch(cortex, batch, general_sp)
        _projected, shadow_nll, contrastive, _nlls = _quality_replay_terms(
            cortex, result, targets, answer_mask
        )
        total_loss = nll_weight * shadow_nll + contrastive_weight * contrastive
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step % 10 == 0 or step == route_steps:
            record = {
                "step": step,
                "shadow_nll": round(float(shadow_nll.detach()), 6),
                "contrastive_loss": round(float(contrastive.detach()), 6),
                "total_loss": round(float(total_loss.detach()), 6),
            }
            history.append(record)
            print(
                f"[route] step {step}/{route_steps}: "
                f"shadow_nll={record['shadow_nll']:.4f} "
                f"contrastive={record['contrastive_loss']:.4f}",
                flush=True,
            )
        del result

    after = _route_snapshot(cortex, eval_rounds, general_sp)
    generation_after = _generation_snapshot(cortex, expanded_ids)
    changed = {
        "hard_nll_delta": round(
            after["hard_route_teacher_forcing_nll"]
            - before["hard_route_teacher_forcing_nll"],
            6,
        ),
        "shadow_nll_delta": round(
            after["shadow_soft_route_teacher_forcing_nll"]
            - before["shadow_soft_route_teacher_forcing_nll"],
            6,
        ),
        "hard_ppl_ratio": round(
            after["hard_route_ppl"] / max(before["hard_route_ppl"], 1e-9),
            6,
        ),
    }

    report = {
        "contract": {
            "seed": SEED,
            "population": "5 dialogue + 4 general + 3 temporary micro specialists",
            "base_population_size": BASE_POPULATION_SIZE,
            "expanded_population_size": len(expanded_ids),
            "specialist_steps_per_member": specialist_steps,
            "route_steps": route_steps,
            "route_parameters": "quality_head only",
            "language_bodies_frozen": True,
            "shared_embedding_frozen": True,
            "field_and_cross_spec_fusion_frozen": True,
            "production_checkpoint_written": False,
            "default_loader_changed": False,
            "shadow_fusion_is_training_only": True,
            "production_hard_route_evaluated": True,
            "route_temperature": ROUTE_TEMPERATURE,
        },
        "data": {
            **data_info,
            "route_train_rounds": len(train_rounds),
            "route_eval_rounds": len(eval_rounds),
            "route_seq_len": ROUTE_SEQ_LEN,
            "route_batch_size": ROUTE_BATCH_SIZE,
        },
        "specialist_reports": specialist_reports,
        "route_before": before,
        "route_after": after,
        "delta": changed,
        "route_loss_trace": history,
        "generation_before": generation_before,
        "generation_after": generation_after,
    }
    del optimizer, cortex, shared
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist-steps", type=int, default=DEFAULT_SPECIALIST_STEPS)
    parser.add_argument("--route-steps", type=int, default=DEFAULT_ROUTE_STEPS)
    parser.add_argument("--train-cap", type=int, default=ROUTE_TRAIN_SAMPLE_CAP)
    parser.add_argument("--eval-cap", type=int, default=ROUTE_EVAL_SAMPLE_CAP)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--nll-weight", type=float, default=DEFAULT_NLL_WEIGHT)
    parser.add_argument("--contrastive-weight", type=float, default=DEFAULT_CONTRASTIVE_WEIGHT)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(
        specialist_steps=args.specialist_steps,
        route_steps=args.route_steps,
        train_cap=args.train_cap,
        eval_cap=args.eval_cap,
        lr=args.lr,
        nll_weight=args.nll_weight,
        contrastive_weight=args.contrastive_weight,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
