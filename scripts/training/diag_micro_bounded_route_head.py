"""临时有界 token-level route head + projected-NLL 监督试验。

本试验只改变内存中的 route head：

* 输入是每个成员的 token hidden，先经过 LayerNorm，消除不同神经元表征的尺度差；
* 输出是每个成员、每个位置的 ``bound * tanh`` trust logit；
* 监督目标是每个答案 token 上各成员 general-space projected NLL 的 softmax；
* 语言主体、embed_adapter、field、跨规格投影和 shared embedding 全冻结。

token-level 返回路径由显式实验开关触发；默认生产路径不变，不保存任何 production checkpoint。
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.training.train_round_level_quality import batch_rounds
from scripts.training.diag_micro_route_fusion_pilot import (
    DEFAULT_SPECIALIST_STEPS,
    ROUTE_EVAL_SAMPLE_CAP,
    ROUTE_SEQ_LEN,
    ROUTE_TRAIN_SAMPLE_CAP,
    SEED,
    _load_route_rounds,
    _masked_teacher_forcing_nll,
    _prepare_population,
    _projected_logits,
    _rounds_from_texts,
    _route_snapshot as _scalar_route_snapshot,
    _freeze_to_quality_heads,
)
from scripts.training.utils import load_dialogue_texts_multi


ROUTE_HEAD_BOUND = 2.0
ROUTE_HEAD_HIDDEN = 128
ROUTE_HEAD_LR = 2e-3
ROUTE_TARGET_TEMPERATURE = 2.0
ROUTE_PREDICT_TEMPERATURE = 1.0
DEFAULT_ROUTE_STEPS = 80


class TokenBoundedRouteHead(nn.Module):
    """输入 token hidden、输出逐位置有界 trust logits 的临时路由头。"""

    def __init__(self, input_dim: int, hidden_dim: int = ROUTE_HEAD_HIDDEN,
                 bound: float = ROUTE_HEAD_BOUND):
        super().__init__()
        self.bound = float(bound)
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.005)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, token_hidden: torch.Tensor) -> torch.Tensor:
        return self.bound * torch.tanh(self.mlp(self.norm(token_hidden)))


def _install_token_bounded_heads(cortex) -> list[torch.nn.Parameter]:
    for neuron in cortex.ensemble.neurons.values():
        input_dim = neuron.config.hidden_size
        neuron.quality_head = TokenBoundedRouteHead(input_dim)
    return _freeze_to_quality_heads(cortex)


def _forward_batch(cortex, rounds, general_sp, token_route: bool = False):
    """实验侧前向；token_route=True 才启用 token-level quality head。"""

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
        targets=None,
        answer_mask=None,
        field_conditioning=True,
        step=0,
        target_domain="general",
        return_individual_logits=True,
        return_quality_tokens=token_route,
    )
    return result, targets, answer_mask


def _projected_member_nll(result, cortex, targets, answer_mask):
    projected = _projected_logits(cortex, result)
    nlls = torch.stack([
        _masked_teacher_forcing_nll(member_logits, targets, answer_mask)
        for member_logits in projected
    ])
    shift_targets = targets[:, 1:].contiguous()
    valid = answer_mask[:, 1:].bool() & shift_targets.ge(0)
    if not bool(valid.any()):
        valid = shift_targets.ge(0)
    token_nlls = []
    for member_logits in projected:
        token_nll = F.cross_entropy(
            member_logits[:, :-1, :].float().reshape(-1, member_logits.shape[-1]),
            shift_targets.reshape(-1),
            reduction="none",
        ).reshape_as(shift_targets)
        token_nlls.append(token_nll[valid])
    return projected, nlls, torch.stack(token_nlls)


def _per_position_target_snapshot(cortex, rounds, general_sp) -> dict:
    """汇总逐 token oracle 目标，检查 micro 是否得到真实互补份额。"""

    member_ids = list(cortex.ensemble.neurons.keys())
    target_sum = torch.zeros(len(member_ids))
    oracle_wins = torch.zeros(len(member_ids), dtype=torch.long)
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(rounds), 1):
            result, targets, answer_mask = _forward_batch(
                cortex, rounds[start:start + 1], general_sp, token_route=True
            )
            _projected, _member_nll, token_nll = _projected_member_nll(
                result, cortex, targets, answer_mask
            )
            ideal = F.softmax(-token_nll / ROUTE_TARGET_TEMPERATURE, dim=0)
            target_sum += ideal.sum(dim=1).cpu()
            oracle_wins += torch.bincount(
                ideal.argmax(dim=0).cpu(), minlength=len(member_ids)
            )
            total_tokens += token_nll.shape[1]
            del result
    total_tokens = max(total_tokens, 1)
    return {
        "samples": len(rounds),
        "answer_tokens": total_tokens,
        "mean_target_weights": {
            nid: round(float(target_sum[i] / total_tokens), 6)
            for i, nid in enumerate(member_ids)
        },
        "oracle_position_winner_fraction": {
            nid: round(float(oracle_wins[i] / total_tokens), 6)
            for i, nid in enumerate(member_ids)
        },
    }


def _masked_shift_teacher_forcing_nll(
    logits: torch.Tensor, targets: torch.Tensor, answer_mask: torch.Tensor
) -> torch.Tensor:
    """对已经右移对齐的 [B, L-1, V] logits 计算 answer NLL。"""

    shift_targets = targets[:, 1:].contiguous()
    valid = answer_mask[:, 1:].bool() & shift_targets.ge(0)
    if not bool(valid.any()):
        valid = shift_targets.ge(0)
    safe_targets = shift_targets.masked_fill(~valid, -100)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / valid.sum().clamp_min(1)


def _token_route_snapshot(cortex, rounds, general_sp) -> dict:
    """评估 token-level trust 的 hard route、soft shadow 和位置权重。"""

    if not rounds:
        return {"samples": 0}
    member_ids = list(cortex.ensemble.neurons.keys())
    hard_nlls = []
    shadow_nlls = []
    weight_sums = {nid: 0.0 for nid in member_ids}
    quality_sums = {nid: 0.0 for nid in member_ids}
    with torch.no_grad():
        for start in range(0, len(rounds), 1):
            result, targets, answer_mask = _forward_batch(
                cortex, rounds[start:start + 1], general_sp, token_route=True
            )
            projected = _projected_logits(cortex, result)
            token_quality = result.get("quality_token_logits")
            if token_quality is None:
                raise RuntimeError("token route head did not return token quality logits")
            route_logits = token_quality[:, :, :-1]
            trust = F.softmax(
                route_logits / ROUTE_PREDICT_TEMPERATURE, dim=0
            )
            shadow_logits = torch.einsum(
                "nbl,nblv->blv", trust, projected[:, :, :-1, :]
            )
            hard_nlls.append(float(_masked_teacher_forcing_nll(
                result["fused_logits"], targets, answer_mask
            ).detach()))
            shadow_nlls.append(float(_masked_shift_teacher_forcing_nll(
                shadow_logits, targets, answer_mask
            ).detach()))
            weights = result.get("weights")
            if weights is not None:
                for nid, value in zip(member_ids, weights):
                    weight_sums[nid] += float(value.detach())
            for nid, value in zip(member_ids, token_quality.mean(dim=(1, 2))):
                quality_sums[nid] += float(value.detach())
            del result, projected
    count = max(len(hard_nlls), 1)
    hard_mean = sum(hard_nlls) / count
    shadow_mean = sum(shadow_nlls) / count
    return {
        "samples": len(rounds),
        "hard_route_teacher_forcing_nll": round(hard_mean, 6),
        "shadow_soft_route_teacher_forcing_nll": round(shadow_mean, 6),
        "hard_route_ppl": round(math.exp(min(hard_mean, 20)), 4),
        "shadow_soft_route_ppl": round(math.exp(min(shadow_mean, 20)), 4),
        "hard_route_mean_weights": {
            nid: round(value / count, 6) for nid, value in weight_sums.items()
        },
        "quality_token_logits_mean": {
            nid: round(value / count, 6) for nid, value in quality_sums.items()
        },
    }


def _route_loss(cortex, result, targets, answer_mask):
    quality_token_logits = result.get("quality_token_logits")
    if quality_token_logits is None:
        raise RuntimeError("token route head did not return token quality logits")
    _projected, member_nll, member_token_nll = _projected_member_nll(
        result, cortex, targets, answer_mask
    )
    per_position_ideal = F.softmax(
        -member_token_nll.detach() / ROUTE_TARGET_TEMPERATURE, dim=0
    )
    ideal = per_position_ideal.mean(dim=1)
    shift_targets = targets[:, 1:].contiguous()
    valid = answer_mask[:, 1:].bool() & shift_targets.ge(0)
    if not bool(valid.any()):
        valid = shift_targets.ge(0)
    predicted_logits = quality_token_logits[:, :, :-1].reshape(
        quality_token_logits.shape[0], -1
    )[:, valid.reshape(-1)]
    if predicted_logits.shape != member_token_nll.shape:
        raise RuntimeError(
            "token route/projected NLL shape mismatch: "
            f"route={tuple(predicted_logits.shape)} nll={tuple(member_token_nll.shape)}"
        )
    predicted_log_probs = F.log_softmax(
        predicted_logits / ROUTE_PREDICT_TEMPERATURE, dim=0
    )
    loss = -(per_position_ideal * predicted_log_probs).sum(dim=0).mean()
    return loss, member_nll.detach(), ideal.detach(), per_position_ideal.detach()


def run(
    specialist_steps: int = DEFAULT_SPECIALIST_STEPS,
    route_steps: int = DEFAULT_ROUTE_STEPS,
    train_cap: int = ROUTE_TRAIN_SAMPLE_CAP,
    eval_cap: int = ROUTE_EVAL_SAMPLE_CAP,
) -> dict:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    random.seed(SEED)

    (
        cortex,
        shared,
        general_sp,
        expanded_ids,
        specialist_reports,
        data_info,
    ) = _prepare_population(specialist_steps, eval_cap)
    for neuron in cortex.ensemble.neurons.values():
        neuron.eval()

    train_rounds, eval_rounds = _load_route_rounds(train_cap, eval_cap)
    hf_eval_rounds = _rounds_from_texts(load_dialogue_texts_multi(
        "data/hf_candidates/moss_003_dialogue",
        filenames=["eval.jsonl"],
        max_texts=eval_cap,
    ))
    if not train_rounds or not eval_rounds:
        raise RuntimeError("bounded route head train/eval rounds are empty")
    if not hf_eval_rounds:
        raise RuntimeError("bounded route head HF eval rounds are empty")

    production_before = _scalar_route_snapshot(cortex, eval_rounds, general_sp)
    production_before_hf = _scalar_route_snapshot(cortex, hf_eval_rounds, general_sp)
    trainable = _install_token_bounded_heads(cortex)
    bounded_before = _token_route_snapshot(cortex, eval_rounds, general_sp)
    bounded_before_hf = _token_route_snapshot(cortex, hf_eval_rounds, general_sp)
    optimizer = torch.optim.AdamW(
        trainable, lr=ROUTE_HEAD_LR, weight_decay=0.01
    )
    generator = torch.Generator().manual_seed(SEED + 19)
    history = []
    for step in range(1, route_steps + 1):
        index = int(torch.randint(0, len(train_rounds), (1,), generator=generator))
        result, targets, answer_mask = _forward_batch(
            cortex, [train_rounds[index]], general_sp, token_route=True
        )
        loss, member_nll, ideal, per_position_ideal = _route_loss(
            cortex, result, targets, answer_mask
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step % 10 == 0 or step == route_steps:
            history.append({
                "step": step,
                "loss": round(float(loss.detach()), 6),
                "best_member_nll": round(float(member_nll.min()), 6),
                "ideal_top1": int(ideal.argmax()),
                "ideal_position_top1_fraction": round(
                    float((per_position_ideal.argmax(dim=0) == ideal.argmax()).float().mean()),
                    6,
                ),
                "quality_min": round(float(
                    result["quality_token_logits"].detach().min()
                ), 6),
                "quality_max": round(float(
                    result["quality_token_logits"].detach().max()
                ), 6),
            })
            print(
                f"[bounded-route] step {step}/{route_steps}: "
                f"loss={history[-1]['loss']:.4f} "
                f"quality=[{history[-1]['quality_min']:.3f},"
                f"{history[-1]['quality_max']:.3f}]",
                flush=True,
            )
        del result

    bounded_after = _token_route_snapshot(cortex, eval_rounds, general_sp)
    bounded_after_hf = _token_route_snapshot(cortex, hf_eval_rounds, general_sp)
    target_current = _per_position_target_snapshot(cortex, eval_rounds, general_sp)
    target_hf = _per_position_target_snapshot(cortex, hf_eval_rounds, general_sp)
    raw_nll = production_before["hard_route_teacher_forcing_nll"]
    after_nll = bounded_after["hard_route_teacher_forcing_nll"]
    report = {
        "contract": {
            "seed": SEED,
            "population": "5 dialogue + 4 general + 3 temporary micro specialists",
            "expanded_population_size": len(expanded_ids),
            "specialist_steps_per_member": specialist_steps,
            "route_steps": route_steps,
            "route_head": "token hidden -> LayerNorm -> Linear/GELU/Linear -> 2*tanh",
            "route_head_bound": ROUTE_HEAD_BOUND,
            "route_target": "per_position_softmax(-projected_token_nll / 2.0)",
            "language_bodies_frozen": True,
            "embed_adapter_frozen": True,
            "field_and_cross_spec_fusion_frozen": True,
            "shared_embedding_frozen": True,
            "production_checkpoint_written": False,
            "default_loader_changed": False,
        },
        "data": {
            **data_info,
            "route_train_rounds": len(train_rounds),
            "route_eval_rounds": len(eval_rounds),
            "hf_eval_rounds": len(hf_eval_rounds),
            "route_seq_len": ROUTE_SEQ_LEN,
        },
        "specialist_reports": specialist_reports,
        "production_before": production_before,
        "production_before_hf": production_before_hf,
        "bounded_before": bounded_before,
        "bounded_before_hf": bounded_before_hf,
        "bounded_after": bounded_after,
        "bounded_after_hf": bounded_after_hf,
        "per_position_target_current": target_current,
        "per_position_target_hf": target_hf,
        "delta": {
            "bounded_after_minus_production_nll": round(after_nll - raw_nll, 6),
            "bounded_after_over_production_ppl_ratio": round(
                math.exp(min(after_nll - raw_nll, 20)), 6
            ),
            "hf_bounded_after_minus_production_nll": round(
                bounded_after_hf["hard_route_teacher_forcing_nll"]
                - production_before_hf["hard_route_teacher_forcing_nll"],
                6,
            ),
            "hf_bounded_after_over_production_ppl_ratio": round(
                math.exp(min(
                    bounded_after_hf["hard_route_teacher_forcing_nll"]
                    - production_before_hf["hard_route_teacher_forcing_nll"],
                    20,
                )),
                6,
            ),
        },
        "route_loss_trace": history,
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
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(
        specialist_steps=args.specialist_steps,
        route_steps=args.route_steps,
        train_cap=args.train_cap,
        eval_cap=args.eval_cap,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
