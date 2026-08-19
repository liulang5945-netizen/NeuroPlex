"""只读的 9+3 oracle projected-NLL 路由上界审计。

该审计使用真实 general-space projected logits 和答案 token 的真实 NLL，
计算一个不可用于推理的 oracle：每个答案位置都选择 NLL 最低的成员。
它不是训练，也不是生产路由方案；用途是区分两种情况：

* oracle 明显优于当前 hard route：群体里有互补能力，下一步应修复信用分配；
* oracle 也没有优势：当前 9+3 群体本身没有可利用的 projected 能力增益，
  不应继续堆 route/fusion 复杂度。
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
import torch.nn.functional as F

from scripts.training.diag_micro_route_fusion_pilot import (
    DEFAULT_SPECIALIST_STEPS,
    ROUTE_EVAL_SAMPLE_CAP,
    ROUTE_TRAIN_SAMPLE_CAP,
    SEED,
    _forward_batch,
    _load_route_rounds,
    _masked_teacher_forcing_nll,
    _prepare_population,
    _projected_logits,
)


def _member_token_nll(projected, targets, answer_mask):
    shift_targets = targets[:, 1:].contiguous()
    valid = answer_mask[:, 1:].bool() & shift_targets.ge(0)
    if not bool(valid.any()):
        valid = shift_targets.ge(0)
    rows = []
    for member_logits in projected:
        token_nll = F.cross_entropy(
            member_logits[:, :-1, :].float().reshape(-1, member_logits.shape[-1]),
            shift_targets.reshape(-1),
            reduction="none",
        ).reshape_as(shift_targets)
        rows.append(token_nll[valid])
    return torch.stack(rows), int(valid.sum())


def _audit(cortex, rounds, general_sp):
    member_ids = list(cortex.ensemble.neurons.keys())
    raw_nlls = []
    oracle_nlls = []
    single_best_nlls = []
    member_token_sum = torch.zeros(len(member_ids))
    member_token_count = torch.zeros(len(member_ids), dtype=torch.long)
    oracle_wins = torch.zeros(len(member_ids), dtype=torch.long)
    total_tokens = 0

    with torch.no_grad():
        for index in range(len(rounds)):
            result, targets, answer_mask = _forward_batch(
                cortex, rounds[index:index + 1], general_sp
            )
            projected = _projected_logits(cortex, result)
            member_nll, valid_tokens = _member_token_nll(
                projected, targets, answer_mask
            )
            raw_nll = _masked_teacher_forcing_nll(
                result["fused_logits"], targets, answer_mask
            )
            oracle_per_token, winner = member_nll.min(dim=0)
            oracle_nll = oracle_per_token.mean()
            single_best_nll = member_nll.mean(dim=1).min()

            raw_nlls.append(float(raw_nll))
            oracle_nlls.append(float(oracle_nll))
            single_best_nlls.append(float(single_best_nll))
            member_token_sum += member_nll.sum(dim=1).float().cpu()
            member_token_count += valid_tokens
            oracle_wins += torch.bincount(
                winner.cpu(), minlength=len(member_ids)
            )
            total_tokens += valid_tokens
            del result, projected, member_nll

    count = max(len(rounds), 1)
    raw_mean = sum(raw_nlls) / count
    oracle_mean = sum(oracle_nlls) / count
    single_best_mean = sum(single_best_nlls) / count
    oracle_winner_fraction = oracle_wins.float() / max(total_tokens, 1)
    return {
        "samples": len(rounds),
        "answer_tokens": total_tokens,
        "raw_hard_route_teacher_forcing_nll": round(raw_mean, 6),
        "single_best_member_nll": round(single_best_mean, 6),
        "oracle_per_position_nll": round(oracle_mean, 6),
        "oracle_minus_raw_nll": round(oracle_mean - raw_mean, 6),
        "oracle_gain_fraction_vs_raw": round(
            (raw_mean - oracle_mean) / max(abs(raw_mean), 1e-9), 6
        ),
        "raw_hard_route_ppl": round(math.exp(min(raw_mean, 20)), 4),
        "oracle_per_position_ppl": round(math.exp(min(oracle_mean, 20)), 4),
        "member_mean_projected_nll": {
            nid: round(float(member_token_sum[i] / max(member_token_count[i], 1)), 6)
            for i, nid in enumerate(member_ids)
        },
        "oracle_winner_fraction": {
            nid: round(float(oracle_winner_fraction[i]), 6)
            for i, nid in enumerate(member_ids)
        },
    }


def run(
    specialist_steps: int = DEFAULT_SPECIALIST_STEPS,
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
    _train_rounds, eval_rounds = _load_route_rounds(train_cap, eval_cap)
    if not eval_rounds:
        raise RuntimeError("oracle audit eval set is empty")
    metrics = _audit(cortex, eval_rounds, general_sp)
    report = {
        "contract": {
            "seed": SEED,
            "population": "5 dialogue + 4 general + 3 temporary micro specialists",
            "expanded_population_size": len(expanded_ids),
            "specialist_steps_per_member": specialist_steps,
            "read_only": True,
            "oracle_uses_answer_targets": True,
            "language_bodies_frozen": True,
            "shared_embedding_frozen": True,
            "production_checkpoint_written": False,
            "default_loader_changed": False,
        },
        "data": {
            **data_info,
            "route_eval_rounds": len(eval_rounds),
        },
        "specialist_reports": specialist_reports,
        "oracle_audit": metrics,
    }
    del cortex, shared
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist-steps", type=int, default=DEFAULT_SPECIALIST_STEPS)
    parser.add_argument("--train-cap", type=int, default=ROUTE_TRAIN_SAMPLE_CAP)
    parser.add_argument("--eval-cap", type=int, default=ROUTE_EVAL_SAMPLE_CAP)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(
        specialist_steps=args.specialist_steps,
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
