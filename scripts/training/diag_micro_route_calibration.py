"""只读的 9+3 route calibration 审计。

上一轮已经证明三个 6.97M 小专家本体能学习，但 quality_head 的成员间偏置和
尺度让真实硬路由长期 100% 选择 ``zh``。本脚本不训练任何参数，而是：

1. 重建同一临时 9+3 群体；
2. 在固定 calibration 回合上统计每个成员的 quality-logit 均值/尺度；
3. 对每个留出样本把 quality-logit 做成员内 zero-centering、单位尺度化并
   用 ``2*tanh(z/2)`` 限幅；
4. 将得到的 trust override 送入真实 hard route，与原始 no-op 对照。

它只回答“消除成员间尺度偏置后，路由是否能脱离 zh 独占并改善留出 NLL”，
不修改 quality_head、语言主体、field、embed_adapter、shared embedding 或
任何生产 checkpoint。
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
)


CALIBRATION_CAP = 8
CALIBRATION_BOUND = 2.0
CALIBRATION_TEMPERATURE = 1.0


def _collect_quality_logits(cortex, rounds, general_sp) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for index in range(len(rounds)):
            result, _targets, _answer_mask = _forward_batch(
                cortex, rounds[index:index + 1], general_sp
            )
            quality = result.get("quality_logits")
            if quality is None:
                raise RuntimeError("quality logits missing during calibration")
            rows.append(quality.detach().float().cpu())
            del result
    if not rows:
        raise RuntimeError("calibration set is empty")
    return torch.stack(rows)


def _calibrated_trust(
    quality_logits: torch.Tensor,
    calibration_mean: torch.Tensor,
    calibration_std: torch.Tensor,
) -> torch.Tensor:
    """成员内 zero-center + bounded trust，保持真实 hard route 不变。"""

    z = (quality_logits - calibration_mean) / calibration_std.clamp_min(1.0)
    bounded = CALIBRATION_BOUND * torch.tanh(z / CALIBRATION_BOUND)
    return F.softmax(bounded / CALIBRATION_TEMPERATURE, dim=0)


def _route_metrics(cortex, rounds, general_sp, calibration_mean, calibration_std):
    member_ids = list(cortex.ensemble.neurons.keys())
    raw_nlls = []
    calibrated_nlls = []
    raw_weights = torch.zeros(len(member_ids))
    calibrated_weights = torch.zeros(len(member_ids))
    raw_quality = []
    calibrated_route_wins = torch.zeros(len(member_ids), dtype=torch.long)
    with torch.no_grad():
        for index in range(len(rounds)):
            batch = rounds[index:index + 1]
            raw, targets, answer_mask = _forward_batch(cortex, batch, general_sp)
            raw_nlls.append(float(_masked_teacher_forcing_nll(
                raw["fused_logits"], targets, answer_mask
            )))
            raw_weights += raw["weights"].detach().float().cpu()
            raw_quality.append(raw["quality_logits"].detach().float().cpu())

            trust = _calibrated_trust(
                raw["quality_logits"].detach().float().cpu(),
                calibration_mean,
                calibration_std,
            )
            calibrated, targets_cal, answer_mask_cal = _forward_batch(
                cortex, batch, general_sp, trust_override=trust
            )
            calibrated_nlls.append(float(_masked_teacher_forcing_nll(
                calibrated["fused_logits"], targets_cal, answer_mask_cal
            )))
            weights = calibrated["weights"].detach().float().cpu()
            calibrated_weights += weights
            calibrated_route_wins += (weights > 0).long()
            del raw, calibrated

    count = max(len(rounds), 1)
    raw_mean = sum(raw_nlls) / count
    calibrated_mean = sum(calibrated_nlls) / count
    raw_quality_tensor = torch.stack(raw_quality)
    return {
        "samples": len(rounds),
        "raw_hard_route_teacher_forcing_nll": round(raw_mean, 6),
        "calibrated_hard_route_teacher_forcing_nll": round(calibrated_mean, 6),
        "raw_hard_route_ppl": round(math.exp(min(raw_mean, 20)), 4),
        "calibrated_hard_route_ppl": round(math.exp(min(calibrated_mean, 20)), 4),
        "nll_delta_calibrated_minus_raw": round(calibrated_mean - raw_mean, 6),
        "ppl_ratio_calibrated_over_raw": round(
            math.exp(min(calibrated_mean - raw_mean, 20)), 6
        ),
        "raw_mean_weights": {
            nid: round(float(raw_weights[i] / count), 6)
            for i, nid in enumerate(member_ids)
        },
        "calibrated_mean_weights": {
            nid: round(float(calibrated_weights[i] / count), 6)
            for i, nid in enumerate(member_ids)
        },
        "calibrated_route_win_counts": {
            nid: int(calibrated_route_wins[i])
            for i, nid in enumerate(member_ids)
        },
        "raw_quality_logits_mean": {
            nid: round(float(raw_quality_tensor[:, i].mean()), 6)
            for i, nid in enumerate(member_ids)
        },
    }


def run(
    specialist_steps: int = DEFAULT_SPECIALIST_STEPS,
    train_cap: int = ROUTE_TRAIN_SAMPLE_CAP,
    eval_cap: int = ROUTE_EVAL_SAMPLE_CAP,
    calibration_cap: int = CALIBRATION_CAP,
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
    calibration_rounds = train_rounds[:calibration_cap]
    if not calibration_rounds or not eval_rounds:
        raise RuntimeError("calibration or eval rounds are empty")

    calibration_logits = _collect_quality_logits(
        cortex, calibration_rounds, general_sp
    )
    calibration_mean = calibration_logits.mean(dim=0)
    calibration_std = calibration_logits.std(dim=0, unbiased=False).clamp_min(1.0)
    eval_metrics = _route_metrics(
        cortex,
        eval_rounds,
        general_sp,
        calibration_mean,
        calibration_std,
    )
    member_ids = list(cortex.ensemble.neurons.keys())
    report = {
        "contract": {
            "seed": SEED,
            "population": "5 dialogue + 4 general + 3 temporary micro specialists",
            "expanded_population_size": len(expanded_ids),
            "specialist_steps_per_member": specialist_steps,
            "read_only": True,
            "quality_head_updated": False,
            "language_bodies_frozen": True,
            "field_and_cross_spec_fusion_updated": False,
            "shared_embedding_frozen": True,
            "production_checkpoint_written": False,
            "default_loader_changed": False,
            "calibration": "member-wise zero-center/unit-scale + 2*tanh(z/2)",
            "calibration_temperature": CALIBRATION_TEMPERATURE,
        },
        "data": {
            **data_info,
            "route_train_rounds": len(train_rounds),
            "calibration_rounds": len(calibration_rounds),
            "route_eval_rounds": len(eval_rounds),
        },
        "specialist_reports": specialist_reports,
        "quality_logit_calibration": {
            "member_ids": member_ids,
            "mean": {
                nid: round(float(calibration_mean[i]), 6)
                for i, nid in enumerate(member_ids)
            },
            "std": {
                nid: round(float(calibration_std[i]), 6)
                for i, nid in enumerate(member_ids)
            },
            "global_min": round(float(calibration_logits.min()), 6),
            "global_max": round(float(calibration_logits.max()), 6),
        },
        "eval": eval_metrics,
    }
    del cortex, shared
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist-steps", type=int, default=DEFAULT_SPECIALIST_STEPS)
    parser.add_argument("--train-cap", type=int, default=ROUTE_TRAIN_SAMPLE_CAP)
    parser.add_argument("--eval-cap", type=int, default=ROUTE_EVAL_SAMPLE_CAP)
    parser.add_argument("--calibration-cap", type=int, default=CALIBRATION_CAP)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(
        specialist_steps=args.specialist_steps,
        train_cap=args.train_cap,
        eval_cap=args.eval_cap,
        calibration_cap=args.calibration_cap,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
