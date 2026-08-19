"""临时有界 route head + projected-NLL 监督试验。

本试验只改变内存中的 route head：

* 输入先经过 LayerNorm，消除不同神经元表征的尺度差；
* 输出经过 ``bound * tanh`` 限幅，避免 quality-logit 膨胀；
* 监督目标是同一回合上各成员 general-space projected NLL 的 softmax；
* 语言主体、embed_adapter、field、跨规格投影和 shared embedding 全冻结。

它不修改 ResonanceNeuron 的生产实现，不保存任何 production checkpoint。
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

from scripts.training.diag_micro_route_fusion_pilot import (
    DEFAULT_SPECIALIST_STEPS,
    ROUTE_EVAL_SAMPLE_CAP,
    ROUTE_SEQ_LEN,
    ROUTE_TRAIN_SAMPLE_CAP,
    SEED,
    _forward_batch,
    _load_route_rounds,
    _masked_teacher_forcing_nll,
    _prepare_population,
    _projected_logits,
    _route_snapshot,
    _freeze_to_quality_heads,
)


ROUTE_HEAD_BOUND = 2.0
ROUTE_HEAD_HIDDEN = 128
ROUTE_HEAD_LR = 2e-3
ROUTE_TARGET_TEMPERATURE = 2.0
ROUTE_PREDICT_TEMPERATURE = 1.0
DEFAULT_ROUTE_STEPS = 80


class BoundedRouteHead(nn.Module):
    """输入归一化、输出有界的临时质量路由头。"""

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

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.bound * torch.tanh(self.mlp(self.norm(pooled)))


def _install_bounded_heads(cortex) -> list[torch.nn.Parameter]:
    for neuron in cortex.ensemble.neurons.values():
        input_dim = neuron.config.hidden_size * 2
        neuron.quality_head = BoundedRouteHead(input_dim)
    return _freeze_to_quality_heads(cortex)


def _projected_member_nll(result, cortex, targets, answer_mask):
    projected = _projected_logits(cortex, result)
    nlls = torch.stack([
        _masked_teacher_forcing_nll(member_logits, targets, answer_mask)
        for member_logits in projected
    ])
    return projected, nlls


def _route_loss(cortex, result, targets, answer_mask):
    quality_logits = result.get("quality_logits")
    if quality_logits is None:
        raise RuntimeError("bounded route head did not return quality logits")
    _projected, member_nll = _projected_member_nll(
        result, cortex, targets, answer_mask
    )
    ideal = F.softmax(-member_nll.detach() / ROUTE_TARGET_TEMPERATURE, dim=0)
    predicted_log_probs = F.log_softmax(
        quality_logits / ROUTE_PREDICT_TEMPERATURE, dim=0
    )
    loss = -(ideal * predicted_log_probs).sum()
    return loss, member_nll.detach(), ideal.detach()


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
    if not train_rounds or not eval_rounds:
        raise RuntimeError("bounded route head train/eval rounds are empty")

    production_before = _route_snapshot(cortex, eval_rounds, general_sp)
    trainable = _install_bounded_heads(cortex)
    bounded_before = _route_snapshot(cortex, eval_rounds, general_sp)
    optimizer = torch.optim.AdamW(
        trainable, lr=ROUTE_HEAD_LR, weight_decay=0.01
    )
    generator = torch.Generator().manual_seed(SEED + 19)
    history = []
    for step in range(1, route_steps + 1):
        index = int(torch.randint(0, len(train_rounds), (1,), generator=generator))
        result, targets, answer_mask = _forward_batch(
            cortex, [train_rounds[index]], general_sp
        )
        loss, member_nll, ideal = _route_loss(
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
                "quality_min": round(float(result["quality_logits"].detach().min()), 6),
                "quality_max": round(float(result["quality_logits"].detach().max()), 6),
            })
            print(
                f"[bounded-route] step {step}/{route_steps}: "
                f"loss={history[-1]['loss']:.4f} "
                f"quality=[{history[-1]['quality_min']:.3f},"
                f"{history[-1]['quality_max']:.3f}]",
                flush=True,
            )
        del result

    bounded_after = _route_snapshot(cortex, eval_rounds, general_sp)
    raw_nll = production_before["hard_route_teacher_forcing_nll"]
    after_nll = bounded_after["hard_route_teacher_forcing_nll"]
    report = {
        "contract": {
            "seed": SEED,
            "population": "5 dialogue + 4 general + 3 temporary micro specialists",
            "expanded_population_size": len(expanded_ids),
            "specialist_steps_per_member": specialist_steps,
            "route_steps": route_steps,
            "route_head": "LayerNorm -> Linear/GELU/Linear -> 2*tanh",
            "route_head_bound": ROUTE_HEAD_BOUND,
            "route_target": "softmax(-projected_member_nll / 2.0)",
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
            "route_seq_len": ROUTE_SEQ_LEN,
        },
        "specialist_reports": specialist_reports,
        "production_before": production_before,
        "bounded_before": bounded_before,
        "bounded_after": bounded_after,
        "delta": {
            "bounded_after_minus_production_nll": round(after_nll - raw_nll, 6),
            "bounded_after_over_production_ppl_ratio": round(
                math.exp(min(after_nll - raw_nll, 20)), 6
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
