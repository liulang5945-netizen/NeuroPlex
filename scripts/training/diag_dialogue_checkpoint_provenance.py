#!/usr/bin/env python3
"""P1.2：审计五个 dialogue checkpoint 的保存状态与训练 provenance。

只读取 checkpoint 元数据（tensor 映射到 meta，不分配大权重）和对应 fine-tune 日志，核对：

* checkpoint 当前 step 与日志最后一次评估；
* ``best_val_ppl`` / ``best_step`` 是否只是历史指标；
* optimizer/scheduler 参数是否与训练入口一致；
* 五个 checkpoint 是否都有 per-neuron shared embedding 和一致结构。

运行：
    python -X utf8 -u scripts/training/diag_dialogue_checkpoint_provenance.py
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


DIALOGUE_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
    "zh_aug3_dialogue", "zh_std0_dialogue",
]
CHECKPOINT_DIR = "data/neurons"
LOG_DIR = "logs"
EVAL_RE = re.compile(r"\[EVAL\] step (\d+): val PPL=([0-9.]+)")


def _latest_log(neuron_id: str) -> str | None:
    paths = glob.glob(os.path.join(LOG_DIR, f"finetune_dialogue_{neuron_id}_*.log"))
    return max(paths, key=os.path.getmtime) if paths else None


def _log_metrics(path: str | None) -> dict:
    if path is None:
        return {"path": None, "evals": []}
    text = open(path, "r", encoding="utf-8").read()
    evals = [
        {"step": int(step), "ppl": float(ppl)}
        for step, ppl in EVAL_RE.findall(text)
    ]
    best = min(evals, key=lambda item: item["ppl"]) if evals else None
    return {
        "path": path,
        "eval_count": len(evals),
        "first_eval": evals[0] if evals else None,
        "last_eval": evals[-1] if evals else None,
        "log_best": best,
    }


def _checkpoint(neuron_id: str) -> dict:
    # Import registers historical pickle aliases used by these checkpoints.
    from neuroplex.loader import assemble_cortex  # noqa: F401

    path = os.path.join(CHECKPOINT_DIR, f"neuron_{neuron_id}.pt")
    ckpt = torch.load(path, map_location="meta", weights_only=False)
    result = ckpt.get("result") or {}
    optimizer = ckpt.get("optimizer_state") or {}
    scheduler = ckpt.get("scheduler_state") or {}
    param_groups = optimizer.get("param_groups") or []
    optimizer_group_summary = [
        {
            key: group.get(key)
            for key in ("lr", "initial_lr", "weight_decay", "betas", "eps")
            if key in group
        }
        for group in param_groups
    ]
    shared = ckpt.get("shared_embedding_state") or {}
    shared_shapes = {
        key: list(value.shape) if torch.is_tensor(value) else type(value).__name__
        for key, value in shared.items()
    }
    body = ckpt.get("state_dict") or {}
    body_shapes = {
        key: list(value.shape) if torch.is_tensor(value) else type(value).__name__
        for key, value in list(body.items())[:5]
    }
    log = _log_metrics(_latest_log(neuron_id))
    checkpoint = {
        "file_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
        "domain": ckpt.get("domain"),
        "data_source": ckpt.get("data_source"),
        "has_shared_embedding": bool(shared),
        "shared_embedding_shapes": shared_shapes,
        "state_dict_sample_shapes": body_shapes,
        "optimizer_state_entries": len(optimizer.get("state", {})),
        "optimizer_groups": optimizer_group_summary,
        "scheduler": {
            key: scheduler.get(key)
            for key in ("last_epoch", "_step_count", "base_lrs", "last_lr")
            if key in scheduler
        },
        "result": {
            key: result.get(key)
            for key in ("steps", "best_step", "best_val_ppl", "base_id", "finetune")
        },
        "log": log,
        "save_state_interpretation": "latest_state_at_last_save; best_val_ppl is historical metric",
    }
    del ckpt
    return checkpoint


def main() -> None:
    logging.disable(logging.CRITICAL)
    report = {
        "training_entry_contract": {
            "steps_default": 12000,
            "lr_default": 1e-4,
            "warmup_steps_default": 100,
            "eval_every_default": 500,
            "shared_embedding_trainable_default": True,
            "checkpoint_save_behavior": "save current state at every eval; no separate best snapshot",
        },
        "neurons": {neuron_id: _checkpoint(neuron_id) for neuron_id in DIALOGUE_IDS},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
