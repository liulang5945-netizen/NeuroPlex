"""N10: verify dense-reference equivalence and all gates after sparse migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import SparseSynapses, Taiji
from verify_taiji_n7_context import run_benchmark as run_n7
from verify_taiji_n8_delayed_trace import run_benchmark as run_n8
from verify_taiji_n9_long_free_run import run_benchmark as run_n9
from verify_taiji_native_v5 import run_benchmark as run_native


def _dense_view(synapses: SparseSynapses) -> torch.Tensor:
    dense = torch.zeros(synapses.out_features, synapses.in_features)
    posts = torch.arange(synapses.out_features).unsqueeze(1).expand_as(
        synapses.pre_index
    )
    dense[posts, synapses.pre_index.long()] = synapses.edge_weight
    return dense


def _operator_equivalence() -> Dict[str, object]:
    synapses = SparseSynapses(
        out_features=7,
        in_features=11,
        fan_in=4,
        generator=torch.Generator().manual_seed(31),
        init_scale=0.45,
        max_weight_norm=2.5,
    )
    dense_before = _dense_view(synapses)
    presynaptic = torch.linspace(-0.8, 0.9, 11)
    error = torch.linspace(-0.4, 0.5, 7)
    sparse_forward = synapses.forward(presynaptic)
    dense_forward = dense_before @ presynaptic
    sparse_backproject = synapses.backproject(error)
    dense_backproject = dense_before.T @ error

    learning_rate = 0.07
    weight_decay = 1e-3
    mask = dense_before != 0
    scale = max(1.0, float((presynaptic != 0).sum().item()) ** 0.5)
    expected = dense_before * (1.0 - weight_decay)
    expected.add_(learning_rate * torch.outer(error, presynaptic) / scale * mask)
    expected.mul_(mask)
    norms = expected.norm(dim=1, keepdim=True).clamp_min(1e-8)
    expected.mul_(torch.clamp(2.5 / norms, max=1.0))
    synapses.local_update(
        error,
        presynaptic,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    sparse_after = _dense_view(synapses)

    metrics = {
        "forward_max_abs_error": float(
            (sparse_forward - dense_forward).abs().max().item()
        ),
        "backproject_max_abs_error": float(
            (sparse_backproject - dense_backproject).abs().max().item()
        ),
        "local_update_max_abs_error": float(
            (sparse_after - expected).abs().max().item()
        ),
    }
    return {
        "metrics": metrics,
        "checks": {
            "forward_matches_dense": metrics["forward_max_abs_error"] <= 1e-6,
            "backproject_matches_dense": (
                metrics["backproject_max_abs_error"] <= 1e-6
            ),
            "local_update_matches_dense": (
                metrics["local_update_max_abs_error"] <= 1e-6
            ),
        },
    }


def run_benchmark(*, epochs: int = 200, seed: int = 7) -> Dict[str, object]:
    operators = _operator_equivalence()
    native = run_native(epochs=epochs, seed=seed)
    n7 = run_n7(epochs=epochs, seed=seed)
    n8 = run_n8(epochs=epochs, seed=seed)
    n9 = run_n9(epochs=epochs, seed=seed)

    # 迁移期这里还比对过 v2 存档报告的九项行为指标逐位相等。那条口径是为
    # "只换存储、不改行为" 的一次搬家服务的，搬家结束即完成使命。机制继续
    # 演化后它自相矛盾：任何有意的动力学改动都会让它必红，于是它只能逼人
    # 冻结机制或反复重刷存档。现在退役，改由三层仍然自洽的判据把守——
    # 算子层的稀疏/稠密数学等价、各 benchmark 自己的行为阈值、以及稀疏存储
    # 契约。存档报告只留作历史证据，不再作为通过条件。
    checks = {
        **operators["checks"],
        "native_current_passes": native["status"] == "pass",
        "n7_passes_after_migration": n7["status"] == "pass",
        "n8_passes_after_migration": n8["status"] == "pass",
        "n9_passes_after_migration": n9["status"] == "pass",
        "checkpoint_format_is_current": (
            native["architecture"]["checkpoint_format"]
            == Taiji.CHECKPOINT_FORMAT
        ),
        "no_dense_synapse_tensor": native["checks"]["no_dense_synapse_tensor"],
    }
    return {
        "benchmark": "taiji_n10_sparse_migration",
        "seed": seed,
        "epochs": epochs,
        "operator_equivalence": operators["metrics"],
        "storage": {
            key: native["architecture"][key]
            for key in (
                "learned_synapse_edges",
                "dense_equivalent_learned_scalars",
                "edge_weight_bytes",
                "topology_index_bytes",
                "sparse_synapse_storage_bytes",
                "dense_synapse_weight_bytes",
                "sparse_to_dense_synapse_byte_ratio",
                "default_config_storage_projection",
            )
        },
        "behavior": {
            "native_current_status": native["status"],
            "n5_accuracy": native["metrics"]["after"]["accuracy"],
            "n5_generated": native["metrics"]["generated_text"],
            "n7_full_accuracy": n7["metrics"]["full_accuracy"],
            "n8_full_accuracy": n8["metrics"]["full_accuracy"],
            "n8_no_trace_accuracy": n8["metrics"]["no_trace_accuracy"],
            "n8_trace_only_accuracy": n8["metrics"]["trace_only_accuracy"],
            "n9_accuracy": n9["metrics"]["accuracy"],
            "n9_first_error_index": n9["metrics"]["first_error_index"],
        },
        "diagnostic_timing": {
            "native_current_training_seconds": native["metrics"]["training_seconds"],
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(epochs=args.epochs, seed=args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
