"""T5: verify sequential retention in Taiji local fast memories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from neuroplex.taiji import EventKind, EventMode, TaijiConfig, TaijiRuntime


def _association_set(count: int, dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cues = F.normalize(torch.randn(count, dim, generator=generator), dim=-1)
    targets = F.normalize(torch.randn(count, dim, generator=generator), dim=-1) * 1.5
    return cues, targets


def _probe(
    runtime: TaijiRuntime,
    cue: torch.Tensor,
    target: torch.Tensor,
    *,
    preserve_memory: bool,
    episode_id: str,
) -> Dict[str, float]:
    runtime.reset_dynamics(
        preserve_fast_memory=preserve_memory,
        episode_id=episode_id,
    )
    event = runtime.make_event(cue, kind=EventKind.SENSORY, source="t5_probe")
    result = runtime.step([event])
    mse = float(torch.mean((result.output - target) ** 2).item())
    cosine = float(F.cosine_similarity(
        result.output.unsqueeze(0), target.unsqueeze(0)
    ).item())
    return {
        "mse": mse,
        "cosine": cosine,
        "active": len(result.active_cell_ids),
        "memory_confidence_max": max(result.memory_confidences.values()),
    }


def _memory_stats(runtime: TaijiRuntime) -> Dict[str, object]:
    state = runtime.snapshot()
    occupancy: Dict[str, int] = {}
    max_offdiag: Dict[str, float] = {}
    mean_offdiag: Dict[str, float] = {}
    for cell_id, cell in state.cells.items():
        active = cell.memory_usage > 0
        keys = cell.memory_keys[active]
        occupancy[cell_id] = int(active.sum().item())
        if keys.shape[0] < 2:
            max_offdiag[cell_id] = 0.0
            mean_offdiag[cell_id] = 0.0
            continue
        units = F.normalize(keys, dim=-1)
        cosine = units @ units.T
        mask = ~torch.eye(keys.shape[0], dtype=torch.bool)
        offdiag = cosine[mask]
        max_offdiag[cell_id] = float(offdiag.max().item())
        mean_offdiag[cell_id] = float(offdiag.mean().item())

    identical_memory_pairs: List[List[str]] = []
    cell_ids = list(state.cells)
    for left_index, left_id in enumerate(cell_ids):
        left = state.cells[left_id]
        if occupancy[left_id] == 0:
            continue
        for right_id in cell_ids[left_index + 1:]:
            right = state.cells[right_id]
            if (
                occupancy[left_id] == occupancy[right_id]
                and torch.equal(left.memory_keys, right.memory_keys)
                and torch.equal(left.memory_values, right.memory_values)
                and torch.equal(left.memory_usage, right.memory_usage)
            ):
                identical_memory_pairs.append([left_id, right_id])
    return {
        "occupancy": occupancy,
        "max_offdiag_cosine": max_offdiag,
        "mean_offdiag_cosine": mean_offdiag,
        "identical_memory_pairs": identical_memory_pairs,
    }


def run_benchmark(
    *,
    association_count: int = 20,
    seed: int = 20260821,
) -> Dict[str, object]:
    config = TaijiConfig(
        event_dim=32,
        state_dim=48,
        field_dim=64,
        dendritic_branches=4,
        fast_memory_slots=32,
        cell_ids=("cell_0", "cell_1", "cell_2"),
        active_budget=2,
        seed=seed,
    )
    cues, targets = _association_set(association_count, config.event_dim, seed + 1)
    runtime = TaijiRuntime(config, episode_id="t5-learn")
    initial_parameters = {
        name: value.detach().clone() for name, value in runtime.state_dict().items()
    }

    baseline_runtime = TaijiRuntime.from_checkpoint(runtime.checkpoint())
    baseline: List[Dict[str, float]] = []
    for index in range(association_count):
        baseline.append(_probe(
            baseline_runtime,
            cues[index],
            targets[index],
            preserve_memory=False,
            episode_id=f"baseline-{index}",
        ))

    activation_counts = {cell_id: 0 for cell_id in config.cell_ids}
    for index in range(association_count):
        runtime.reset_dynamics(
            preserve_fast_memory=True,
            episode_id=f"learn-{index}",
        )
        cue = runtime.make_event(
            cues[index], kind=EventKind.SENSORY, source="t5_cue"
        )
        outcome = runtime.make_event(
            targets[index],
            kind=EventKind.SENSORY,
            source="t5_environment",
            mode=EventMode.REAL,
            delay=1,
        )
        learned = runtime.learn_association(cue, outcome, reward=1.0)
        for cell_id in learned.active_cell_ids:
            activation_counts[cell_id] += 1

    parameters_unchanged = all(
        torch.equal(before, runtime.state_dict()[name])
        for name, before in initial_parameters.items()
    )
    memory_stats = _memory_stats(runtime)
    inactive_cell_ids = [
        cell_id for cell_id, count in activation_counts.items() if count == 0
    ]
    learned_checkpoint = runtime.checkpoint()

    retained: List[Dict[str, float]] = []
    retained_flags: List[bool] = []
    for index in range(association_count):
        metrics = _probe(
            runtime,
            cues[index],
            targets[index],
            preserve_memory=True,
            episode_id=f"retention-{index}",
        )
        retained.append(metrics)
        retained_flags.append(
            metrics["mse"] <= baseline[index]["mse"] * 0.70
            and metrics["cosine"] >= 0.80
        )

    first_count = max(1, association_count // 4)
    first_retention = sum(retained_flags[:first_count]) / first_count
    total_retention = sum(retained_flags) / association_count

    lesioned = TaijiRuntime.from_checkpoint(learned_checkpoint)
    lesioned.reset(episode_id="t5-memory-lesion")
    lesion_flags: List[bool] = []
    lesion_metrics: List[Dict[str, float]] = []
    for index in range(association_count):
        metrics = _probe(
            lesioned,
            cues[index],
            targets[index],
            preserve_memory=False,
            episode_id=f"lesion-{index}",
        )
        lesion_metrics.append(metrics)
        lesion_flags.append(
            metrics["mse"] <= baseline[index]["mse"] * 0.70
            and metrics["cosine"] >= 0.80
        )
    lesion_retention = sum(lesion_flags) / association_count

    checks = {
        "first_quarter_retention": first_retention >= 0.70,
        "overall_retention": total_retention >= 0.80,
        "memory_lesion_causal": lesion_retention <= total_retention - 0.50,
        "slow_parameters_unchanged": parameters_unchanged,
        "no_slot_overflow": all(
            occupied <= config.fast_memory_slots
            for occupied in memory_stats["occupancy"].values()
        ),
    }
    return {
        "benchmark": "taiji_t5_continual_memory",
        "seed": seed,
        "association_count": association_count,
        "first_quarter_count": first_count,
        "config": config.to_dict(),
        "metrics": {
            "first_quarter_retention": first_retention,
            "overall_retention": total_retention,
            "memory_lesion_retention": lesion_retention,
            "mean_baseline_mse": sum(item["mse"] for item in baseline) / association_count,
            "mean_retained_mse": sum(item["mse"] for item in retained) / association_count,
            "mean_lesion_mse": sum(item["mse"] for item in lesion_metrics) / association_count,
            "activation_counts": activation_counts,
            **memory_stats,
        },
        "diagnostics": {
            "all_cells_participated": not inactive_cell_ids,
            "inactive_cell_ids": inactive_cell_ids,
            "population_specialization_observed": (
                not inactive_cell_ids
                and not memory_stats["identical_memory_pairs"]
            ),
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(association_count=args.count, seed=args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
