"""最小可复现群体基线。

该入口验证的是群体网络的工程闭环，不是语言模型质量：

* 用固定随机种子创建一组极小的独立神经元；
* 对同一批合成输入比较单神经元、稠密协作和稀疏协作；
* 记录路由命中、激活数量、场贡献和共振分；
* 通过临时 checkpoint round-trip 验证 Cortex 装配；
* 可选地验证 API 健康入口。

因此报告中的质量数字属于 ``synthetic_probe_only``，不能替代真实训练
checkpoint 上的 PPL、EMERGE 或生成质量评估。真实模型评估应在本基线稳定后
复用同一指标格式接入。

Usage:
    python scripts/verify_population_baseline.py
    python scripts/verify_population_baseline.py --output reports/population_baseline.json
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from neuroplex.brain.cortex import Cortex
from neuroplex.resonance import NeuronConfig, ResonanceEnsemble, ResonanceField, ResonanceNeuron


DEFAULT_SEED = 20260819
NEURON_IDS = ("probe_alpha", "probe_beta", "probe_gamma")
BASE_EMBED_DIM = 16
FIELD_DIM = 16
VOCAB_SIZE = 32
SEQ_LEN = 6
BATCH_SIZE = 4


def _probe_config(neuron_id: str) -> NeuronConfig:
    """Return a tiny but real Transformer neuron configuration."""
    return NeuronConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        base_embed_dim=BASE_EMBED_DIM,
        field_dim=FIELD_DIM,
        score_dim=8,
        spec="population_baseline_probe",
        neuron_id=neuron_id,
        refractory_cooldown=0,
    )


def _make_population(seed: int) -> Dict[str, ResonanceNeuron]:
    torch.manual_seed(seed)
    population: Dict[str, ResonanceNeuron] = {}
    for neuron_id in NEURON_IDS:
        neuron = ResonanceNeuron(_probe_config(neuron_id)).eval()
        population[neuron_id] = neuron
    return population


def _make_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    embeddings = torch.randn(
        BATCH_SIZE, SEQ_LEN, BASE_EMBED_DIM, generator=generator
    )
    targets = torch.randint(
        0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=generator
    )
    return embeddings, targets


def _clone_population(
    population: Dict[str, ResonanceNeuron],
) -> Dict[str, ResonanceNeuron]:
    """Clone via checkpoint state, avoiding runtime locks inside neurons."""
    cloned: Dict[str, ResonanceNeuron] = {}
    for neuron_id, neuron in population.items():
        replica = ResonanceNeuron(neuron.config).eval()
        replica.load_state_dict(neuron.state_dict(), strict=True)
        cloned[neuron_id] = replica
    return cloned


def _clone_field(field: ResonanceField) -> ResonanceField:
    """Clone field parameters/buffers without copying runtime bookkeeping."""
    cloned = ResonanceField(dim=field.dim)
    cloned.load_state_dict(field.state_dict(), strict=True)
    return cloned


def _token_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
    )
    return float(loss.item())


def _ppl(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    return float(value)


def _result_logits(result: Dict[str, Any]) -> torch.Tensor:
    weighted = result.get("weighted_logits")
    if weighted is not None:
        return weighted
    neuron_logits = result.get("neuron_logits") or result.get("round1_logits")
    if not neuron_logits:
        raise RuntimeError("population baseline produced no logits")
    return next(iter(neuron_logits.values()))


def _score_map(result: Dict[str, Any]) -> Dict[str, float]:
    return {nid: _as_float(score) for nid, score in result.get("final_scores", {}).items()}


def _field_contribution_norms(ensemble: ResonanceEnsemble) -> Dict[str, float]:
    # The task-local field is exposed through the public ``field`` property after
    # forward(). Contributions are intentionally read for observability only.
    contributions = getattr(ensemble.field, "_contributions", {})
    return {
        nid: float(value.detach().norm(dim=-1).mean().item())
        for nid, value in contributions.items()
    }


def _route_metrics(
    ensemble: ResonanceEnsemble,
    population_ids: tuple[str, ...],
) -> Dict[str, Any]:
    router_result = ensemble._last_router_result
    if router_result is None:
        return {
            "engaged": False,
            "average_active_round2": float(len(population_ids)),
            "selected_counts": {nid: 0 for nid in population_ids},
            "top_k_ids": [],
        }

    active_ids = list(ensemble._router_active_ids or population_ids)
    selected_counts = {nid: 0 for nid in population_ids}
    top_k_ids = []
    for row in router_result.get("top_k_ids", []):
        selected = []
        for index in row:
            if 0 <= index < len(active_ids):
                nid = active_ids[index]
                selected.append(nid)
                selected_counts[nid] = selected_counts.get(nid, 0) + 1
        top_k_ids.append(selected)

    k_per_sample = router_result.get("k_per_sample")
    average_active = (
        float(k_per_sample.float().mean().item())
        if isinstance(k_per_sample, torch.Tensor)
        else float(len(population_ids))
    )
    return {
        "engaged": True,
        "average_active_round2": average_active,
        "selected_counts": selected_counts,
        "top_k_ids": top_k_ids,
    }


def _run_ensemble(
    population: Dict[str, ResonanceNeuron],
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    *,
    sparse: bool,
    field: ResonanceField,
) -> Dict[str, Any]:
    ensemble = ResonanceEnsemble(
        population,
        field,
        max_rounds=2,
        use_sparse_router=sparse,
        sparse_router_top_k=1,
        sparse_router_warmup_steps=0,
    )
    with torch.inference_mode():
        result = ensemble.forward(
            shared_embeddings=embeddings,
            return_logits=True,
            active_filter=False,
            field_conditioning=True,
            fusion_mode="soft",
        )
    logits = _result_logits(result)
    loss = _token_loss(logits, targets)
    route = _route_metrics(ensemble, tuple(population.keys()))
    return {
        "loss": loss,
        "ppl": _ppl(loss),
        "n_rounds": int(result.get("n_rounds", 0)),
        "n_active_history": [int(v) for v in result.get("n_active_history", [])],
        "final_scores": _score_map(result),
        "field_contribution_norms": _field_contribution_norms(ensemble),
        "route": route,
    }


def _individual_metrics(
    population: Dict[str, ResonanceNeuron],
    embeddings: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    with torch.inference_mode():
        for neuron_id, neuron in population.items():
            loss = _token_loss(neuron(embeddings, return_logits=True)["logits"], targets)
            metrics[neuron_id] = {"loss": loss, "ppl": _ppl(loss)}
    return metrics


def _cortex_roundtrip(
    population: Dict[str, ResonanceNeuron],
    embeddings: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, Any]:
    """Round-trip tiny checkpoints in memory and exercise Cortex.think().

    The managed runner can execute source edits but denies runtime-created
    checkpoint files. BytesIO preserves the actual torch serialization contract
    while keeping the baseline self-contained and side-effect free.
    """
    restored: Dict[str, ResonanceNeuron] = {}
    for neuron_id, neuron in population.items():
        buffer = io.BytesIO()
        torch.save(
            {
                "neuron_config": neuron.config,
                "state_dict": neuron.state_dict(),
            },
            buffer,
        )
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
        replica = ResonanceNeuron(checkpoint["neuron_config"]).eval()
        replica.load_state_dict(checkpoint["state_dict"], strict=True)
        restored[neuron_id] = replica

    # Instantiate the real Cortex container without asking it to write or scan
    # any fixture files, then replace its empty runtime population with the
    # restored checkpoint set and rebuild the matching small ensemble.
    cortex = Cortex(
        neurons_dir="data/__population_baseline_missing__",
        device="cpu",
        max_rounds=2,
    )
    cortex.neurons = restored
    cortex.field = ResonanceField(dim=FIELD_DIM)
    cortex.ensemble = ResonanceEnsemble(
        cortex.neurons,
        cortex.field,
        max_rounds=2,
        coaction=cortex.coaction,
    )
    cortex.is_loaded = True
    with torch.inference_mode():
        result = cortex.think(
            shared_embeddings=embeddings,
            fusion_mode="soft",
            collab_mode="fusion",
        )
    logits = _result_logits(result)
    loss = _token_loss(logits, targets)
    return {
        "roundtrip_ok": set(cortex.neurons) == set(population),
        "loaded_neurons": list(cortex.neurons),
        "n_rounds": int(result.get("n_rounds", 0)),
        "loss": loss,
        "ppl": _ppl(loss),
    }


def _api_health_smoke() -> Dict[str, Any]:
    try:
        from fastapi.testclient import TestClient
        from api.app import create_app

        response = TestClient(create_app(startup_tasks=False)).get("/api/health")
        return {
            "available": True,
            "status_code": response.status_code,
            "status": response.json().get("status"),
            "ok": response.status_code == 200,
        }
    except Exception as exc:  # pragma: no cover - dependency-specific fallback
        return {
            "available": False,
            "status_code": None,
            "status": "skipped",
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _core_probe(seed: int, include_cortex: bool = True) -> Dict[str, Any]:
    population = _make_population(seed)
    embeddings, targets = _make_inputs(seed)
    individual = _individual_metrics(population, embeddings, targets)

    # Clone the same population and field for a fair dense/sparse comparison.
    dense_population = _clone_population(population)
    sparse_population = _clone_population(population)
    torch.manual_seed(seed + 2)
    dense_field = ResonanceField(dim=FIELD_DIM)
    sparse_field = _clone_field(dense_field)
    dense = _run_ensemble(
        dense_population, embeddings, targets,
        sparse=False, field=dense_field,
    )
    sparse = _run_ensemble(
        sparse_population, embeddings, targets,
        sparse=True, field=sparse_field,
    )

    result: Dict[str, Any] = {
        "individual": individual,
        "dense": dense,
        "sparse": sparse,
    }
    if include_cortex:
        result["cortex"] = _cortex_roundtrip(population, embeddings, targets)
    return result


def run_baseline(
    seed: int = DEFAULT_SEED,
    *,
    output: Optional[str] = None,
    include_api: bool = True,
) -> Dict[str, Any]:
    """Run the deterministic population baseline and return a JSON-safe report."""
    probe = _core_probe(seed, include_cortex=True)
    repeat = _core_probe(seed, include_cortex=False)

    dense = probe["dense"]
    sparse = probe["sparse"]
    deterministic = all(
        abs(probe[branch]["loss"] - repeat[branch]["loss"]) < 1e-7
        for branch in ("dense", "sparse")
    )
    sparse_route = sparse["route"]
    dense_active = len(NEURON_IDS)
    sparse_active = sparse_route["average_active_round2"]

    report: Dict[str, Any] = {
        "schema_version": 1,
        "architecture": "population_resonance",
        "quality_scope": "synthetic_probe_only",
        "seed": seed,
        "config": {
            "population": list(NEURON_IDS),
            "hidden_size": 32,
            "layers": 1,
            "vocab_size": VOCAB_SIZE,
            "base_embed_dim": BASE_EMBED_DIM,
            "field_dim": FIELD_DIM,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "sparse_top_k": 1,
        },
        "metrics": probe,
        "checks": {
            "deterministic": deterministic,
            "sparse_router_engaged": bool(sparse_route["engaged"]),
            "sparse_activation_reduced": sparse_active < dense_active,
            "cortex_roundtrip_ok": bool(probe["cortex"]["roundtrip_ok"]),
        },
    }
    if include_api:
        report["api_health"] = _api_health_smoke()
        report["checks"]["api_health_ok"] = bool(report["api_health"]["ok"])

    report["status"] = (
        "pass" if all(report["checks"].values()) else "fail"
    )
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible population baseline")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    report = run_baseline(seed=args.seed, output=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
