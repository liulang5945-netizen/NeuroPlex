"""Compatibility routes for clients that still call the former model-upgrade API.

The active runtime is a population resonance network. These routes keep legacy
URLs readable while directing clients to population growth.
"""

from fastapi import APIRouter, HTTPException

from neuroplex.core.app_state import app_state

router = APIRouter()


def _population_status() -> dict:
    model = app_state.model
    neurons = getattr(model, "neurons", {}) if model is not None else {}
    field = getattr(model, "field", None) if model is not None else None
    ensemble = getattr(model, "ensemble", None) if model is not None else None
    active_nids = list(neurons.keys())
    if ensemble is not None:
        active_nids = list(getattr(ensemble, "neurons", neurons).keys())

    return {
        "status": "active" if app_state.is_taiji() else "inactive",
        "architecture": "population_resonance",
        "neuron_count": len(neurons),
        "active_neurons": active_nids,
        "field_dim": getattr(field, "dim", None),
        "routing": "sparse_population",
        "growth_strategy": "neurogenesis_specialization",
    }


@router.get("/api/taiji_model/status")
def get_taiji_model_status():
    """Return population status under the legacy URL."""
    return _population_status()


@router.get("/api/taiji_model/upgrade_check")
def check_upgrade():
    """Explain that capability grows through the neuron population."""
    status = _population_status()
    return {
        **status,
        "can_upgrade": False,
        "recommended_action": "population_growth",
        "message": (
            "NeuroPlex 通过神经元专业化、新增成员、同伴协作训练或"
            "生命周期管理扩展群体能力。"
        ),
    }


@router.get("/api/taiji_model/capability")
def get_capability():
    """Expose population-level capability facts for legacy clients."""
    status = _population_status()
    return {
        "status": status["status"],
        "architecture": status["architecture"],
        "capability": {
            "neuron_count": status["neuron_count"],
            "active_neurons": status["active_neurons"],
            "routing": status["routing"],
        },
        "bottleneck": {
            "strategy": "population_growth",
            "message": "通过群体评估决定新增、专业化、隔离或修剪神经元。",
        },
    }


@router.post("/api/taiji_model/upgrade")
def start_upgrade():
    """Reject the deprecated upgrade operation explicitly."""
    raise HTTPException(
        status_code=410,
        detail=(
            "旧升级接口已退役。请使用 Cortex 的神经元群体增长、"
            "协作训练和生命周期接口。"
        ),
    )


@router.get("/api/taiji_model/upgrade_progress")
def get_upgrade_progress():
    """Return a terminal compatibility state for old polling clients."""
    return {
        "state": "deprecated",
        "progress": 0,
        "message": "旧升级接口已退役，当前采用群体神经元网络。",
        "recommended_action": "population_growth",
    }
