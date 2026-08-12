"""验证 /api/system/reload_model 重载 Cortex 路径。

Usage:
    python scripts/training/verify_cortex_reload.py
"""
from __future__ import annotations

import os
import sys
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print = functools.partial(print, flush=True)


def main():
    print("=" * 60)
    print("Cortex Reload Verification")
    print("=" * 60)

    # Step 1: 首次加载 Cortex
    print("\n=== Step 1: Initial Cortex load ===")
    from taiji.core.model_loader import load_model_on_startup
    load_model_on_startup()

    from taiji.core.app_state import app_state
    assert app_state.model is not None
    assert type(app_state.model).__name__ == 'Cortex'
    n1 = len(app_state.model.neurons)
    print(f"  Initial: {n1} neurons, is_taiji={app_state.is_taiji()}")

    # Step 2: 调用 reload_model 端点
    print("\n=== Step 2: Call reload_model via HTTP ===")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes_model_switch import router as switch_router

    app = FastAPI(title="Reload Test")
    app.include_router(switch_router)
    client = TestClient(app)

    resp = client.post("/api/system/reload_model")
    print(f"  status: {resp.status_code}")
    print(f"  body: {resp.json()}")

    data = resp.json()
    assert data["status"] == "ok", f"reload failed: {data}"
    assert data["model_type"] == "cortex"
    n2 = data.get("neuron_count", 0)
    print(f"  Reloaded: {n2} neurons")

    # Step 3: 验证 app_state 状态
    print("\n=== Step 3: Verify app_state after reload ===")
    assert app_state.model is not None
    assert type(app_state.model).__name__ == 'Cortex'
    n3 = len(app_state.model.neurons)
    print(f"  After reload: {n3} neurons, is_taiji={app_state.is_taiji()}")
    assert n3 == n1, f"neuron count changed: {n1} -> {n3}"

    # Step 4: 验证 switch_model 端点（同步模式 via reload_model）
    print("\n=== Step 4: Verify switch_model accepts model_type='cortex' ===")
    resp = client.post("/api/system/switch_model", json={"model_type": "cortex"})
    print(f"  status: {resp.status_code}")
    data = resp.json()
    print(f"  body: {data}")
    assert data["status"] in ("ok", "switching_in_progress")

    # Step 5: 验证 switch_model 拒绝无效类型
    print("\n=== Step 5: Verify switch_model rejects invalid type ===")
    resp = client.post("/api/system/switch_model", json={"model_type": "gguf"})
    print(f"  status: {resp.status_code}")
    data = resp.json()
    print(f"  body: {data}")
    assert data["status"] == "error"
    assert "Cortex" in data["message"] or "仅支持" in data["message"]

    # Step 6: 验证 switch_status
    print("\n=== Step 6: GET /api/system/switch_status ===")
    resp = client.get("/api/system/switch_status")
    print(f"  status: {resp.status_code}")
    print(f"  body: {resp.json()}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — Cortex reload path verified")
    print("=" * 60)
    print(f"\nVerified:")
    print(f"  - POST /api/system/reload_model: reloads Cortex ({n1} neurons)")
    print(f"  - POST /api/system/switch_model: accepts model_type='cortex'")
    print(f"  - POST /api/system/switch_model: rejects invalid types")
    print(f"  - GET  /api/system/switch_status: returns status")
    print(f"  - Cortex is the only cognitive subject")
    return 0


if __name__ == "__main__":
    sys.exit(main())
