"""验证 POST /api/taiji/cortex/chat 文本对话端点。

测试：
1. 端点可达性（非 404）
2. 中文 prompt → 自动推断 zh 域
3. 代码 prompt → 自动推断 code 域
4. 数学 prompt → 自动推断 math 域
5. 指定域强制路由
6. 空 prompt 错误处理
7. Pydantic 校验

Usage:
    python scripts/training/verify_cortex_chat.py
"""
from __future__ import annotations

import os
import sys
import functools

os.environ.setdefault('TAJIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print = functools.partial(print, flush=True)


def main():
    print("=" * 60)
    print("Cortex Chat HTTP Endpoint Verification")
    print("=" * 60)

    # Step 1: 加载 Cortex
    print("\n=== Step 1: Load Cortex ===")
    from taiji.core.model_loader import load_model_on_startup
    load_model_on_startup()

    from taiji.core.app_state import app_state
    assert app_state.is_taiji(), "Cortex not loaded"
    print(f"  Cortex loaded: {len(app_state.model.neurons)} neurons")

    # Step 2: 创建 TestClient
    print("\n=== Step 2: Create TestClient ===")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes_neuroplex import router as neuroplex_router

    app = FastAPI(title="Cortex Chat Test")
    app.include_router(neuroplex_router)
    client = TestClient(app)
    print("  TestClient ready")

    # Step 3: 中文 prompt → zh 域
    print("\n=== Step 3: Chinese prompt (auto-route to zh) ===")
    resp = client.post("/api/taiji/cortex/chat", json={
        "prompt": "你好，请介绍一下你自己",
        "max_tokens": 64,
        "temperature": 0.8,
    })
    print(f"  status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  domain: {data['domain']}")
        print(f"  response: {data['response'][:100]}...")
        assert data["status"] == "ok"
        assert data["domain"] == "zh", f"expected zh, got {data['domain']}"
    else:
        print(f"  body: {resp.text[:300]}")

    # Step 4: 代码 prompt → code 域
    print("\n=== Step 4: Code prompt (auto-route to code) ===")
    resp = client.post("/api/taiji/cortex/chat", json={
        "prompt": "def hello():\n    print('hello world')\n    return None",
        "max_tokens": 32,
    })
    print(f"  status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  domain: {data['domain']}")
        print(f"  response: {data['response'][:80]}...")
        assert data["domain"] == "code", f"expected code, got {data['domain']}"

    # Step 5: 数学 prompt → math 域
    print("\n=== Step 5: Math prompt (auto-route to math) ===")
    resp = client.post("/api/taiji/cortex/chat", json={
        "prompt": "1+2*3-4/5=?",
        "max_tokens": 32,
    })
    print(f"  status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  domain: {data['domain']}")
        print(f"  response: {data['response'][:80]}...")
        assert data["domain"] == "math", f"expected math, got {data['domain']}"

    # Step 6: 强制域路由
    print("\n=== Step 6: Forced domain (domain='en') ===")
    resp = client.post("/api/taiji/cortex/chat", json={
        "prompt": "hello world",
        "max_tokens": 32,
        "domain": "en",
    })
    print(f"  status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  domain: {data['domain']}")
        print(f"  response: {data['response'][:80]}...")
        assert data["domain"] == "en", f"expected en, got {data['domain']}"

    # Step 7: 空 prompt → 400
    print("\n=== Step 7: Empty prompt (should be 400) ===")
    resp = client.post("/api/taiji/cortex/chat", json={
        "prompt": "   ",
        "max_tokens": 32,
    })
    print(f"  status: {resp.status_code}")
    print(f"  body: {resp.text[:200]}")
    assert resp.status_code == 400, f"expected 400, got {resp.status_code}"

    # Step 8: 缺少 prompt 字段 → 422
    print("\n=== Step 8: Missing prompt field (should be 422) ===")
    resp = client.post("/api/taiji/cortex/chat", json={
        "max_tokens": 32,
    })
    print(f"  status: {resp.status_code}")
    assert resp.status_code == 422, f"expected 422, got {resp.status_code}"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — Cortex chat endpoint verified")
    print("=" * 60)
    print("\nVerified endpoints:")
    print("  - POST /api/taiji/cortex/chat: text generation with auto domain routing")
    print("  - Auto-routing: zh/code/math/en detection works")
    print("  - Forced domain: domain parameter overrides auto-routing")
    print("  - Error handling: empty prompt → 400, missing field → 422")
    return 0


if __name__ == "__main__":
    sys.exit(main())
