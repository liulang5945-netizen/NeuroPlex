"""阶段 4 产品接入冒烟测试：Seed 原生运行时的聊天/切换/健康检查分支。

不加载真实 800K 检查点——用默认小配置 Seed 注入运行时单例，验证
路由分支、SSE 事件协议与健康状态字段。
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def seed_client(monkeypatch):
    from api.app import create_app
    import api.seed_runtime as seed_runtime
    from api.seed_runtime import SeedRuntime
    from seed import Seed

    runtime = SeedRuntime(Seed(episode_id="smoke"))
    monkeypatch.setattr(seed_runtime, "_runtime", runtime)

    # 健康检查只在启动完成后返回完整负载，测试里直接标记就绪。
    from neuroplex.core.app_state import app_state
    monkeypatch.setattr(app_state, "startup_complete", True)
    monkeypatch.setattr(app_state, "startup_error", None)

    app = create_app(startup_tasks=False)
    client = TestClient(app)
    yield client
    monkeypatch.setattr(seed_runtime, "_runtime", None)


def test_health_reports_seed_active(seed_client):
    response = seed_client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["seed_active"] is True
    assert payload["model_loaded"] is True


def test_chat_stream_uses_seed_branch(seed_client):
    response = seed_client.post(
        "/api/chat/stream",
        json={"prompt": "你好", "history": []},
    )
    assert response.status_code == 200
    body = response.text
    # SSE 协议：final 事件 + 结束标记
    assert '"type": "final"' in body or '"type":"final"' in body
    assert "[DONE]" in body


def test_chat_stream_with_history(seed_client):
    response = seed_client.post(
        "/api/chat/stream",
        json={"prompt": "再见", "history": [["你好", "你好，很高兴见到你。"]]},
    )
    assert response.status_code == 200
    assert "[DONE]" in response.text


def test_switch_model_rejects_unknown_type():
    from api.app import create_app

    client = TestClient(create_app(startup_tasks=False))
    response = client.post(
        "/api/system/switch_model", json={"model_type": "unknown"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert "cortex" in payload["message"] and "seed" in payload["message"]
