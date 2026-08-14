"""真实 checkpoint 契约测试（slow）。

加载真实训练产物（GB 级），验证生产装配链路的关键契约：

- R1 契约：collab ckpt 持久化 field_w_cond（W_cond 训练闭环的落盘端）
- hub 规格契约：EXPERT + general 256K + field 4096（联合皮层设计决策 4）
- dialogue 装配契约：生产默认装配消费的 ckpt 可加载且含 body
- C26 记忆组件契约：write_gate / anchor_projector 产物结构

ckpt 不存在时 skip（CI 无 data/ 也能跑，本地跑需 data/ 目录）。
运行: python -m pytest tests/resonance/test_real_checkpoint_slow.py -v
"""
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"
NEURONS = DATA / "neurons"
SLEEP_DATA = PROJECT_ROOT / "taiji_data" / "sleep_data"


def _load_ckpt(path: Path):
    if not path.exists():
        pytest.skip(f"ckpt 不存在: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.slow
def test_collab_ckpt_field_w_cond_contract():
    """R1: W_cond 训练闭环的落盘契约——collab ckpt 必须持久化 field_w_cond。"""
    ck = _load_ckpt(NEURONS / "hub_collab_v2.ckpt.pt")
    assert "field_w_cond" in ck, "R1 契约破坏：collab ckpt 缺 field_w_cond"
    w = ck["field_w_cond"]
    assert w.dim() == 2 and w.shape[0] == w.shape[1], f"W_cond 应为方阵: {tuple(w.shape)}"
    assert torch.isfinite(w).all(), "W_cond 含 NaN/Inf"


@pytest.mark.slow
def test_hub_ckpt_spec_contract():
    """hub neuron 规格契约：EXPERT + general 256K + field 4096。"""
    ck = _load_ckpt(DATA / "hub_neuron" / "neuron_hub.pt")
    cfg = ck.get("neuron_config")
    assert cfg is not None, "hub ckpt 缺 neuron_config"
    assert cfg.spec == "expert", f"spec={cfg.spec}"
    assert cfg.hidden_size == 1024, f"hidden={cfg.hidden_size}"
    assert cfg.field_dim == 4096, f"field_dim={cfg.field_dim}"
    assert cfg.vocab_size == 256000, f"vocab={cfg.vocab_size}"
    assert ck.get("hub_neuron") is True, "缺 hub_neuron 标记"
    assert len(ck["state_dict"]) > 100, f"state_dict 条目异常: {len(ck['state_dict'])}"


@pytest.mark.slow
def test_dialogue_std0_ckpt_loadable():
    """生产默认装配的 dialogue neuron ckpt 可加载且含 body 与配置。"""
    ck = _load_ckpt(NEURONS / "neuron_zh_std0_dialogue.pt")
    assert "neuron_config" in ck and "state_dict" in ck, "dialogue ckpt 结构异常"
    sd = ck["state_dict"]
    assert "embed_adapter.weight" in sd, "缺 embed_adapter"
    body_keys = [k for k in sd if k.startswith("layers.")]
    assert len(body_keys) > 20, f"body 层数异常: {len(body_keys)}"
    # 生产装配（loader.py）依赖的 shared_embedding 副本必须存在
    assert "shared_embedding_state" in ck, "缺 shared_embedding_state（loader 装配依赖）"


def test_memory_component_products():
    """C26 记忆组件产物契约（轻量 ~4MB，非 slow）。"""
    gate = _load_ckpt(SLEEP_DATA / "write_gate.pt")
    anchor = _load_ckpt(SLEEP_DATA / "anchor_projector.pt")
    assert "state_dict" in gate and "in_dim" in gate, "write_gate 产物结构异常"
    assert (
        "state_dict" in anchor and "in_dim" in anchor and "proj_dim" in anchor
    ), "anchor_projector 产物结构异常"
    assert gate["in_dim"] == anchor["in_dim"], (
        f"gate/projector 输入维度不一致: {gate['in_dim']} vs {anchor['in_dim']}"
    )
