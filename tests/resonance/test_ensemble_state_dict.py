"""P2-3: ResonanceEnsemble 聚合 state_dict 接口契约测试。

ResonanceEnsemble 非 nn.Module（历史设计），本文件验证聚合接口
state_dict()/load_state_dict() 的 round-trip、跳过语义与混合规格覆盖。
"""
import torch

from taiji.resonance import (
    ResonanceEnsemble,
    ResonanceField,
    ResonanceNeuron,
    get_domain_neuron_config,
)


def _make_ensemble(mixed_spec: bool = True):
    cfg_zh = get_domain_neuron_config("zh", spec="compact")  # field_dim=2048
    spec2 = "standard" if mixed_spec else "compact"  # standard field_dim=3072
    cfg_code = get_domain_neuron_config("code", spec=spec2)
    n_zh = ResonanceNeuron(cfg_zh)
    n_zh.config.neuron_id = "zh0"
    n_code = ResonanceNeuron(cfg_code)
    n_code.config.neuron_id = "code0"
    field = ResonanceField(dim=3072)  # unified = max(field_dim)
    ens = ResonanceEnsemble(
        neurons={"zh0": n_zh, "code0": n_code}, field=field, max_rounds=2)
    return ens, n_zh, n_code


def test_state_dict_roundtrip_field():
    """state_dict → load_state_dict round-trip：field.W_cond 数值精确一致。"""
    ens, _, _ = _make_ensemble(mixed_spec=False)
    torch.manual_seed(0)
    sd = ens.state_dict()
    assert "field" in sd, "state_dict 应含 field（W_cond R1 持久化端）"

    w_before = ens._field.W_cond.detach().clone()
    # 注意：标准 state_dict 语义下 sd 内张量为 live 引用，原地破坏会污染 sd；
    # 这里保存干净副本（等价于 torch.save/load 的落盘-回读路径）
    sd_clean = {
        "field": {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in sd["field"].items()
        }
    }
    ens._field.W_cond.data.uniform_(-1, 1)  # 破坏后再恢复
    skipped = ens.load_state_dict(sd_clean)
    assert skipped == [], f"round-trip 不应有跳过项: {skipped}"
    assert torch.equal(ens._field.W_cond, w_before), "round-trip 后 W_cond 应精确一致"


def test_state_dict_includes_cross_spec_mixed():
    """混合规格装配：state_dict 应含 per-nid 跨规格投影层。"""
    ens, _, _ = _make_ensemble(mixed_spec=True)
    sd = ens.state_dict()
    assert "cross_spec_projectors" in sd, "混合规格应有正向投影层"
    assert "cross_spec_back_projectors" in sd, "混合规格应有反向投影层"
    assert "zh0" in sd["cross_spec_projectors"], "compact(2048) 应有 2048→3072 投影"


def test_load_state_dict_skips_missing_module():
    """缺失模块（当前 ensemble 未创建）应跳过并返回清单，不抛异常。"""
    ens, _, _ = _make_ensemble(mixed_spec=False)
    # field_score_proj 默认 None（score_dim=None 时），加载其状态应跳过
    fake_state = {"field_score_proj": {"weight": torch.ones(4, 4)}}
    skipped = ens.load_state_dict(fake_state)
    assert any("field_score_proj" in s for s in skipped), f"应跳过未创建模块: {skipped}"


def test_load_state_dict_skips_unknown_neuron():
    """未知 neuron 的投影状态应跳过（热插拔场景）。"""
    ens, _, _ = _make_ensemble(mixed_spec=True)
    fake_proj = next(iter(ens._cross_spec_projectors.values())).state_dict()
    fake_state = {"cross_spec_projectors": {"ghost": fake_proj}}
    skipped = ens.load_state_dict(fake_state)
    assert any("ghost" in s for s in skipped), f"应跳过未知 neuron: {skipped}"
