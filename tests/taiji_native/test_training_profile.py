"""阶段 1 放大配置的失败测试：训练用放大画像必须存在且可构造。

计划要求：在 ``taiji/config.py`` 增加训练用放大配置（区域数/维度/边密度）。
放大画像不是新的动力学——它只把基底做大，全部校验规则与默认画像共用，
因此任何画像都必须原地通过 ``TaijiConfig`` 的全部 ``__post_init__`` 约束。
"""

from taiji import Taiji, TaijiConfig


def test_training_profile_scales_regions_dimensions_and_edge_density() -> None:
    profile = TaijiConfig.training_profile()
    default = TaijiConfig()

    assert profile.region_sizes > default.region_sizes
    assert profile.synapse_fan_in > default.synapse_fan_in
    assert profile.motor_fan_in > default.motor_fan_in
    assert profile.memory_units > default.memory_units
    assert profile.memory_fan_in > default.memory_fan_in
    assert profile.cortical_context_dim > default.cortical_context_dim


def test_training_profile_builds_and_steps() -> None:
    model = Taiji(TaijiConfig.training_profile(seed=311))
    assert model.parameter_count() > Taiji(TaijiConfig(seed=311)).parameter_count()

    model.observe(97, learn=True)
    step = model.observe(98, learn=True)
    assert step.probabilities.shape == (model.config.alphabet_size,)


def test_training_profile_rejects_non_positive_scale() -> None:
    try:
        TaijiConfig.training_profile(scale=0)
    except ValueError:
        pass
    else:
        raise AssertionError("scale must be positive")
