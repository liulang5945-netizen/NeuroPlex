"""S12 多轮对话状态管理 smoke test.

验证 DialogueState + field save/load round_state 的正确性：
1. field.save_round_state / load_round_state round-trip
2. DialogueState 多轮 start/end round 的 field_state 持久化
3. max_rounds 滑动窗口（旧轮次被丢弃）
4. round_token 前缀插入（第 2 轮及以后）
5. reset 清空状态
6. 序列化/反序列化
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.resonance.field import ResonanceField
from taiji.resonance.dialogue_state import DialogueState


def test_field_round_state_roundtrip():
    """[1] field.save_round_state / load_round_state round-trip。"""
    print("\n[1] field round_state round-trip")
    field = ResonanceField(dim=256)
    field.reset(batch_size=1)

    # 写入一些状态
    vec1 = torch.randn(256)
    vec2 = torch.randn(256)
    field.write("n0", vec1)
    field.write("n1", vec2)

    state_before = field.state.clone()
    mask_before = field.inhibitory_mask.clone()

    # 保存
    snapshot = field.save_round_state()
    assert torch.equal(snapshot["state"], state_before), "保存的 state 应匹配"
    assert torch.equal(snapshot["inhibitory_mask"], mask_before), "保存的 mask 应匹配"
    assert len(snapshot["contributions"]) == 2, "应有 2 个 neuron 贡献"

    # 修改 field
    field.reset(batch_size=1)
    assert field.state.norm().item() < 1e-6, "reset 后 state 应为 0"

    # 加载
    field.load_round_state(snapshot)
    assert torch.equal(field.state, state_before), "加载后 state 应匹配"
    assert torch.equal(field.inhibitory_mask, mask_before), "加载后 mask 应匹配"
    assert len(field._contributions) == 2, "应有 2 个 neuron 贡献恢复"

    print(f"  PASS: round-trip 完整恢复 state + mask + contributions")


def test_dialogue_state_multi_round():
    """[2] DialogueState 多轮 start/end round 的 field_state 持久化。"""
    print("\n[2] 多轮对话 field_state 持久化")
    field = ResonanceField(dim=256)
    dialogue = DialogueState(max_rounds=5)

    # 第 1 轮
    field.reset(batch_size=1)
    dialogue.start_round(field)  # 无历史，不加载
    assert dialogue.current_round == 0
    field.write("n0", torch.randn(256))
    state_round1 = field.state.clone()
    dialogue.end_round(field)  # 保存第 1 轮 state
    assert dialogue.n_rounds == 1

    # 第 2 轮
    field.reset(batch_size=1)  # 模拟新 forward 的 reset
    assert field.state.norm().item() < 1e-6, "reset 后 state 应为 0"
    dialogue.start_round(field)  # 加载第 1 轮 state
    assert dialogue.current_round == 1
    assert torch.equal(field.state, state_round1), "第 2 轮应加载第 1 轮的 state"

    print(f"  PASS: 第 2 轮正确加载第 1 轮的 field_state")


def test_max_rounds_sliding_window():
    """[3] max_rounds 滑动窗口（旧轮次被丢弃）。"""
    print("\n[3] 滑动窗口")
    field = ResonanceField(dim=256)
    dialogue = DialogueState(max_rounds=2)  # 只保留最近 2 轮

    # 模拟 3 轮对话
    for round_num in range(3):
        field.reset(batch_size=1)
        dialogue.start_round(field)
        field.write(f"n{round_num}", torch.randn(256) * (round_num + 1))
        dialogue.end_round(field)

    assert dialogue.n_rounds == 2, f"max_rounds=2 应只保留 2 轮, 实际 {dialogue.n_rounds}"
    assert dialogue.current_round == 2, f"当前应为第 3 轮 (index=2), 实际 {dialogue.current_round}"

    # 第 4 轮：加载的应是第 3 轮的 state（第 1 轮已被丢弃）
    field.reset(batch_size=1)
    # 先保存第 3 轮的 state 用于比较
    round3_state = dialogue._round_states[-1]["state"].clone()
    dialogue.start_round(field)
    assert torch.equal(field.state, round3_state), "第 4 轮应加载第 3 轮的 state"

    print(f"  PASS: 滑动窗口保留最近 {dialogue.max_rounds} 轮")


def test_round_token_prepend():
    """[4] round_token 前缀插入（第 2 轮及以后）。"""
    print("\n[4] round_token 前缀插入")
    dialogue = DialogueState(max_rounds=5, round_token_id=999)

    # 第 1 轮：不应插入 token
    dialogue.start_round()
    assert not dialogue.should_prepend_round_token(), "第 1 轮不应插入 round_token"
    ids = dialogue.prepend_round_token([1, 2, 3])
    assert ids == [1, 2, 3], "第 1 轮 ids 应不变"

    # 第 2 轮：应插入 token
    dialogue.end_round.__wrapped__ if hasattr(dialogue.end_round, '__wrapped__') else None
    # 模拟 end_round（需要 field，这里用 None 跳过）
    # 直接手动调用内部逻辑
    dialogue._round_states.append({"state": torch.zeros(1), "inhibitory_mask": torch.ones(1),
                                    "contributions": {}, "inhibit_contributions": {}, "batch_size": 1})
    dialogue.start_round()
    assert dialogue.should_prepend_round_token(), "第 2 轮应插入 round_token"
    ids = dialogue.prepend_round_token([1, 2, 3])
    assert ids == [999, 1, 2, 3], f"第 2 轮应插入 round_token=999, 实际 {ids}"

    print(f"  PASS: round_token 在第 2 轮及以后正确插入")


def test_reset():
    """[5] reset 清空状态。"""
    print("\n[5] reset 清空状态")
    field = ResonanceField(dim=256)
    dialogue = DialogueState(max_rounds=5)

    # 模拟 2 轮对话
    for i in range(2):
        field.reset(batch_size=1)
        dialogue.start_round(field)
        field.write(f"n{i}", torch.randn(256))
        dialogue.end_round(field)

    assert dialogue.n_rounds == 2
    assert dialogue.current_round == 1

    # reset
    dialogue.reset()
    assert dialogue.n_rounds == 0, "reset 后应无轮次"
    assert dialogue.current_round == -1, "reset 后 current_round 应为 -1"

    # 新会话第 1 轮：不应加载任何 state
    field.reset(batch_size=1)
    loaded = dialogue.start_round(field)
    assert loaded is None, "reset 后第 1 轮不应加载 state"

    print(f"  PASS: reset 清空所有状态")


def test_serialization():
    """[6] 序列化/反序列化。"""
    print("\n[6] 序列化/反序列化")
    field = ResonanceField(dim=256)
    dialogue = DialogueState(max_rounds=5, round_token_id=42)

    # 模拟 2 轮
    for i in range(2):
        field.reset(batch_size=1)
        dialogue.start_round(field)
        field.write(f"n{i}", torch.randn(256))
        dialogue.end_round(field)
    dialogue.add_dialogue_entry("user", "你好")
    dialogue.add_dialogue_entry("assistant", "你好！")

    # 序列化
    state = dialogue.get_state_dict()
    assert state["max_rounds"] == 5
    assert state["round_token_id"] == 42
    assert state["current_round"] == 1
    assert len(state["round_states"]) == 2
    assert len(state["dialogue_history"]) == 2

    # 反序列化到新实例
    dialogue2 = DialogueState()
    dialogue2.load_state_dict(state)
    assert dialogue2.max_rounds == 5
    assert dialogue2.round_token_id == 42
    assert dialogue2.current_round == 1
    assert dialogue2.n_rounds == 2
    assert len(dialogue2.get_dialogue_history()) == 2

    print(f"  PASS: 序列化/反序列化完整恢复")


def main():
    print("=" * 70)
    print("S12 多轮对话状态管理 smoke test")
    print("=" * 70)

    test_field_round_state_roundtrip()
    test_dialogue_state_multi_round()
    test_max_rounds_sliding_window()
    test_round_token_prepend()
    test_reset()
    test_serialization()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
