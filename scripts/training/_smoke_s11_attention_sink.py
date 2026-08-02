"""S11 attention sink + 滑动窗口 smoke test.

验证长上下文 KV cache 管理的正确性：
1. sink/window=0 时与原始行为一致（向后兼容，KV cache 无限增长）
2. sink+window 启用时，KV cache 超限后自动驱逐到 max_len
3. 驱逐后保留前 sink_size + 最近 window_size token
4. 训练时（无 kv_cache）不受 sink/window 影响
5. checkpoint 兼容：sink/window=0 的旧 ckpt 加载到新模型
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import torch
import torch.nn as nn

from taiji.layers import GroupedQueryAttention, TransformerBlock
from taiji.resonance.config import TINY_TEST, NeuronConfig
from taiji.resonance.neuron import ResonanceNeuron


def test_backward_compat_no_sink():
    """[1] sink/window=0 时与原始行为一致（KV cache 无限增长）。"""
    print("\n[1] sink/window=0 向后兼容性")
    torch.manual_seed(42)
    # sink/window=0（默认）
    attn_legacy = GroupedQueryAttention(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        attention_sink_size=0, sliding_window_size=0,
    )
    attn_legacy.eval()

    assert attn_legacy.kv_cache_max_len == 0, "sink/window=0 时 kv_cache_max_len 应为 0"

    x = torch.randn(1, 4, 256)
    kv_cache = None
    # 模拟 5 步推理，每步 1 个 token，KV cache 持续增长
    for step in range(5):
        with torch.no_grad():
            out, kv_cache = attn_legacy(x, kv_cache=kv_cache, use_cache=True)
        # KV cache 长度应等于 (step+1) * seqlen，无驱逐
        expected_len = (step + 1) * 4
        actual_len = kv_cache[0].shape[1]
        assert actual_len == expected_len, f"step={step}: KV cache 应为 {expected_len}, 实际 {actual_len}"

    print(f"  PASS: sink/window=0 时 KV cache 无限增长（5步后长度={kv_cache[0].shape[1]}）")


def test_kv_cache_eviction():
    """[2] sink+window 启用时，KV cache 超限后自动驱逐到 max_len。"""
    print("\n[2] KV cache 自动驱逐")
    torch.manual_seed(42)
    # sink=2, window=4 → max_len=6
    sink_size, window_size = 2, 4
    max_len = sink_size + window_size
    attn = GroupedQueryAttention(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        attention_sink_size=sink_size, sliding_window_size=window_size,
    )
    attn.eval()

    assert attn.kv_cache_max_len == max_len, f"kv_cache_max_len 应为 {max_len}"

    x = torch.randn(1, 2, 256)  # 每步 2 个 token
    kv_cache = None

    # step 0: +2 token → KV=2 (未超限)
    with torch.no_grad():
        out, kv_cache = attn(x, kv_cache=kv_cache, use_cache=True)
    assert kv_cache[0].shape[1] == 2, f"step 0: KV 应为 2, 实际 {kv_cache[0].shape[1]}"

    # step 1: +2 token → KV=4 (未超限)
    with torch.no_grad():
        out, kv_cache = attn(x, kv_cache=kv_cache, use_cache=True)
    assert kv_cache[0].shape[1] == 4, f"step 1: KV 应为 4, 实际 {kv_cache[0].shape[1]}"

    # step 2: +2 token → KV=6 (达到 max_len，未超限)
    with torch.no_grad():
        out, kv_cache = attn(x, kv_cache=kv_cache, use_cache=True)
    assert kv_cache[0].shape[1] == 6, f"step 2: KV 应为 6, 实际 {kv_cache[0].shape[1]}"

    # step 3: +2 token → KV=8 > 6，驱逐到 max_len=6
    with torch.no_grad():
        out, kv_cache = attn(x, kv_cache=kv_cache, use_cache=True)
    assert kv_cache[0].shape[1] == max_len, f"step 3: 驱逐后 KV 应为 {max_len}, 实际 {kv_cache[0].shape[1]}"

    # step 4: +2 token → 持续驱逐
    with torch.no_grad():
        out, kv_cache = attn(x, kv_cache=kv_cache, use_cache=True)
    assert kv_cache[0].shape[1] == max_len, f"step 4: 持续驱逐后 KV 应为 {max_len}, 实际 {kv_cache[0].shape[1]}"

    print(f"  PASS: KV cache 驱逐生效（max_len={max_len}，多步后稳定）")


def test_sink_preserved():
    """[3] 驱逐后保留前 sink_size + 最近 window_size token。"""
    print("\n[3] sink + window 保留策略")
    torch.manual_seed(42)
    sink_size, window_size = 2, 3
    max_len = sink_size + window_size
    attn = GroupedQueryAttention(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        attention_sink_size=sink_size, sliding_window_size=window_size,
    )
    attn.eval()

    # 构造可识别的 KV cache：每个 token 的 K 值唯一
    # step 0: tokens 0,1 → KV=[0,1]
    # step 1: tokens 2,3 → KV=[0,1,2,3]
    # step 2: tokens 4,5 → KV=[0,1,2,3,4,5] (max_len=5，未超限)
    # step 3: tokens 6,7 → KV=[0..7] → 驱逐到 [0,1,5,6,7] (sink=2, window=3)
    kv_cache = None
    for step in range(4):
        # 每步 2 个 token，用 step 标识便于追踪
        x = torch.randn(1, 2, 256)
        with torch.no_grad():
            out, kv_cache = attn(x, kv_cache=kv_cache, use_cache=True)

    # step 3 后：驱逐发生，KV 应为 [token0, token1, token5, token6, token7]
    # 验证：检查驱逐前后 sink 部分是否保持不变
    # 由于无法直接从 KV tensor 识别 token，改用行为验证：
    # 驱逐后 KV 长度 = max_len
    assert kv_cache[0].shape[1] == max_len, f"驱逐后 KV 长度应为 {max_len}"

    # 验证 _evict_kv_cache 逻辑：直接调用
    fake_k = torch.randn(1, 10, 2, 64)  # 10 个 token
    fake_v = torch.randn(1, 10, 2, 64)
    evicted_k, evicted_v = attn._evict_kv_cache(fake_k, fake_v)
    assert evicted_k.shape[1] == max_len, f"驱逐后长度应为 {max_len}, 实际 {evicted_k.shape[1]}"

    # 验证 sink 部分（前 sink_size 个）确实来自原 cache 的前部
    assert torch.equal(evicted_k[:, :sink_size], fake_k[:, :sink_size]), "sink 部分应保持原前部"
    # 验证 window 部分（后 window_size 个）来自原 cache 的后部
    assert torch.equal(evicted_k[:, -window_size:], fake_k[:, -window_size:]), "window 部分应保持原后部"

    print(f"  PASS: sink (前{sink_size}) + window (后{window_size}) 正确保留")


def test_training_unaffected():
    """[4] 训练时（无 kv_cache）不受 sink/window 影响。"""
    print("\n[4] 训练模式不受 sink/window 影响")
    torch.manual_seed(42)
    # sink/window=0（标准）
    attn_std = GroupedQueryAttention(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        attention_sink_size=0, sliding_window_size=0,
    )
    # sink/window 启用
    attn_sink = GroupedQueryAttention(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        attention_sink_size=4, sliding_window_size=1024,
    )
    # 复制权重
    attn_sink.load_state_dict(attn_std.state_dict(), strict=False)

    attn_std.eval()
    attn_sink.eval()
    # 训练时无 kv_cache，无 use_cache
    x = torch.randn(2, 16, 256)
    mask = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        out_std, _ = attn_std(x, mask=mask)
        out_sink, _ = attn_sink(x, mask=mask)

    diff = (out_std - out_sink).abs().max().item()
    assert diff < 1e-6, f"训练模式（无 kv_cache）输出应一致, diff={diff}"
    print(f"  PASS: 训练模式输出一致 (diff={diff:.2e})")


def test_checkpoint_compat():
    """[5] checkpoint 兼容：sink/window=0 的旧 ckpt 加载到新模型。"""
    print("\n[5] checkpoint 兼容性")
    # 旧模型（无 sink/window 参数，但新代码默认 sink/window=0）
    torch.manual_seed(42)
    attn_old = GroupedQueryAttention(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        attention_sink_size=0, sliding_window_size=0,
    )
    old_sd = attn_old.state_dict()

    # 新模型（sink/window 启用，但参数名相同）
    attn_new = GroupedQueryAttention(
        hidden_size=256, num_heads=4, num_kv_heads=2,
        attention_sink_size=4, sliding_window_size=512,
    )
    # sink/window 不是 nn.Parameter，只是 Python 属性，不影响 state_dict
    # 所以 strict=True 加载应该成功
    attn_new.load_state_dict(old_sd, strict=True)
    print(f"  PASS: 旧 ckpt 加载到新模型 (strict=True 成功)")


def test_neuron_level():
    """[6] neuron 级别：配置 sink/window 后构建正确。"""
    print("\n[6] neuron 级别配置")
    cfg = copy.deepcopy(TINY_TEST)
    cfg.vocab_size = 100
    cfg.neuron_id = "n_sink"
    cfg.attention_sink_size = 4
    cfg.sliding_window_size = 256
    torch.manual_seed(42)
    neuron = ResonanceNeuron(cfg)

    # 检查每层 attention 都配置了 sink/window
    for i, layer in enumerate(neuron.layers):
        attn = layer.attention
        assert attn.attention_sink_size == 4, f"layer {i}: sink_size 应为 4"
        assert attn.sliding_window_size == 256, f"layer {i}: window_size 应为 256"
        assert attn.kv_cache_max_len == 260, f"layer {i}: max_len 应为 260"

    # 训练模式不受影响
    neuron.eval()
    shared_emb = torch.randn(2, 16, 512)
    with torch.no_grad():
        result = neuron.forward(shared_emb, return_logits=True)
    assert "logits" in result, "neuron forward 应返回 logits"
    print(f"  PASS: neuron 级别 sink/window 配置生效（每层 max_len=260）")


def main():
    print("=" * 70)
    print("S11 attention sink + 滑动窗口 smoke test")
    print("=" * 70)

    test_backward_compat_no_sink()
    test_kv_cache_eviction()
    test_sink_preserved()
    test_training_unaffected()
    test_checkpoint_compat()
    test_neuron_level()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
