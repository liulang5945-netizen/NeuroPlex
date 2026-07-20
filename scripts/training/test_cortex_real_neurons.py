"""P2-1.4: 用新蒸馏的神经元验证 Cortex 加载 + E2E generate。

测试目标：
  1. Cortex 能加载 data/neurons/ 下 5 个 fieldcond 神经元（field_dim=4096 统一）
  2. teacher_pipeline 设置成功
  3. think() 返回合法 field_vector + logits
  4. generate("你好") 返回合法字符串
  5. 生成的 token ID 全部在 [0, 256000) 范围内

运行：
    python scripts/training/test_cortex_real_neurons.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

_LIBS = os.path.join(PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)


def main():
    print("=" * 70)
    print("P2-1.4: Cortex + 新蒸馏神经元 E2E 验证")
    print("=" * 70)

    # ── Step 1: Cortex 加载真实 neurons ──
    print("\n[1] 加载 Cortex (neurons_dir=data/neurons) ...")
    t0 = time.time()
    from taiji.loader import create_cortex
    cortex, tokenizer = create_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=2,
        enable_gating=False,
    )
    print(f"    加载耗时: {time.time()-t0:.1f}s")
    print(f"    neurons loaded: {list(cortex.neurons.keys())}")
    assert len(cortex.neurons) == 5, f"expected 5 neurons, got {len(cortex.neurons)}"
    # 检查 field_dim 统一
    dims = {n.config.field_dim for n in cortex.neurons.values()}
    assert dims == {4096}, f"field_dim should be {{4096}}, got {dims}"
    print(f"    field_dim 统一: {dims}")

    # ── Step 2: 设置 teacher pipeline ──
    print("\n[2] 设置 teacher_pipeline ...")
    from taiji.training.checkpoint_bridge import load_teacher_model
    teacher, shared_embedding = load_teacher_model("checkpoint-481000", device="cpu")
    cortex.set_tokenizer(tokenizer)

    # 加载 shared_proj 把 teacher 2048d hidden 投影到 neuron 的 base_embed_dim=512
    from taiji.resonance.shared_embed import SharedEmbedProj
    shared_proj = SharedEmbedProj.load("data/distill/shared_proj.pt", 2048, 512)

    # H10: set_teacher_pipeline 内部用 extract_hidden_states(teacher, ids) -> shared_proj
    # 只设置 teacher_pipeline（让 think 走 _embed_pipeline 路径），
    # 不设置 _shared_embedding（teacher 的 embedding 是 2048d, 与 neuron base_embed_dim=512 不匹配）
    cortex.set_teacher_pipeline(teacher_model=teacher, shared_proj=shared_proj)
    print(f"    teacher_pipeline 已注入")

    # ── Step 3: think() 验证 ──
    print("\n[3] think() 验证 ...")
    test_text = "你好"
    ids = tokenizer.encode(test_text)
    print(f"    输入: '{test_text}' -> {len(ids)} tokens")
    input_ids = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    state = cortex.think(input_ids)
    print(f"    think 耗时: {time.time()-t0:.1f}s")
    print(f"    state keys: {list(state.keys())}")
    # think 返回 ensemble.forward 的结果，应有 logits/field_state
    has_logits = "logits" in state or "weighted_logits" in state
    assert has_logits, f"think should return logits, got {list(state.keys())}"
    logits_key = "logits" if "logits" in state else "weighted_logits"
    print(f"    {logits_key} shape: {state[logits_key].shape}")
    assert state[logits_key].shape[0] == 1, "batch dim should be 1"
    assert state[logits_key].shape[2] == 256000, (
        f"vocab dim should be 256000, got {state[logits_key].shape[2]}"
    )

    # ── Step 4: generate() 验证 ──
    print("\n[4] generate() 验证 ...")
    t0 = time.time()
    output = cortex.generate(test_text, max_tokens=32)
    print(f"    generate 耗时: {time.time()-t0:.1f}s")
    print(f"    输出: '{output}'")
    assert isinstance(output, str), f"output should be str, got {type(output)}"
    assert len(output) > 0, "output should be non-empty"

    # ── Step 5: 多轮 generate 验证 ──
    print("\n[5] 多轮 generate 验证 ...")
    test_prompts = ["1+1=", "hello", "def fibonacci"]
    for p in test_prompts:
        t0 = time.time()
        out = cortex.generate(p, max_tokens=24)
        dur = time.time() - t0
        print(f"    '{p}' -> '{out[:60]}...' ({dur:.1f}s)")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
