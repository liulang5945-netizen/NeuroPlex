"""P1-4: 完整 sleep 闭环测试（feed→sleep→PPL下降→neurogenesis）。

测试目标：
  1. 构建 MinimalCortex + StubFeedEngine + MockLifecycle
  2. 测量 sleep 前的初始 PPL（eval 模式跑相同样本）
  3. 调用 sleep_engine.sleep() 跑完整 6 阶段
  4. 验证 SleepReport.phases_completed 包含 'model_training'
  5. 验证 SleepReport.training_loss 有限
  6. 测量 sleep 后的 PPL，断言 PPL 下降（训练有效）
  7. 验证 neurogenesis 触发（mock 高 error_rate）

设计要点：
  - 不依赖磁盘 neurons/，直接构造 MinimalCortex.neurons dict
  - 不依赖真实 sentencepiece，用 SimpleTokenizer（字符级）
  - field_dim=4096（项目硬约束）
  - 训练样本用固定 pattern "abcabc..." 让 neuron 能学会
  - max_training_steps 限制为 15 步以加速

运行：
    python scripts/training/test_sleep_cycle.py
"""

from __future__ import annotations

import os
import sys
import math
import types
import logging
from typing import Dict, List, Optional, Any

# 项目根加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# sentencepiece 沙箱兜底
_LIBS = os.path.join(_PROJECT_ROOT, "_libs")
if os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

import torch
import torch.nn as nn
import torch.nn.functional as F


# 配置 logging 让 SleepEngine 的 info 输出可见
logging.basicConfig(level=logging.INFO, format="  [%(name)s] %(message)s")


# ═══════════════════════════════════════════════════════════
# 1. Stubs & Mocks
# ═══════════════════════════════════════════════════════════

VOCAB = 64                 # 测试用极小 vocab
BASE_EMBED_DIM = 32        # 等于 neuron.base_embed_dim → 无需投影
HIDDEN = 64
FIELD_DIM = 4096           # 项目硬约束


class SimpleTokenizer:
    """字符级 tokenizer：text → [4, VOCAB) 范围的 ID 列表。

    保留 0-3 作为 control token (pad/unk/bos/eos)。
    """

    def __init__(self, vocab_size: int = VOCAB):
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = []
        for ch in text:
            tid = (ord(ch) % (self.vocab_size - 4)) + 4
            ids.append(tid)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        chars = []
        for i in ids:
            if 4 <= i < self.vocab_size:
                chars.append(chr((i - 4) + ord("a")))
        return "".join(chars)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.encode(token)[0] if token else 0


class StubFeedEngine:
    """最小 FeedEngine stub。

    提供 SleepEngine._train_cortex_neurons 所需接口：
      - get_pending_samples_by_domain() -> Dict[str, list]
      - get_domain_error_rates() -> Dict[str, float]
      - clear_pending_samples()
    """

    def __init__(self, domain_samples: Dict[str, list], error_rates: Optional[Dict[str, float]] = None):
        self._samples = domain_samples
        self._error_rates = error_rates or {}
        self.cleared = False

    def get_pending_samples_by_domain(self) -> Dict[str, list]:
        # 返回副本，clear 后再调用应返回空
        return {d: list(s) for d, s in self._samples.items()} if not self.cleared else {}

    def get_domain_error_rates(self) -> Dict[str, float]:
        return dict(self._error_rates)

    def clear_pending_samples(self):
        self.cleared = True


class MockApoptosis:
    def __init__(self):
        self.ppl_records: Dict[str, list] = {}
        self._failure_counts: Dict[str, int] = {}
        self.activation_triggered: List[str] = []

    def record_ppl(self, domain: str, ppl: float):
        self.ppl_records.setdefault(domain, []).append(ppl)

    def check_activation(self, nid: str, activation_count: int, total_rounds: int) -> bool:
        # 测试中不触发凋亡
        return False


class MockNeurogenesis:
    def __init__(self):
        self.error_records: Dict[str, list] = {}
        self.created: List[str] = []

    def record_domain_error(self, domain: str, error_rate: float) -> bool:
        self.error_records.setdefault(domain, []).append(error_rate)
        # error_rate >= 0.5 触发新生
        return error_rate >= 0.5

    def create_neuron_for_domain(self, domain: str) -> dict:
        self.created.append(domain)
        return {
            "success": True,
            "neuron_id": f"neuron_{domain}_new",
            "teacher_type": "mock_teacher",
        }


class MockMaturity:
    def __init__(self):
        self.ticked = 0

    def tick_all(self):
        self.ticked += 1


class MockLifecycle:
    """组合 apoptosis + neurogenesis + maturity 的 lifecycle stub。"""

    def __init__(self):
        self.apoptosis = MockApoptosis()
        self.neurogenesis = MockNeurogenesis()
        self.maturity = MockMaturity()


class MockNeurogenesisCreator:
    """NeurogenesisCreator stub。"""

    def __init__(self):
        self.created: List[str] = []

    def create_neuron_for_domain(self, domain: str) -> dict:
        self.created.append(domain)
        return {
            "success": True,
            "neuron_id": f"neuron_{domain}_new",
            "teacher_type": "mock",
        }


class MinimalCortex:
    """最小 Cortex stub，满足 SleepEngine._train_cortex_neurons 接口。

    SleepEngine 访问的属性：
      - cortex.neurons: Dict[str, ResonanceNeuron]
      - cortex._shared_embedding: nn.Embedding
      - cortex._tokenizer
      - cortex.device
    """

    def __init__(self, neurons: Dict[str, Any], shared_embedding: nn.Embedding, tokenizer):
        self.neurons = neurons
        self._shared_embedding = shared_embedding
        self._tokenizer = tokenizer
        self._embed_pipeline = None
        self.device = "cpu"


# ═══════════════════════════════════════════════════════════
# 2. 辅助函数
# ═══════════════════════════════════════════════════════════

def _build_tiny_neuron(domain: str):
    """构造一个极小 ResonanceNeuron 用于测试。"""
    from taiji.resonance import ResonanceNeuron, NeuronConfig

    cfg = NeuronConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        vocab_size=VOCAB,
        base_embed_dim=BASE_EMBED_DIM,
        field_dim=FIELD_DIM,
        spec="tiny_test",
        lm_head_rank=0,         # 禁用低秩，训练完整 lm_head
        neuron_id=f"neuron_{domain}",
    )
    return ResonanceNeuron(cfg)


def _make_learnable_samples(pattern: str = "abcabcabcabcabcabc", n: int = 8) -> list:
    """生成可学习的训练样本（固定 pattern，neuron 应能学会 next-token）。"""
    return [{"task": pattern, "content": pattern} for _ in range(n)]


def _measure_ppl(neuron, shared_embedding, tokenizer, samples: list) -> float:
    """测量 neuron 在 samples 上的平均 PPL（eval 模式）。

    Returns:
        平均 cross_entropy loss（越低越好）；如果无有效样本，返回 inf。
    """
    neuron.eval()
    losses = []
    with torch.no_grad():
        for sample in samples:
            text = sample.get("task") or sample.get("content") or ""
            if not text or len(text) < 5:
                continue
            ids = tokenizer.encode(text)
            if len(ids) < 5:
                continue
            ids = ids[:64]
            input_ids = torch.tensor([ids], dtype=torch.long)
            shared_emb = shared_embedding(input_ids)
            output = neuron.forward(shared_emb, return_logits=True)
            logits = output.get("logits")
            if logits is None:
                continue
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )
            losses.append(loss.item())
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


# ═══════════════════════════════════════════════════════════
# 3. 测试主体
# ═══════════════════════════════════════════════════════════

def test_sleep_cycle_ppl_decrease():
    """主测试：完整 sleep 周期 → PPL 下降。

    Steps:
      1. 构建 MinimalCortex（1 个 zh neuron）+ StubFeedEngine（提供 pattern 样本）
      2. 注入 SleepEngine
      3. 测量 sleep 前 PPL
      4. 调用 sleep()
      5. 测量 sleep 后 PPL
      6. 断言 PPL 下降
    """
    from taiji.life.sleep_engine import SleepEngine, SleepConfig

    print("\n── Step 1: 构建 Cortex + FeedEngine ──")
    torch.manual_seed(42)
    tokenizer = SimpleTokenizer(VOCAB)
    shared_embedding = nn.Embedding(VOCAB, BASE_EMBED_DIM)

    # 1 个 zh neuron + 学习样本
    neuron = _build_tiny_neuron("zh")
    cortex = MinimalCortex(
        neurons={"zh": neuron},
        shared_embedding=shared_embedding,
        tokenizer=tokenizer,
    )

    samples = _make_learnable_samples("abcabcabcabcabcabc", n=8)
    feed_engine = StubFeedEngine(
        domain_samples={"zh": samples},
        error_rates={},  # 第一轮不触发 neurogenesis
    )
    print(f"  Cortex: 1 neuron (zh), {len(samples)} samples, vocab={VOCAB}")

    print("\n── Step 2: 注入 SleepEngine ──")
    lifecycle = MockLifecycle()
    sleep_config = SleepConfig(
        auto_sleep_enabled=False,
        training_enabled=True,
        max_training_steps=15,  # 限制步数加速
        save_checkpoints=False,
    )
    sleep_engine = SleepEngine(config=sleep_config, data_dir="_test_sleep_data")
    sleep_engine.set_brain_interfaces(
        cortex=cortex,
        lifecycle=lifecycle,
        feed_engine=feed_engine,
    )
    # 手动注入 neurogenesis_creator（绕过自动创建的 try/except）
    sleep_engine._neurogenesis_creator = MockNeurogenesisCreator()
    print(f"  SleepEngine 注入完成: max_steps={sleep_config.max_training_steps}")

    print("\n── Step 3: 测量 sleep 前 PPL ──")
    ppl_before = _measure_ppl(neuron, shared_embedding, tokenizer, samples)
    print(f"  PPL before sleep: {ppl_before:.4f}  (loss, lower=better)")
    assert math.isfinite(ppl_before), "PPL before sleep should be finite"

    print("\n── Step 4: 调用 sleep() ──")
    report = sleep_engine.sleep(reason="test")
    print(f"  phases_completed: {report.phases_completed}")
    print(f"  training_samples_used: {report.training_samples_used}")
    print(f"  training_loss: {report.training_loss}")
    print(f"  health_status: {report.health_status}")

    # ── 验证 Phase 2 完成 ──
    assert "model_training" in report.phases_completed, (
        f"Phase 2 'model_training' should complete, got {report.phases_completed}"
    )
    assert report.training_samples_used > 0, (
        f"should use > 0 samples, got {report.training_samples_used}"
    )
    assert report.training_loss is not None and math.isfinite(report.training_loss), (
        f"training_loss should be finite, got {report.training_loss}"
    )

    # ── 验证 PPL 记录到 apoptosis tracker ──
    assert "zh" in lifecycle.apoptosis.ppl_records, (
        f"apoptosis.record_ppl should be called for 'zh', got {lifecycle.apoptosis.ppl_records}"
    )
    print(f"  apoptosis PPL records: {lifecycle.apoptosis.ppl_records}")

    # ── 验证 maturity.tick_all 被调用 ──
    assert lifecycle.maturity.ticked > 0, "maturity.tick_all should be called"

    # ── 验证 feed_engine 样本被清除 ──
    assert feed_engine.cleared, "feed_engine.clear_pending_samples should be called"

    print("\n── Step 5: 测量 sleep 后 PPL ──")
    ppl_after = _measure_ppl(neuron, shared_embedding, tokenizer, samples)
    print(f"  PPL after sleep:  {ppl_after:.4f}")

    # ── 关键断言：PPL 应下降（训练有效）──
    assert ppl_after < ppl_before, (
        f"PPL should decrease after sleep training: before={ppl_before:.4f}, after={ppl_after:.4f}"
    )
    delta = ppl_before - ppl_after
    delta_pct = delta / ppl_before * 100
    print(f"  PPL decrease: {delta:.4f} ({delta_pct:.1f}%)")

    print("\n[PASS] test_sleep_cycle_ppl_decrease")
    print(f"        PPL: {ppl_before:.4f} → {ppl_after:.4f}  (Δ={delta:.4f}, {delta_pct:.1f}%)")
    print(f"        Phases: {len(report.phases_completed)}/6")
    print(f"        Training loss: {report.training_loss:.4f}")

    return {
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "report": report,
        "lifecycle": lifecycle,
    }


def test_neurogenesis_triggered_on_high_error():
    """验证 neurogenesis 触发：高 error_rate 域应触发新神经元创建。

    Steps:
      1. 复用主测试的 cortex
      2. FeedEngine 报告某域 error_rate=0.8（>0.5 阈值）
      3. 调用 sleep()
      4. 断言 NeurogenesisCreator.create_neuron_for_domain 被调用
    """
    from taiji.life.sleep_engine import SleepEngine, SleepConfig

    print("\n── Neurogenesis 触发测试 ──")
    torch.manual_seed(42)
    tokenizer = SimpleTokenizer(VOCAB)
    shared_embedding = nn.Embedding(VOCAB, BASE_EMBED_DIM)

    neuron = _build_tiny_neuron("math")
    cortex = MinimalCortex(
        neurons={"math": neuron},
        shared_embedding=shared_embedding,
        tokenizer=tokenizer,
    )

    # 高错误率域：math error_rate=0.8（>0.5 阈值）
    samples = _make_learnable_samples("123123123123", n=4)
    feed_engine = StubFeedEngine(
        domain_samples={"math": samples},
        error_rates={"math": 0.8},  # 触发 neurogenesis
    )

    lifecycle = MockLifecycle()
    sleep_config = SleepConfig(
        auto_sleep_enabled=False,
        training_enabled=True,
        max_training_steps=5,
        save_checkpoints=False,
    )
    sleep_engine = SleepEngine(config=sleep_config, data_dir="_test_sleep_data_ng")
    sleep_engine.set_brain_interfaces(
        cortex=cortex,
        lifecycle=lifecycle,
        feed_engine=feed_engine,
    )
    neurogenesis_creator = MockNeurogenesisCreator()
    sleep_engine._neurogenesis_creator = neurogenesis_creator

    print(f"  setup: 1 math neuron, error_rate=0.8 (should trigger neurogenesis)")

    report = sleep_engine.sleep(reason="test_neurogenesis")

    # ── 验证 neurogenesis 被触发 ──
    assert "math" in lifecycle.neurogenesis.error_records, (
        f"neurogenesis.record_domain_error should be called for 'math', "
        f"got {lifecycle.neurogenesis.error_records}"
    )
    assert lifecycle.neurogenesis.record_domain_error("math", 0.8), (
        "error_rate=0.8 should trigger neurogenesis (>=0.5 threshold)"
    )

    # ── 验证 NeurogenesisCreator.create_neuron_for_domain 被调用 ──
    assert "math" in neurogenesis_creator.created, (
        f"NeurogenesisCreator.create_neuron_for_domain should be called for 'math', "
        f"got {neurogenesis_creator.created}"
    )

    # ── 验证 recommendations 包含神经新生提示 ──
    ng_recs = [r for r in report.recommendations if "神经新生" in r]
    assert len(ng_recs) >= 1, (
        f"recommendations should contain neurogenesis event, got {report.recommendations}"
    )

    print(f"  neurogenesis_creator.created: {neurogenesis_creator.created}")
    print(f"  recommendations: {ng_recs}")

    print("\n[PASS] test_neurogenesis_triggered_on_high_error")


def test_sleep_all_phases_complete():
    """验证 sleep 完整 6 阶段都能完成（即使某些 stage 走 try/except fallback）。

    期望 phases_completed 至少包含:
      - memory_consolidation
      - model_training
      - knowledge_integration
      - evaluation
    （knowledge_distillation 和 recursive_improvement 可能因 ImportError 跳过，可接受）
    """
    from taiji.life.sleep_engine import SleepEngine, SleepConfig

    print("\n── 完整 6 阶段验证 ──")
    torch.manual_seed(42)
    tokenizer = SimpleTokenizer(VOCAB)
    shared_embedding = nn.Embedding(VOCAB, BASE_EMBED_DIM)

    neuron = _build_tiny_neuron("general")
    cortex = MinimalCortex(
        neurons={"general": neuron},
        shared_embedding=shared_embedding,
        tokenizer=tokenizer,
    )

    samples = _make_learnable_samples("hellohellohello", n=4)
    feed_engine = StubFeedEngine(
        domain_samples={"general": samples},
        error_rates={},
    )

    sleep_config = SleepConfig(
        auto_sleep_enabled=False,
        training_enabled=True,
        max_training_steps=5,
        save_checkpoints=False,
    )
    sleep_engine = SleepEngine(config=sleep_config, data_dir="_test_sleep_data_phases")
    sleep_engine.set_brain_interfaces(
        cortex=cortex,
        lifecycle=MockLifecycle(),
        feed_engine=feed_engine,
    )
    sleep_engine._neurogenesis_creator = MockNeurogenesisCreator()

    report = sleep_engine.sleep(reason="test_phases")

    expected_phases = {"memory_consolidation", "model_training", "knowledge_integration", "evaluation"}
    completed = set(report.phases_completed)
    missing = expected_phases - completed
    assert not missing, (
        f"missing required phases: {missing}, completed={completed}"
    )

    print(f"  completed phases: {sorted(completed)}")
    print(f"  duration: {report.duration_seconds}s")

    print("\n[PASS] test_sleep_all_phases_complete")


# ═══════════════════════════════════════════════════════════
# 清理辅助
# ═══════════════════════════════════════════════════════════

def _cleanup():
    """清理测试产生的临时目录。"""
    import shutil
    for d in ["_test_sleep_data", "_test_sleep_data_ng", "_test_sleep_data_phases"]:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("P1-4: 完整 sleep 闭环测试（feed→sleep→PPL下降→neurogenesis）")
    print("=" * 72)

    try:
        test_sleep_cycle_ppl_decrease()
        test_neurogenesis_triggered_on_high_error()
        test_sleep_all_phases_complete()
        print()
        print("=" * 72)
        print("ALL TESTS PASSED")
        print("=" * 72)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
