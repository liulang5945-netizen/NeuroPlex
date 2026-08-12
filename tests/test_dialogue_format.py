"""对话口径契约回归测试（2026-08-12 机制化）。

防口径漂移：覆盖 build_dialogue_prompt 构造 + dialogue_prompt_requires_guard 守卫判断。

背景：这是从 scripts/training/ 的 87 个一次性 verify/_smoke 脚本中，
第一个收敛到 tests/ 的「可回归契约」。口径错误（07-29 评估集失真、
07-31 token ID 错位、08-12 裸 prompt）反复出现的根因是验证脚本各自为政、
无统一契约 + 无回归保障。此后口径契约的任何改动必须先过本测试。

运行（标准库 unittest，零第三方依赖，pytest 亦可收集）：
    python -m unittest tests.test_dialogue_format -v
"""
import unittest

from taiji.resonance.dialogue_format import (
    SFT_ANSWER_MARKER,
    Q_MARKER,
    build_dialogue_prompt,
    dialogue_prompt_requires_guard,
)

DIALOGUE_NIDS = ["zh_aug0_dialogue", "zh_std0_dialogue"]
DOMAIN_NIDS = ["zh", "en"]  # 无 _dialogue 后缀的域 neuron


class TestBuildDialoguePrompt(unittest.TestCase):
    def test_exact_format(self):
        self.assertEqual(build_dialogue_prompt("你好"), "问：你好\n答：")

    def test_structure(self):
        p = build_dialogue_prompt("什么是人工智能")
        self.assertTrue(p.startswith(Q_MARKER))
        self.assertIn(f"\n{SFT_ANSWER_MARKER}", p)
        self.assertTrue(p.rstrip().endswith(SFT_ANSWER_MARKER))

    def test_empty_question(self):
        self.assertEqual(build_dialogue_prompt(""), f"{Q_MARKER}\n{SFT_ANSWER_MARKER}")


class TestDialoguePromptGuard(unittest.TestCase):
    def test_blocks_bare_prompt_with_dialogue_neuron(self):
        # 裸 prompt + zh 域 + dialogue neuron → 必须拦截（防换行死循环假退化）
        self.assertTrue(dialogue_prompt_requires_guard("你好", "zh", DIALOGUE_NIDS))

    def test_allows_formatted_prompt(self):
        p = build_dialogue_prompt("你好")
        self.assertFalse(dialogue_prompt_requires_guard(p, "zh", DIALOGUE_NIDS))

    def test_allows_non_dialogue_neuron(self):
        # 纯域 neuron（无 _dialogue 后缀）评估裸 prompt 不拦截
        self.assertFalse(dialogue_prompt_requires_guard("你好", "zh", DOMAIN_NIDS))

    def test_allows_non_zh_domain(self):
        self.assertFalse(dialogue_prompt_requires_guard("hello", "en", DIALOGUE_NIDS))

    def test_allows_plain_prompt_override(self):
        # base/域 neuron 评估显式放行
        self.assertFalse(
            dialogue_prompt_requires_guard("你好", "zh", DIALOGUE_NIDS, allow_plain_prompt=True)
        )

    def test_allows_empty_active_nids(self):
        self.assertFalse(dialogue_prompt_requires_guard("你好", "zh", None))
        self.assertFalse(dialogue_prompt_requires_guard("你好", "zh", []))


class TestContractInvariant(unittest.TestCase):
    def test_build_output_never_triggers_guard(self):
        """核心不变量：build_dialogue_prompt 的输出绝不能被守卫拦截。

        若本测试失败，说明构造与守卫契约漂移（例如 marker 改了但守卫没同步）。
        """
        for q in ["你好", "什么是神经网络", "", "帮我写代码"]:
            p = build_dialogue_prompt(q)
            self.assertFalse(dialogue_prompt_requires_guard(p, "zh", DIALOGUE_NIDS))


if __name__ == "__main__":
    unittest.main()
