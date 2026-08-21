# scripts/archive — 历史脚本（不可运行）

本目录是 Legacy NeuroPlex 时期的验证/诊断脚本存档，**全部视为不可运行的历史记录**，只用于追溯当时做过哪些检查。

## 为什么这里的 `from taiji.<...>` 不是 Bug

98 个文件里共 301 处 `from taiji.resonance / taiji.brain / taiji.life / taiji.loader ...`。这里的 `taiji` 是 **`neuroplex` 包的历史 import 别名**，不是当前顶层 `taiji/` 新基底。命名口径见 `plans/active/ARCHITECTURE_DIRECTION_2026_08.md` §0。

已决定 **不重写、不批量改名**，理由：

1. 改名只解决“包名”这一层。这些脚本依赖的 Legacy 符号与数据路径本身早已不存在（`architecture_verification.py:8-10` 自证：`TribalMetrics`/`compute_initial_D` 不再导出，训练数据路径也不存在）。重写会产出“能 import、仍然跑不了”的假活代码，比明确的历史存档更危险。
2. 目录内混有不可机械替换的字面路径：`taiji_data/...`、`taiji/tokenizer/sentencepiece.model`、`taiji/domains/general/sp_general.model`、User-Agent `taiji-neuron/1.0`。批量替换会破坏它们。
3. 少量文件（`legacy_convert_dense_model_format.py`、`legacy_train_teacher_alignment.py`、`_smoke_r7_distillation.py`）已改用 `neuroplex`，与别名版本并存，进一步说明这里是分层沉积的历史，不是一致的代码库。

## 风险边界（已核实）

- 本目录 **没有 `test_*.py`**，pytest 不会收集。
- CI 只执行 `scripts/training/verify_taiji_*.py`、`pytest tests/taiji_native`、`pytest tests/`，都不触及本目录。
- 无任何在用代码引用本目录；仅 `neuroplex/resonance/neuron.py:896` 把一段死代码标注为“仅 scripts/archive/”。

因此这些 import 不会让构建变红，唯一风险是**人（或 agent）误以为它们引用的是新基底 `taiji/`**。本文件即为消除该误解而存在。

## 需要复现历史行为时

用 `git log -- scripts/archive/<file>` 找到当时的提交，在该提交上运行；不要在当前 HEAD 上修补这些脚本。历史 checkpoint 的加载走 `neuroplex/legacy_checkpoint.py`（在受控作用域内临时映射 `taiji` → `neuroplex`，退出后恢复）。
