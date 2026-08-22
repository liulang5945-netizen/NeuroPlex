# diagnostics：阶段 3 睡眠损伤诊断脚本存档（不可从当前 HEAD 复现全部结论）

本目录保存阶段 3（sleep/play）期间 A2 睡眠损伤定位过程的全部临时诊断脚本
（原位于 `scripts/training/_diag_*.py`，2026-08-23 归档至此）。

性质与 `scripts/archive/` 主目录一致：**历史调查记录，不是可运行资产**。
它们绑定当时的检查点状态与配置数值（如 `replay_outcome_slow_scale`、
`replay_restructure_trust`、`replay_maturity_ticks` 的演化序列），部分脚本在
当前 HEAD 上仍可执行，但结论以 `reports/` 与提交信息为准。

## 诊断编号与结论链（摘要）

- `_diag_sleep_damage*.py` / `_diag_sleep_fix_probe.py`：损伤定位起点。
- `_diag_a2_worst.py`：损伤来自 experience 写入经回注，而非回放本身。
- `_diag_a2_decay_sweep.py`：`memory_confidence_decay=5e-3` 消除写入损伤。
- `_diag_a2_scale.py` / `_diag_a2_fast.py`：快通路加量只增损伤；存在剂量无关底盘。
- `_diag_a2_nolearn.py`：损伤由学习更新产生，单次巩固即饱和。
- `_diag_a2_noise.py`：再生噪声假说证伪。
- `_diag_a2_channel.py`：慢通路证据通道排除（mute 后损伤不变）。
- `_diag_a2_struct.py`：换线全禁后 Δ=-0.025、unfamiliar=+0.06、targets=+0.12——
  换线确认为主损伤源。
- `_diag_a2_trust.py`：门控信号选型（write_count 信任度 → 场成熟度）。

最终机制：`replay_maturity_ticks` 场成熟度门控——新场保留换线修复与 outcome
慢通路写入（M7 outcome leg），成熟场冻结两者以保护清醒面板（A2）。
机制代码见 `taiji/model.py::consolidate` 与 `taiji/config.py`。
