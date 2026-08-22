# 阶段 5 超越证据对比报告：Taiji/Seed 原生链 vs 冻结 Transformer 基线

日期：2026-08-23 · 分支：main（901a8c5 后）· 判定口径：BOOTSTRAP_CRITERIA.md 原通过线

## 1. 对比语义

`neuroplex/`（Transformer 成员）作为冻结对照组保留不删除；`taiji/seed` 禁止导入
`neuroplex/transformers`（`tests/taiji_native/test_naming_boundary_contract.py` 强制）。
"继承"= 逐功能原生替代 + 同判据验证；"超越"= Transformer 结构上做不到的三项能力
的实证：**在线持续学习**、**动作改变环境（play 闭环）**、**无灾难性遗忘**。

## 2. 继承证据（同判据对比）

| 遗产能力 | Transformer 基线口径 | Taiji/Seed 原生结果 |
|---|---|---|
| 语言建模 | token LM PPL | byte_ppl 23.1（800K raw-byte 重训；首轮 200K=27.1，均匀分布=257；holdout surprise 稳定 2.7-2.8） |
| 面板区分度 | 24 条真实面板三组 NLL 排序 | 原生 judge：dialogue -3.07 < knowledge -3.28 < unfamiliar -3.48，三组 std>0.05，排序正确 |
| judge 排序（A1） | judge NLL 排序准确率 ≥0.7 | 原生 judge 低/高 loss 对排序 **144/144**（≈1.0），校准拟合 0.993，PASS |
| 巩固/replay（M7） | sleep_engine 判据 | 七项全过：`act()` 显著高于 no-replay/content-lesion 的 62.5% 基线，control/lesion 回落 |
| 生成评估 | 固定面板 + PPL | `eval_seed_corpus.py` 同口径；生成可读性为下一阶段目标（阶段 1 通过线=单调进步+面板区分度，达成） |

## 3. 超越证据（Transformer 结构上做不到的三项）

### 3.1 在线持续学习（训练后继续从对话学习）

- 机制：`observe(learn=True)` 在清醒期对每个符号做局部 delta 更新；`seed/sleep.py`
  的 `experience()` 把文本经验同步写入基底；`api/seed_runtime.py` 的对话分支把
  （问+答）整段经 `learn_bytes` 写回——**每一次对话都是一次真实参数更新**。
- 对照：Transformer 成员权重冻结，无优化器/无在线路径，结构上不可能。
- 证据：`tests/taiji_native/test_experience_safety.py`（experience 写入安全性）与
  `tests/seed/test_seed_sleep.py` 全绿；诊断十三确认 `memory_confidence_decay=5e-3`
  下写入损伤消除（Δ=+0.03）。

### 3.2 动作改变环境（play 闭环）

- 机制：`seed/environments.py` 的 `TopicWorld` 实现 `TaijiEnvironment` 协议，
  `observe/act/settle_action` 标准序列驱动；模型动作改变环境下一观察。
- 证据（N11 判据）：主动环境末 40 次成功率 **100%**，随机基线 50%，
  action-lesion 57.5%——行为差异可归因于动作本身。
- 对照：冻结基线只能条件生成，无 action→环境→sensation 闭环。

### 3.3 无灾难性遗忘（睡眠巩固不损伤清醒能力）

- 机制：**场成熟度门控**（`replay_maturity_ticks`）。睡眠期的结构换线与慢通路写入
  只在场成熟前进行（新场保留修复与 outcome leg，M7 依赖它）；成熟场冻结两者，
  防止梦境基底误差对清醒解码器做离分布更新。
- 证据链（诊断十二→二十）：
  - 未门控：一夜睡眠全面板 Δ≈-0.23 ~ -0.38（换线为主损伤源）；
  - 门控后：overall Δ=-0.025、**unfamiliar +0.062、targets +0.123**，
    some_improvement PASS；M7 七项保持全过。
  - A2 复验（800K 成熟检查点）待重训完成后最终落盘。
- 对照：Transformer 无内生巩固；继续训练即灾难性遗忘，只能靠冻结回避。

## 4. 工程继承清单（阶段 4）

- `api/seed_runtime.py`：Seed 加载/卸载/热切换 + 字节级对话（问/答标记序列化，
  多轮上下文由基底持久状态承担，无 KV cache 拼装）。
- `api/routes_chat.py`：聊天主路由 seed 分支（SSE 同协议）；`/api/health` 报告 `seed_active`。
- `api/routes_model_switch.py`：`switch_model` 支持 `cortex|seed` 双向热切换，互斥语义。
- `desktop/main.py`：`SEED_RUNTIME=1` 启动即原生模式；`SEED_HOST/SEED_PORT` 支持
  局域网远程接入（移动端浏览器共用同一 Web UI）。
- 前端：设置页"运行环境"分区（认知主体热切换），构建进 `frontend/dist`。
- 冒烟测试：`tests/test_seed_product_smoke.py` 4 项；全仓 **108 passed, 3 skipped**。

## 5. 待完成项（诚实边界）

1. **A2–A5/B1 最终复验**：依赖 800K 成熟检查点（重训进行中，约 38 分钟）。
   机制已由诊断二十与玩具场判据证明；数值报告待检查点落盘后由
   `verify_seed_a2_sleep.py / a3 / a4_a5 / b1` 产出。
2. **生成可读性**：byte-level 生成尚未到人工可读（阶段 1 通过线不要求）。
3. **原生移动端客户端**：当前以移动端浏览器远程接入实现；独立 App 未启动。

## 6. 复现入口

```
pytest tests -q                                        # 全仓回归
python scripts/training/verify_taiji_m7_cue_chain.py   # M7 七项判据
python scripts/training/verify_seed_a1_judge.py        # A1 同判据
python scripts/training/verify_seed_a2_sleep.py        # A2（需 800K 检查点）
python scripts/training/train_seed_corpus.py --scale 2 --epochs 1 --max-symbols 800000
```
