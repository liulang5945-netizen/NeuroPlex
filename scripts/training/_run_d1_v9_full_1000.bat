@echo off
REM ============================================================
REM  D1-fix v9-N 1000 步完整长程测试
REM  N 方案：修复 baseline 初始化
REM  - 旧 v4-v8: baseline=第一次 LoRA L2 测量值，LoRA 刚 init 时为 0
REM    → baseline 锁 0 → ceiling 机制数学上不可触发
REM  - 新 v9-N: baseline=前 N 步 (warmup_n=50) LoRA L2 均值
REM    → 跳过 0.0 噪声，ceiling 真正可触发
REM  - 配合 hysteresis N=2 + ceiling 1.6 + DECAY 0.85
REM
REM  运行: _run_d1_v9_full_1000.bat
REM  报告: reports/play_engine_d1_fix_v9_baseline_fix_YYYYMMDD.json
REM ============================================================
setlocal
cd /d e:\taiji-neuron

set D1_MICRO_N=1000
set D1_DECISION_EVERY=50
set D1_DECAY=0.85
set D1_JUDGE_DRIVEN_DECAY=1
set D1_DECAY_MIN_STD=0.05
set D1_DECAY_SAMPLE_N=3
set D1_HYSTERESIS_N=2
set D1_CEILING_RATIO=1.6
set D1_BASELINE_INIT=first_n_steps_mean
set D1_BASELINE_WARMUP_N=50
set D1_EPSILON=0.10
set D1_FORCE_STREAK=5
set D1_RECENCY_BONUS=0.5
set D1_SAMPLE_EVERY=100

echo === D1-fix v9-N 1000 步 (baseline=first_n_steps_mean warmup_n=50) ===
echo === DECAY=0.85, CEILING=1.6, HYSTERESIS_N=2, BASELINE_INIT=first_n_steps_mean ===

python -u scripts\training\verify_play_engine_d1_long_run.py

endlocal
