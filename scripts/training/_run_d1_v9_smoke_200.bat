@echo off
REM Smoke test: D1-fix v9-N 200 步快速验证
REM 验证 v9 路径分支正确激活 + baseline 修复生效
setlocal
cd /d e:\taiji-neuron

set D1_MICRO_N=200
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
set D1_SAMPLE_EVERY=50

echo === v9 Smoke 200 步 (baseline=first_n_steps_mean) ===
python -u scripts\training\verify_play_engine_d1_long_run.py

endlocal
