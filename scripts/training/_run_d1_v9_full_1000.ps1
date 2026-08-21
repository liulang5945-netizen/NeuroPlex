# D1-fix v9-N 1000 步完整长程测试
# N 方案：修复 baseline 初始化
#  - 旧 v4-v8: baseline=第一次 LoRA L2 测量值，LoRA 刚 init 时为 0
#    → baseline 锁 0 → ceiling 机制数学上不可触发
#  - 新 v9-N: baseline=前 N 步 (warmup_n=50) LoRA L2 均值
#    → 跳过 0.0 噪声，ceiling 真正可触发
#  - 配合 hysteresis N=2 + ceiling 1.6 + DECAY 0.85
#
# 运行: powershell -ExecutionPolicy Bypass -File scripts/training/_run_d1_v9_full_1000.ps1
# 报告: reports/play_engine_d1_fix_v9_baseline_fix_YYYYMMDD.json
$env:D1_MICRO_N = "1000"
$env:D1_DECISION_EVERY = "50"
$env:D1_DECAY = "0.85"
$env:D1_JUDGE_DRIVEN_DECAY = "1"
$env:D1_DECAY_MIN_STD = "0.05"
$env:D1_DECAY_SAMPLE_N = "3"
$env:D1_HYSTERESIS_N = "2"
$env:D1_CEILING_RATIO = "1.6"
$env:D1_BASELINE_INIT = "first_n_steps_mean"
$env:D1_BASELINE_WARMUP_N = "50"
$env:D1_EPSILON = "0.10"
$env:D1_FORCE_STREAK = "5"
$env:D1_RECENCY_BONUS = "0.5"
$env:D1_SAMPLE_EVERY = "100"

Set-Location e:\taiji-neuron
Write-Host "=== D1-fix v9-N 1000 步 (baseline=first_n_steps_mean warmup_n=50) ==="
Write-Host "=== DECAY=0.85 CEILING=1.6 HYSTERESIS_N=2 BASELINE_INIT=first_n_steps_mean ==="
& python -u scripts\training\verify_play_engine_d1_long_run.py
