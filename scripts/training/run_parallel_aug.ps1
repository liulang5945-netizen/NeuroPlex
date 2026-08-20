# 4 路并行训练 compact 神经元（数据增强 + 差异化数据 + 独立 shared_embedding）
# 每路独立训练自己的 shared_embedding 副本，无串行依赖。
#
# 启动方式（PowerShell）:
#   powershell -ExecutionPolicy Bypass -File scripts\training\run_parallel_aug.ps1
#
# 日志各自写入 logs/training/zh_augN_*.log，结束后可对比。

Set-Location -Path "$PSScriptRoot\..\.."
New-Item -ItemType Directory -Path "logs/training" -Force | Out-Null

Write-Output "=== 启动 4 路并行训练 ==="
Write-Output "每路 3 threads，共 12 threads，16000 步/路，数据增强 ON，dropout=0.2"
Write-Output ""

# zh_aug0: 全量数据
$proc0 = Start-Process -FilePath "python" `
    -ArgumentList "-u","scripts/training/train_compact_parallel.py",
        "--neuron_id","zh_aug0",
        "--data_files","simple_zh_texts.jsonl",
        "--shared_emb_mode","train",
        "--steps","16000",
        "--dropout","0.2",
        "--threads","3",
        "--eval_every","2000" `
    -RedirectStandardOutput "logs/training/zh_aug0_train.log" `
    -RedirectStandardError "logs/training/zh_aug0_err.log" `
    -NoNewWindow -PassThru
Write-Output "[zh_aug0] PID=$($proc0.Id) -> logs/training/zh_aug0_train.log (787K full)"

# zh_aug1: shared_core + class_a_chinese
$proc1 = Start-Process -FilePath "python" `
    -ArgumentList "-u","scripts/training/train_compact_parallel.py",
        "--neuron_id","zh_aug1",
        "--data_files","shared_core.jsonl","class_a_chinese.jsonl",
        "--shared_emb_mode","train",
        "--steps","16000",
        "--dropout","0.2",
        "--threads","3",
        "--eval_every","2000" `
    -RedirectStandardOutput "logs/training/zh_aug1_train.log" `
    -RedirectStandardError "logs/training/zh_aug1_err.log" `
    -NoNewWindow -PassThru
Write-Output "[zh_aug1] PID=$($proc1.Id) -> logs/training/zh_aug1_train.log (249K chinese)"

# zh_aug2: shared_core + class_b_encyclopedia
$proc2 = Start-Process -FilePath "python" `
    -ArgumentList "-u","scripts/training/train_compact_parallel.py",
        "--neuron_id","zh_aug2",
        "--data_files","shared_core.jsonl","class_b_encyclopedia.jsonl",
        "--shared_emb_mode","train",
        "--steps","16000",
        "--dropout","0.2",
        "--threads","3",
        "--eval_every","2000" `
    -RedirectStandardOutput "logs/training/zh_aug2_train.log" `
    -RedirectStandardError "logs/training/zh_aug2_err.log" `
    -NoNewWindow -PassThru
Write-Output "[zh_aug2] PID=$($proc2.Id) -> logs/training/zh_aug2_train.log (341K encyclopedia)"

# zh_aug3: shared_core + class_c_story
$proc3 = Start-Process -FilePath "python" `
    -ArgumentList "-u","scripts/training/train_compact_parallel.py",
        "--neuron_id","zh_aug3",
        "--data_files","shared_core.jsonl","class_c_story.jsonl",
        "--shared_emb_mode","train",
        "--steps","16000",
        "--dropout","0.2",
        "--threads","3",
        "--eval_every","2000" `
    -RedirectStandardOutput "logs/training/zh_aug3_train.log" `
    -RedirectStandardError "logs/training/zh_aug3_err.log" `
    -NoNewWindow -PassThru
Write-Output "[zh_aug3] PID=$($proc3.Id) -> logs/training/zh_aug3_train.log (670K story)"

Write-Output ""
Write-Output "所有进程已启动。等待完成..."
Write-Output "  监控: Get-Content logs/training/zh_aug0_train.log -Wait -Tail 10"
Write-Output "  停止: Stop-Process -Id $($proc0.Id),$($proc1.Id),$($proc2.Id),$($proc3.Id)"
Write-Output ""

# 等待所有完成
$proc0.WaitForExit()
$proc1.WaitForExit()
$proc2.WaitForExit()
$proc3.WaitForExit()
Write-Output "=== 全部完成 ==="
