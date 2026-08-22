#!/usr/bin/env python3
"""阶段 3 门槛 B1-bis（原生版）：探索自主性（突破锁定）。

同判据移植：主题池环境（``seed/environments.py`` 的 ``TopicWorld``）中，
有机体用自己的运动策略选择探索方向（动作真实改变环境走向），环境提供
防锁定支架（连续同主题超限排除 + 近因窗口，与 B1-bis 的 ε-greedy/
force_switch/recency 同语义），但选哪个永远由有机体自己的策略决定。

通过线（原 B1-bis 口径）：
    20 次主题决策覆盖全部 6 主题（distinct=6/6），
    switch_count >= 5（远超阈值），
    最高频主题占比 <= 70%，
    0 崩溃。

不写检查点。输出 ``reports/seed_b1_explore_<date>.json``。

运行：python -X utf8 -u scripts/training/verify_seed_b1_explore.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import _seed_verify_common as common  # noqa: E402

import torch  # noqa: E402

from seed import TopicWorld, play  # noqa: E402


# 6 主题池：中文短句；首字节互不相同保证选择时刻候选可区分。
TOPICS = [
    "0 太阳每天从东方升起，照亮山川。".encode("utf-8"),
    "1 河水向东流入大海，永不停歇。".encode("utf-8"),
    "2 风穿过森林，树叶轻轻摇摆。".encode("utf-8"),
    "3 星星在夜里闪烁，像远方的灯。".encode("utf-8"),
    "4 雨水落在田野，庄稼慢慢长大。".encode("utf-8"),
    "5 月亮绕着地球转动，周而复始。".encode("utf-8"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="B1-bis 原生版：探索自主性")
    parser.add_argument(
        "--checkpoint",
        default=str(common.REPO / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    started = time.time()
    print("=" * 64, flush=True)
    print("B1-bis 原生版：探索自主性（主题池 + 防锁定支架）", flush=True)
    print("=" * 64, flush=True)

    torch.manual_seed(args.seed)
    model = common.load_model(args.checkpoint)
    print(
        f"[1/3] 检查点 = {args.checkpoint}（tick={model.tick}），"
        f"决策次数 = {args.decisions}",
        flush=True,
    )

    world = TopicWorld(
        TOPICS,
        boundary_symbol=model.substrate.config.boundary_symbol,
        force_switch_streak=5,
        recency_window=2,
    )
    stats = play(model, world, episodes=args.decisions, sample=True, learn=True)

    sequence = [int(index) for index in stats["topic_sequence"]]
    counts = Counter(sequence)
    distinct = len(counts)
    switch_count = sum(
        1 for a, b in zip(sequence, sequence[1:]) if a != b
    )
    top_share = max(counts.values()) / max(1, len(sequence))
    finite = all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in model.substrate.parameter_tensors()
    )

    print("\n[2/3] 探索轨迹", flush=True)
    print(f"  主题序列 = {sequence}", flush=True)
    print(f"  各主题次数 = {dict(sorted(counts.items()))}", flush=True)
    print(f"  切换次数 = {switch_count}，强制换向触发 = {stats['forced_switches']}", flush=True)
    print(f"  平均 reward = {stats['mean_reward']:.4f}，动作数 = {stats['actions']}", flush=True)

    coverage = distinct == len(TOPICS)
    switches_ok = switch_count >= 5
    share_ok = top_share <= 0.70
    no_crash = stats["crashes"] == 0
    b1_pass = coverage and switches_ok and share_ok and no_crash and finite

    print("\n[3/3] 判定", flush=True)
    print(
        f"  覆盖全部主题 = {'PASS' if coverage else 'FAIL'}"
        f"（distinct={distinct}/{len(TOPICS)}）",
        flush=True,
    )
    print(
        f"  切换次数 >= 5 = {'PASS' if switches_ok else 'FAIL'}（{switch_count}）",
        flush=True,
    )
    print(
        f"  最高频占比 <= 70% = {'PASS' if share_ok else 'FAIL'}（{top_share:.2f}）",
        flush=True,
    )
    print(f"  0 崩溃 = {'PASS' if no_crash else 'FAIL'}（{stats['crashes']}）", flush=True)
    print(f"  参数数值有限 = {'PASS' if finite else 'FAIL'}", flush=True)
    print("=" * 64, flush=True)
    print(f"B1-bis 原生版 判定: {'PASS' if b1_pass else 'FAIL'}", flush=True)
    print("=" * 64, flush=True)

    out_path = common.write_report(
        "seed_b1_explore",
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "B1-bis 原生版：探索自主性（突破锁定）",
            "checkpoint": str(args.checkpoint),
            "decisions": args.decisions,
            "topics": [topic.decode("utf-8") for topic in TOPICS],
            "play_stats": {
                key: value
                for key, value in stats.items()
                if key != "topic_sequence"
            },
            "topic_sequence": sequence,
            "topic_counts": {str(k): v for k, v in sorted(counts.items())},
            "distinct_topics": distinct,
            "switch_count": switch_count,
            "top_share": top_share,
            "criteria": {
                "coverage": coverage,
                "switches_ok": switches_ok,
                "share_ok": share_ok,
                "no_crash": no_crash,
                "finite": finite,
            },
            "b1_pass": b1_pass,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"报告已写入: {out_path}", flush=True)
    sys.exit(0 if b1_pass else 1)


if __name__ == "__main__":
    main()
