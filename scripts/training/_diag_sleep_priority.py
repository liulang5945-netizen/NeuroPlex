"""诊断七：toy 睡眠测试回放优先级分量分解。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from seed import Seed, SeedConfig  # noqa: E402
from seed.judge import SeedJudge  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402
from taiji import TaijiConfig  # noqa: E402

PATTERN = "ababcdcdabcd"


def main() -> None:
    config = SeedConfig(
        taiji=TaijiConfig(
            region_sizes=(12, 8),
            synapse_fan_in=4,
            motor_fan_in=6,
            memory_units=16,
            memory_fan_in=4,
            memory_readout_fan_in=6,
            memory_meta_dim=6,
            memory_iterations=2,
            memory_time_dim=4,
            memory_episode_dim=4,
            lateral_fan_in=4,
            seed=45,
        )
    )
    model = Seed(config, episode_id="sleep-test")
    model.learn_bytes(PATTERN.encode("ascii"), epochs=8)
    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)
    report = scheduler.experience(PATTERN.encode("ascii"))
    print(f"experience reward={report['reward']:.3f}", flush=True)
    substrate = model.substrate
    memory_state = substrate._state.memory
    for index in range(4):
        memory_state, replay = substrate.memory.replay(
            memory_state, tick=100 + index, generator=substrate._memory_rng
        )
        print(
            f"cycle {index}: priority={replay.priority:.4f} "
            f"novelty={replay.novelty:.3f} value={replay.value:.3f} "
            f"accepted={replay.accepted} "
            f"conf={memory_state.last_confidence:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
