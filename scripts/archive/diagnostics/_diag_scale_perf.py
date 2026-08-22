"""Profile Taiji construction and per-observe cost at candidate scales."""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji, TaijiConfig  # noqa: E402

CANDIDATES = {
    "default": dict(),
    "scale-2": dict(
        region_sizes=(256, 192, 128),
        synapse_fan_in=48,
        motor_fan_in=96,
        memory_units=384,
        memory_fan_in=64,
    ),
    "scale-4": dict(
        region_sizes=(512, 384, 256),
        synapse_fan_in=96,
        motor_fan_in=192,
        memory_units=768,
        memory_fan_in=128,
    ),
}


def main() -> None:
    data = ("你好，世界。这是一个性能测试。" * 4).encode("utf-8")
    for name, overrides in CANDIDATES.items():
        start = time.perf_counter()
        model = Taiji(TaijiConfig(**overrides))
        build = time.perf_counter() - start
        # warmup
        model.learn_bytes(data, epochs=1)
        start = time.perf_counter()
        stats = model.learn_bytes(data, epochs=1)
        elapsed = time.perf_counter() - start
        ticks = int(stats["observations"]) + 1
        print(
            f"{name}: build={build:.2f}s params={model.parameter_count()/1e6:.2f}M "
            f"per-observe={elapsed / ticks * 1000:.2f}ms "
            f"acc={stats['online_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
