"""诊断二：破坏到底在参数还是动力学/测量路径。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from seed import Seed, SeedJudge  # noqa: E402
from seed.sleep import SeedSleepScheduler  # noqa: E402


def load_model() -> Seed:
    state = torch.load(REPO / "checkpoints" / "seed_corpus.pt", weights_only=False)
    return Seed.from_checkpoint(state)


def probes() -> list[bytes]:
    texts = []
    path = REPO / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 4:
                break
            texts.append(json.loads(line)["text"].encode("utf-8"))
    return texts


def measure(model: Seed, texts: list[bytes]) -> float:
    return sum(model.score_bytes(t)["mean_surprise"] for t in texts) / len(texts)


def param_fingerprint(model: Seed) -> float:
    return float(sum(t.detach().abs().sum().item() for t in model.substrate.parameter_tensors()))


def main() -> None:
    torch.manual_seed(7)
    model = load_model()
    texts = probes()
    fp0 = param_fingerprint(model)
    base = measure(model, texts)
    print(f"基线 surprise={base:.3f} param_fp={fp0:.2f} tick={model.tick}", flush=True)

    judge = SeedJudge(model)
    scheduler = SeedSleepScheduler(model, judge)
    scheduler.experience(texts[0], learn=False)  # 只读经历，零学习
    after_exp = measure(model, texts)
    fp1 = param_fingerprint(model)
    print(f"仅 experience(learn=False): surprise={after_exp:.3f} param_fp={fp1:.2f} tick={model.tick}", flush=True)

    model.reset_dynamics(episode_id="probe")
    after_reset = measure(model, texts)
    print(f"experience 后 reset_dynamics: surprise={after_reset:.3f}", flush=True)

    # 检查点往返：保存 → 破坏 → 恢复
    snapshot = model.checkpoint()
    scheduler.night([texts[0]], cycles_per_text=2, learn=True)
    damaged = measure(model, texts)
    model.restore(snapshot)
    restored = measure(model, texts)
    fp2 = param_fingerprint(model)
    print(f"night(learn=True) 后: surprise={damaged:.3f}", flush=True)
    print(f"restore(checkpoint) 后: surprise={restored:.3f} param_fp={fp2:.2f} (原始 {fp0:.2f})", flush=True)


if __name__ == "__main__":
    main()
