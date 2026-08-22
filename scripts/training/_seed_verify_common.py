"""阶段 3 判据验证共用工具：检查点加载 / judge 校准 / 面板测量 / 报告落盘。

口径与阶段 2 的 ``verify_seed_a1_judge.py`` 保持一致：质量信号只来自原生
器官 ``SeedJudge``（以 -loss 为目标的闭式岭回归局部校准），面板为 A1 冻结
的 24 条真实提示（dialogue/knowledge/unfamiliar 三组）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from seed import Seed, SeedJudge  # noqa: E402


def load_eval_module():
    script = REPO / "scripts" / "training" / "eval_seed_corpus.py"
    spec = importlib.util.spec_from_file_location("eval_seed_corpus", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_paths() -> List[Path]:
    base = REPO / "data" / "simple_zh"
    return [
        base / "dialogue_extended_clean.jsonl",
        base / "alpaca_zh_sft_clean.jsonl",
        base / "class_a_chinese.jsonl",
    ]


def corpus_documents(limit: int, skip: int = 0) -> List[str]:
    documents: List[str] = []
    seen = 0
    for path in corpus_paths():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                if seen >= skip:
                    documents.append(text)
                    if len(documents) >= limit:
                        return documents
                seen += 1
    return documents


def load_model(checkpoint: str) -> Seed:
    state = torch.load(checkpoint, weights_only=False)
    return Seed.from_checkpoint(state)


def calibrated_judge(model: Seed, documents: int = 48) -> SeedJudge:
    """Judge 校准与 A1 同口径：以 -loss 为目标做闭式岭回归。"""

    judge = SeedJudge(model)
    calibration = []
    for text in corpus_documents(documents):
        data = text.encode("utf-8")
        loss = model.score_bytes(data)["mean_surprise"]
        calibration.append((judge.features(data), -loss))
    judge.calibrate(calibration)
    return judge


def measure_panel(model: Seed, judge: SeedJudge) -> Dict[str, object]:
    """24 条冻结面板的质量测量（三组 mean/std + 全体均值）。"""

    panel = load_eval_module().PROMPT_PANEL
    groups: Dict[str, Dict[str, object]] = {}
    all_means: List[float] = []
    for group_name, prompts in panel.items():
        qualities = [
            float(judge.score(text.encode("utf-8"))["quality"])
            for text in prompts
        ]
        mean = sum(qualities) / len(qualities)
        std = (
            sum((value - mean) ** 2 for value in qualities) / len(qualities)
        ) ** 0.5
        groups[group_name] = {"mean": mean, "std": std, "qualities": qualities}
        all_means.append(mean)
    return {"groups": groups, "overall_mean": sum(all_means) / len(all_means)}


def parameter_delta(model: Seed, before: List[torch.Tensor]) -> float:
    after = model.substrate.parameter_tensors()
    return float(
        sum(
            float((a.detach() - b.detach()).abs().sum().item())
            for a, b in zip(after, before)
        )
    )


def write_report(name: str, payload: Dict[str, object]) -> Path:
    reports = REPO / "reports"
    reports.mkdir(exist_ok=True)
    out_path = reports / f"{name}_{time.strftime('%Y%m%d')}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return out_path


def panel_texts_by_quality(
    judge: SeedJudge,
) -> List[Tuple[str, float, bytes]]:
    """全部 24 条面板文本按 judge 质量升序（最差的在前）。"""

    panel = load_eval_module().PROMPT_PANEL
    scored: List[Tuple[str, float, bytes]] = []
    for group_name, prompts in panel.items():
        for text in prompts:
            data = text.encode("utf-8")
            quality = float(judge.score(data)["quality"])
            scored.append((group_name, quality, data))
    scored.sort(key=lambda item: item[1])
    return scored
