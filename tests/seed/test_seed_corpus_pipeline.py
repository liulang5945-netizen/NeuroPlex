"""阶段 1 原生数据管线的失败测试。

计划要求：``scripts/training/train_seed_corpus.py`` 复用 ``data/simple_zh/``
语料，但以 raw-byte 流喂入 ``Seed.learn_bytes`` 同款的 ``observe`` 通道；
会话边界用 ``boundary_symbol``，对话结构序列化为文本标记（不引入
tokenizer）；分片流式多 epoch 训练 + 周期 ``checkpoint()`` 落盘
（seed-native-v1 信封）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from seed import Seed, SeedConfig
from taiji import TaijiConfig

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "train_seed_corpus.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "train_seed_corpus", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_config() -> SeedConfig:
    return SeedConfig(
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
            seed=41,
        )
    )


def _write_corpus(path: Path) -> None:
    documents = [
        {"text": "问：你好。\n答：你好，很高兴见到你。"},
        {"text": "水的沸点在标准大气压下是一百摄氏度。"},
        {"text": "问：推荐一本书。\n答：可以读一读《活着》。"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


def test_corpus_stream_is_raw_bytes_separated_by_boundary_symbols(tmp_path) -> None:
    trainer = _module()
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    boundary = TaijiConfig().boundary_symbol

    symbols = list(trainer.iter_corpus_symbols([corpus]))

    assert symbols[0] == boundary
    assert symbols.count(boundary) == 3  # one session boundary per document
    assert all(0 <= value <= 256 for value in symbols)
    payload = [value for value in symbols if value != boundary]
    expected = "".join(
        "问：你好。\n答：你好，很高兴见到你。"
        "水的沸点在标准大气压下是一百摄氏度。"
        "问：推荐一本书。\n答：可以读一读《活着》。"
    ).encode("utf-8")
    assert bytes(payload) == expected


def test_training_run_streams_checkpoints_and_progress(tmp_path) -> None:
    trainer = _module()
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    checkpoint_path = tmp_path / "seed.pt"
    progress_path = tmp_path / "progress.jsonl"

    summary = trainer.run_training(
        corpus_paths=[corpus],
        config=_small_config(),
        epochs=2,
        checkpoint_path=checkpoint_path,
        progress_path=progress_path,
        checkpoint_every=48,
        progress_every=48,
    )

    assert checkpoint_path.is_file()
    restored = Seed.from_checkpoint(
        __import__("torch").load(checkpoint_path, weights_only=False)
    )
    assert restored.checkpoint()["format"] == "seed-native-v1"

    entries = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) >= 2
    assert entries[-1]["ticks"] > entries[0]["ticks"]
    assert "online_accuracy" in entries[-1]
    assert "mean_surprise" in entries[-1]
    assert summary["ticks"] == entries[-1]["ticks"]
