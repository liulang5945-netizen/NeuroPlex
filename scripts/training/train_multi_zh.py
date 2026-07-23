"""训练多个同域 zh 神经元——验证"小神经元协作"核心假设。

实验设计：
  - 将 zh_texts.jsonl (3MB, ~13000条) 均分 3 份
  - 每份 ~1MB，训练一个独立 zh 神经元（zh_1, zh_2, zh_3）
  - 每个 neuron 只看到 1/3 的数据，个体更弱
  - 但 3 个 neuron 合计覆盖同样 3MB 数据
  - 对比：1×3MB 单干 vs 3×1MB 协作 → 协作能否涌现集体智能？

复用 train_neuron.py 的 train_one_neuron()，仅修改：
  - 数据来源：手动分割 zh_texts.jsonl
  - 保存名：neuron_zh_1.pt, neuron_zh_2.pt, neuron_zh_3.pt
  - neuron config domain 仍为 "zh"（共享 zh tokenizer）
"""
import sys
import os
sys.path.insert(0, "e:/taiji-neuron")

import torch
import sentencepiece as spm

from taiji.resonance import ResonanceNeuron, get_domain_neuron_config

# 复用 train_neuron.py 的函数
from scripts.training.train_neuron import (
    train_one_neuron,
    load_domain_tokenizer,
    load_general_tokenizer,
    load_or_create_shared_embedding,
    save_shared_embedding,
    DATA_DIR,
    OUTPUT_DIR,
)


def split_zh_texts(n_parts: int = 3) -> list[list[str]]:
    """将 zh_texts.jsonl 均分为 n_parts 份。"""
    cache_path = os.path.join(DATA_DIR, "zh_texts.jsonl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"zh 数据未缓存: {cache_path}，请先运行 train_neuron.py --domain zh")

    all_texts = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_texts.append(line)

    print(f"  总数据: {len(all_texts)} 条文本")

    # 均分（不 shuffle，保持原始分布差异）
    chunk_size = len(all_texts) // n_parts
    parts = []
    for i in range(n_parts):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_parts - 1 else len(all_texts)
        part = all_texts[start:end]
        parts.append(part)
        print(f"  Part {i+1}: {len(part)} 条 (index {start}-{end})")

    return parts


def main():
    NUM_NEURONS = 3
    NUM_STEPS = 2000  # 与单 neuron 实验一致
    SPEC = "compact"  # 与现有 neurons 一致

    print("=" * 60)
    print(f"多同域 zh 神经元训练 ({NUM_NEURONS} × 1MB)")
    print(f"  目标：验证共振场协作能否涌现集体智能")
    print(f"  对比：1×3MB 单干 (PPL=62.4) vs {NUM_NEURONS}×1MB 协作")
    print("=" * 60)

    # 1. 分割数据
    print(f"\n[1] 分割 zh_texts.jsonl → {NUM_NEURONS} 份...")
    parts = split_zh_texts(NUM_NEURONS)

    # 2. 加载共享资源
    print(f"\n[2] 加载共享资源...")
    device = "cpu"
    shared_embedding = load_or_create_shared_embedding(device)
    general_sp = load_general_tokenizer()
    domain_sp = load_domain_tokenizer("zh")
    print(f"  zh tokenizer vocab={domain_sp.vocab_size()}")
    print(f"  general tokenizer vocab={general_sp.vocab_size()}")

    # 3. 训练每个 neuron
    cfg_template = get_domain_neuron_config("zh", SPEC)

    for i, texts in enumerate(parts):
        neuron_id = f"zh_{i+1}"
        print(f"\n{'='*60}")
        print(f"[{neuron_id}] 训练 (part {i+1}/{NUM_NEURONS}, {len(texts)} texts)...")

        # 每个 neuron 用独立 config（不同随机初始化）
        cfg = get_domain_neuron_config("zh", SPEC)
        neuron = ResonanceNeuron(cfg).to(device)
        print(f"  Created: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
              f"vocab={cfg.vocab_size}, params≈{cfg.approx_params_m:.0f}M")

        save_path = os.path.join(OUTPUT_DIR, f"neuron_{neuron_id}.pt")
        result = train_one_neuron(
            neuron=neuron,
            texts=texts,
            domain="zh",  # config domain 仍为 zh（共享 tokenizer）
            shared_embedding=shared_embedding,
            domain_sp=domain_sp,
            general_sp=general_sp,
            num_steps=NUM_STEPS,
            batch_size=4,
            lr=5e-4,
            device=device,
            log_every=100,
            save_path=save_path,
        )
        print(f"  [{neuron_id}] done: loss={result['final_loss']:.4f}, PPL={result['final_ppl']:.1f}")

    # 保存 shared embedding
    save_shared_embedding(shared_embedding)

    # 总结
    print(f"\n{'='*60}")
    print("训练完成！对比基准：")
    print(f"  单 neuron (3MB): PPL=62.4")
    print(f"  本实验: 3 × 1MB neurons, 各自更弱")
    print(f"  下一步: 运行 test_collaboration.py 测试协作生成质量")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
