"""smoke test: 验证 EOS 注入 + 统计筛选后数据量。

验证项：
1. batch_align_and_embed 返回的 targets 末尾是 domain EOS
2. sft_mask 末尾位置为 True（EOS 计入 loss）
3. 统计 max_answer_chars=150 筛选后数据量
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import sentencepiece as spm
import torch

from taiji.resonance.translator import batch_align_and_embed
from scripts.training.utils import load_dialogue_texts_multi

# 1. 加载 tokenizers
general_sp = spm.SentencePieceProcessor()
general_sp.Load(os.path.join(PROJECT_ROOT, "taiji", "domains", "general", "sp_general.model"))
domain_sp = spm.SentencePieceProcessor()
domain_sp.Load(os.path.join(PROJECT_ROOT, "taiji", "domains", "zh", "sp_zh.model"))

general_eos = general_sp.eos_id()
domain_eos = domain_sp.eos_id()
print(f"general EOS id={general_eos} piece={general_sp.id_to_piece(general_eos)!r}")
print(f"domain   EOS id={domain_eos} piece={domain_sp.id_to_piece(domain_eos)!r}")

# 2. 构造测试文本
texts = [
    "问：你好\n答：你好，很高兴认识你。",
    "问：1+1等于几？\n答：1+1等于2。",
]

shared_emb = torch.nn.Embedding(general_sp.GetPieceSize(), 512)
torch.nn.init.normal_(shared_emb.weight, mean=0.0, std=0.02)

emb, targets, mask, sft_mask = batch_align_and_embed(
    texts, domain_sp, general_sp, shared_emb,
    answer_marker="答：",
    answer_marker_mode="last",
)

print(f"\n[1] shape: emb={emb.shape}, targets={targets.shape}, sft_mask={sft_mask.shape}")

# 3. 验证每条样本末尾是 domain EOS
ok = 0
for b in range(len(texts)):
    L = int(mask[b].sum().item())
    last_tgt = int(targets[b, L-1].item())
    last_sft = bool(sft_mask[b, L-1].item())
    print(f"  样本{b}: L={L}, 末尾target={last_tgt}(EOS={domain_eos}), sft={last_sft}")
    if last_tgt == domain_eos and last_sft:
        ok += 1

print(f"\n[结果] {ok}/{len(texts)} 样本 EOS 注入正确")
assert ok == len(texts), "EOS 注入失败"

# 4. 统计筛选后数据量
print("\n[2] 统计筛选后数据量...")
dialogue_dir = os.path.join(PROJECT_ROOT, "data", "simple_zh")
for threshold in [100, 150, 200, 300, 0]:
    texts = load_dialogue_texts_multi(
        dialogue_dir,
        max_texts=1000000,
        max_answer_chars=threshold,
    )
    label = f"≤{threshold}字" if threshold > 0 else "不筛选"
    print(f"  max_answer_chars={threshold:>3} ({label:>8}): {len(texts):>6} 条")

print("\n=== smoke test 通过 ===")
