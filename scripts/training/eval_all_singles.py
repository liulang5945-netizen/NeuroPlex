"""快速评估所有 10 个神经元的个体 argmax + PPL，找最强个体。

诊断目的：
- 若最强个体 argmax ≈ 协作 argmax (74%) → 协作未带来增益，瓶颈在容量
- 若最强个体 argmax > 协作 argmax → 融合机制在伤害性能
- 若最强个体 argmax < 协作 argmax → 融合有效但容量仍不足
"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance import ResonanceNeuron
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer, load_or_create_shared_embedding,
)
import torch.nn.functional as F

device = 'cpu'
domain_sp = load_domain_tokenizer('zh')
general_sp = load_general_tokenizer()
shared_emb = load_or_create_shared_embedding(device)

# 加载测试文本（与 eval_collab.py 完全一致的 100 条）
texts = []
with open('data/distill/zh_texts.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if len(line) >= 20:
            texts.append(line)
test_texts = texts[-100:]
print(f'测试集: {len(test_texts)} 条文本\n')

results = []
for i in range(10):
    ckpt_path = f'data/neurons/neuron_zh_j{i}.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['neuron_config']
    cfg.dropout = 0.0
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.load_state_dict(ckpt['state_dict'], strict=False)
    neuron.eval()

    correct, total = 0, 0
    total_ce = 0.0
    with torch.no_grad():
        for text in test_texts:
            shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
            result = neuron.forward(shared, return_logits=True)
            logits = result['logits']
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            valid = shift_mask.bool()
            correct += (preds[valid] == shift_targets[valid]).sum().item()
            total += valid.sum().item()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1), ignore_index=-100,
            )
            total_ce += ce.item()

    argmax = correct / max(total, 1) * 100
    ppl = math.exp(min(total_ce / len(test_texts), 20))
    best_loss = ckpt.get('result', {}).get('best_loss', '?')
    best_step = ckpt.get('result', {}).get('best_step', '?')
    print(f'zh_j{i}: argmax={argmax:.1f}%, PPL={ppl:.2f}, best_loss={best_loss}@{best_step}')
    results.append((f'zh_j{i}', argmax, ppl, best_loss, best_step))

# 汇总
print('\n' + '=' * 70)
print('个体评估汇总（100 条测试集）')
print('=' * 70)
argmaxes = [r[1] for r in results]
ppls = [r[2] for r in results]
print(f'  argmax 范围: {min(argmaxes):.1f}% ~ {max(argmaxes):.1f}%')
print(f'  argmax 平均: {sum(argmaxes)/len(argmaxes):.1f}%')
print(f'  PPL 范围: {min(ppls):.2f} ~ {max(ppls):.2f}')
print(f'  PPL 平均: {sum(ppls)/len(ppls):.2f}')

best = max(results, key=lambda x: x[1])
worst = min(results, key=lambda x: x[1])
print(f'\n  最强个体: {best[0]} argmax={best[1]:.1f}% PPL={best[2]:.2f}')
print(f'  最弱个体: {worst[0]} argmax={worst[1]:.1f}% PPL={worst[2]:.2f}')

print(f'\n  对比协作: argmax=73.9%, PPL=4.45')
print(f'  对比目标: argmax>=85%, PPL<2.0')

if best[1] > 73.9:
    print(f'\n  ⚠️ 最强个体 {best[1]:.1f}% > 协作 73.9% → 融合机制在伤害性能！')
elif best[1] < 73.9 - 2:
    print(f'\n  ✅ 协作 73.9% > 最强个体 {best[1]:.1f}% → 融合有效，瓶颈在容量')
else:
    print(f'\n  ➡️ 协作 ≈ 最强个体 → 协作未带来增益，瓶颈在容量')
