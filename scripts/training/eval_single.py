"""快速评估单个神经元的 argmax + PPL。

Usage:
    python -u scripts/training/eval_single.py --neuron_id 0
    python -u scripts/training/eval_single.py --neuron_id 0 --compare_before
"""
import sys, os, torch, math, argparse
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

# 加载测试文本
texts = []
with open('data/distill/zh_texts.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if len(line) >= 20:
            texts.append(line)
test_texts = texts[-100:]
print(f'测试集: {len(test_texts)} 条文本\n')


def eval_neuron(ckpt_path, label=''):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['neuron_config']
    cfg.dropout = 0.0
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.load_state_dict(ckpt['state_dict'], strict=False)
    neuron.eval()

    old_result = ckpt.get('result', {})
    print(f'--- {label} ---')
    print(f'  ckpt: best_loss={old_result.get("best_loss", "?")}, '
          f'best_step={old_result.get("best_step", "?")}')

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
    print(f'  argmax: {argmax:.1f}%')
    print(f'  PPL: {ppl:.2f}')
    print(f'  目标: argmax>=85%, PPL<2.0\n')
    return argmax, ppl


parser = argparse.ArgumentParser()
parser.add_argument('--neuron_id', type=int, default=0)
parser.add_argument('--compare_before', action='store_true',
                    help='同时评估续训前后的对比（需备份了旧ckpt）')
args = parser.parse_args()

ckpt_path = f'data/neurons/neuron_zh_j{args.neuron_id}.pt'
eval_neuron(ckpt_path, label=f'zh_j{args.neuron_id} (当前)')

if args.compare_before:
    backup_path = f'data/neurons/neuron_zh_j{args.neuron_id}_before_resume.pt'
    if os.path.exists(backup_path):
        eval_neuron(backup_path, label=f'zh_j{args.neuron_id} (续训前)')
    else:
        print(f'未找到备份: {backup_path}')
