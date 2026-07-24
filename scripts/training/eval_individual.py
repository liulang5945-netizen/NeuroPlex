"""评估单独训练的神经元：argmax 准确率 + 生成测试。"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance import ResonanceNeuron
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer, load_or_create_shared_embedding
)
import torch.nn.functional as F

# 加载
domain_sp = load_domain_tokenizer('zh')
general_sp = load_general_tokenizer()
shared_emb = load_or_create_shared_embedding('cpu')

# 加载测试文本（从数据末尾取，避免和训练数据重叠）
texts = []
with open('data/distill/zh_texts.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if len(line) >= 20:
            texts.append(line)
# 取最后 100 条作为测试集
test_texts = texts[-100:]
print(f'测试集: {len(test_texts)} 条文本')
print('=' * 70)
print('argmax 评估')
print('=' * 70)

all_argmax = []
for i in range(10):
    ckpt_path = f'data/neurons/neuron_zh_j{i}.pt'
    if not os.path.exists(ckpt_path):
        continue
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['neuron_config']
    cfg.dropout = 0.0  # 评估时关闭 dropout
    neuron = ResonanceNeuron(cfg)
    neuron.load_state_dict(ckpt['state_dict'], strict=False)
    neuron.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for text in test_texts:
            shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
            result = neuron.forward(shared, return_logits=True)
            logits = result['logits']
            shift_logits = logits[:, :-1, :]
            shift_targets = targets[:, 1:]
            shift_mask = mask[:, 1:]
            preds = shift_logits.argmax(dim=-1)
            valid = shift_mask.bool()
            correct += (preds[valid] == shift_targets[valid]).sum().item()
            total += valid.sum().item()

    argmax_acc = correct / max(total, 1) * 100
    all_argmax.append(argmax_acc)
    best_loss = ckpt['result']['best_loss']
    print(f'  zh_j{i}: argmax={argmax_acc:.1f}%  best_loss={best_loss:.4f}')

avg_argmax = sum(all_argmax) / len(all_argmax)
print(f'\n  平均 argmax: {avg_argmax:.1f}%')
print(f'  最高 argmax: {max(all_argmax):.1f}%')
print(f'  最低 argmax: {min(all_argmax):.1f}%')
print(f'\n  目标: >= 85% (连贯生成)')
print(f'  之前联合训练: 68.1%')

# 生成测试
print('\n' + '=' * 70)
print('生成测试（最强神经元）')
print('=' * 70)

best_idx = all_argmax.index(max(all_argmax))
ckpt = torch.load(f'data/neurons/neuron_zh_j{best_idx}.pt', map_location='cpu', weights_only=False)
cfg = ckpt['neuron_config']
cfg.dropout = 0.0
neuron = ResonanceNeuron(cfg)
neuron.load_state_dict(ckpt['state_dict'], strict=False)
neuron.eval()

prompts = ['今天天气', '中国的首都', '人工智能是']
for prompt in prompts:
    print(f'\n  提示: "{prompt}"')
    with torch.no_grad():
        shared, targets, mask = batch_align_and_embed([prompt], domain_sp, general_sp, shared_emb)
        result = neuron.forward(shared, return_logits=True)
        logits = result['logits'][0]  # [L, V]

        # 自回归生成 20 个 token
        generated_ids = []
        for step in range(20):
            last_logits = logits[-1]  # [V]
            next_id = last_logits.argmax().item()
            generated_ids.append(next_id)
            # decode
            piece = domain_sp.id_to_piece(next_id)
            print(f'    step {step+1}: id={next_id} piece="{piece}" p={F.softmax(last_logits, dim=-1)[next_id].item():.3f}')

            # 用新 token 继续生成
            next_text = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])
            shared, _, _ = batch_align_and_embed([next_text], domain_sp, general_sp, shared_emb)
            result = neuron.forward(shared, return_logits=True)
            logits = result['logits'][0]

        full_output = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])
        print(f'  完整输出: {full_output}')
