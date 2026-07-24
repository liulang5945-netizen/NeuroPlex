"""快速生成测试 —— 多轮共振 vs 单轮，多种解码策略。

测试态极架构的核心特色：多轮共振（神经元通过场反复交互）能否提升生成质量。
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer, load_or_create_shared_embedding,
)
import torch.nn.functional as F

device = 'cpu'
domain_sp = load_domain_tokenizer('zh')
general_sp = load_general_tokenizer()
shared_emb = load_or_create_shared_embedding(device)

# 加载 10 个神经元
neurons = {}
for i in range(10):
    ckpt = torch.load(f'data/neurons/neuron_zh_j{i}.pt', map_location=device, weights_only=False)
    cfg = ckpt['neuron_config']
    cfg.dropout = 0.0
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.load_state_dict(ckpt['state_dict'], strict=False)
    neuron.eval()
    neurons[f'zh_j{i}'] = neuron

field_dim = next(iter(neurons.values())).config.field_dim

# 创建两个 ensemble：1轮（基线）和 3轮（多轮共振）
ens_1round = ResonanceEnsemble(neurons, ResonanceField(dim=field_dim), max_rounds=1)
ens_3round = ResonanceEnsemble(neurons, ResonanceField(dim=field_dim), max_rounds=3)
print(f'已加载 {len(neurons)} 个神经元，创建 1轮/3轮 ensemble\n')


def top_p_sample(logits, p=0.9, temperature=0.8, rep_ids=None, rep_penalty=0.5):
    logits = logits / temperature
    if rep_ids:
        for tid in rep_ids:
            logits[tid] *= rep_penalty
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs <= p
    mask[0] = True
    sorted_probs[~mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled = torch.multinomial(sorted_probs, 1)
    return sorted_indices[sampled].item()


def generate(ensemble, prompt, mode='greedy', max_steps=50, p=0.9, temp=0.8):
    current_text = prompt
    generated_ids = []

    for step in range(max_steps):
        with torch.no_grad():
            shared, _, _ = batch_align_and_embed([current_text], domain_sp, general_sp, shared_emb)
            # 用 forward() 支持多轮共振
            result = ensemble.forward(
                shared_embeddings=shared, return_logits=True,
                fusion_mode='residual', active_filter=False,
            )
            last_logits = result['weighted_logits'][0, -1]

            rep_ids = generated_ids[-8:] if len(generated_ids) >= 2 else None

            if mode == 'greedy':
                if rep_ids:
                    for tid in rep_ids:
                        last_logits[tid] *= 0.5
                next_id = last_logits.argmax().item()
            else:
                next_id = top_p_sample(last_logits, p=p, temperature=temp,
                                       rep_ids=rep_ids, rep_penalty=0.5)

            piece = domain_sp.id_to_piece(next_id)
            generated_ids.append(next_id)
            current_text = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])

            if piece in ['。', '\n', '！', '？', '▁。', '▁\n']:
                break

    return prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])


prompts = ['今天天气', '中国的首都', '人工智能是']

# ── 测试 1：单轮 vs 多轮（贪婪解码）──
for rounds, ensemble in [(1, ens_1round), (3, ens_3round)]:
    print('=' * 70)
    print(f'{rounds}轮共振 | 贪婪解码 + 重复惩罚')
    print('=' * 70)
    for prompt in prompts:
        output = generate(ensemble, prompt, mode='greedy')
        print(f'  [{prompt}] → {output}')
    print()

# ── 测试 2：多轮 + top-p sampling ──
print('=' * 70)
print(f'3轮共振 | top-p(p=0.9, t=0.8) + 重复惩罚')
print('=' * 70)
for prompt in prompts:
    output = generate(ens_3round, prompt, mode='top_p', p=0.9, temp=0.8)
    print(f'  [{prompt}] → {output}')
print()

# ── 测试 3：多轮 + 低温 top-p（更保守）──
print('=' * 70)
print(f'3轮共振 | top-p(p=0.8, t=0.5) + 重复惩罚（保守）')
print('=' * 70)
for prompt in prompts:
    output = generate(ens_3round, prompt, mode='top_p', p=0.8, temp=0.5)
    print(f'  [{prompt}] → {output}')
