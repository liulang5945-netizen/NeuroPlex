"""测试 10 个单独训练的神经元协作效果：协作 argmax + 生成测试。

对比：
1. 个体 argmax（已测：73.4% 平均）
2. 协作 argmax（共振场聚合，全员参与）
3. 生成测试（协作 vs 个体）
"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble, NeuronConfig
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer, load_or_create_shared_embedding
)
import torch.nn.functional as F

device = 'cpu'

# 加载资源
domain_sp = load_domain_tokenizer('zh')
general_sp = load_general_tokenizer()
shared_emb = load_or_create_shared_embedding(device)

# 加载 10 个神经元
neurons = {}
for i in range(10):
    ckpt_path = f'data/neurons/neuron_zh_j{i}.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['neuron_config']
    cfg.dropout = 0.0
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.load_state_dict(ckpt['state_dict'], strict=False)
    neuron.eval()
    neurons[f'zh_j{i}'] = neuron
    print(f'  加载 zh_j{i}: argmax 实测中...')

print(f'\n已加载 {len(neurons)} 个神经元')

# 创建共振场和 ensemble
field_dim = next(iter(neurons.values())).config.field_dim
field = ResonanceField(dim=field_dim)
ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)

# 加载测试文本
texts = []
with open('data/distill/zh_texts.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if len(line) >= 20:
            texts.append(line)
test_texts = texts[-100:]
print(f'测试集: {len(test_texts)} 条文本')

# ── 1. 协作 argmax + PPL 评估（残差模式，与训练一致）──
print('\n' + '=' * 70)
print('协作 argmax + PPL 评估（残差模式 fusion=residual）')
print('=' * 70)

correct, total = 0, 0
total_ce = 0.0
with torch.no_grad():
    for idx, text in enumerate(test_texts):
        shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
        result = ensemble.forward_train(shared, temperature=1.0, fusion_mode='residual')
        fused_logits = result['fused_logits']

        shift_logits = fused_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()
        preds = shift_logits.argmax(dim=-1)
        valid = shift_mask.bool()
        correct += (preds[valid] == shift_targets[valid]).sum().item()
        total += valid.sum().item()

        # PPL 计算
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1), ignore_index=-100,
        )
        total_ce += ce.item()

        if (idx + 1) % 20 == 0:
            acc = correct / max(total, 1) * 100
            print(f'  进度: {idx+1}/100, 当前 argmax={acc:.1f}%')

collab_argmax = correct / max(total, 1) * 100
collab_ppl = math.exp(min(total_ce / len(test_texts), 20))
print(f'\n  协作 argmax: {collab_argmax:.1f}%')
print(f'  协作 PPL: {collab_ppl:.2f}')
print(f'  个体平均 argmax: 73.4%, 个体 PPL: 6.6~8.0')
print(f'  之前联合训练协作: argmax=68.1%, PPL=11.0')
print(f'  目标: argmax >= 85%')

# ── 2. 生成测试（协作，top-p sampling + 重复惩罚）──
print('\n' + '=' * 70)
print('协作生成测试（top-p sampling, p=0.9, temp=0.8, rep_penalty=0.6）')
print('=' * 70)

def top_p_sample(logits, p=0.9, temperature=0.8, rep_ids=None, rep_penalty=0.6):
    """Top-p (nucleus) sampling with repetition penalty."""
    logits = logits / temperature
    # 重复惩罚：对已生成的 token 降低概率
    if rep_ids:
        for tid in rep_ids:
            logits[tid] *= rep_penalty
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # 保留累计概率 <= p 的 token（至少保留 1 个）
    mask = cumulative - sorted_probs <= p
    mask[0] = True
    sorted_probs[~mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled = torch.multinomial(sorted_probs, 1)
    next_id = sorted_indices[sampled].item()
    return next_id

prompts = ['今天天气', '中国的首都', '人工智能是']
for prompt in prompts:
    print(f'\n  提示: "{prompt}"')
    with torch.no_grad():
        current_text = prompt
        generated_ids = []

        for step in range(50):
            shared, _, _ = batch_align_and_embed([current_text], domain_sp, general_sp, shared_emb)
            result = ensemble.forward_train(shared, temperature=1.0, fusion_mode='residual')
            fused_logits = result['fused_logits'][0]
            last_logits = fused_logits[-1]

            # top-p sampling + 重复惩罚（最近 8 个 token）
            rep_ids = generated_ids[-8:] if len(generated_ids) >= 2 else None
            next_id = top_p_sample(last_logits, p=0.9, temperature=0.8,
                                   rep_ids=rep_ids, rep_penalty=0.6)
            piece = domain_sp.id_to_piece(next_id)
            generated_ids.append(next_id)
            current_text = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])

            if step < 15 or step % 10 == 0:
                print(f'    step {step+1}: piece="{piece}"')

            # 遇到句号/换行停止
            if piece in ['。', '\n', '！', '？', '▁。', '▁\n']:
                break

        full_output = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])
        print(f'  完整输出: {full_output}')

# ── 3. 残差预测编码模式测试 ──
print('\n' + '=' * 70)
print('残差预测编码模式 argmax')
print('=' * 70)

correct_res, total_res = 0, 0
with torch.no_grad():
    for idx, text in enumerate(test_texts[:50]):  # 50 条快速测试
        shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
        result = ensemble.forward_train(shared, temperature=1.0, fusion_mode='residual')
        fused_logits = result['fused_logits']

        shift_logits = fused_logits[:, :-1, :]
        shift_targets = targets[:, 1:]
        shift_mask = mask[:, 1:]
        preds = shift_logits.argmax(dim=-1)
        valid = shift_mask.bool()
        correct_res += (preds[valid] == shift_targets[valid]).sum().item()
        total_res += valid.sum().item()

res_argmax = correct_res / max(total_res, 1) * 100
print(f'  残差模式 argmax: {res_argmax:.1f}% (50 条)')
print(f'  软加权模式 argmax: {collab_argmax:.1f}% (100 条)')
