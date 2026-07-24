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

# ── 1. 协作 argmax 评估 ──
print('\n' + '=' * 70)
print('协作 argmax 评估（forward_train 软加权融合）')
print('=' * 70)

correct, total = 0, 0
with torch.no_grad():
    for idx, text in enumerate(test_texts):
        shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
        result = ensemble.forward_train(shared, temperature=1.0)
        fused_logits = result['fused_logits']

        shift_logits = fused_logits[:, :-1, :]
        shift_targets = targets[:, 1:]
        shift_mask = mask[:, 1:]
        preds = shift_logits.argmax(dim=-1)
        valid = shift_mask.bool()
        correct += (preds[valid] == shift_targets[valid]).sum().item()
        total += valid.sum().item()

        if (idx + 1) % 20 == 0:
            acc = correct / max(total, 1) * 100
            print(f'  进度: {idx+1}/100, 当前 argmax={acc:.1f}%')

collab_argmax = correct / max(total, 1) * 100
print(f'\n  协作 argmax: {collab_argmax:.1f}%')
print(f'  个体平均 argmax: 73.4%')
print(f'  之前联合训练协作: 68.1%')
print(f'  目标: >= 85%')

# ── 2. 生成测试（协作） ──
print('\n' + '=' * 70)
print('协作生成测试')
print('=' * 70)

prompts = ['今天天气', '中国的首都', '人工智能是']
for prompt in prompts:
    print(f'\n  提示: "{prompt}"')
    with torch.no_grad():
        # 用提示文本做初始 forward
        current_text = prompt
        generated_pieces = []

        for step in range(30):
            shared, _, _ = batch_align_and_embed([current_text], domain_sp, general_sp, shared_emb)
            result = ensemble.forward_train(shared, temperature=1.0)
            fused_logits = result['fused_logits'][0]
            last_logits = fused_logits[-1]

            # 简单重复惩罚：降低已生成 token 的概率
            if len(generated_pieces) > 0:
                recent = set(generated_pieces[-5:])  # 最近 5 个 token
                for tid in recent:
                    last_logits[tid] *= 0.5

            next_id = last_logits.argmax().item()
            piece = domain_sp.id_to_piece(next_id)
            p = F.softmax(last_logits, dim=-1)[next_id].item()
            generated_pieces.append(next_id)
            current_text = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_pieces])

            if step < 10 or step % 5 == 0:
                print(f'    step {step+1}: piece="{piece}" p={p:.3f}')

            # 遇到句号/换行停止
            if piece in ['。', '\n', '。', '！', '？']:
                break

        full_output = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_pieces])
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
