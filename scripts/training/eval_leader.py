"""评估 standard 族长神经元 + 组装 1+9 混合规格协作测试。

评估流程：
1. 族长个体评估：argmax + PPL（对比 compact 73% 天花板）
2. 生成测试：top-p sampling + 重复惩罚
3. 1+9 混合协作评估：standard 族长 + 9 compact 跟随者（突触投影）
"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer, load_or_create_shared_embedding,
)
import torch.nn as nn
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

# ── 1. 族长个体评估 ──
print('=' * 70)
print('1. Standard 族长个体评估')
print('=' * 70)

leader_path = 'data/neurons/neuron_zh_leader0.pt'
ckpt = torch.load(leader_path, map_location=device, weights_only=False)
cfg = ckpt['neuron_config']
cfg.dropout = 0.0
leader = ResonanceNeuron(cfg).to(device)
leader.load_state_dict(ckpt['state_dict'], strict=False)
leader.eval()
n_params = sum(p.numel() for p in leader.parameters())
print(f'  族长: spec={cfg.spec}, params={n_params/1e6:.1f}M, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}')
print(f'  field_dim={cfg.field_dim}, unified_field_dim={cfg.unified_field_dim}')
print(f'  best_loss={ckpt["result"]["best_loss"]:.4f}@step{ckpt["result"]["best_step"]}')

correct, total = 0, 0
total_ce = 0.0
with torch.no_grad():
    for idx, text in enumerate(test_texts):
        shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
        result = leader.forward(shared, return_logits=True)
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
        if (idx + 1) % 20 == 0:
            acc = correct / max(total, 1) * 100
            print(f'  进度: {idx+1}/100, 当前 argmax={acc:.1f}%')

leader_argmax = correct / max(total, 1) * 100
leader_ppl = math.exp(min(total_ce / len(test_texts), 20))
print(f'\n  族长 argmax: {leader_argmax:.1f}%')
print(f'  族长 PPL: {leader_ppl:.2f}')
print(f'  对比 compact 平均: argmax=72.9%, PPL=3.39')
print(f'  对比 compact 最强: argmax=73.5%, PPL=3.20')
print(f'  目标: argmax >= 85%')

# ── 2. 生成测试 ──
print('\n' + '=' * 70)
print('2. 族长生成测试（top-p sampling, p=0.9, temp=0.8, rep_penalty=0.6）')
print('=' * 70)

def top_p_sample(logits, p=0.9, temperature=0.8, rep_ids=None, rep_penalty=0.6):
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

prompts = ['今天天气', '中国的首都', '人工智能是']
for prompt in prompts:
    print(f'\n  提示: "{prompt}"')
    with torch.no_grad():
        generated_ids = []
        current_text = prompt
        for step in range(50):
            shared, _, _ = batch_align_and_embed([current_text], domain_sp, general_sp, shared_emb)
            result = leader.forward(shared, return_logits=True)
            last_logits = result['logits'][0, -1]
            rep_ids = generated_ids[-8:] if len(generated_ids) >= 2 else None
            next_id = top_p_sample(last_logits, p=0.9, temperature=0.8,
                                   rep_ids=rep_ids, rep_penalty=0.6)
            piece = domain_sp.id_to_piece(next_id)
            generated_ids.append(next_id)
            current_text = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])
            if step < 15 or step % 10 == 0:
                print(f'    step {step+1}: piece="{piece}"')
            if piece in ['。', '\n', '！', '？', '▁。', '▁\n']:
                break
        full_output = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])
        print(f'  完整输出: {full_output}')

# ── 3. 1+9 混合规格协作评估（族长+9 compact 跟随者）──
print('\n' + '=' * 70)
print('3. 1+9 混合规格协作评估（standard 族长 + 9 compact 跟随者）')
print('=' * 70)

# 加载 9 个 compact 跟随者
neurons = {'zh_leader0': leader}
for i in range(9):  # j0~j8 作为跟随者
    ckpt_path = f'data/neurons/neuron_zh_j{i}.pt'
    if not os.path.exists(ckpt_path):
        continue
    ckpt_i = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_i = ckpt_i['neuron_config']
    cfg_i.dropout = 0.0
    # 用原始 config 加载（无 unified_field_dim，保持 field_read_layers [512, 2048]）
    neuron_i = ResonanceNeuron(cfg_i).to(device)
    neuron_i.load_state_dict(ckpt_i['state_dict'], strict=False)
    neuron_i.eval()

    # 手动升级到 unified_field_dim=4096（identity 初始化，保留原始训练权重）
    old_field_dim = cfg_i.field_dim  # 2048
    unified_dim = 4096
    # 1. 创建 field_projector: Linear(2048 → 4096), identity-like 初始化
    projector = nn.Linear(old_field_dim, unified_dim, bias=False)
    with torch.no_grad():
        projector.weight.zero_()
        for j in range(old_field_dim):
            projector.weight[j, j] = 1.0  # identity block
    neuron_i.field_projector = projector
    # 2. 扩展 field_read_layers: [512, 2048] → [512, 4096], 零填充
    new_read_layers = nn.ModuleList()
    for old_layer in neuron_i.field_read_layers:
        new_layer = nn.Linear(unified_dim, cfg_i.hidden_size, bias=False)
        with torch.no_grad():
            new_layer.weight.zero_()
            new_layer.weight[:, :old_field_dim] = old_layer.weight  # 保留原始权重
        new_read_layers.append(new_layer)
    neuron_i.field_read_layers = new_read_layers
    # 更新 config 记录
    neuron_i.config.unified_field_dim = unified_dim

    neurons[f'zh_j{i}'] = neuron_i

print(f'  加载 {len(neurons)} 个神经元: 1 standard + {len(neurons)-1} compact')
print(f'  族长 field_dim={neurons["zh_leader0"].config.field_dim}, '
      f'跟随者 field_dim={neurons["zh_j0"].config.field_dim}')
print(f'  unified_field_dim=4096（突触投影处理混合规格）')

# 创建共振场和 ensemble
field_dim = 4096  # 使用统一场维度
field = ResonanceField(dim=field_dim)
ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)

# 协作 argmax + PPL
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
print(f'\n  1+9 协作 argmax: {collab_argmax:.1f}%')
print(f'  1+9 协作 PPL: {collab_ppl:.2f}')
print(f'  族长个体 argmax: {leader_argmax:.1f}%, PPL: {leader_ppl:.2f}')
print(f'  对比 10×compact 协作: argmax=73.9%, PPL=4.45')
print(f'  目标: argmax >= 85%')

if collab_argmax > leader_argmax:
    print(f'\n  ✅ 协作 > 族长个体 (+{collab_argmax - leader_argmax:.1f}%) → 跟随者有增益')
elif collab_argmax < leader_argmax:
    print(f'\n  ⚠️ 协作 < 族长个体 ({collab_argmax - leader_argmax:.1f}%) → 融合有噪声')
else:
    print(f'\n  ➡️ 协作 ≈ 族长个体 → 跟随者无增益')

print(f'\n{"="*70}')
print('评估完成')
print(f'{"="*70}')
