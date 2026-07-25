"""约束生成测试：屏蔽 byte fallback token，验证 74% argmax 能否生成可读中文。

核心改进（对比 eval_leader.py 的生成测试）：
1. 屏蔽 byte fallback token（<0xXX> 格式）—— 生成乱码的直接原因
2. 屏蔽纯符号 token（>, ], ), 等非语言符号）
3. 只允许有效中文 token + 常用标点

测试对象：standard 族长（argmax=73.8%, PPL=3.05）+ 最强 compact（zh_j2, argmax=73.5%）
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

# ── 分析 domain tokenizer 的 token 分布 ──
print('=' * 70)
print('0. Domain tokenizer token 分析')
print('=' * 70)

vocab_size = domain_sp.vocab_size()
byte_tokens = []     # <0xXX> byte fallback
symbol_tokens = []   # 纯符号
chinese_tokens = []  # 包含中文
other_tokens = []    # 其他

for tid in range(vocab_size):
    piece = domain_sp.id_to_piece(tid)
    if piece.startswith('<0x') and piece.endswith('>'):
        byte_tokens.append(tid)
    elif piece.startswith('▁'):
        # SentencePiece 用 ▁ 表示空格前缀，检查去掉后是否中文
        core = piece[1:]
        if any('\u4e00' <= c <= '\u9fff' for c in core):
            chinese_tokens.append(tid)
        elif all(c in '，。！？、；：""''（）【】《》…—' or c.isalpha() for c in core):
            chinese_tokens.append(tid)  # 包含中文标点和字母
        else:
            other_tokens.append(tid)
    else:
        if any('\u4e00' <= c <= '\u9fff' for c in piece):
            chinese_tokens.append(tid)
        elif piece in ['，', '。', '！', '？', '、', '；', '：', '"', '"', ''', ''', '（', '）', '【', '】', '《', '》', '…', '—', '\n']:
            chinese_tokens.append(tid)  # 中文标点
        elif len(piece) > 0 and all(not c.isalnum() and not c.isspace() for c in piece):
            symbol_tokens.append(tid)
        else:
            other_tokens.append(tid)

print(f'  vocab_size: {vocab_size}')
print(f'  byte fallback tokens: {len(byte_tokens)} (将屏蔽)')
print(f'  纯符号 tokens: {len(symbol_tokens)} (将屏蔽)')
print(f'  有效中文 tokens: {len(chinese_tokens)} (保留)')
print(f'  其他 tokens: {len(other_tokens)} (保留)')

# 构建屏蔽 mask（True = 允许生成）
allow_mask = torch.ones(vocab_size, dtype=torch.bool)
for tid in byte_tokens:
    allow_mask[tid] = False
for tid in symbol_tokens:
    allow_mask[tid] = False

print(f'  屏蔽后可用 tokens: {allow_mask.sum().item()} / {vocab_size}')
print(f'  示例屏蔽 byte: {[domain_sp.id_to_piece(t) for t in byte_tokens[:5]]}')
print(f'  示例屏蔽符号: {[domain_sp.id_to_piece(t) for t in symbol_tokens[:5]]}')

# ── 加载 standard 族长 ──
print('\n' + '=' * 70)
print('1. 加载 standard 族长')
print('=' * 70)

leader_path = 'data/neurons/neuron_zh_leader0.pt'
ckpt = torch.load(leader_path, map_location=device, weights_only=False)
cfg = ckpt['neuron_config']
cfg.dropout = 0.0
leader = ResonanceNeuron(cfg).to(device)
leader.load_state_dict(ckpt['state_dict'], strict=False)
leader.eval()
print(f'  族长: spec={cfg.spec}, argmax=73.8%, PPL=3.05')

# ── 约束生成函数 ──
def constrained_top_p_sample(logits, allow_mask, p=0.9, temperature=0.8, rep_ids=None, rep_penalty=0.6):
    """Top-p sampling with byte token masking + repetition penalty."""
    logits = logits / temperature

    # 屏蔽不允许的 token（设为 -inf）
    logits[~allow_mask] = float('-inf')

    # 重复惩罚
    if rep_ids:
        for tid in rep_ids:
            if tid < len(logits):
                logits[tid] *= rep_penalty

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # 移除概率为 0 的 token（被屏蔽的）
    nonzero = sorted_probs > 0
    sorted_probs = sorted_probs[nonzero]
    sorted_indices = sorted_indices[nonzero]

    # Top-p sampling
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs <= p
    mask[0] = True
    sorted_probs[~mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled = torch.multinomial(sorted_probs, 1)
    return sorted_indices[sampled].item()

# ── 2. 约束生成测试 ──
print('\n' + '=' * 70)
print('2. 约束生成测试（屏蔽 byte + 符号 token）')
print('=' * 70)

prompts = ['今天天气', '中国的首都', '人工智能是', '请问你叫什么名字', '解释一下机器学习']

for prompt in prompts:
    print(f'\n  提示: "{prompt}"')
    with torch.no_grad():
        generated_ids = []
        current_text = prompt

        for step in range(80):
            shared, _, _ = batch_align_and_embed([current_text], domain_sp, general_sp, shared_emb)
            result = leader.forward(shared, return_logits=True)
            last_logits = result['logits'][0, -1]

            rep_ids = generated_ids[-10:] if len(generated_ids) >= 2 else None
            next_id = constrained_top_p_sample(
                last_logits, allow_mask, p=0.9, temperature=0.8,
                rep_ids=rep_ids, rep_penalty=0.6
            )

            piece = domain_sp.id_to_piece(next_id)
            generated_ids.append(next_id)
            current_text = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])

            if step < 20 or step % 10 == 0:
                print(f'    step {step+1}: piece="{piece}"')

            # 遇到句号/换行停止
            if piece in ['。', '\n', '！', '？', '▁。', '▁\n', '▁！', '▁？']:
                break

        full_output = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])
        print(f'  完整输出: {full_output}')

# ── 3. 对比：无约束 vs 约束 ──
print('\n' + '=' * 70)
print('3. 对比：无约束 vs 约束生成')
print('=' * 70)

def unconstrained_top_p_sample(logits, p=0.9, temperature=0.8, rep_ids=None, rep_penalty=0.6):
    """无约束 top-p sampling（对比基线）。"""
    logits = logits / temperature
    if rep_ids:
        for tid in rep_ids:
            if tid < len(logits):
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

prompt = '今天天气'
print(f'\n  提示: "{prompt}"')

# 无约束
with torch.no_grad():
    generated_ids = []
    current_text = prompt
    for step in range(50):
        shared, _, _ = batch_align_and_embed([current_text], domain_sp, general_sp, shared_emb)
        result = leader.forward(shared, return_logits=True)
        last_logits = result['logits'][0, -1]
        rep_ids = generated_ids[-8:] if len(generated_ids) >= 2 else None
        next_id = unconstrained_top_p_sample(last_logits, p=0.9, temperature=0.8,
                                              rep_ids=rep_ids, rep_penalty=0.6)
        piece = domain_sp.id_to_piece(next_id)
        generated_ids.append(next_id)
        current_text = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])
        if piece in ['。', '\n', '！', '？']:
            break
    unconstrained_output = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])

# 约束
with torch.no_grad():
    generated_ids = []
    current_text = prompt
    for step in range(50):
        shared, _, _ = batch_align_and_embed([current_text], domain_sp, general_sp, shared_emb)
        result = leader.forward(shared, return_logits=True)
        last_logits = result['logits'][0, -1]
        rep_ids = generated_ids[-8:] if len(generated_ids) >= 2 else None
        next_id = constrained_top_p_sample(last_logits, allow_mask, p=0.9, temperature=0.8,
                                            rep_ids=rep_ids, rep_penalty=0.6)
        piece = domain_sp.id_to_piece(next_id)
        generated_ids.append(next_id)
        current_text = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])
        if piece in ['。', '\n', '！', '？']:
            break
    constrained_output = prompt + ''.join([domain_sp.id_to_piece(id) for id in generated_ids])

print(f'\n  无约束: {unconstrained_output}')
print(f'  约束:   {constrained_output}')

print(f'\n{"="*70}')
print('评估完成')
print(f'{"="*70}')
