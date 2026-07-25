"""深入分析 argmax 73% 天花板的根因。

诊断三个维度：
1. Top-k 准确率分布：top-1/3/5/10 → 判断模型是否"理解了但不够精确"
2. 错误 token 类型分布：byte fallback / 标点 / 汉字 / 数字 / 英文 → 判断是否分词器问题
3. 正确答案排名分布：错误时正确答案排第几 → 判断是"差一点"还是"完全错"

输出诊断报告，指导下一步方向：
- 若 top-5 > 95% → top-p sampling 可生成连贯文本，无需 argmax 85%
- 若错误集中 byte fallback → 修复分词器
- 若错误均匀分布 → 训练/架构问题
"""
from __future__ import annotations

import os
import sys
import math
import torch
import torch.nn.functional as F
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sentencepiece as spm
from taiji.resonance import ResonanceNeuron
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer, load_or_create_shared_embedding,
)

device = 'cpu'
domain_sp = load_domain_tokenizer('zh')
general_sp = load_general_tokenizer()
shared_emb = load_or_create_shared_embedding(device)


def classify_token(piece: str) -> str:
    """对 token piece 分类。"""
    # byte fallback tokens: <0xXX>
    if piece.startswith('<0x') and piece.endswith('>'):
        return 'byte_fallback'
    # 标点
    puncts = set('。，！？、；：""''（）【】《》〈〉「」『』…—·,.!?;:"\'()[]{}<>')
    if all(c in puncts for c in piece if c):
        return 'punctuation'
    # 纯数字
    if piece.replace('▁', '').isdigit():
        return 'digit'
    # 纯英文/拉丁字母
    stripped = piece.replace('▁', '')
    if stripped.isascii() and stripped.isalpha():
        return 'english'
    # 汉字（含 ▁ 前缀）
    han_chars = [c for c in piece if '\u4e00' <= c <= '\u9fff']
    if han_chars:
        return 'chinese'
    # 空白/特殊
    if piece in ['▁', '\n', '▁\n', '', ' ']:
        return 'whitespace'
    return 'other'


def analyze_ceiling():
    # 加载测试文本
    texts = []
    with open('data/distill/zh_texts.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) >= 20:
                texts.append(line)
    test_texts = texts[-100:]
    print(f'测试集: {len(test_texts)} 条文本\n')

    # 加载 standard 族长（最强模型）
    leader_path = 'data/neurons/neuron_zh_leader0.pt'
    ckpt = torch.load(leader_path, map_location=device, weights_only=False)
    cfg = ckpt['neuron_config']
    cfg.dropout = 0.0
    neuron = ResonanceNeuron(cfg).to(device)
    neuron.load_state_dict(ckpt['state_dict'], strict=False)
    neuron.eval()
    print(f'模型: spec={cfg.spec}, params={sum(p.numel() for p in neuron.parameters())/1e6:.1f}M')
    print(f'      hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, vocab={cfg.vocab_size}\n')

    # ── 收集所有预测结果 ──
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_tokens = 0

    # 错误分析统计
    error_types = Counter()        # 错误 token 的类型
    correct_types = Counter()      # 正确 token 的类型
    error_target_ranks = []        # 错误时正确答案的排名
    error_target_pieces = []       # 错误时正确答案的 piece（采样前 200 个）
    correct_confidences = []       # 正确预测的置信度
    all_confidences = []           # 所有预测（正确+错误）的置信度

    # per-type top-1 准确率
    type_correct = Counter()
    type_total = Counter()

    # 每个预测的 (is_correct, confidence) 用于置信度分析
    prediction_records: list[tuple[bool, float]] = []

    print('=' * 70)
    print('开始逐 token 分析（100 条测试集）...')
    print('=' * 70)

    with torch.no_grad():
        for idx, text in enumerate(test_texts):
            shared, targets, mask = batch_align_and_embed([text], domain_sp, general_sp, shared_emb)
            result = neuron.forward(shared, return_logits=True)
            logits = result['logits']  # [1, L, vocab]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()

            # Top-k
            topk_vals, topk_ids = torch.topk(shift_logits, k=10, dim=-1)  # [1, L-1, 10]
            probs = F.softmax(shift_logits, dim=-1)  # [1, L-1, vocab]

            for pos in range(shift_logits.size(1)):
                if not shift_mask[0, pos]:
                    continue
                target_id = shift_targets[0, pos].item()
                pred_top1 = topk_ids[0, pos, 0].item()
                pred_top3 = topk_ids[0, pos, :3].tolist()
                pred_top5 = topk_ids[0, pos, :5].tolist()
                pred_top10 = topk_ids[0, pos, :10].tolist()

                # Top-k 准确率
                top1_correct += (pred_top1 == target_id)
                top3_correct += (target_id in pred_top3)
                top5_correct += (target_id in pred_top5)
                top10_correct += (target_id in pred_top10)
                total_tokens += 1

                # 置信度
                pred_conf = probs[0, pos, pred_top1].item()
                all_confidences.append(pred_conf)
                is_correct = (pred_top1 == target_id)
                prediction_records.append((is_correct, pred_conf))
                if is_correct:
                    correct_confidences.append(pred_conf)

                # Token 类型分析（防止 target_id 超出域词表范围）
                domain_vocab_size = domain_sp.get_piece_size()
                if 0 <= target_id < domain_vocab_size:
                    target_piece = domain_sp.id_to_piece(target_id)
                else:
                    target_piece = f'<id_{target_id}>'
                if 0 <= pred_top1 < domain_vocab_size:
                    pred_piece = domain_sp.id_to_piece(pred_top1)
                else:
                    pred_piece = f'<id_{pred_top1}>'
                t_type = classify_token(target_piece)

                type_total[t_type] += 1
                if pred_top1 == target_id:
                    type_correct[t_type] += 1
                    correct_types[t_type] += 1
                else:
                    error_types[t_type] += 1
                    # 正确答案的排名
                    if target_id in pred_top10:
                        rank = pred_top10.index(target_id) + 1
                    else:
                        rank = 11  # 不在 top-10
                    error_target_ranks.append(rank)
                    if len(error_target_pieces) < 200:
                        error_target_pieces.append((target_piece, pred_piece, t_type, rank))

            if (idx + 1) % 20 == 0:
                acc = top1_correct / max(total_tokens, 1) * 100
                print(f'  进度: {idx+1}/100, 当前 top-1={acc:.1f}%')

    # ── 输出诊断报告 ──
    print('\n' + '=' * 70)
    print('诊断报告：argmax 73% 天花板根因分析')
    print('=' * 70)

    # 1. Top-k 准确率
    print('\n## 1. Top-k 准确率分布')
    print(f'  Top-1  = {top1_correct/total_tokens*100:.1f}%')
    print(f'  Top-3  = {top3_correct/total_tokens*100:.1f}%')
    print(f'  Top-5  = {top5_correct/total_tokens*100:.1f}%')
    print(f'  Top-10 = {top10_correct/total_tokens*100:.1f}%')
    top1_to_top5_gap = (top5_correct - top1_correct) / total_tokens * 100
    print(f'  Top-1→Top-5 提升 = +{top1_to_top5_gap:.1f}%')

    if top5_correct / total_tokens > 0.95:
        print('  → ✅ Top-5 > 95%：模型理解了但 top-1 不够精确 → top-p sampling 可能生成连贯文本')
    elif top5_correct / total_tokens > 0.85:
        print('  → ⚠️ Top-5 85-95%：部分理解，top-p sampling 可能有改善但不够')
    else:
        print('  → ❌ Top-5 < 85%：模型理解不足，非 top-1 精度问题')

    # 2. 错误 token 类型分布
    print('\n## 2. 错误 token 类型分布')
    print(f'  {"类型":<15} {"总数":>6} {"正确":>6} {"错误":>6} {"准确率":>8} {"错误占比":>8}')
    print(f'  {"-"*15} {"-"*6} {"-"*6} {"-"*6} {"-"*8} {"-"*8}')
    for t_type in sorted(type_total.keys(), key=lambda x: -type_total[x]):
        total_t = type_total[t_type]
        correct_t = type_correct[t_type]
        error_t = total_t - correct_t
        acc_t = correct_t / total_t * 100 if total_t > 0 else 0
        err_share = error_t / max(top1_correct, 1)
        err_share = (total_tokens - top1_correct)
        err_pct = error_t / max(err_share, 1) * 100 if err_share > 0 else 0
        print(f'  {t_type:<15} {total_t:>6} {correct_t:>6} {error_t:>6} {acc_t:>7.1f}% {err_pct:>7.1f}%')

    # 3. 错误时正确答案的排名分布
    print('\n## 3. 错误时正确答案的排名分布')
    if error_target_ranks:
        rank_counter = Counter(error_target_ranks)
        print(f'  总错误数: {len(error_target_ranks)}')
        for rank in sorted(rank_counter.keys()):
            count = rank_counter[rank]
            pct = count / len(error_target_ranks) * 100
            bar = '█' * int(pct / 2)
            rank_label = f'#{rank}' if rank <= 10 else '#>10'
            print(f'  排名 {rank_label:>5}: {count:>5} ({pct:>5.1f}%) {bar}')

        in_top5 = sum(1 for r in error_target_ranks if r <= 5)
        in_top10 = sum(1 for r in error_target_ranks if r <= 10)
        beyond_top10 = sum(1 for r in error_target_ranks if r > 10)
        print(f'\n  错误中正确答案在 Top-5 内: {in_top5} ({in_top5/len(error_target_ranks)*100:.1f}%)')
        print(f'  错误中正确答案在 Top-10 内: {in_top10} ({in_top10/len(error_target_ranks)*100:.1f}%)')
        print(f'  错误中正确答案不在 Top-10: {beyond_top10} ({beyond_top10/len(error_target_ranks)*100:.1f}%)')

    # 4. 置信度分析
    print('\n## 4. 置信度分析')
    if correct_confidences and all_confidences and prediction_records:
        avg_correct_conf = sum(correct_confidences) / len(correct_confidences)
        avg_all_conf = sum(all_confidences) / len(all_confidences)
        # 错误预测的平均置信度（用 prediction_records 准确计算）
        error_confs = [c for is_c, c in prediction_records if not is_c]
        avg_error_conf = sum(error_confs) / len(error_confs) if error_confs else 0
        print(f'  正确预测平均置信度: {avg_correct_conf:.3f}')
        print(f'  错误预测平均置信度: {avg_error_conf:.3f}')
        print(f'  所有预测平均置信度: {avg_all_conf:.3f}')
        # 置信度分布
        high_conf = sum(1 for c in all_confidences if c > 0.9)
        mid_conf = sum(1 for c in all_confidences if 0.5 <= c <= 0.9)
        low_conf = sum(1 for c in all_confidences if c < 0.5)
        print(f'  高置信度(>0.9): {high_conf}/{len(all_confidences)} ({high_conf/len(all_confidences)*100:.1f}%)')
        print(f'  中置信度(0.5-0.9): {mid_conf}/{len(all_confidences)} ({mid_conf/len(all_confidences)*100:.1f}%)')
        print(f'  低置信度(<0.5): {low_conf}/{len(all_confidences)} ({low_conf/len(all_confidences)*100:.1f}%)')
        # 高置信度但预测错误的占比（过度自信）
        high_conf_wrong = sum(1 for is_c, c in prediction_records if c > 0.9 and not is_c)
        print(f'  高置信度但预测错误: {high_conf_wrong} (过度自信占比 {high_conf_wrong/max(high_conf,1)*100:.1f}%)')

    # 5. 错误样例
    print('\n## 5. 错误样例（前 30 个）')
    print(f'  {"目标token":<12} {"预测token":<12} {"类型":<12} {"正确排名":>8}')
    print(f'  {"-"*12} {"-"*12} {"-"*12} {"-"*8}')
    for target, pred, t_type, rank in error_target_pieces[:30]:
        target_disp = target.replace('\n', '\\n').replace('▁', '_')[:10]
        pred_disp = pred.replace('\n', '\\n').replace('▁', '_')[:10]
        rank_str = f'#{rank}' if rank <= 10 else '>10'
        print(f'  {target_disp:<12} {pred_disp:<12} {t_type:<12} {rank_str:>8}')

    # 6. 结论
    print('\n' + '=' * 70)
    print('## 6. 诊断结论')
    print('=' * 70)
    top5_pct = top5_correct / total_tokens * 100
    byte_err_pct = error_types.get('byte_fallback', 0) / max(sum(error_types.values()), 1) * 100
    beyond10_pct = sum(1 for r in error_target_ranks if r > 10) / max(len(error_target_ranks), 1) * 100

    print(f'  Top-1={top1_correct/total_tokens*100:.1f}%, Top-5={top5_pct:.1f}%, Top-10={top10_correct/total_tokens*100:.1f}%')
    print(f'  byte_fallback 错误占比: {byte_err_pct:.1f}%')
    print(f'  正确答案不在 Top-10 的错误占比: {beyond10_pct:.1f}%')

    if top5_pct > 95:
        print('\n  → 🎯 核心发现：Top-5 > 95%！模型理解语言但 top-1 不够精确')
        print('  → 建议：放弃 argmax 85% 目标，改用 top-p sampling 生成（p=0.9 已覆盖 Top-5）')
    elif byte_err_pct > 30:
        print(f'\n  → 🎯 核心发现：byte_fallback 错误占比 {byte_err_pct:.1f}%！分词器是主要瓶颈')
        print('  → 建议：修复分词器，消除 byte fallback token')
    elif beyond10_pct > 50:
        print(f'\n  → 🎯 核心发现：{beyond10_pct:.1f}% 错误的正确答案不在 Top-10！模型确实不理解')
        print('  → 建议：增加训练量/数据量，提升模型理解能力')
    else:
        print(f'\n  → 🎯 核心发现：错误分散，无单一主导因素')
        print('  → 建议：综合提升（训练+数据+分词器）')

    print(f'\n{"="*70}')
    print('分析完成')
    print(f'{"="*70}')


if __name__ == '__main__':
    analyze_ceiling()
