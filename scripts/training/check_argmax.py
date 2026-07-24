"""快速检查 argmax 准确率 —— 判断生成乱码根因。

使用与 eval_joint.py 相同的 batch_align_and_embed 官方对齐函数。
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from taiji.resonance import ResonanceField, ResonanceEnsemble
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import load_domain_texts, load_domain_tokenizer, load_general_tokenizer
from scripts.training.eval_joint import load_joint_neurons

DOMAIN = "zh"

def main():
    neurons, shared_embedding, cfg = load_joint_neurons(10, DOMAIN, "cpu", spec="compact")
    domain_sp = load_domain_tokenizer(DOMAIN)
    general_sp = load_general_tokenizer()

    field = ResonanceField(dim=cfg.field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)

    texts = load_domain_texts(DOMAIN, max_texts=100)
    eval_texts = texts[-77:] if len(texts) > 77 else texts
    print(f"评估 {len(eval_texts)} 条文本", flush=True)

    correct = 0
    total = 0
    top5_correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for text in eval_texts[:50]:
            shared_emb, targets, mask = batch_align_and_embed(
                [text], domain_sp, general_sp, shared_embedding,
            )
            out = ensemble.forward_train(shared_emb, temperature=1.0, fusion_mode="soft")
            fused = out["fused_logits"]
            shift_logits = fused[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()

            pred = shift_logits.argmax(dim=-1)
            top5 = shift_logits.topk(5, dim=-1).indices

            for t in range(shift_targets.shape[1]):
                if shift_mask[0, t] == 0:
                    continue
                tgt = shift_targets[0, t].item()
                total += 1
                loss = F.cross_entropy(shift_logits[0, t], shift_targets[0, t])
                total_loss += loss.item()
                if pred[0, t].item() == tgt:
                    correct += 1
                if tgt in top5[0, t].tolist():
                    top5_correct += 1

    argmax_acc = correct / max(total, 1) * 100
    top5_acc = top5_correct / max(total, 1) * 100
    avg_loss = total_loss / max(total, 1)
    ppl = math.exp(min(avg_loss, 20))

    print(f"\n软加权融合模式 (50 条文本):")
    print(f"  argmax 准确率: {argmax_acc:.1f}% ({correct}/{total})")
    print(f"  top-5 准确率:  {top5_acc:.1f}% ({top5_correct}/{total})")
    print(f"  PPL: {ppl:.1f} (avg_ce={avg_loss:.4f})")
    print()
    if argmax_acc >= 85:
        print("  → argmax ≥ 85%，生成乱码是管线问题（token 转换错误）")
    elif argmax_acc >= 50:
        print("  → argmax 50-85%，训练不足，需继续训练或增加数据")
    else:
        print("  → argmax < 50%，严重训练不足")

if __name__ == "__main__":
    main()
