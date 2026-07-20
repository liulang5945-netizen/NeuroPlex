"""验证 teacher PPL 36 万是否因 tokenizer 不匹配。"""
import os
import sys
import math
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from taiji.tokenizer_native_v2 import TaijiNativeTokenizerV2
from taiji.training.checkpoint_bridge import load_teacher_model


def main():
    print("=" * 70)
    print("Teacher PPL 36 万: checkpoint 加载完整性检查")
    print("=" * 70)

    # 加载 teacher(不需要 tokenizer)
    print("\n[1] 加载 teacher")
    teacher, emb = load_teacher_model("e:/taiji-neuron/checkpoint-481000", device="cpu")

    # 对比 data/real
    print("\n[2] teacher PPL on data/real")
    for domain in ["zh", "en", "code"]:
        path = f"data/real/{domain}.pt"
        if not os.path.exists(path):
            continue
        data = torch.load(path, map_location="cpu", weights_only=True)
        sample = data[:2]
        with torch.no_grad():
            out = teacher(sample)
            logits = out.logits if hasattr(out, "logits") else out
        shift = logits[:, :-1, :].contiguous()
        targets = sample[:, 1:].contiguous()
        loss = F.cross_entropy(shift.view(-1, shift.size(-1)), targets.view(-1))
        ppl = math.exp(min(loss.item(), 15.0))
        print(f"  {domain}: PPL={ppl:.2f}, loss={loss.item():.4f}, log(256K)={math.log(256000):.4f}")
        print(f"    first 10 ids: {sample[0, :10].tolist()}")

    # 检查 teacher 权重统计量(直接用已加载的 teacher)
    print("\n[3] teacher 权重统计量(判断是否训练过)")
    print(f"  init 用 std=0.02,训练后通常 std > 0.05")

    # embedding
    emb_w = teacher.backbone.embedding.weight.data
    print(f"\n  embedding.weight: mean={emb_w.mean().item():.6f}, std={emb_w.std().item():.6f}")

    # 各层 weight
    layers = teacher.backbone.layers
    for i in [0, len(layers)//2, len(layers)-1]:
        layer = layers[i]
        for pname, p in layer.named_parameters():
            print(f"  layer[{i}].{pname}: std={p.std().item():.6f}, mean={p.mean().item():.6f}")

    # lm_head
    lm_head_w = teacher.lm_head.weight.data
    print(f"\n  lm_head.weight: mean={lm_head_w.mean().item():.6f}, std={lm_head_w.std().item():.6f}")
    print(f"  (与 embedding 共享内存: {lm_head_w.data_ptr() == emb_w.data_ptr()})")

    # 检查 training_state.json
    print("\n[4] training_state.json")
    import json
    ts_path = "e:/taiji-neuron/checkpoint-481000/training_state.json"
    if os.path.exists(ts_path):
        with open(ts_path, "r", encoding="utf-8") as f:
            ts = json.load(f)
        print(f"  content: {ts}")
    else:
        print(f"  不存在")

    # 检查 tokenizer_contract.json
    print("\n[5] tokenizer_contract.json")
    tc_path = "e:/taiji-neuron/checkpoint-481000/tokenizer_contract.json"
    if os.path.exists(tc_path):
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        print(f"  text_vocab_size: {tc.get('text_vocab_size')}")
        print(f"  total_vocab_size: {tc.get('total_vocab_size')}")
        print(f"  contract_version: {tc.get('contract_version')}")
    else:
        print(f"  不存在")

    # 结论
    print("\n" + "=" * 70)
    print("[结论]")
    print("=" * 70)
    emb_std = emb_w.std().item()
    if emb_std < 0.03:
        print(f"  embedding std={emb_std:.6f} 接近初始化值 0.02")
        print("  → teacher checkpoint-481000 的 embedding 未训练!")
        print("  → teacher PPL 30 万是因为模型权重是随机的")
        print("  → 需要找到充分训练的 teacher checkpoint 或重新训练 teacher")
    elif emb_std < 0.05:
        print(f"  embedding std={emb_std:.6f} 介于初始化和训练后之间")
        print("  → teacher 可能训练了但未收敛")
    else:
        print(f"  embedding std={emb_std:.6f} 看起来已训练")
        print("  → PPL 30 万可能是 tokenizer 不匹配或其他问题")


if __name__ == "__main__":
    main()
