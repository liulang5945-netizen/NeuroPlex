"""调查 teacher PPL 30 万的根因。

假设:
1. tokenizer 不匹配(data 用 checkpoint-400000 SP 编码,teacher 是 checkpoint-481000)
2. teacher checkpoint 未训练好
3. weight tying 被破坏
4. token id 范围超出 vocab_size
"""
import os
import sys
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from taiji.training.checkpoint_bridge import load_teacher_model

TEACHER_DIR = "e:/taiji-neuron/checkpoint-481000"
DATA_DIR = "data/real"


def main():
    print("=" * 70)
    print("Teacher PPL 30 万根因调查")
    print("=" * 70)

    # ── 1. 检查数据 token id 范围 ─────────────────────────────
    print("\n[1] 数据 token id 范围检查")
    for domain in ["zh", "en", "code", "math", "general"]:
        path = os.path.join(DATA_DIR, f"{domain}.pt")
        if not os.path.exists(path):
            continue
        data = torch.load(path, map_location="cpu", weights_only=True)
        print(f"  {domain}: shape={tuple(data.shape)}, min={data.min().item()}, max={data.max().item()}, vocab_size=256000")

    # ── 2. 加载 teacher ──────────────────────────────────────
    print("\n[2] 加载 teacher")
    teacher, embedding = load_teacher_model(TEACHER_DIR, device="cpu")
    print(f"  teacher vocab_size: {teacher.config.vocab_size}")
    print(f"  teacher hidden_size: {teacher.config.hidden_size}")
    print(f"  embedding.weight shape: {tuple(embedding.weight.shape)}")

    # ── 3. 检查 weight tying ─────────────────────────────────
    print("\n[3] weight tying 检查")
    lm_head = getattr(teacher, "lm_head", None)
    if lm_head is None:
        print("  [error] teacher 没有 lm_head 属性")
        return
    print(f"  lm_head.weight shape: {tuple(lm_head.weight.shape)}")
    print(f"  embedding.weight shape: {tuple(embedding.weight.shape)}")
    print(f"  weight tying (同一内存): {lm_head.weight.data_ptr() == embedding.weight.data_ptr()}")

    # ── 4. 检查 lm_head.weight 的统计量 ──────────────────────
    print("\n[4] lm_head.weight 统计量")
    w = lm_head.weight.data
    print(f"  mean: {w.mean().item():.6f}")
    print(f"  std: {w.std().item():.6f}")
    print(f"  min: {w.min().item():.6f}")
    print(f"  max: {w.max().item():.6f}")
    print(f"  norm: {w.norm().item():.4f}")

    # ── 5. 单样本 forward 检查 logits 分布 ───────────────────
    print("\n[5] 单样本 forward 检查")
    data = torch.load(os.path.join(DATA_DIR, "zh.pt"), map_location="cpu", weights_only=True)
    sample = data[:1]  # [1, 256]
    print(f"  输入 sample shape: {tuple(sample.shape)}")
    print(f"  输入 token ids (前10): {sample[0, :10].tolist()}")

    with torch.no_grad():
        output = teacher(sample)
        logits = output.logits if hasattr(output, "logits") else output
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"  logits mean: {logits.mean().item():.6f}")
    print(f"  logits std: {logits.std().item():.6f}")
    print(f"  logits min: {logits.min().item():.6f}")
    print(f"  logits max: {logits.max().item():.6f}")

    # 检查第一个 token 的预测分布
    first_token_logits = logits[0, 0, :]  # [vocab]
    probs = F.softmax(first_token_logits, dim=-1)
    print(f"  first token probs: max={probs.max().item():.6f}, entropy={(-probs * probs.log()).sum().item():.4f}")
    print(f"  first token top-5 logits: {first_token_logits.topk(5).values.tolist()}")
    print(f"  first token top-5 indices: {first_token_logits.topk(5).indices.tolist()}")
    print(f"  真实 next token: {sample[0, 1].item()}")

    # ── 6. 检查 checkpoint-400000 vs 481000 的 SP 模型 ──────
    print("\n[6] tokenizer 检查")
    sp_400 = "e:/taiji/checkpoint-400000/sentencepiece.model"
    sp_481 = "e:/taiji-neuron/checkpoint-481000/sentencepiece.model"
    sp_taiji = "e:/taiji-neuron/taiji/tokenizer_native_v2/sentencepiece.model"

    for name, path in [("checkpoint-400000", sp_400), ("checkpoint-481000", sp_481), ("taiji/tokenizer_native_v2", sp_taiji)]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            # 比较 hash
            import hashlib
            with open(path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            print(f"  {name}: size={size}, md5={md5[:16]}")
        else:
            print(f"  {name}: 不存在")

    # ── 7. 检查 config.json 对比 ─────────────────────────────
    print("\n[7] config.json 对比")
    import json
    cfg_400_path = "e:/taiji/checkpoint-400000/config.json"
    cfg_481_path = "e:/taiji-neuron/checkpoint-481000/config.json"
    for name, path in [("checkpoint-400000", cfg_400_path), ("checkpoint-481000", cfg_481_path)]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print(f"  {name}: vocab_size={cfg.get('vocab_size')}, hidden_size={cfg.get('hidden_size')}, num_layers={cfg.get('num_hidden_layers')}")
        else:
            print(f"  {name}: 不存在")

    # ── 8. 检查 teacher 的 training_state ────────────────────
    print("\n[8] teacher checkpoint 训练状态")
    ts_path = os.path.join(TEACHER_DIR, "training_state.pt")
    if os.path.exists(ts_path):
        ts = torch.load(ts_path, map_location="cpu", weights_only=True)
        if isinstance(ts, dict):
            print(f"  keys: {list(ts.keys())[:10]}")
            for k in ["step", "epoch", "global_step", "best_loss", "lr"]:
                if k in ts:
                    print(f"  {k}: {ts[k]}")
    else:
        print(f"  training_state.pt 不存在")

    # ── 9. 用 checkpoint-400000 作为 teacher 对比 ────────────
    print("\n[9] 尝试用 checkpoint-400000 作为 teacher")
    alt_teacher_dir = "e:/taiji/checkpoint-400000"
    if os.path.exists(alt_teacher_dir):
        try:
            alt_teacher, alt_emb = load_teacher_model(alt_teacher_dir, device="cpu")
            with torch.no_grad():
                output = alt_teacher(sample)
                alt_logits = output.logits if hasattr(output, "logits") else output
            shift = alt_logits[:, :-1, :].contiguous()
            targets = sample[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), targets.view(-1))
            import math
            ppl = math.exp(min(loss.item(), 15.0))
            print(f"  checkpoint-400000 teacher PPL on zh: {ppl:.2f}")
            print(f"  logits std: {alt_logits.std().item():.6f}")
            print(f"  first token top-5: {alt_logits[0, 0, :].topk(5).indices.tolist()}")
        except Exception as e:
            print(f"  [error] {e}")
    else:
        print(f"  {alt_teacher_dir} 不存在")


if __name__ == "__main__":
    main()
