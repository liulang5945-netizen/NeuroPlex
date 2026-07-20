"""PPL 过高根因诊断脚本。

诊断 3 个关键假设:
1. teacher baseline PPL(走 teacher 完整 forward 含 lm_head)
2. student 当前 PPL(无 W_base,已知 8000+)
3. student 注入 W_base 后 PPL(SVD top-384,不训练)

运行:
    python scripts/training/diagnose_ppl.py
"""
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 确保项目根在 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.shared_embed import SharedEmbedProj


TEACHER_DIR = "e:/taiji-neuron/checkpoint-481000"
SHARED_PROJ_PATH = "data/distill/shared_proj.pt"
NEURONS_DIR = "data/neurons"
DATA_DIR = "data/real"
DOMAINS = ["zh", "en", "code", "math", "general"]
BATCH_SIZE = 2
MAX_BATCHES = 5


def load_data(domain: str) -> torch.Tensor:
    """加载某个 domain 的数据。"""
    path = os.path.join(DATA_DIR, f"{domain}.pt")
    if not os.path.exists(path):
        print(f"  [warn] {path} 不存在,跳过")
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.dim() == 1:
        data = data.unsqueeze(0)
    print(f"  加载 {domain}: shape={tuple(data.shape)}, dtype={data.dtype}")
    return data


def compute_teacher_ppl(teacher, data: torch.Tensor, device: str) -> float:
    """实验1: teacher baseline PPL。

    走 teacher 完整 forward:embedding → layers → norm → lm_head → cross_entropy。
    这告诉我们 teacher 在这批数据上的真实 PPL。
    """
    teacher.eval()
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= MAX_BATCHES:
                break
            ids = batch[0].to(device)
            # teacher 完整 forward(含 lm_head),返回 ModelOutput
            output = teacher(ids)
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, dict):
                logits = output.get("logits")
            elif isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                logits = output
            shift = logits[:, :-1, :].contiguous()
            targets = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift.view(-1, shift.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg = total_loss / max(total_tokens, 1)
    return math.exp(min(avg, 15.0))


def compute_student_ppl(
    neuron: ResonanceNeuron,
    data: torch.Tensor,
    teacher,
    proj: SharedEmbedProj,
    device: str,
) -> float:
    """测 student PPL。

    链路:ids → teacher.extract_hidden_states → SharedEmbedProj → neuron.forward → logits → cross_entropy。
    """
    neuron.eval()
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= MAX_BATCHES:
                break
            ids = batch[0].to(device)
            # shared embedding
            hidden = extract_hidden_states(teacher, ids)  # [B, L, 2048]
            shared_emb = proj(hidden)  # [B, L, 512]
            # neuron forward
            out = neuron.forward(shared_emb, return_logits=True)
            logits = out["logits"] if isinstance(out, dict) else out
            shift = logits[:, :-1, :].contiguous()
            targets = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift.view(-1, shift.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg = total_loss / max(total_tokens, 1)
    return math.exp(min(avg, 15.0))


def build_w_base_from_teacher_svd(
    teacher,
    target_hidden_size: int,
    device: str,
) -> nn.Linear:
    """从 teacher 的 lm_head.weight 做 SVD,构造 W_base。

    teacher.lm_head.weight: [vocab_size, hidden_size] = [256000, 2048]
    目标:W_base: [vocab_size, target_hidden_size] = [256000, 384]

    方法:对 W_teacher.T [2048, 256000] 做 SVD_lowrank(top-k=target_hidden_size)
    W_base = W_teacher @ V  其中 V 是 [2048, target_hidden_size] 的右奇异向量
    等价于 W_base = U[:, :k] @ diag(S[:k])
    """
    # 获取 teacher 的 lm_head weight
    lm_head = getattr(teacher, "lm_head", None)
    if lm_head is None:
        # 尝试 backbone.lm_head 或其他路径
        backbone = getattr(teacher, "backbone", None)
        if backbone is not None:
            lm_head = getattr(backbone, "lm_head", None)

    if lm_head is None:
        raise AttributeError("无法找到 teacher 的 lm_head")

    W_teacher = lm_head.weight.data  # [vocab_size, hidden_size] = [256000, 2048]
    print(f"  teacher lm_head.weight shape: {tuple(W_teacher.shape)}")

    # SVD: W_teacher = U S V^T
    # 对 W_teacher.T [hidden, vocab] 做 svd_lowrank 更高效
    Wt = W_teacher.T.to(device).float()  # [2048, 256000]
    q = min(target_hidden_size, Wt.shape[0])
    U, S, V = torch.svd_lowrank(Wt, q=q)  # U:[2048,q], S:[q], V:[256000,q]

    # W_teacher ≈ V @ diag(S) @ U^T
    # W_base = W_teacher @ U[:, :q] = V @ diag(S)  (因为 W_teacher @ U = V @ diag(S))
    # W_base_weight shape: [vocab, q] = [256000, 384]
    W_base_weight = V @ torch.diag(S)  # [256000, q]

    # 创建 W_base Linear
    # nn.Linear(in=q, out=vocab) 的 weight shape = [out, in] = [vocab, q]
    # W_base_weight 已经是 [vocab, q],直接赋值(不需要转置)
    w_base = nn.Linear(q, W_teacher.shape[0], bias=False)  # in=q, out=vocab
    w_base.weight.data = W_base_weight.contiguous().to(w_base.weight.device)
    w_base = w_base.to(device)

    # 验证形状
    print(f"  W_base shape: {tuple(w_base.weight.shape)} (期望 [{W_teacher.shape[0]}, {q}])")

    # 计算 SVD 保留的能量比例
    total_energy = (W_teacher.float() ** 2).sum().item()
    retained_energy = (W_base_weight ** 2).sum().item()
    ratio = retained_energy / total_energy if total_energy > 0 else 0
    print(f"  SVD top-{q} 保留能量: {ratio:.4f} ({ratio*100:.1f}%)")

    return w_base


def main():
    print("=" * 70)
    print("PPL 过高根因诊断")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")

    # ── 加载 teacher ──────────────────────────────────────────
    print("\n[1/5] 加载 teacher 模型...")
    teacher, teacher_embedding = load_teacher_model(TEACHER_DIR, device=device)
    print(f"  teacher: {type(teacher).__name__}, hidden_size={teacher_embedding.weight.shape[1]}")

    # ── 加载 SharedEmbedProj ──────────────────────────────────
    print("\n[2/5] 加载 SharedEmbedProj...")
    # 优先用 distill 目录(蒸馏时实际用的)
    if os.path.exists(SHARED_PROJ_PATH):
        proj = SharedEmbedProj.load(SHARED_PROJ_PATH)
        print(f"  加载自 {SHARED_PROJ_PATH}")
    else:
        # fallback: data/shared_proj.pt
        alt_path = "data/shared_proj.pt"
        if os.path.exists(alt_path):
            proj = SharedEmbedProj.load(alt_path)
            print(f"  加载自 {alt_path}")
        else:
            print(f"  [error] SharedEmbedProj 不存在")
            return
    proj = proj.to(device)

    # ── 加载数据 ──────────────────────────────────────────────
    print("\n[3/5] 加载评估数据...")
    datasets = {}
    for domain in DOMAINS:
        data = load_data(domain)
        if data is not None:
            datasets[domain] = data

    # ── 实验1: teacher baseline PPL ───────────────────────────
    print("\n" + "=" * 70)
    print("[实验1] teacher baseline PPL(走 teacher 完整 forward 含 lm_head)")
    print("=" * 70)
    teacher_ppls = {}
    for domain, data in datasets.items():
        ppl = compute_teacher_ppl(teacher, data, device)
        teacher_ppls[domain] = ppl
        print(f"  {domain}: teacher PPL = {ppl:.2f}")

    # ── 实验2: student 当前 PPL(无 W_base)────────────────────
    print("\n" + "=" * 70)
    print("[实验2] student 当前 PPL(无 W_base,lm_head_delta 初始化 std=0.01)")
    print("=" * 70)
    student_ppls_no_wbase = {}
    for domain in DOMAINS:
        neuron_path = os.path.join(NEURONS_DIR, f"neuron_{domain}.pt")
        if not os.path.exists(neuron_path):
            print(f"  {domain}: ckpt 不存在,跳过")
            continue
        ckpt = torch.load(neuron_path, map_location="cpu", weights_only=False)
        cfg = ckpt["neuron_config"]
        neuron = ResonanceNeuron(cfg)
        neuron.load_state_dict(ckpt["state_dict"], strict=False)
        neuron.v1_compat = True
        neuron.to(device)
        # 确认 W_base 未注入
        assert neuron.lm_head_base is None, f"{domain} 的 lm_head_base 应为 None"
        ppl = compute_student_ppl(neuron, datasets[domain], teacher, proj, device)
        student_ppls_no_wbase[domain] = ppl
        print(f"  {domain}: student PPL(无 W_base) = {ppl:.2f}")

    # ── 实验3: student 注入 W_base 后 PPL(不训练)─────────────
    print("\n" + "=" * 70)
    print("[实验3] student 注入 W_base 后 PPL(SVD top-k,不训练)")
    print("=" * 70)
    student_ppls_with_wbase = {}
    # W_base 的 target_hidden_size 取 student 的 hidden_size
    for domain in DOMAINS:
        neuron_path = os.path.join(NEURONS_DIR, f"neuron_{domain}.pt")
        if not os.path.exists(neuron_path):
            print(f"  {domain}: ckpt 不存在,跳过")
            continue
        ckpt = torch.load(neuron_path, map_location="cpu", weights_only=False)
        cfg = ckpt["neuron_config"]
        neuron = ResonanceNeuron(cfg)
        neuron.load_state_dict(ckpt["state_dict"], strict=False)
        neuron.v1_compat = True
        neuron.to(device)

        # 构造并注入 W_base
        target_hidden = cfg.hidden_size
        print(f"\n  构造 {domain} 的 W_base(target_hidden={target_hidden})...")
        w_base = build_w_base_from_teacher_svd(teacher, target_hidden, device)
        neuron.set_shared_lm_head(w_base)
        assert neuron.lm_head_base is not None

        ppl = compute_student_ppl(neuron, datasets[domain], teacher, proj, device)
        student_ppls_with_wbase[domain] = ppl
        print(f"  {domain}: student PPL(注入 W_base) = {ppl:.2f}")

    # ── 汇总报告 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[汇总报告] PPL 对比")
    print("=" * 70)
    print(f"{'domain':<10} {'teacher':<12} {'student(无W_base)':<18} {'student(有W_base)':<18} {'W_base改善':<12}")
    print("-" * 70)
    for domain in datasets:
        t_ppl = teacher_ppls.get(domain, float("nan"))
        s_no = student_ppls_no_wbase.get(domain, float("nan"))
        s_yes = student_ppls_with_wbase.get(domain, float("nan"))
        improvement = s_no / s_yes if s_yes > 0 else float("nan")
        print(f"{domain:<10} {t_ppl:<12.2f} {s_no:<18.2f} {s_yes:<18.2f} {improvement:<12.2f}x")

    # ── 结论分析 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[结论分析]")
    print("=" * 70)
    if student_ppls_with_wbase:
        avg_improvement = sum(
            student_ppls_no_wbase[d] / student_ppls_with_wbase[d]
            for d in student_ppls_with_wbase
        ) / len(student_ppls_with_wbase)
        print(f"  W_base 注入平均改善: {avg_improvement:.1f}x")

        if avg_improvement > 10:
            print("  → W_base 缺失是 PPL 过高的【主因】,必须修复蒸馏流程注入 W_base")
        elif avg_improvement > 2:
            print("  → W_base 缺失是【重要原因】,但非唯一")
        else:
            print("  → W_base 注入效果有限,主因在别处(容量/投影/数据)")

    if teacher_ppls:
        avg_teacher = sum(teacher_ppls.values()) / len(teacher_ppls)
        print(f"\n  teacher 平均 PPL: {avg_teacher:.2f}")
        if avg_teacher > 100:
            print("  → teacher 本身 PPL 偏高,可能 ckpt 未充分训练或数据不匹配")
        else:
            print("  → teacher PPL 正常,问题在 student 侧")


if __name__ == "__main__":
    main()
