"""C16b 冒烟验证：per-neuron NLL z-score 监督是否修复 code 独占（临时脚本，用完删除）。

验证目标：预热 EMA（25 步各域文本）后，对 5 个测试 prompt 检查
forward_train 返回的 per_neuron_nll_z + ideal_weights（softmax(-z/0.5)）——
修复前 dialogue/zh 文本上 ideal 是 code one-hot，修复后应分散到对应域 neuron。
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

from scripts.training.train_cross_domain_collab import (
    load_neuron, load_shared_lm_head, load_shared_embedding,
)
from scripts.training.utils import (
    load_general_tokenizer, create_shared_embedding,
)
from scripts.training.gen_test_collab import DOMAINS, DIALOGUE_IDS, DIALOGUE_DIR, GENERAL_DIR
from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.resonance.geometry import NeuronGeometry
from taiji.resonance.topology import build_topology, establish_topology_channels
from taiji.resonance.translator import batch_align_and_embed

CKPT = os.path.join(PROJECT_ROOT, "data", "neurons", "collab_v3_c16.ckpt.pt")
SEQ_LEN = 128

# 预热文本（各域轮转，模拟训练分布）
WARMUP_TEXTS = [
    "Write a Python function that returns the sum of two numbers",
    "If a car travels at 50 miles per hour, how far in 2 hours?",
    "今天天气非常好，我们去公园散步吧",
    "The capital city of Japan is Tokyo, a vibrant metropolis",
    "你好，请问你会做什么事情呢",
    "def add(a, b): return a + b",
    "A rectangle has length 6 and width 4, what is its area?",
    "学习是一种乐趣，每天进步一点点",
]

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("dialogue", "你好，请介绍一下你自己"),
    ("en", "What is the capital of France?"),
]


def main():
    torch.manual_seed(42)
    general_sp = load_general_tokenizer()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    shared_lm_head = load_shared_lm_head(GENERAL_DIR, 512, "cpu")
    neurons, embeddings = {}, {}
    for nid in DOMAINS + DIALOGUE_IDS:
        if nid in DOMAINS:
            n = load_neuron(nid, GENERAL_DIR, "cpu", shared_lm_head=shared_lm_head)
            emb = load_shared_embedding(GENERAL_DIR, "cpu")
        else:
            ckp = torch.load(os.path.join(DIALOGUE_DIR, f"neuron_{nid}.pt"),
                             map_location="cpu", weights_only=False)
            cfg = ckp["neuron_config"]; cfg.unified_field_dim = None
            n = ResonanceNeuron(cfg)
            n.load_state_dict(ckp["state_dict"], strict=False)
            emb = create_shared_embedding("cpu")
            ses = ckp.get("shared_embedding_state", {})
            w = ses["weight"] if isinstance(ses, dict) else ses
            emb.weight.data.copy_(w)
        neurons[nid] = n
        embeddings[nid] = emb

    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)

    # tokenizer hub（forward_train 跨 vocab 转译需要）
    from taiji.resonance.translator import TokenizerHub
    from scripts.training.utils import load_domain_tokenizer
    hub = TokenizerHub()
    for dom in DOMAINS:
        hub.register_domain(dom, load_domain_tokenizer(dom))
    hub.register_domain("zh", load_domain_tokenizer("zh"))
    hub.register_domain("general", general_sp)
    ens.set_tokenizer_hub(hub)

    # 注入 ckpt 训练分量（head/lora/cross/side/scale）
    for nid, hd in ck.get("head_state", {}).items():
        if nid in neurons and getattr(neurons[nid], "quality_head", None) is not None:
            neurons[nid].quality_head.load_state_dict(hd)
    for nid, ls in ck.get("lora_state", {}).items():
        if nid not in neurons:
            continue
        n = neurons[nid]
        if len(n.lora_adapters) == 0:
            rank = 0
            for k, v in ls.items():
                if k.endswith(".a.weight"):
                    rank = max(rank, v.shape[0])
            n.enable_lora(rank if rank > 0 else 16, layers=None)
        n.lora_adapters.load_state_dict(ls)
    for nid, sd in ck.get("cross_spec_state", {}).get("forward", {}).items():
        if nid in ens._cross_spec_projectors:
            ens._cross_spec_projectors[nid].load_state_dict(sd)
    for nid, sd in ck.get("cross_spec_state", {}).get("backward", {}).items():
        if nid in ens._cross_spec_back_projectors:
            ens._cross_spec_back_projectors[nid].load_state_dict(sd)
    for n in neurons.values():
        n.eval()

    def run_ft(batch_texts, step):
        neuron_embeddings = {}
        targets = mask = None
        for nid, emb in embeddings.items():
            out = batch_align_and_embed(batch_texts, general_sp, general_sp, emb,
                                        max_seq_len=SEQ_LEN)
            neuron_embeddings[nid] = out[0]
            if targets is None:
                targets, mask = out[1], out[2]
        return ens.forward_train(
            neuron_embeddings=neuron_embeddings,
            n_rounds=2, fusion_mode="soft", targets=targets,
            field_conditioning=True, step=step, target_domain="general",
        )

    nids = list(neurons.keys())
    # 预热 EMA（25 步，各域文本轮转）
    print("预热 EMA（25 步）...")
    for s in range(25):
        r = run_ft([WARMUP_TEXTS[s % len(WARMUP_TEXTS)]], step=s)
    print(f"EMA 统计已建立: { {nid: round(v['mean'], 2) for nid, v in ens._nll_ema.items()} }\n")

    # 测试 5 个 prompt
    GATE_FACTOR = 50.0
    print(f"{'prompt':<10} | gate+z-score ideal top3（gate={GATE_FACTOR}: 绝对 NLL 超 batch 最优 {GATE_FACTOR}× 排除）")
    for tag, prompt in PROMPTS:
        r = run_ft([prompt], step=99)
        z = r["per_neuron_nll_z"]
        nll = r["per_neuron_nll"]
        min_nll = nll.min()
        # gate: 绝对 NLL 远超 batch 最优的 neuron 排除（转译 neuron 高基线保护）
        gate = (nll < min_nll * GATE_FACTOR)
        z_gated = z.clone()
        z_gated[~gate] = 1e9
        ideal = F.softmax(-z_gated / 0.5, dim=0)
        order = torch.argsort(ideal, descending=True)
        top3 = " > ".join(
            f"{nids[i]}(ideal={ideal[i].item():.3f},z={z[i].item():.2f},nll={nll[i].item():.1f},{'G' if gate[i] else 'X'})"
            for i in order[:3].tolist())
        print(f"[{tag:<8}] {top3}")
    print("\n验证判据：code 文本→code；math/en 文本→math/en（dialogue 被 gate 排除）；"
          "中文对话→zh_aug（gate 不排除，z-score 竞争赢）")


if __name__ == "__main__":
    main()
