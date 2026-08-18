"""C16 路由诊断：quality_logit 分布 vs 真实 NLL 对照（临时脚本，用完删除）。

目的：确认"code=1.00 独占"是 quality_head 信号失效（偏置膨胀/监督失效）还是
max-prob 天然差异。逐文本打印 9 neuron 的 quality_logit、softmax trust、
真实 NLL 与 max-prob。
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
from scripts.training.utils import load_general_tokenizer, create_shared_embedding
from scripts.training.gen_test_collab import DOMAINS, DIALOGUE_IDS, DIALOGUE_DIR, GENERAL_DIR

PROMPTS = [
    ("code", "Write a Python function to compute the Fibonacci sequence"),
    ("math", "If a train travels at 60 mph for 3 hours, how many miles does it travel?"),
    ("zh", "写一个 Python 函数计算斐波那契数列"),
    ("dialogue", "你好，请介绍一下你自己"),
    ("en", "What is the capital of France?"),
]

CKPT = os.path.join(PROJECT_ROOT, "data", "neurons", "collab_v3_c16.ckpt.pt")


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
            from taiji.resonance.neuron import ResonanceNeuron
            n = ResonanceNeuron(cfg)
            n.load_state_dict(ckp["state_dict"], strict=False)
            emb = create_shared_embedding("cpu")
            ses = ckp.get("shared_embedding_state", {})
            w = ses["weight"] if isinstance(ses, dict) else ses
            emb.weight.data.copy_(w)
        neurons[nid] = n
        embeddings[nid] = emb

    # 注入 head + lora（C16 训练分量）
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
    for n in neurons.values():
        n.eval()

    nids = list(neurons.keys())
    print(f"{'prompt':<8} | " + " | ".join(f"{nid:<18}" for nid in nids))
    for tag, prompt in PROMPTS:
        ids_t = torch.tensor([general_sp.EncodeAsIds(prompt)], dtype=torch.long)
        qvals, nlls, mps = {}, {}, {}
        with torch.no_grad():
            for nid in nids:
                h = embeddings[nid](ids_t)
                r = neurons[nid].forward(h, round_num=1, return_logits=True)
                q = r["quality_logit"].item()
                logits = r["logits"]
                if logits.shape[-1] != 256000:
                    from taiji.resonance.translator import build_logits_alignment_matrix
                    from scripts.training.utils import load_domain_tokenizer
                    src_sp = load_domain_tokenizer("zh")
                    m = build_logits_alignment_matrix(src_sp, general_sp, "zh", "general",
                                                      cache={}, source_vocab_size=logits.shape[-1])
                    b, l, vi = logits.shape
                    logits = torch.sparse.mm(logits.reshape(-1, vi), m.to(logits.dtype)).reshape(b, l, 256000)
                # 目标 token 是输入 shifted（纯预测质量近似：预测输入下一个 token）
                tgt = ids_t[:, 1:].contiguous()
                lg = logits[:, :-1, :].contiguous()
                ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1))
                qvals[nid] = q
                nlls[nid] = ce.item()
                mps[nid] = F.softmax(logits, dim=-1).max(dim=-1).values.mean().item()
        qv = torch.tensor([qvals[nid] for nid in nids])
        trust = F.softmax(qv / 0.15, dim=0)
        print(f"\n── [{tag}] {prompt}")
        print(f"  {'neuron':<18} {'quality':>9} {'trust@0.15':>11} {'NLL(shift)':>11} {'maxprob':>9}")
        for i, nid in enumerate(nids):
            print(f"  {nid:<18} {qvals[nid]:>9.3f} {trust[i].item():>11.3f} {nlls[nid]:>11.3f} {mps[nid]:>9.3f}")


if __name__ == "__main__":
    main()
