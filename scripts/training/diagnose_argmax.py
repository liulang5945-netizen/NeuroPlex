"""对比个体 vs 协作的 argmax 准确率——判断融合是否拉低 argmax。

若 协作argmax < 最强个体argmax → 融合平滑了分布，需改融合策略(如leader-takes-all)
若 协作argmax ≥ 最强个体argmax → 融合有效，问题纯在训练量
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F
from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble, get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import load_domain_texts, load_domain_tokenizer, load_general_tokenizer, OUTPUT_DIR

DOMAIN = "zh"
dev = "cpu"
cfg = get_domain_neuron_config(DOMAIN)
neurons = {}
for i in range(5):
    nid = f"{DOMAIN}_j{i}"
    ck = torch.load(os.path.join(OUTPUT_DIR, f"neuron_{nid}.pt"), map_location=dev, weights_only=False)
    n = ResonanceNeuron(cfg).to(dev); n.load_state_dict(ck["state_dict"], strict=False); n.eval(); neurons[nid] = n
emb = torch.nn.Embedding(256000, 512)
emb.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "shared_embedding_joint.pt"), map_location=dev))
emb.to(dev).eval()
field = ResonanceField(dim=cfg.field_dim)
ens = ResonanceEnsemble(neurons, field, max_rounds=1)
dsp = load_domain_tokenizer(DOMAIN); gsp = load_general_tokenizer()
texts = load_domain_texts(DOMAIN, max_texts=40)[-40:]
print(f"评估 {len(texts)} 条文本\n", flush=True)

def acc_for(get_logits_fn, label):
    total = correct = 0
    with torch.no_grad():
        for text in texts:
            se, tg, mk = batch_align_and_embed([text], dsp, gsp, emb)
            se, tg, mk = se.to(dev), tg.to(dev), mk.to(dev)
            logits = get_logits_fn(se)  # [1, L, V]
            sl = logits[:, :-1, :]; st = tg[:, 1:]; sm = mk[:, 1:]
            valid = sm & (st != -100)
            preds = sl.argmax(dim=-1)
            correct += (preds[valid] == st[valid]).sum().item()
            total += valid.sum().item()
    print(f"  {label}: argmax top-1 = {correct/max(total,1)*100:.1f}% ({correct}/{total})", flush=True)
    return correct / max(total, 1)

# 个体
print("个体 argmax 准确率:", flush=True)
indiv_accs = {}
for nid, n in neurons.items():
    indiv_accs[nid] = acc_for(lambda se, n=n: n.forward(se, return_logits=True)["logits"], f"  个体[{nid}]")

# 协作 (forward_train 融合)
print("\n协作 argmax 准确率:", flush=True)
collab_acc = acc_for(lambda se: ens.forward_train(se, temperature=1.0)["fused_logits"], "  协作[融合]")

# 协作 (族长主导: 取共振分最高neuron的logits)
print("\n族长主导 argmax 准确率:", flush=True)
def leader_logits(se):
    r = ens.forward_train(se, temperature=1.0)
    w = r["weights"]
    leader = max(range(len(w)), key=lambda i: w[i].item())
    nid = list(neurons.keys())[leader]
    return r["individual_logits"][nid]
leader_acc = acc_for(leader_logits, "  协作[族长主导]")

print("\n" + "=" * 60, flush=True)
best_nid = max(indiv_accs, key=indiv_accs.get)
print(f"最强个体: {best_nid} = {indiv_accs[best_nid]*100:.1f}%", flush=True)
print(f"协作融合: {collab_acc*100:.1f}%", flush=True)
print(f"族长主导: {leader_acc*100:.1f}%", flush=True)
if collab_acc < indiv_accs[best_nid]:
    print(f"\n⚠️ 协作融合 argmax({collab_acc*100:.1f}%) < 最强个体({indiv_accs[best_nid]*100:.1f}%)", flush=True)
    print(f"   → 融合平滑分布拉低了 argmax！族长主导({leader_acc*100:.1f}%)可能更适合生成", flush=True)
else:
    print(f"\n✅ 协作融合 argmax({collab_acc*100:.1f}%) ≥ 最强个体({indiv_accs[best_nid]*100:.1f}%)", flush=True)
    print(f"   → 融合有效，问题在训练量(需更多数据+步数提升准确率)", flush=True)
