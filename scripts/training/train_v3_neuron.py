"""Train single neuron with shared embedding (standalone, called per-domain)."""
import math, os, sys, time, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print = functools.partial(print, flush=True)
import torch, torch.nn as nn, torch.nn.functional as F
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.config import get_domain_neuron_config
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import load_domain_tokenizer, load_domain_texts
import sentencepiece as spm

domain = sys.argv[1] if len(sys.argv) > 1 else "code"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300

# Load tokenizers
domain_sp = load_domain_tokenizer(domain)
general_path = os.path.join("taiji/domains", "general", "sp_general.model")
general_sp = spm.SentencePieceProcessor()
general_sp.Load(general_path)

# Load texts
texts = load_domain_texts(domain, max_texts=2000)

# Load shared embedding
emb_path = "data/verify_v3/shared_embedding.pt"
shared_embedding = nn.Embedding(256000, 512)
if os.path.exists(emb_path):
    shared_embedding.weight.data.copy_(torch.load(emb_path, map_location="cpu", weights_only=True))

# Create neuron
cfg = get_domain_neuron_config(domain, "compact")
neuron = ResonanceNeuron(cfg)
neuron.train()
n_texts = len(texts)
params = list(neuron.parameters()) + list(shared_embedding.parameters())
optimizer = torch.optim.AdamW(params, lr=5e-4)
total_loss, t0 = 0.0, time.time()

print(f"[{domain}] Training {steps} steps, {n_texts} texts, batch=4")

for step in range(1, steps + 1):
    idx = torch.randint(0, n_texts, (4,))
    batch_texts = [texts[int(i)] for i in idx]
    shared_emb, targets, mask = batch_align_and_embed(batch_texts, domain_sp, general_sp, shared_embedding)
    targets = targets.clone(); targets[~mask] = -100

    result = neuron.forward(shared_emb, return_logits=True)
    logits = result["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=-100)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    total_loss += loss.item()

    if step % 100 == 0:
        avg = total_loss / step
        print(f"  [{domain}] step {step:>4d} loss={loss.item():.4f} avg={avg:.4f} PPL={math.exp(min(avg,20)):.1f} ({time.time()-t0:.0f}s)")

avg_loss = total_loss / steps
ppl = math.exp(min(avg_loss, 20))
print(f"[{domain}] DONE: avg_loss={avg_loss:.4f} PPL={ppl:.1f} ({time.time()-t0:.0f}s)")

os.makedirs("data/verify_v3", exist_ok=True)
torch.save({"neuron_config": neuron.config, "state_dict": neuron.state_dict(), "domain": domain},
           f"data/verify_v3/neuron_{domain}.pt")
torch.save(shared_embedding.weight.data, f"data/verify_v3/shared_embedding.pt")
print(f"[{domain}] Saved")
