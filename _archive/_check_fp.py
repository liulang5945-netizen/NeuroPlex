"""Quick verification: field_vector fingerprint diversity."""
import torch

for d in ['zh', 'code']:
    c = torch.load(f'data/neurons_v2/neuron_{d}.pt', map_location='cpu', weights_only=False)
    fw = c['state_dict']['field_write.weight']
    r = c['result']
    fp = fw.mean(dim=0)
    fp = fp / (fp.norm() + 1e-8)
    torch.save(fp, f'data/neurons_v2/_fp_{d}.pt')
    print(f'{d}: PPL={r["ppl_own"]:.0f} field_loss={r.get("field_loss",0):.4f} fp_ok')

# Compute cosine
fp_zh = torch.load('data/neurons_v2/_fp_zh.pt', weights_only=True)
fp_code = torch.load('data/neurons_v2/_fp_code.pt', weights_only=True)
cos = float(torch.dot(fp_zh, fp_code))
print(f'zh-code cos: {cos:.4f} | diverse={abs(cos) < 0.7}')
