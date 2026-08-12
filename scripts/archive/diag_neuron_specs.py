import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

for p in ["data/neurons/neuron_zh_std0_dialogue.pt",
          "data/foundation_v1_dual/neuron_zh.pt"]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", {})
    keys = list(sd.keys())
    print("=" * 60)
    print(p)
    print("  top-level keys:", list(ck.keys())[:12])
    print("  state_dict n:", len(keys))
    for k in keys:
        if any(x in k for x in ["lm_head", "judge", "embed_adapter", "field_write", "body", "transformer"]) and ".weight" in k:
            print("    %s: %s" % (k, tuple(sd[k].shape)))
