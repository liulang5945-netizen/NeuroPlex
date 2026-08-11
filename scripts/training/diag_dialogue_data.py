import os, sys, glob, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 找所有 zh 相关 base neuron 的 lm_head vocab
for d in ["data/neurons", "data/neurons/pre_t12_backup",
          "data/foundation_v1", "data/foundation_v1_general",
          "data/foundation_v1_sft", "data/foundation_v1_dual"]:
    if not os.path.isdir(d):
        continue
    for p in sorted(glob.glob(os.path.join(d, "neuron_zh*.pt"))):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", {})
            lh = sd.get("lm_head.weight")
            cfg = ck.get("neuron_config")
            spec = getattr(cfg, "spec", "?") if cfg else "?"
            hidden = getattr(cfg, "hidden_size", "?") if cfg else "?"
            r = ck.get("result", {})
            print("%-62s vocab=%s spec=%s hidden=%s best_ppl=%s steps=%s" % (
                p, lh.shape[0] if lh is not None else "?", spec, hidden,
                r.get("best_val_ppl"), r.get("steps", "?")))
        except Exception as e:
            print("%s ERR %s" % (p, e))
