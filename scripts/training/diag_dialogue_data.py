import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

for nid in ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
            "zh_aug3_dialogue", "zh_std0_dialogue"]:
    p = os.path.join("data/neurons", "neuron_%s.pt" % nid)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    r = ck.get("result", {})
    ds = ck.get("data_source", "?")
    print("%-20s data_source=%s result=%s" % (nid, ds, r))
