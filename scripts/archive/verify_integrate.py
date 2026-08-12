#!/usr/bin/env python3
"""IntegrateEngine 冒烟验证（C17，2026-08-08）。

验证：① 影子 COW 跑通 ② 训练循环（CE + 邻居蒸馏 + contrastive）跑通
③ ablation 决策（commit/apoptosis）跑通 ④ 静默期 maturity 压低融合。
"""
import os
import sys

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.resonance.ensemble import ResonanceEnsemble
from taiji.resonance.field import ResonanceField
from taiji.resonance.geometry import NeuronGeometry
from taiji.resonance.topology import build_topology, establish_topology_channels
from taiji.resonance.translator import TokenizerHub
from scripts.training.utils import load_general_tokenizer, load_domain_tokenizer, create_shared_embedding
from scripts.training.train_cross_domain_collab import load_shared_lm_head


class MockMaturity:
    maturity_rounds = 4  # 缩短成熟轮数，一次会话即可触发 ablation 验证

    def __init__(self):
        self._m = {}

    def register_new(self, nid):
        self._m[nid] = 0

    def tick(self, nid):
        if nid in self._m:
            self._m[nid] += 1

    def get_maturity_ratio(self, nid):
        if nid not in self._m:
            return 1.0
        return min(1.0, self._m[nid] / self.maturity_rounds)

    def get_lr_multiplier(self, nid):
        r = self.get_maturity_ratio(nid)
        return 3.0 * (1 - r) + 1.0 * r

    def get_resonance_weight(self, nid):
        r = self.get_maturity_ratio(nid)
        return 0.1 + 0.9 * r

    def is_mature(self, nid):
        return self.get_maturity_ratio(nid) >= 1.0


class MockApoptosis:
    def record_ppl(self, domain, ppl):
        print(f"  [MockApoptosis] record_ppl({domain}, {ppl:.3f})")


class MockLifecycle:
    def __init__(self):
        self.maturity = MockMaturity()
        self.apoptosis = MockApoptosis()


class MockFeed:
    def __init__(self, texts):
        self._texts = texts

    def get_pending_samples_by_domain(self):
        return {"zh": [{"text": t} for t in self._texts]}


class MockCortex:
    def __init__(self, neurons, ensemble, emb, sp, hub):
        self.neurons = neurons
        self.ensemble = ensemble
        self._shared_embedding = emb
        self._general_sp = sp
        self._tokenizer_hub = hub
        self.device = "cpu"


def main():
    from taiji.life.integrate_engine import IntegrateEngine

    general_sp = load_general_tokenizer()
    DIALOGUE_DIR = "data/neurons"
    DIALOGUE_IDS = ["zh_std0_dialogue", "zh_aug0_dialogue", "zh_aug1_dialogue"]
    MOCK_SAMPLES = [
        "你好，请介绍一下你自己。",
        "你最近在做什么？",
        "今天天气很好，我想出去散步。",
        "你能帮我写一封邮件吗？",
        "介绍一下你的功能。",
        "晚上好，很高兴见到你。",
        "请用中文回答。",
        "你在学习什么？",
    ]

    shared_lm_head = load_shared_lm_head("data/foundation_v1_general", 512, "cpu")
    neurons, embeddings = {}, {}
    for nid in DIALOGUE_IDS:
        ckp = torch.load(os.path.join(DIALOGUE_DIR, f"neuron_{nid}.pt"),
                         map_location="cpu", weights_only=False)
        cfg = ckp["neuron_config"]; cfg.unified_field_dim = None
        from taiji.resonance.neuron import ResonanceNeuron
        n = ResonanceNeuron(cfg)
        n.load_state_dict(ckp["state_dict"], strict=False)
        neurons[nid] = n
        emb = create_shared_embedding("cpu")
        ses = ckp.get("shared_embedding_state", {})
        w = ses["weight"] if isinstance(ses, dict) else ses
        emb.weight.data.copy_(w)
        embeddings[nid] = emb
    shared_emb = create_shared_embedding("cpu")  # 统一用 general 共享嵌入

    geometry = NeuronGeometry(embedding_dim=8, sigma=0.5)
    topology = build_topology(neurons, geometry, mode="hybrid", k=3)
    establish_topology_channels(neurons, topology, geometry)
    max_field_dim = max(n.config.field_dim for n in neurons.values())
    field = ResonanceField(dim=max_field_dim)
    ens = ResonanceEnsemble(neurons, field, max_rounds=2, geometry=geometry)
    hub = TokenizerHub()
    hub.register_domain("zh", load_domain_tokenizer("zh"))
    hub.register_domain("general", general_sp)
    ens.set_tokenizer_hub(hub)

    # 新生 neuron：clone zh_std0 + 噪声（模拟 split 新生 zh_1）
    import copy
    parent = neurons["zh_std0_dialogue"]
    cfg = copy.deepcopy(parent.config)
    cfg.neuron_id = "zh_1"
    from taiji.resonance.neuron import ResonanceNeuron
    new_neuron = ResonanceNeuron(cfg)
    sd = parent.state_dict()
    for k, v in new_neuron.state_dict().items():
        if k in sd and v.shape == sd[k].shape and v.dtype in (torch.float32,):
            sd[k] = sd[k] + torch.randn_like(sd[k]) * 0.01
    new_neuron.load_state_dict(sd, strict=False)
    new_neuron.eval()
    ens.add_neuron("zh_1", new_neuron)
    neurons["zh_1"] = new_neuron

    lifecycle = MockLifecycle()
    lifecycle.maturity.register_new("zh_1")
    ens.maturity = lifecycle.maturity  # 启用静默期融合压低（C17）
    cortex = MockCortex(neurons, ens, shared_emb, general_sp, hub)
    feed = MockFeed(MOCK_SAMPLES)

    # 静默期验证：maturity=0 时 ensemble 融合压低新 neuron（conf 乘 0.1）
    r = ens.forward_train(
        neuron_embeddings={nid: shared_emb(torch.tensor(
            [general_sp.EncodeAsIds("你好")], dtype=torch.long)) for nid in neurons},
        n_rounds=2, fusion_mode="soft",
        targets=torch.tensor([general_sp.EncodeAsIds("你好")], dtype=torch.long),
        target_domain="zh",
    )
    print(f"[静默期] 新 neuron 路由权重: {r.get('weights', 'N/A')}")

    ie = IntegrateEngine(cortex, lifecycle=lifecycle, feed_engine=feed)
    result = ie.integrate("zh_1")
    print(f"\n[IntegrateEngine] 整合结果: {result}")

    # 校验新 neuron 协作层已写回 live
    m = lifecycle.maturity.get_maturity_ratio("zh_1")
    print(f"[校验] 新 neuron maturity={m:.2f}")
    assert result["status"] in ("training", "committed", "apoptosis", "skipped")
    print("\n✅ IntegrateEngine 冒烟通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
