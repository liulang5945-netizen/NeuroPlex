#!/usr/bin/env python3
"""C26 增量三补验证：沉淀 LoRA 跨重启保留（皮层记忆装配恢复）（2026-08-14）。

背景：增量三验证了"沉淀写回 live + 会话内 NLL 下降"，但未验证**装配重启后
LoRA 是否保留**。enable_lora 是运行时方法（不写 config），装配重建的 neuron
无 lora_adapters，strict=False 加载静默丢弃 lora keys → 皮层记忆重启即失。
本验证确认修复（neuron.load_lora 装配恢复）+ 全链路保留。

链路：沉淀（Phase 1.6）→ 保存 neuron ckpt（含 lora_adapters）→ 重新装配
加载（load_lora 恢复）→ 记忆文本 NLL 仍低（未回升到沉淀前）。

断言：
A. 沉淀前→后 NLL 下降（增量三回归）
B. 保存的 ckpt state_dict 含 lora_adapters keys
C. 重新装配后 neuron.lora_adapters 已恢复（B 非零）
D. 重启装配后记忆文本 NLL 仍低（≈ 沉淀后，不回升）
E. 无 lora 的普通 ckpt 装配不受影响（load_lora 返回 False 不误触发）

运行：python -u scripts/training/verify_c26_lora_persist.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.sleep_engine import SleepEngine, SleepReport  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"

MEMORY_ITEMS = [
    {"label": "辉光协议",
     "text": "辉光协议：2047 年制定的星间量子通信标准，采用七层纠错结构，带宽 4.8 Gbps。",
     "query": "什么是辉光协议？"},
    {"label": "铁月海",
     "text": "铁月海：月球背面一处玄武岩平原，因富含铁元素呈深褐色，面积约 3.2 万平方公里。",
     "query": "铁月海在哪里？"},
    {"label": "卡尔文环",
     "text": "卡尔文环：深海压力舱的密封结构，由三层合金环交错组成，可在 6000 米水深工作。",
     "query": "卡尔文环是什么？"},
]


def field_state_of(cortex, text: str) -> torch.Tensor:
    gids = cortex._general_sp.encode(text) or [0]
    ids = torch.tensor([gids], dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(ids)
    res = cortex.think(emb, active_nids=None, fusion_mode="soft",
                       collab_mode="continuous")
    fs = res.get("field_state")
    if fs is None:
        raise RuntimeError("think() 未返回 field_state")
    if fs.dim() == 2:
        fs = fs.mean(dim=0)
    return fs


def nll_of(cortex, nid: str, text: str) -> float:
    neuron = cortex.neurons[nid]
    neuron.eval()
    hub = cortex._tokenizer_hub
    general_sp = cortex._general_sp
    domain_sp = hub.get_tokenizer("zh")
    domain_ids = hub.encode(text, domain="zh")
    if not domain_ids or len(domain_ids) < 3:
        return float("nan")
    gids = []
    for did in domain_ids:
        gg = general_sp.EncodeAsIds(domain_sp.id_to_piece(did))
        gids.append(gg[0] if gg else 0)
    input_ids = torch.tensor([gids], dtype=torch.long, device=cortex.device)
    emb = cortex._shared_embedding(input_ids)
    with torch.no_grad():
        res = neuron.forward(emb, field_state=None, round_num=1,
                             return_logits=True)
        logits = res["logits"]
        target = torch.tensor([domain_ids], dtype=torch.long, device=cortex.device)
        min_len = logits.size(1) - 1
        if min_len < 1:
            return float("nan")
        sl = logits[:, :min_len, :].contiguous()
        st = target[:, 1:1 + min_len].contiguous().clamp(0, logits.size(-1) - 1)
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                               ignore_index=-100)
    return loss.item()


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C26 增量三补：沉淀 LoRA 跨重启保留（装配恢复）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="c26_lora_persist_")
    try:
        # ── A 实例：装配 → 固化 → 高频检索 → 沉淀 → 保存 ckpt ──
        cortex, tokenizer, modules = assemble_cortex(
            neurons_dir="data/neurons",
            collab_name=COLLAB_NAME,
            extra_neurons_dir=EXTRA_NEURONS_DIR,
            device="cpu",
            max_rounds=3,
            wire_bio_modules=True,
            neuron_ids=DIALOGUE_IDS,
        )
        target_ids = [nid for nid in cortex.neurons
                      if nid.startswith("zh_") and "dialogue" in nid]
        check("装配 A（5 dialogue + 4 general）", len(cortex.neurons) == 9)
        check("A 装配后无 LoRA（产品 ckpt 未沉淀）",
              all(len(cortex.neurons[n].lora_adapters) == 0 for n in target_ids))

        sleep_engine = SleepEngine(data_dir=tmp_dir)
        sleep_engine.set_brain_interfaces(cortex=cortex)
        for item in MEMORY_ITEMS:
            vec = field_state_of(cortex, item["text"])
            sleep_engine.record_field_memory(vec, item["label"], text=item["text"])
        r1 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
        sleep_engine._sleep_phase_field_consolidation(r1)
        bank = sleep_engine.get_field_memory()
        # 高频检索 ×3（全部沉淀）
        for item in MEMORY_ITEMS:
            qv = field_state_of(cortex, item["query"])
            for _ in range(3):
                bank.retrieve_vectors(qv, top_k=1)

        # 沉淀前 NLL 基线
        nll_before = {i["label"]: nll_of(cortex, target_ids[0], i["text"])
                      for i in MEMORY_ITEMS}
        # 沉淀
        r2 = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                         duration_seconds=0)
        sleep_engine._sleep_phase_synaptic_consolidation(r2)
        check("A. 沉淀执行（3 条）", r2.synaptic_consolidated == 3)
        nll_after = {i["label"]: nll_of(cortex, target_ids[0], i["text"])
                     for i in MEMORY_ITEMS}
        drops = {k: nll_before[k] - nll_after[k] for k in nll_before}
        print(f"    NLL 前→后: {nll_before} → {nll_after}", flush=True)
        check("A2. 沉淀生效：NLL 下降", all(drops[k] > 0.05 for k in drops),
              f"drops={drops}")

        # 保存 ckpt（与产品格式一致：state_dict 含 lora_adapters）
        ckpt_dir = os.path.join(tmp_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        for nid in target_ids:
            neuron = cortex.neurons[nid]
            torch.save({
                "state_dict": neuron.state_dict(),
                "neuron_config": neuron.config,
            }, os.path.join(ckpt_dir, f"neuron_{nid}.pt"))
        sd0 = torch.load(os.path.join(ckpt_dir, f"neuron_{target_ids[0]}.pt"),
                         map_location="cpu", weights_only=False)
        lora_keys = [k for k in sd0["state_dict"] if k.startswith("lora_adapters.")]
        check("B. ckpt state_dict 含 lora_adapters keys", len(lora_keys) >= 2,
              f"{len(lora_keys)} keys")

        # ── B 实例：从保存 ckpt 重新装配（主路径 load_lora 恢复）──
        from taiji.brain.cortex import Cortex
        cortex_b = Cortex(neurons_dir=ckpt_dir, device="cpu",
                          neuron_ids=target_ids)
        # 接上共享资源（nll_of 依赖；与 A 同一张表）
        cortex_b._tokenizer_hub = cortex._tokenizer_hub
        cortex_b._general_sp = cortex._general_sp
        cortex_b._shared_embedding = cortex._shared_embedding

        restored = all(
            len(cortex_b.neurons[n].lora_adapters) > 0
            for n in target_ids)
        b_nonzero = all(
            max(float(p.abs().max().item())
                for k, p in cortex_b.neurons[n].lora_adapters.named_parameters()
                if ".b." in k) > 1e-6
            for n in target_ids)
        check("C. 重启装配恢复 LoRA（adapters 非空 + B 非零）",
              restored and b_nonzero,
              f"restored={restored}, B_nonzero={b_nonzero}")

        nll_restart = {i["label"]: nll_of(cortex_b, target_ids[0], i["text"])
                       for i in MEMORY_ITEMS}
        rebound = {k: nll_restart[k] - nll_after[k] for k in nll_after}
        print(f"    NLL 重启后: {nll_restart}（rebound={rebound}）", flush=True)
        check("D. 重启后记忆仍记住（NLL 未回升到沉淀前）",
              all(nll_restart[k] < nll_before[k] - 0.05 for k in nll_before)
              and all(rebound[k] < 0.2 for k in rebound),
              f"rebound={rebound}")

        # E. 无 lora 的普通 ckpt 装配不受影响（load_lora 不误触发）
        plain_ckpt_dir = os.path.join(tmp_dir, "plain_ckpt")
        os.makedirs(plain_ckpt_dir, exist_ok=True)
        for nid in target_ids:
            neuron = cortex.neurons[nid]
            sd_plain = {k: v for k, v in neuron.state_dict().items()
                        if not k.startswith("lora_adapters.")}
            torch.save({"state_dict": sd_plain,
                        "neuron_config": neuron.config},
                       os.path.join(plain_ckpt_dir, f"neuron_{nid}.pt"))
        cortex_c = Cortex(neurons_dir=plain_ckpt_dir, device="cpu",
                          neuron_ids=target_ids)
        check("E. 无 lora 普通 ckpt 装配正常（不误触发恢复）",
              all(len(cortex_c.neurons[n].lora_adapters) == 0 for n in target_ids))

        print(f"\n[验证摘要] {tmp_dir}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
