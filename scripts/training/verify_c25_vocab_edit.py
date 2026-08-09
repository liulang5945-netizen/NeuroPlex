#!/usr/bin/env python3
"""C25 词库实时编辑冒烟验证（2026-08-09 用户决策：容量不限 + 实时编辑 → 不需要热插拔）。

验证目标：
1. EditableVocabulary：add_token → encode 新词优先整词 → decode 还原 → vocab_size 扩展
2. tokenizer_fingerprint 变化（对齐/转译表缓存失效机制）
3. piece_to_id / id_to_pipe 合并扩展区（对齐表构建依赖接口）
4. TokenizerHub API：to_editable / add_tokens / unregister_domain
5. resize_lm_head_for_vocab：neuron lm_head 扩展且旧行权重保留
6. 持久化 save/load 还原

运行：python -u scripts/training/verify_c25_vocab_edit.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from sentencepiece import SentencePieceProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance.translator import (
    TokenizerHub,
    EditableVocabulary,
    tokenizer_fingerprint,
    build_logits_alignment_matrix,
    resize_linear_for_vocab,
    resize_lm_head_for_vocab,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "taiji", "domains"))

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


def load_sp(domain: str) -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    sp.Load(os.path.join(DOMAINS_DIR, domain, f"sp_{domain}.model"))
    return sp


def main() -> None:
    print("=" * 60)
    print("C25 词库实时编辑冒烟验证")
    print("=" * 60)

    zh_sp = load_sp("zh")
    base = int(zh_sp.GetPieceSize())
    print(f"\n[0] zh base vocab = {base}", flush=True)

    # ---- 1. EditableVocabulary 核心 ----
    print("\n[1] EditableVocabulary 核心行为", flush=True)
    ev = EditableVocabulary(zh_sp)
    check("初始 vocab == base", ev.vocab_size() == base)

    # 注意：中文 SP 对部分词已在 base（如 "量子计算"→21632）——base 复用
    # 是设计行为；"超导"/"张量网络" 不在 base → 分配 ext id
    new_pieces = ["量子计算", "超导", "张量网络"]
    ids = ev.add_tokens(new_pieces)
    check("base 已含词复用 base id", ids[0] < base, f"量子计算→{ids[0]}")
    check("新词分配 ext id (≥base)", ids[1] >= base and ids[2] >= base,
          f"超导→{ids[1]} 张量网络→{ids[2]}")
    check("vocab_size 扩展 +2", ev.vocab_size() == base + 2)
    check("重复 add 幂等", ev.add_token("量子计算") == ids[0]
          and ev.add_token("超导") == ids[1])

    text = "量子计算与超导和张量网络"
    enc = ev.encode(text)
    check("encode 含 3 个目标词 id", all(i in enc for i in ids), f"enc={enc}")
    check("decode 还原原文", ev.decode(enc) == text, f"got={ev.decode(enc)!r}")
    check("新词整词编码", ev.encode("超导") == [ids[1]])

    # base token 编码不受影响（未含新词文本走 SP）
    plain = "普通中文文本不包含新词"
    enc_plain = ev.encode(plain)
    check("base 文本 encode 等价 SP", enc_plain == zh_sp.encode(plain))

    # ---- 2. fingerprint 变化（缓存失效机制） ----
    print("\n[2] tokenizer_fingerprint 缓存失效", flush=True)
    fp0 = tokenizer_fingerprint(zh_sp)
    fp1 = tokenizer_fingerprint(ev)
    check("指纹随 vocab 扩展变化", fp0 != fp1)
    ev2 = EditableVocabulary(zh_sp, ext_pieces=["超导", "张量网络"])
    check("同扩展内容指纹一致（缓存命中）", tokenizer_fingerprint(ev2) == fp1)

    # ---- 3. 对齐/转译表依赖接口 ----
    print("\n[3] 对齐/转译表依赖接口", flush=True)
    check("piece_to_id 查 ext 区", ev.piece_to_id("超导") == ids[1])
    check("id_to_piece 读 ext 区", ev.id_to_piece(ids[1]) == "超导")
    check("base piece 仍走 SP", ev.piece_to_id("量子计算") == ids[0])
    check("GetPieceSize 兼容", ev.GetPieceSize() == base + 2)

    # 端到端：转译矩阵随 target vocab 扩展自动重建（fingerprint 失效 → 缓存 miss）
    code_sp = load_sp("code")
    cache: dict = {}
    m0 = build_logits_alignment_matrix(code_sp, zh_sp, "code", "zh", cache=cache)
    m1 = build_logits_alignment_matrix(code_sp, ev, "code", "zh", cache=cache)
    check("target vocab 扩展 → 矩阵重建列数+2", m1.shape[1] == m0.shape[1] + 2,
          f"{tuple(m0.shape)} → {tuple(m1.shape)}")

    # ---- 4. TokenizerHub API ----
    print("\n[4] TokenizerHub 实时编辑 API", flush=True)
    hub = TokenizerHub()
    hub.register_domain("zh", load_sp("zh"))
    ev_hub = hub.to_editable("zh")
    check("to_editable 返回可编辑实例", isinstance(ev_hub, EditableVocabulary))
    got = hub.add_tokens("zh", ["边缘计算"])
    check("hub.add_tokens 分配 ext id", got[0] >= base)
    check("hub 内词表已扩展", hub.vocab_size("zh") == base + 1)
    check("unregister_domain 移除", hub.unregister_domain("zh") is True)
    check("移除后 fallback None", hub.get_tokenizer("zh") is None)

    # ---- 5. neuron lm_head resize ----
    print("\n[5] resize_lm_head_for_vocab", flush=True)
    head = torch.nn.Linear(512, base, bias=False)
    with torch.no_grad():
        head.weight.data.normal_(0, 0.02)
    old_row = head.weight.data[0].clone()
    new_head = resize_linear_for_vocab(head, base + 5)
    check("resize 后 out_features 扩展", new_head.out_features == base + 5)
    check("旧行权重保留", torch.equal(new_head.weight.data[0], old_row))

    class FakeNeuron:
        lm_head_rank = 0
        config = type("Cfg", (), {"neuron_id": "zh"})()
    fn = FakeNeuron()
    fn.lm_head = torch.nn.Linear(512, base, bias=False)
    ok = resize_lm_head_for_vocab(fn, base + 3)
    check("neuron lm_head resize 生效", ok and fn.lm_head.out_features == base + 3)
    ok2 = resize_lm_head_for_vocab(fn, base + 1)
    check("目标小于当前不重复 shrink", ok2 is False and fn.lm_head.out_features == base + 3)

    # ---- 6. 持久化 ----
    print("\n[6] 持久化 save/load", flush=True)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sp_zh_ext.json")
        ev.save_ext(p)
        ev3 = EditableVocabulary(load_sp("zh"), ext_path=p)
        check("load 还原扩展 token 数", ev3.vocab_size() == ev.vocab_size())
        check("load 后 encode 一致", ev3.encode("量子计算与超导和张量网络") == enc)

    print("\n" + "=" * 60)
    print(f"结果: {passed} PASS / {failed} FAIL")
    print("=" * 60)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
