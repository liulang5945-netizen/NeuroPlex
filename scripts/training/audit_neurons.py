"""神经元质量摸底——盘点所有 neuron 质量，识别好的，整合验证。

步骤：
  1. 元数据摸底：读每个 neuron ckpt 的 PPL/loss/saved
  2. 单干生成：每个 neuron 用各自 domain 生成，评估连贯性
  3. 汇总：哪些是"质量好"的
"""
import sys, os, glob
sys.path.insert(0, "e:/taiji-neuron")

import torch

NEURONS_DIR = "e:/taiji-neuron/data/neurons"


def audit_metadata():
    """读取所有 neuron checkpoint 元数据。"""
    print("=" * 70, flush=True)
    print("[1] 神经元元数据摸底", flush=True)
    print("=" * 70, flush=True)
    ckpts = sorted(glob.glob(os.path.join(NEURONS_DIR, "neuron_*.pt")))
    rows = []
    for path in ckpts:
        name = os.path.basename(path)
        if "_fieldcond" in name or name.startswith("_"):
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("neuron_config")
        result = ckpt.get("result", {})
        domain = ckpt.get("domain", "?")
        final_ppl = result.get("final_ppl", "N/A")
        best_loss = result.get("best_loss", "N/A")
        saved = result.get("saved", "N/A")
        spec = getattr(cfg, "spec", "?") if cfg else "?"
        row = (name, domain, spec, final_ppl, best_loss, saved)
        rows.append(row)
        print(f"  {name:28s} domain={domain:8s} spec={str(spec):14s} "
              f"final_PPL={final_ppl} best_loss={best_loss} saved={saved}", flush=True)
    return rows


def gen_test(cortex):
    """对每个 neuron 单干生成（domain 匹配），评估质量。"""
    print("\n" + "=" * 70, flush=True)
    print("[2] 单干生成质量测试（各 neuron 用各自 domain）", flush=True)
    print("=" * 70, flush=True)
    DOMAIN_PROMPTS = {
        "zh":      "你好，请介绍一下自己",
        "en":      "Hello, please introduce yourself",
        "code":    "def fibonacci(n):",
        "math":    "Solve the equation x^2 + 2x - 3 = 0",
        "general": "你好，请介绍一下你自己",
    }
    for nid in cortex.neurons:
        domain = nid.split("_")[0] if "_" in nid else nid
        prompt = DOMAIN_PROMPTS.get(domain, "Hello")
        print(f"\n--- [{nid}] domain={domain} ---", flush=True)
        try:
            out = cortex.generate(
                prompt=prompt, active_nids=[nid], collab_mode="leader",
                max_tokens=50, temperature=0.8, top_k=40,
                domain=domain, repetition_penalty=1.2,
            )
        except Exception as e:
            out = f"[ERROR] {e}"
        print(f"  prompt: {prompt}", flush=True)
        print(f"  output: {out[:160] if out else '(empty)'}", flush=True)


def main():
    rows = audit_metadata()

    print("\n装配 Cortex 做生成测试...", flush=True)
    from taiji.loader import assemble_cortex
    cortex, _, _ = assemble_cortex()
    print(f"  neurons: {list(cortex.neurons.keys())}", flush=True)

    gen_test(cortex)

    print("\n" + "=" * 70, flush=True)
    print("[3] 汇总判读", flush=True)
    print("=" * 70, flush=True)
    print("  质量判断标准：", flush=True)
    print("  - 输出 domain 匹配（中文 neuron 输出中文，非符号乱码）", flush=True)
    print("  - 有连贯片段（非纯词组堆砌或乱码）", flush=True)
    print("  - PPL < 100 为可用基线", flush=True)
    print("  → 识别'质量好'的 neuron，下一步整合（族长主导多域协作）验证", flush=True)


if __name__ == "__main__":
    main()
