"""Read-only fusion-mode A/B for the real nine-member production population."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from neuroplex.loader import assemble_cortex


DIALOGUE_IDS = [
    "zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
    "zh_aug3_dialogue", "zh_std0_dialogue",
]
QUESTIONS = ["你好", "你是谁？", "今天天气怎么样？", "帮我写一首关于春天的诗"]
FUSION_MODES = ("soft", "per_position", "residual", "division")
SEED = 20260819


def _reset(cortex) -> None:
    cortex.field.reset()
    cortex.clear_dialogue_state()
    if cortex.gamma_oscillator is not None and hasattr(cortex.gamma_oscillator, "reset"):
        cortex.gamma_oscillator.reset()


def run() -> dict:
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(6)
    started = time.time()
    cortex, _, _ = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c24v2.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_dual",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    population = list(cortex.neurons.keys())
    expected = set(DIALOGUE_IDS) | {"code", "en", "math", "zh"}
    if set(population) != expected:
        raise RuntimeError(f"production population mismatch: {population}")
    report = {
        "contract": {
            "population": population,
            "population_shape": "5 dialogue + 4 general",
            "questions": QUESTIONS,
            "fusion_modes": list(FUSION_MODES),
            "temperature": 0.55,
            "top_k": 15,
            "repetition_penalty": 1.4,
            "max_tokens": 8,
            "seed": SEED,
            "writes_checkpoint": False,
        },
        "outputs": {question: {} for question in QUESTIONS},
    }
    for question in QUESTIONS:
        prompt = f"问：{question}\n答："
        for mode in FUSION_MODES:
            _reset(cortex)
            torch.manual_seed(SEED)
            try:
                output = cortex.generate(
                    prompt=prompt,
                    max_tokens=8,
                    temperature=0.55,
                    top_k=15,
                    domain="zh",
                    repetition_penalty=1.4,
                    fusion_mode=mode,
                    auto_memory=False,
                    instance_routing=False,
                )
                error = None
            except Exception as exc:
                output = None
                error = repr(exc)
            report["outputs"][question][mode] = {
                "text": output,
                "error": error,
            }
            print(f"{question} | fusion={mode} | {output or error}", flush=True)
    report["elapsed_s"] = round(time.time() - started, 2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
