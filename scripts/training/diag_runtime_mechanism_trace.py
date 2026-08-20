#!/usr/bin/env python3
"""Trace the real production and PlayEngine mechanism paths without training.

This is intentionally a diagnostic script, not a new runtime abstraction.  It
monkey-patches the existing class methods for one short run, records the
observed state transitions, restores the methods, and writes a compact JSON
report.  The purpose is to distinguish a callable mechanism from a mechanism
that is actually reached by the assembled production path.

Example:
    python -u scripts/training/diag_runtime_mechanism_trace.py --max-tokens 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, Dict

import torch

from neuroplex.loader import assemble_cortex
from neuroplex.life.sleep_engine import get_sleep_engine
from neuroplex.resonance.field import ResonanceField
from neuroplex.resonance.neuron import ResonanceNeuron
from neuroplex.resonance.ensemble import ResonanceEnsemble
from neuroplex.resonance.dialogue_format import build_dialogue_prompt


DEFAULT_PROMPT = "请用一句话说明你现在的任务。"
DEFAULT_REPORT = "reports/runtime_mechanism_trace_20260820.json"


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "norm": float(value.detach().float().norm().cpu().item()),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _field_norm(field: Any) -> float | None:
    state = getattr(field, "state", None)
    if state is None:
        return None
    return float(state.detach().float().norm().cpu().item())


def _install_trace(events: Dict[str, list]):
    """Install wrappers around real methods and return a restore callback."""
    originals = {
        "neuron_forward": ResonanceNeuron.forward,
        "field_reset": ResonanceField.reset,
        "field_write": ResonanceField.write,
        "field_update": ResonanceField.update,
        "field_write_inhibit": ResonanceField.write_inhibit,
        "ensemble_continuous": ResonanceEnsemble.continuous_forward,
        "ensemble_parallel": ResonanceEnsemble._parallel_forward,
    }

    def append(bucket: str, value: dict, limit: int = 4000) -> None:
        if len(events[bucket]) < limit:
            events[bucket].append(value)

    def traced_neuron_forward(self, *args, **kwargs):
        result = originals["neuron_forward"](self, *args, **kwargs)
        field_state = kwargs.get("field_state")
        side_signals = kwargs.get("side_signals")
        append(
            "neuron_forwards",
            {
                "nid": getattr(self, "neuron_id", None),
                "round_num": kwargs.get("round_num", 1),
                "field_state_present": field_state is not None,
                "side_signal_count": len(side_signals or {}),
                "output_keys": sorted(result.keys()) if isinstance(result, dict) else [],
                "has_resonance_score": (
                    isinstance(result, dict) and "resonance_score" in result
                ),
                "has_last_field_state": hasattr(self, "_last_field_state"),
            },
        )
        return result

    def traced_field_reset(self, *args, **kwargs):
        result = originals["field_reset"](self, *args, **kwargs)
        append(
            "field_events",
            {"op": "reset", "dim": getattr(self, "dim", None), "state_norm": _field_norm(self)},
        )
        return result

    def traced_field_write(self, neuron_id, vector, scale=1.0, *args, **kwargs):
        result = originals["field_write"](self, neuron_id, vector, scale=scale, *args, **kwargs)
        append(
            "field_events",
            {
                "op": "write",
                "nid": str(neuron_id),
                "scale": _scalar(scale),
                "vector_norm": _scalar(vector),
                "state_norm": _field_norm(self),
            },
        )
        return result

    def traced_field_update(self, neuron_id, vector, scale=1.0, *args, **kwargs):
        result = originals["field_update"](self, neuron_id, vector, scale=scale, *args, **kwargs)
        append(
            "field_events",
            {
                "op": "update",
                "nid": str(neuron_id),
                "scale": _scalar(scale),
                "vector_norm": _scalar(vector),
                "state_norm": _field_norm(self),
            },
        )
        return result

    def traced_field_write_inhibit(self, neuron_id, vector, weight=1.0, *args, **kwargs):
        result = originals["field_write_inhibit"](
            self, neuron_id, vector, weight=weight, *args, **kwargs
        )
        append(
            "field_events",
            {
                "op": "write_inhibit",
                "nid": str(neuron_id),
                "weight": _scalar(weight),
                "vector_norm": _scalar(vector),
                "state_norm": _field_norm(self),
            },
        )
        return result

    def traced_continuous(self, *args, **kwargs):
        active = kwargs.get("active_nids")
        seed_memories = kwargs.get("seed_memories")
        append(
            "ensemble_calls",
            {
                "path": "continuous_forward.enter",
                "active_nids": list(active) if active is not None else None,
                "seed_memory_count": len(seed_memories or []),
            },
        )
        result = originals["ensemble_continuous"](self, *args, **kwargs)
        append(
            "ensemble_calls",
            {
                "path": "continuous_forward.exit",
                "result_keys": sorted(result.keys()),
                "n_steps": result.get("n_steps"),
                "final_score_count": len(result.get("final_scores") or {}),
                "phase_mean": _scalar(result.get("phase_mean")),
                "phase_lock": _scalar(result.get("phase_lock")),
                "field_state": _scalar(result.get("field_state")),
            },
        )
        return result

    def traced_parallel(self, *args, **kwargs):
        active_ids = args[0] if args else kwargs.get("active_ids")
        round_num = kwargs.get("round_num")
        if len(args) >= 4:
            round_num = args[3]
        append(
            "parallel_calls",
            {
                "active_nids": list(active_ids or []),
                "round_num": round_num,
                "field_state_present": (
                    kwargs.get("field_state") is not None
                    if "field_state" in kwargs
                    else (args[2] is not None if len(args) >= 3 else False)
                ),
                "side_signal_count": len(kwargs.get("side_signals") or {}),
            },
        )
        return originals["ensemble_parallel"](self, *args, **kwargs)

    ResonanceNeuron.forward = traced_neuron_forward
    ResonanceField.reset = traced_field_reset
    ResonanceField.write = traced_field_write
    ResonanceField.update = traced_field_update
    ResonanceField.write_inhibit = traced_field_write_inhibit
    ResonanceEnsemble.continuous_forward = traced_continuous
    ResonanceEnsemble._parallel_forward = traced_parallel

    def restore() -> None:
        ResonanceNeuron.forward = originals["neuron_forward"]
        ResonanceField.reset = originals["field_reset"]
        ResonanceField.write = originals["field_write"]
        ResonanceField.update = originals["field_update"]
        ResonanceField.write_inhibit = originals["field_write_inhibit"]
        ResonanceEnsemble.continuous_forward = originals["ensemble_continuous"]
        ResonanceEnsemble._parallel_forward = originals["ensemble_parallel"]

    return restore


def _trace_one(cortex, modules: dict, prompt: str, max_tokens: int) -> dict:
    events: Dict[str, list] = {
        "neuron_forwards": [],
        "field_events": [],
        "ensemble_calls": [],
        "parallel_calls": [],
        "think_calls": [],
        "coaction_updates": [],
    }
    restore = _install_trace(events)

    original_think = cortex.think
    original_coaction_update = getattr(cortex.coaction, "update", None)

    def traced_think(*args, **kwargs):
        result = original_think(*args, **kwargs)
        events["think_calls"].append(
            {
                "active_nids": kwargs.get("active_nids"),
                "collab_mode": kwargs.get("collab_mode"),
                "result_keys": sorted(result.keys()),
                "field_state": _scalar(result.get("field_state")),
            }
        )
        return result

    def traced_coaction(*args, **kwargs):
        ids = args[0] if args else kwargs.get("ids")
        events["coaction_updates"].append({"ids": list(ids or []), "kwargs": dict(kwargs)})
        return original_coaction_update(*args, **kwargs)

    cortex.think = traced_think
    if original_coaction_update is not None:
        cortex.coaction.update = traced_coaction

    sleep_engine = get_sleep_engine()
    pending_before = len(getattr(sleep_engine, "pending_field_memories", []))
    replay_before = len(getattr(modules.get("sleep_consolidator"), "_replay_buffer", []))
    started = time.time()
    generated = None
    play_result = None
    play_error = None
    play_line_trace = []
    play_engine = modules.get("play_engine")
    play_preconditions = {
        "module_present": play_engine is not None,
        "cortex_set": bool(getattr(play_engine, "_cortex", None)) if play_engine else False,
        "neuron_count": len(getattr(getattr(play_engine, "_cortex", None), "neurons", {}))
        if play_engine else 0,
        "tokenizer_hub_present": bool(
            getattr(getattr(play_engine, "_cortex", None), "_tokenizer_hub", None)
        ) if play_engine else False,
        "shared_embedding_present": bool(
            getattr(getattr(play_engine, "_cortex", None), "_shared_embedding", None)
        ) if play_engine else False,
    }
    play_probe = {}
    if play_engine is not None and play_engine._cortex is not None:
        hub = getattr(play_engine._cortex, "_tokenizer_hub", None)
        shared = getattr(play_engine._cortex, "_shared_embedding", None)
        topic = "人工智能能有意识吗？"
        if hub is not None:
            for domain in ("general", "en"):
                try:
                    ids = hub.encode(topic, domain=domain)
                    play_probe[f"{domain}_encode"] = {
                        "length": len(ids),
                        "max_id": max(ids) if ids else None,
                    }
                except Exception as exc:
                    play_probe[f"{domain}_encode_error"] = f"{type(exc).__name__}: {exc}"
        if shared is not None and play_probe.get("en_encode", {}).get("length", 0):
            try:
                ids = hub.encode(topic, domain="en")
                emb = shared(torch.tensor([ids[:128]], dtype=torch.long, device=shared.weight.device))
                play_probe["shared_embedding_probe"] = {"shape": list(emb.shape)}
            except Exception as exc:
                play_probe["shared_embedding_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        generated = cortex.generate(
            build_dialogue_prompt(prompt),
            max_tokens=max_tokens,
            domain="zh",
            collab_mode="continuous",
            auto_memory=True,
            instance_routing=False,
        )
        if play_engine is not None:
            previous_trace = sys.gettrace()

            def _trace_play(frame, event, arg):
                if frame.f_code.co_filename.endswith("play_engine.py"):
                    if frame.f_code.co_name == "_free_resonance_session" and event == "line":
                        play_line_trace.append(frame.f_lineno)
                    if frame.f_code.co_name == "_free_resonance_session" and event == "exception":
                        exc_type, exc_value, _ = arg
                        play_line_trace.append({
                            "exception_type": exc_type.__name__,
                            "exception": str(exc_value),
                            "line": frame.f_lineno,
                        })
                    return _trace_play
                return _trace_play

            sys.settrace(_trace_play)
            try:
                play_result = play_engine._free_resonance_session()
            except Exception as exc:  # diagnostic result, not a hidden failure
                play_error = f"{type(exc).__name__}: {exc}"
            finally:
                sys.settrace(previous_trace)
    finally:
        cortex.think = original_think
        if original_coaction_update is not None:
            cortex.coaction.update = original_coaction_update
        restore()

    pending_after = len(getattr(sleep_engine, "pending_field_memories", []))
    replay_after = len(getattr(modules.get("sleep_consolidator"), "_replay_buffer", []))
    neuron_rounds = Counter(
        (row.get("nid"), row.get("round_num")) for row in events["neuron_forwards"]
    )
    return {
        "prompt": prompt,
        "generated": generated,
        "elapsed_seconds": round(time.time() - started, 3),
        "population": sorted(cortex.neurons.keys()),
        "default_collab_mode": "continuous",
        "pending_field_memories": {"before": pending_before, "after": pending_after},
        "sleep_replay_buffer": {"before": replay_before, "after": replay_after},
        "play_result": repr(play_result) if play_result is not None else None,
        "play_error": play_error,
        "play_line_trace": play_line_trace,
        "play_preconditions": play_preconditions,
        "play_probe": play_probe,
        "counts": {
            "think_calls": len(events["think_calls"]),
            "ensemble_calls": len(events["ensemble_calls"]),
            "parallel_calls": len(events["parallel_calls"]),
            "neuron_forward_calls": len(events["neuron_forwards"]),
            "field_events": len(events["field_events"]),
            "coaction_updates": len(events["coaction_updates"]),
        },
        "neuron_round_histogram": {
            f"round={round_num}": sum(
                count for (nid, round_value), count in neuron_rounds.items()
                if round_value == round_num
            )
            for round_num in sorted({round_value for _, round_value in neuron_rounds})
        },
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--skip-play", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be >= 1")
    logging.disable(logging.CRITICAL)
    started = time.time()
    cortex, _, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name="collab_v3_c24v2.ckpt.pt",
        extra_neurons_dir="data/foundation_v1_dual",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=None,
    )
    if args.skip_play:
        modules.pop("play_engine", None)
    result = _trace_one(cortex, modules, args.prompt, args.max_tokens)
    result["elapsed_total_seconds"] = round(time.time() - started, 3)
    result["contract"] = {
        "writes_checkpoint": False,
        "trains": False,
        "uses_assemble_cortex": True,
        "uses_cortex_generate": True,
        "play_path_attempted": not args.skip_play,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
