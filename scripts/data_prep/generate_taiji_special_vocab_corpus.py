#!/usr/bin/env python3
"""Generate Taiji special-format text for tokenizer corpus balancing."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

TOOLS = [
    "search",
    "read_file",
    "write_file",
    "run_command",
    "inspect_image",
    "query_memory",
    "save_memory",
    "plan_task",
]

GOALS = [
    "check_training_data_quality",
    "fix_script_parameter_errors",
    "verify_tokenizer_contract_integrity",
    "analyze_long_term_user_goal",
    "prepare_next_training_plan",
    "verify_remote_logs_are_healthy",
    "generate_concise_status_report",
    "compare_zh_en_data_ratio",
]

CAUSES = [
    "input_path_missing",
    "bucket_mix_unbalanced",
    "tool_returned_empty_payload",
    "mirror_timeout",
    "checkpoint_missing_required_files",
    "user_goal_changed",
    "sample_duplication_too_high",
    "missing_required_argument",
]

CORRECTIONS = [
    "inspect_project_tree_before_running_commands",
    "rebalance_zh_en_code_math_tech_taiji_special_sampling",
    "reduce_shard_count_and_record_gap",
    "switch_to_mirror_friendly_source",
    "revalidate_model_config_and_tokenizer_files",
    "restate_current_goal_and_refresh_plan",
    "deduplicate_before_writing_training_corpus",
    "fill_missing_arguments_then_retry",
]

OBSERVATIONS = [
    "sentencepiece_bpe_training_is_running",
    "code_and_math_quota_are_filled",
    "tokenizer_contract_validation_passed",
    "data_disk_space_is_tight",
    "user_prioritizes_1b_quality",
    "english_data_exists_but_tech_docs_are_light",
    "multimodal_tokens_are_reserved_but_alignment_not_started",
    "active_training_job_should_not_be_interrupted",
]

MEMORY_TYPES = ["short_term", "long_term", "episodic", "semantic"]
MODALITIES = ["image", "audio", "video", "screen"]


def build_records(limit: int) -> list[str]:
    records: list[str] = []

    for tool, goal in product(TOOLS, GOALS):
        payload = {"name": tool, "args": {"goal": goal, "dry_run": True}}
        records.append(f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>")
        records.append(f"<tool_result>tool {tool} finished: {goal}</tool_result>")

    for goal in GOALS:
        records.append(
            f"<plan><goal>{goal}</goal><step>understand_goal</step><step>run_checks</step>"
            f"<step>verify_output</step><plan_done></plan_done></plan>"
        )
        records.append(f"<think>choose the smallest reliable action set for {goal}</think>")
        records.append(f"<final_answer>{goal} completed with reviewable evidence.</final_answer>")

    for cause, correction in product(CAUSES, CORRECTIONS):
        records.append(f"<reflect><cause>{cause}</cause><correct>{correction}</correct></reflect>")

    for memory_type, goal in product(MEMORY_TYPES, GOALS):
        records.append(f"<mem_read><{memory_type}>read memory about {goal}</{memory_type}></mem_read>")
        records.append(f"<mem_write><{memory_type}>record verified outcome for {goal}</{memory_type}></mem_write>")

    for modality, observation in product(MODALITIES, OBSERVATIONS):
        records.append(f"<{modality}>{observation}</{modality}>")
        records.append(f"<observe><{modality}>{observation}</{modality}></observe>")

    for observation in OBSERVATIONS:
        records.append(f"<observe>{observation}</observe>")
        records.append(f"<inner_voice>verify facts first: {observation}</inner_voice>")
        records.append(f"<context>{observation}</context>")

    deduped = list(dict.fromkeys(records))
    if len(deduped) >= limit:
        return deduped[:limit]

    expanded: list[str] = []
    round_idx = 0
    while len(expanded) < limit:
        for record in deduped:
            if len(expanded) >= limit:
                break
            expanded.append(f"{record} <state>variant={round_idx}</state>")
        round_idx += 1
    return expanded


def write_records(output: Path, records: int) -> dict[str, str | int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    built_records = build_records(records)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for text in built_records:
            handle.write(json.dumps({"text": text, "source": "taiji_special_vocab"}, ensure_ascii=False) + "\n")
    return {"output": str(output), "records": len(built_records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Taiji special-format tokenizer text")
    parser.add_argument(
        "--output",
        default="taiji_data/training_data/pretrain_12b/tokenizer_sample/taiji_special_vocab.jsonl",
    )
    parser.add_argument("--records", type=int, default=20_000)
    args = parser.parse_args()

    result = write_records(Path(args.output), args.records)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
