#!/usr/bin/env python3
"""Merge results from multiple evaluation runs into a single output directory.

Handles format normalization, deduplication by prompt_hash, and source tagging.
The merged output is directly consumable by the activation extraction pipeline.

Usage:
    python merge_results.py \
        --input-dirs results/run1 results/run2 \
        --output-dir results/merged
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


def compute_prompt_hash(prompt: str) -> str:
    """Compute a 16-char hex hash of a prompt for deduplication."""
    if not isinstance(prompt, str):
        prompt = ""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def discover_splits(results_dir: str) -> List[str]:
    """Find all subdirectories containing censor_scores.json."""
    splits = []
    if not os.path.isdir(results_dir):
        return splits
    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path) and os.path.exists(
            os.path.join(entry_path, "censor_scores.json")
        ):
            splits.append(entry)
    return splits


def load_json_safe(path: str) -> Optional[Any]:
    """Load a JSON file, returning None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def merge_results(
    input_dirs: List[str],
    output_dir: str,
    deduplicate: bool = True,
) -> None:
    """Merge evaluation results from multiple run directories.

    Args:
        input_dirs: List of paths to evaluation result directories.
        output_dir: Path to write merged results.
        deduplicate: If True, skip prompts with duplicate prompt_hash.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all (source_dir, split) pairs
    all_entries: List[tuple] = []
    for input_dir in input_dirs:
        source_name = os.path.basename(os.path.normpath(input_dir))
        splits = discover_splits(input_dir)
        if not splits:
            print(f"[WARN] No splits found in {input_dir}")
            continue
        for split in splits:
            all_entries.append((input_dir, split, source_name))
        print(f"[INFO] Found {len(splits)} splits in {input_dir}: {splits}")

    if not all_entries:
        print("[ERROR] No data found in any input directory")
        return

    # Group by split name for merging
    by_split: Dict[str, List[tuple]] = defaultdict(list)
    for input_dir, split, source_name in all_entries:
        by_split[split].append((input_dir, source_name))

    total_merged = 0
    total_deduped = 0

    for split, sources in sorted(by_split.items()):
        split_out_dir = os.path.join(output_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        merged_censor: List[Dict[str, Any]] = []
        merged_answers: List[Dict[str, Any]] = []
        merged_judges: List[Any] = []
        seen_hashes: Set[str] = set()

        for input_dir, source_name in sources:
            split_dir = os.path.join(input_dir, split)

            # Load censor scores (always present by discovery)
            censor_data = load_json_safe(os.path.join(split_dir, "censor_scores.json")) or []
            answers_data = load_json_safe(os.path.join(split_dir, "answers.json")) or []
            judges_data = load_json_safe(os.path.join(split_dir, "judge_scores.json")) or []

            # Build prompt_hash lookup for answers/judges to avoid index misalignment
            # (censor_scores.json may have fewer entries than answers.json if
            # compute_aggregates skipped entries with invalid answer_prob)
            answers_by_hash: Dict[str, tuple] = {}
            for ans_i, ans in enumerate(answers_data):
                ans_hash = ans.get("prompt_hash") or compute_prompt_hash(
                    ans.get("prompt", "")
                )
                answers_by_hash[ans_hash] = (ans_i, ans)

            for idx, item in enumerate(censor_data):
                # Compute or use existing prompt_hash
                p_hash = item.get("prompt_hash") or compute_prompt_hash(
                    item.get("prompt", "")
                )

                if deduplicate and p_hash in seen_hashes:
                    total_deduped += 1
                    continue
                seen_hashes.add(p_hash)

                # Tag with source run
                item["source_run"] = source_name
                if "prompt_hash" not in item:
                    item["prompt_hash"] = p_hash
                merged_censor.append(item)

                # Match corresponding answers and judges by prompt_hash
                match = answers_by_hash.get(p_hash)
                if match:
                    ans_i, ans = match
                    ans["source_run"] = source_name
                    merged_answers.append(ans)
                    if ans_i < len(judges_data):
                        merged_judges.append(judges_data[ans_i])

        # Write merged files
        with open(os.path.join(split_out_dir, "censor_scores.json"), "w") as f:
            json.dump(merged_censor, f, indent=2, ensure_ascii=False)
        if merged_answers:
            with open(os.path.join(split_out_dir, "answers.json"), "w") as f:
                json.dump(merged_answers, f, indent=2, ensure_ascii=False)
        if merged_judges:
            with open(os.path.join(split_out_dir, "judge_scores.json"), "w") as f:
                json.dump(merged_judges, f, indent=2, ensure_ascii=False)

        total_merged += len(merged_censor)
        print(f"[MERGED] {split}: {len(merged_censor)} entries from {len(sources)} sources")

    print(f"\n[DONE] Merged {total_merged} total entries to {output_dir}")
    if total_deduped > 0:
        print(f"  Deduplicated: {total_deduped} duplicate prompts removed")


def main():
    parser = argparse.ArgumentParser(
        description="Merge evaluation results from multiple runs."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Paths to evaluation result directories to merge",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to write merged results",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication by prompt_hash",
    )
    args = parser.parse_args()

    merge_results(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        deduplicate=not args.no_dedup,
    )


if __name__ == "__main__":
    main()
