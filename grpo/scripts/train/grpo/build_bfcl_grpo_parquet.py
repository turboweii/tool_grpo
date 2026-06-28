"""Build leakage-safe GRPO parquet files for BFCL v4 multi-turn."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.bfcl_wrapper import first_turn_messages, load_bfcl_entries  # noqa: E402

INTERACTION_NAME = "bfcl_v4_multi_turn"


def build_rows(entries: list[dict], ids: list[int], split: str) -> list[dict]:
    rows = []
    for out_idx, entry_idx in enumerate(ids):
        entry = entries[entry_idx]
        entry_id = str(entry.get("id", entry_idx))
        rows.append({
            "prompt": first_turn_messages(entry),
            "extra_info": {
                "index": out_idx,
                "task_id": entry_idx,
                "entry_id": entry_id,
                "split": split,
                "interaction_kwargs": {
                    "name": INTERACTION_NAME,
                    "task_id": entry_idx,
                    "entry_id": entry_id,
                },
            },
            "data_source": INTERACTION_NAME,
            "reward_model": {"ground_truth": entry.get("ground_truth", [])},
            "ability": INTERACTION_NAME,
        })
    return rows


def load_split(path: str | None, all_ids: list[int], holdout_size: int) -> tuple[list[int], list[int]]:
    if path:
        with open(path) as f:
            split = json.load(f)
        seen_ids = [int(x) for x in split.get("seen_task_ids", [])]
        unseen_ids = [int(x) for x in split.get("unseen_task_ids", [])]
        if not unseen_ids:
            unseen_ids = [idx for idx in all_ids if idx not in set(seen_ids)]
    else:
        holdout = max(1, min(int(holdout_size), len(all_ids) - 1))
        seen_ids = all_ids[:-holdout]
        unseen_ids = all_ids[-holdout:]

    seen_set = set(seen_ids)
    unseen_set = set(unseen_ids)
    overlap = seen_set & unseen_set
    if overlap:
        raise RuntimeError(f"BFCL split leakage: ids appear in both seen and unseen: {sorted(overlap)[:20]}")
    if not unseen_ids:
        raise RuntimeError("BFCL split has no unseen tasks; refusing leakage-prone validation.")
    if not seen_ids:
        raise RuntimeError("BFCL split has no seen tasks for training.")
    return seen_ids, unseen_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--seen-task-ids-from", default=None)
    parser.add_argument("--holdout-size", type=int, default=40)
    parser.add_argument("--output-train", default="experiments/bfcl_v4_multi_turn/train.parquet")
    parser.add_argument("--output-val-unseen", default="experiments/bfcl_v4_multi_turn/val_unseen.parquet")
    parser.add_argument("--output-val-seen", default="experiments/bfcl_v4_multi_turn/val_seen.parquet")
    parser.add_argument("--output-val-all", default="experiments/bfcl_v4_multi_turn/val_all.parquet")
    args = parser.parse_args()

    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    entries = load_bfcl_entries(categories)
    all_ids = list(range(len(entries)))
    seen_ids, unseen_ids = load_split(args.seen_task_ids_from, all_ids, args.holdout_size)

    train_rows = build_rows(entries, seen_ids, "seen")
    val_seen_rows = build_rows(entries, seen_ids, "seen")
    val_unseen_rows = build_rows(entries, unseen_ids, "unseen")
    val_all_rows = val_seen_rows + val_unseen_rows

    train_path = Path(args.output_train)
    val_unseen_path = Path(args.output_val_unseen)
    val_seen_path = Path(args.output_val_seen)
    val_all_path = Path(args.output_val_all)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_unseen_rows).to_parquet(val_unseen_path, index=False)
    pd.DataFrame(val_seen_rows).to_parquet(val_seen_path, index=False)
    pd.DataFrame(val_all_rows).to_parquet(val_all_path, index=False)

    with open(train_path.parent / "split.json", "w") as f:
        json.dump({
            "categories": categories,
            "seen_task_ids": seen_ids,
            "unseen_task_ids": unseen_ids,
            "total_tasks": len(entries),
            "train_file": str(train_path),
            "val_unseen_file": str(val_unseen_path),
            "val_seen_file": str(val_seen_path),
            "val_all_file": str(val_all_path),
            "trainer_val_files": [str(val_seen_path), str(val_unseen_path)],
        }, f, indent=2)

    print(f"Train:      {len(train_rows)} -> {train_path}")
    print(f"Val unseen: {len(val_unseen_rows)} -> {val_unseen_path}")
    print(f"Val seen:   {len(val_seen_rows)} -> {val_seen_path}")
    print(f"Val all:    {len(val_all_rows)} -> {val_all_path}  (offline report only; do not combine with seen+unseen val_files)")
    print(f"Seen={len(seen_ids)}, unseen={len(unseen_ids)}")


if __name__ == "__main__":
    main()
