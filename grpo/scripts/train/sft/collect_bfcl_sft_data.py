"""Collect SFT trajectories for BFCL v4 multi-turn with best-of-N sampling."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.bfcl_wrapper import BFCLWrapper  # noqa: E402
from src.models.vllm_policy import VLLMPolicy  # noqa: E402

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")


def make_policy(cfg: dict, temperature: float) -> VLLMPolicy:
    return VLLMPolicy(
        model_name=cfg["policy"]["model_name"],
        base_url=cfg["policy"]["base_url"],
        api_key=cfg["policy"].get("api_key", "EMPTY"),
        temperature=temperature,
        top_p=cfg["policy"].get("top_p", 0.9),
        max_tokens=cfg["policy"].get("max_tokens", 512),
    )


def collect_one_task(task_idx: int, wrapper: BFCLWrapper, cfg: dict, output_dir: Path) -> dict:
    out_file = output_dir / f"task_{task_idx:04d}.jsonl"
    meta_file = output_dir / f"task_{task_idx:04d}.meta.json"
    contaminated_file = output_dir / f"task_{task_idx:04d}_contaminated.jsonl"
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)

    best_of_n = cfg["collect"]["best_of_n"]
    temps = cfg["collect"]["temperatures"]
    max_turns = cfg["collect"].get("max_turns", 30)
    assert len(temps) == best_of_n

    successes = []
    contaminated = []
    attempts = []
    for sample_idx, temp in enumerate(temps):
        policy = make_policy(cfg, temp)
        try:
            traj = wrapper.run_single_task(task_idx, policy, max_turns=max_turns)
        except Exception as exc:  # noqa: BLE001
            attempts.append({
                "sample_idx": sample_idx,
                "temperature": temp,
                "success": False,
                "reward": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        is_contaminated = traj.was_contaminated_from_turn is not None
        attempts.append({
            "sample_idx": sample_idx,
            "temperature": temp,
            "entry_id": traj.entry_id,
            "success": traj.success,
            "reward": traj.reward,
            "num_turns": traj.num_turns,
            "num_tool_calls": traj.num_tool_calls,
            "error": traj.error,
            "was_contaminated": is_contaminated,
            "bfcl_error_type": traj.score_detail.get("error_type", ""),
        })
        if is_contaminated:
            contaminated.append((traj, sample_idx, temp))
        elif traj.success:
            successes.append((traj, sample_idx, temp))

    with open(out_file, "w") as f:
        for traj, sample_idx, temp in successes:
            f.write(json.dumps({
                "task_id": traj.task_id,
                "entry_id": traj.entry_id,
                "sample_idx": sample_idx,
                "temperature": temp,
                "success": traj.success,
                "reward": traj.reward,
                "num_turns": traj.num_turns,
                "num_tool_calls": traj.num_tool_calls,
                "messages": traj.raw_messages,
            }, ensure_ascii=False) + "\n")

    with open(contaminated_file, "w") as f:
        for traj, sample_idx, temp in contaminated:
            f.write(json.dumps({
                "task_id": traj.task_id,
                "entry_id": traj.entry_id,
                "sample_idx": sample_idx,
                "temperature": temp,
                "success": traj.success,
                "reward": traj.reward,
                "was_contaminated_from_turn": traj.was_contaminated_from_turn,
                "messages": traj.raw_messages,
            }, ensure_ascii=False) + "\n")

    meta = {
        "task_id": task_idx,
        "entry_id": attempts[0].get("entry_id") if attempts else None,
        "best_of_n": best_of_n,
        "num_successes": len(successes),
        "num_contaminated": len(contaminated),
        "any_success": bool(successes),
        "attempts": attempts,
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def parse_task_range(raw: str | None, total: int, tiny: bool) -> list[int]:
    if raw:
        if ":" in raw:
            lo, hi = raw.split(":")
            return list(range(int(lo), int(hi)))
        return [int(x) for x in raw.split(",") if x.strip()]
    return list(range(min(2, total))) if tiny else list(range(total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--task-range", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.tiny:
        cfg["collect"]["best_of_n"] = 2
        cfg["collect"]["temperatures"] = [0.0, 0.8]
        cfg["output"]["dir"] += "_tiny"

    wrapper = BFCLWrapper(cfg["bfcl"].get("categories", ["multi_turn_base"]))
    task_ids = parse_task_range(args.task_range, wrapper.get_num_tasks(), args.tiny)
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "collect_config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    print("=== BFCL SFT collection ===")
    print(f"categories: {cfg['bfcl'].get('categories')}")
    print(f"tasks:      {len(task_ids)} / {wrapper.get_num_tasks()}")
    print(f"best_of_n:  {cfg['collect']['best_of_n']}")
    print(f"output:     {output_dir}")

    t0 = time.time()
    metas = []
    with ThreadPoolExecutor(max_workers=cfg["collect"].get("num_workers", 4)) as executor:
        futures = {executor.submit(collect_one_task, idx, wrapper, cfg, output_dir): idx for idx in task_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Collecting BFCL"):
            metas.append(fut.result())

    metas.sort(key=lambda item: item["task_id"])
    holdout_size = int(cfg["collect"].get("holdout_size", 0))
    if holdout_size <= 0 or holdout_size >= len(task_ids):
        raise RuntimeError(
            "BFCL SFT collection requires 0 < holdout_size < num_tasks so unseen eval is non-empty."
        )
    holdout_ids = task_ids[-holdout_size:]
    seen_ids = [idx for idx in task_ids if idx not in set(holdout_ids)]

    train_file = output_dir / "train.jsonl"
    holdout_file = output_dir / "holdout_train.jsonl"
    n_train = 0
    n_holdout = 0
    with open(train_file, "w") as train_f, open(holdout_file, "w") as hold_f:
        for idx in task_ids:
            task_file = output_dir / f"task_{idx:04d}.jsonl"
            if not task_file.exists():
                continue
            target = train_f if idx in seen_ids else hold_f
            with open(task_file) as f:
                for line in f:
                    target.write(line)
                    if idx in seen_ids:
                        n_train += 1
                    else:
                        n_holdout += 1

    summary = {
        "categories": cfg["bfcl"].get("categories"),
        "num_tasks_attempted": len(metas),
        "num_tasks_with_success": sum(1 for item in metas if item["any_success"]),
        "total_success_trajectories": sum(item["num_successes"] for item in metas),
        "total_contaminated_trajectories": sum(item["num_contaminated"] for item in metas),
        "seen_task_ids": seen_ids,
        "unseen_task_ids": holdout_ids,
        "n_train_trajectories": n_train,
        "n_holdout_trajectories": n_holdout,
        "elapsed_seconds": time.time() - t0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / "split.json", "w") as f:
        json.dump({
            "seen_task_ids": seen_ids,
            "unseen_task_ids": holdout_ids,
            "total_tasks": len(task_ids),
            "leakage_policy": "SFT/GRPO train use seen only; unbiased validation uses unseen only.",
        }, f, indent=2)
    print(f"SFT train -> {train_file} ({n_train})")
    print(f"Holdout   -> {holdout_file} ({n_holdout})")


if __name__ == "__main__":
    main()
