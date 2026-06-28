"""BFCL v4 multi-turn helpers for SFT collection and GRPO dataset building."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from src.envs.bfcl_tools import execute_bfcl_calls, function_call_to_python, parse_tool_arguments


def load_bfcl_entries(categories: list[str]) -> list[dict[str, Any]]:
    try:
        from bfcl_eval.utils import load_dataset_entry
    except ImportError as exc:
        raise ImportError(
            "BFCL support requires the official package. Install from the Gorilla "
            "berkeley-function-call-leaderboard directory or `pip install bfcl-eval`."
        ) from exc

    entries: list[dict[str, Any]] = []
    for category in categories:
        entries.extend(load_dataset_entry(category))
    return sorted(entries, key=lambda item: str(item.get("id", "")))


def normalize_tool_schema(function_doc: dict[str, Any]) -> dict[str, Any]:
    if function_doc.get("type") == "function" and "function" in function_doc:
        return copy.deepcopy(function_doc)
    doc = copy.deepcopy(function_doc)
    if "parameters" not in doc:
        doc["parameters"] = {"type": "object", "properties": {}}
    return {"type": "function", "function": doc}


def first_turn_messages(test_entry: dict[str, Any]) -> list[dict[str, Any]]:
    question = test_entry.get("question") or []
    if not question:
        return [{"role": "user", "content": str(test_entry.get("id", ""))}]
    return copy.deepcopy(question[0])


def turn_to_user_text(turn_messages: list[dict[str, Any]]) -> str:
    parts = []
    for msg in turn_messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            parts.append(str(msg.get("content", "")))
    if not parts:
        parts = [str(msg.get("content", "")) for msg in turn_messages if isinstance(msg, dict)]
    return "\n".join(part for part in parts if part)


def make_bfcl_state(test_entry: dict[str, Any], run_id: str) -> dict[str, Any]:
    question = test_entry.get("question") or []
    return {
        "request_id": run_id,
        "test_entry": test_entry,
        "test_entry_id": str(test_entry.get("id", run_id)),
        "test_category": str(test_entry.get("id", "multi_turn")).rsplit("_", 1)[0],
        "current_turn": 0,
        "num_tool_calls": 0,
        "num_user_turns": 0,
        "done": False,
        "model_name": f"bfcl_collect_{run_id}",
        "model_result": [[] for _ in question],
        "last_score_detail": None,
    }


def score_bfcl_state(state: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker

    test_entry = state["test_entry"]
    test_category = state["test_category"]
    ground_truth = test_entry.get("ground_truth", [])
    result = multi_turn_checker(
        state["model_result"],
        ground_truth,
        test_entry,
        test_category,
        state["model_name"],
    )
    score = 1.0 if result.get("valid", False) else 0.0
    state["last_score_detail"] = result
    return score, result


@dataclass
class BFCLTrajectoryResult:
    task_id: int
    entry_id: str
    success: bool
    reward: float
    num_turns: int
    num_tool_calls: int
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    model_result: list[Any] = field(default_factory=list)
    score_detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    was_contaminated_from_turn: int | None = None


class BFCLWrapper:
    def __init__(self, categories: list[str]):
        self.categories = categories
        self.entries = load_bfcl_entries(categories)

    def get_num_tasks(self) -> int:
        return len(self.entries)

    def get_entry(self, task_idx: int) -> dict[str, Any]:
        return copy.deepcopy(self.entries[int(task_idx)])

    def run_single_task(self, task_idx: int, policy, max_turns: int = 30) -> BFCLTrajectoryResult:
        entry = self.get_entry(task_idx)
        if hasattr(policy, "set_tools"):
            policy.set_tools([normalize_tool_schema(fn) for fn in entry.get("function", [])])
        if hasattr(policy, "was_truncated"):
            policy.was_truncated = False

        state = make_bfcl_state(entry, f"{task_idx}_{id(policy)}")
        messages = first_turn_messages(entry)
        error = None
        contaminated_from = None

        try:
            while not state["done"] and state["current_turn"] < len(entry.get("question", [])):
                assistant_msg = policy(messages)
                messages.append(assistant_msg)
                if getattr(policy, "was_truncated", False) and contaminated_from is None:
                    contaminated_from = state["current_turn"]

                tool_calls = assistant_msg.get("tool_calls") or []
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("function", {}).get("name")
                        arguments = parse_tool_arguments(tc.get("function", {}).get("arguments"))
                        call = function_call_to_python(name, arguments)
                        state["model_result"][state["current_turn"]].append([call])
                        state["num_tool_calls"] += 1
                        try:
                            results = execute_bfcl_calls(state, [call])
                            obs = results[0] if results else ""
                        except Exception as exc:  # noqa: BLE001
                            obs = f"Error during execution: {type(exc).__name__}: {exc}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", f"call_{state['num_tool_calls']}"),
                            "name": name,
                            "content": str(obs),
                        })
                else:
                    state["current_turn"] += 1
                    state["num_user_turns"] += 1
                    if state["current_turn"] >= len(entry.get("question", [])):
                        state["done"] = True
                        break
                    messages.append({
                        "role": "user",
                        "content": turn_to_user_text(entry["question"][state["current_turn"]]),
                    })

                if state["num_user_turns"] + state["num_tool_calls"] >= max_turns:
                    state["done"] = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        reward, detail = score_bfcl_state(state)
        return BFCLTrajectoryResult(
            task_id=task_idx,
            entry_id=str(entry.get("id", task_idx)),
            success=reward >= 1.0,
            reward=reward,
            num_turns=state["num_user_turns"],
            num_tool_calls=state["num_tool_calls"],
            raw_messages=messages,
            model_result=state["model_result"],
            score_detail=detail,
            error=error,
            was_contaminated_from_turn=contaminated_from,
        )
