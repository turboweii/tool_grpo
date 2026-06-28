"""Budgeted NDSR helpers for tau-bench GRPO.

The module is intentionally opt-in.  Ordinary GRPO should not observe any of
this unless ``B_NDSR_ENABLED`` is set to a truthy value in the environment.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y", "on"}

READ_TOOLS = frozenset(
    {
        "list_all_airports",
        "search_direct_flight",
        "search_onestop_flight",
        "get_user_details",
        "get_reservation_details",
        "calculate",
    }
)

WRITE_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "update_reservation_flights",
        "send_certificate",
    }
)




def env_tool_set(name: str, default: frozenset[str]) -> frozenset[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.strip() == '':
        return frozenset()
    values = {part.strip() for part in raw.split(',') if part.strip()}
    return frozenset(values)


def is_read_tool(tool_name: str) -> bool:
    tools = env_tool_set('B_NDSR_READ_TOOLS', READ_TOOLS)
    return '*' in tools or tool_name in tools


def is_write_tool(tool_name: str) -> bool:
    tools = env_tool_set('B_NDSR_WRITE_TOOLS', WRITE_TOOLS)
    return '*' in tools or tool_name in tools

def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


def is_enabled() -> bool:
    return env_flag("B_NDSR_ENABLED", False)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    values = tuple(float(part.strip()) for part in raw.split(',') if part.strip())
    if not values:
        return default
    return values


@dataclass(frozen=True)
class BNDSRConfig:
    root_min_samples: int = 4
    root_max_samples: int = 8
    root_increment: int = 2
    total_budget_per_task: int = 12
    suffix_min_samples: int = 4
    root_temperatures: tuple[float, ...] = (0.45, 0.65, 0.80, 0.95, 0.55, 1.00, 0.70, 1.05)
    root_top_ps: tuple[float, ...] = (0.85, 0.90, 0.92, 0.95, 0.88, 0.96, 0.90, 0.97)

    @classmethod
    def from_env(cls) -> "BNDSRConfig":
        return cls(
            root_min_samples=env_int("B_NDSR_ROOT_MIN_SAMPLES", 4),
            root_max_samples=env_int("B_NDSR_ROOT_MAX_SAMPLES", 8),
            root_increment=env_int("B_NDSR_ROOT_INCREMENT", 2),
            total_budget_per_task=env_int("B_NDSR_TOTAL_BUDGET_PER_TASK", 12),
            suffix_min_samples=env_int("B_NDSR_SUFFIX_MIN_SAMPLES", 4),
            root_temperatures=env_float_tuple(
                'B_NDSR_ROOT_TEMPERATURES',
                (0.45, 0.65, 0.80, 0.95, 0.55, 1.00, 0.70, 1.05),
            ),
            root_top_ps=env_float_tuple(
                'B_NDSR_ROOT_TOP_PS',
                (0.85, 0.90, 0.92, 0.95, 0.88, 0.96, 0.90, 0.97),
            ),
        )

    def root_sampling_params(self, sample_idx: int) -> tuple[float, float]:
        temperature = self.root_temperatures[sample_idx % len(self.root_temperatures)]
        top_p = self.root_top_ps[sample_idx % len(self.root_top_ps)]
        return temperature, top_p


def binary_label(score: float) -> int:
    return 1 if float(score) >= 0.5 else 0


def has_variance(scores: list[float]) -> bool:
    labels = {binary_label(score) for score in scores}
    return len(labels) > 1


def all_success(scores: list[float]) -> bool:
    return bool(scores) and all(binary_label(score) == 1 for score in scores)


def all_failure(scores: list[float]) -> bool:
    return bool(scores) and all(binary_label(score) == 0 for score in scores)


def make_replay_action(kind: str, **kwargs: Any) -> dict[str, Any]:
    action = {"kind": kind}
    action.update(kwargs)
    return action


def make_checkpoint(
    *,
    kind: str,
    messages: list[dict[str, Any]],
    replay_actions: list[dict[str, Any]],
    turn_idx: int,
    tool_name: str | None = None,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "messages": copy.deepcopy(messages),
        "replay_actions": copy.deepcopy(replay_actions),
        "turn_idx": int(turn_idx),
        "tool_name": tool_name,
        "features": dict(features or {}),
    }


def trajectory_is_replayable(flags: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> bool:
    if flags.get('json_error', False):
        return False
    if flags.get('invalid_tool_name', False):
        return False
    if flags.get('tool_execution_error', False):
        return False
    if flags.get('early_final_answer', False):
        return False
    if flags.get('repeated_tool_loop', False):
        return False
    if int(flags.get('read_count', 0)) <= 0:
        return False
    if flags.get('write_executed', False):
        return bool(checkpoint and checkpoint.get('kind') == 'before_write_tool')
    return True


def checkpoint_score(checkpoint: dict[str, Any], flags: dict[str, Any]) -> float:
    features = checkpoint.get("features", {})
    kind = checkpoint.get("kind", "")

    score = 0.0
    score += 2.0 if int(flags.get("read_count", 0)) > 0 else 0.0
    score += 1.0 * len(set(flags.get("distinct_read_tools", []) or []))
    score += 0.5 * int(flags.get("valid_tool_calls", 0))
    score += 1.5 if flags.get("has_key_entity", False) else 0.0
    score += 1.0 if int(checkpoint.get("turn_idx", 0)) >= 2 else 0.0

    if kind == "before_write_tool":
        score += 4.0
    elif kind == "after_read_tool":
        score += 2.0

    tool_name = checkpoint.get("tool_name")
    if tool_name in {"get_reservation_details", "get_user_details"}:
        score += 1.0
    elif tool_name in {"search_direct_flight", "search_onestop_flight"}:
        score += 0.75
    elif tool_name == "list_all_airports":
        score -= 0.5

    if features.get("tool_error", False):
        score -= 3.0
    return score


def select_best_checkpoint(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trace:
        return None
    flags = trace.get('flags', {})

    checkpoints = trace.get('checkpoints', []) or []
    if not checkpoints:
        return None

    before_write = [
        ckpt
        for ckpt in checkpoints
        if ckpt.get('kind') == 'before_write_tool' and trajectory_is_replayable(flags, ckpt)
    ]
    after_read = [
        ckpt
        for ckpt in checkpoints
        if ckpt.get('kind') == 'after_read_tool' and trajectory_is_replayable(flags, ckpt)
    ]
    candidates = before_write or after_read
    if not candidates:
        return None

    return max(candidates, key=lambda ckpt: checkpoint_score(ckpt, flags))


def select_best_failed_prefix(traces: list[dict[str, Any] | None]) -> tuple[int, dict[str, Any]] | None:
    best: tuple[float, int, dict[str, Any]] | None = None
    for idx, trace in enumerate(traces):
        ckpt = select_best_checkpoint(trace)
        if ckpt is None:
            continue
        score = checkpoint_score(ckpt, trace.get("flags", {}))
        if best is None or score > best[0]:
            best = (score, idx, ckpt)
    if best is None:
        return None
    _, idx, ckpt = best
    return idx, ckpt
