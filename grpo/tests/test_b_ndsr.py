"""Standalone unit tests for b_ndsr.py pure helpers.

Runnable without GPU / vLLM / tau-bench / pytest:

    python3 grpo/tests/test_b_ndsr.py

Also pytest-compatible (`pytest grpo/tests/test_b_ndsr.py`) if pytest is installed.
"""
from __future__ import annotations

import os
import sys

# Make `from src.training import b_ndsr` resolve when run as a bare script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../grpo
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.training import b_ndsr  # noqa: E402


# --------------------------------------------------------------------------- #
# binary_label / has_variance / all_success / all_failure
# --------------------------------------------------------------------------- #
def test_binary_label():
    assert b_ndsr.binary_label(0.0) == 0
    assert b_ndsr.binary_label(0.49) == 0
    assert b_ndsr.binary_label(0.5) == 1
    assert b_ndsr.binary_label(0.7) == 1
    assert b_ndsr.binary_label(1.0) == 1


def test_has_variance():
    assert b_ndsr.has_variance([0, 0, 0]) is False
    assert b_ndsr.has_variance([1, 1, 1]) is False
    assert b_ndsr.has_variance([0, 1]) is True
    assert b_ndsr.has_variance([1, 0, 1]) is True
    # Scores below/above the 0.5 label threshold still split into {0,1}.
    assert b_ndsr.has_variance([0.2, 0.6]) is True
    assert b_ndsr.has_variance([0.4, 0.45]) is False
    # Edge: empty / single -> no variance.
    assert b_ndsr.has_variance([]) is False
    assert b_ndsr.has_variance([0.3]) is False


def test_all_success_all_failure():
    assert b_ndsr.all_success([1, 1]) is True
    assert b_ndsr.all_success([1, 0]) is False
    assert b_ndsr.all_success([0, 0]) is False
    assert b_ndsr.all_success([]) is False

    assert b_ndsr.all_failure([0, 0, 0]) is True
    assert b_ndsr.all_failure([0, 1]) is False
    assert b_ndsr.all_failure([]) is False


# --------------------------------------------------------------------------- #
# trajectory_is_replayable  (the keep/discard filter)
# --------------------------------------------------------------------------- #
def _clean_flags(**overrides):
    base = {
        "json_error": False,
        "invalid_tool_name": False,
        "tool_execution_error": False,
        "early_final_answer": False,
        "repeated_tool_loop": False,
        "write_executed": False,
        "read_count": 1,
        "distinct_read_tools": ["get_user_details"],
        "valid_tool_calls": 1,
        "has_key_entity": False,
    }
    base.update(overrides)
    return base


def test_replayable_clean_prefix():
    assert b_ndsr.trajectory_is_replayable(_clean_flags()) is True


def test_replayable_rejects_each_hard_invalid():
    for bad in (
        "json_error",
        "invalid_tool_name",
        "tool_execution_error",
        "early_final_answer",
        "repeated_tool_loop",
    ):
        assert b_ndsr.trajectory_is_replayable(_clean_flags(**{bad: True})) is False, bad

    # No reads at all -> not replayable.
    assert b_ndsr.trajectory_is_replayable(_clean_flags(read_count=0)) is False


def test_replayable_write_executed_requires_before_write_checkpoint():
    flags = _clean_flags(write_executed=True)
    # write already executed, no checkpoint -> cannot salvage.
    assert b_ndsr.trajectory_is_replayable(flags, checkpoint=None) is False
    # checkpoint AFTER a read while a write already happened -> state likely dirty.
    after_read = {"kind": "after_read_tool"}
    assert b_ndsr.trajectory_is_replayable(flags, checkpoint=after_read) is False
    # checkpoint right BEFORE the write -> salvageable.
    before_write = {"kind": "before_write_tool"}
    assert b_ndsr.trajectory_is_replayable(flags, checkpoint=before_write) is True


# --------------------------------------------------------------------------- #
# checkpoint_score  (heuristic ranking)
# --------------------------------------------------------------------------- #
def test_checkpoint_score_before_write_high():
    flags = {
        "read_count": 2,
        "distinct_read_tools": ["get_user_details", "search_direct_flight"],
        "valid_tool_calls": 3,
        "has_key_entity": True,
    }
    ckpt = {
        "kind": "before_write_tool",
        "turn_idx": 3,
        "tool_name": "update_reservation_flights",
        "features": {},
    }
    # 2 + 2 + 1.5 + 1.5 + 1 + 4 = 12.0
    assert b_ndsr.checkpoint_score(ckpt, flags) == 12.0


def test_checkpoint_score_after_read_with_tool_error():
    flags = {
        "read_count": 1,
        "distinct_read_tools": ["get_user_details"],
        "valid_tool_calls": 1,
        "has_key_entity": False,
    }
    ckpt = {
        "kind": "after_read_tool",
        "turn_idx": 1,
        "tool_name": "get_user_details",
        "features": {"tool_error": True},
    }
    # 2 + 1 + 0.5 + 0 + 0 + 2 + 1 - 3 = 3.5
    assert b_ndsr.checkpoint_score(ckpt, flags) == 3.5


def test_checkpoint_score_penalises_list_airports():
    flags = {"read_count": 1, "distinct_read_tools": ["list_all_airports"], "valid_tool_calls": 1}
    ckpt = {"kind": "after_read_tool", "turn_idx": 1, "tool_name": "list_all_airports", "features": {}}
    # 2 + 1 + 0.5 + 2 - 0.5 = 5.0
    assert b_ndsr.checkpoint_score(ckpt, flags) == 5.0


# --------------------------------------------------------------------------- #
# select_best_checkpoint / select_best_failed_prefix
# --------------------------------------------------------------------------- #
def test_select_best_checkpoint_none_when_empty():
    assert b_ndsr.select_best_checkpoint(None) is None
    assert b_ndsr.select_best_checkpoint({}) is None
    assert b_ndsr.select_best_checkpoint({"flags": _clean_flags(), "checkpoints": []}) is None


def test_select_best_checkpoint_prefers_before_write_over_after_read():
    # An after_read checkpoint with a higher raw score than the before_write,
    # but before_write must still win because it is strictly preferred.
    flags = _clean_flags(read_count=1, distinct_read_tools=["get_user_details"], valid_tool_calls=1)
    after_read = {"kind": "after_read_tool", "turn_idx": 5, "tool_name": "get_user_details", "features": {}}
    before_write = {"kind": "before_write_tool", "turn_idx": 1, "tool_name": "book_reservation", "features": {}}
    trace = {"flags": flags, "checkpoints": [after_read, before_write]}
    chosen = b_ndsr.select_best_checkpoint(trace)
    assert chosen is not None
    assert chosen["kind"] == "before_write_tool"


def test_select_best_checkpoint_picks_max_score_within_kind():
    flags = _clean_flags(read_count=1, distinct_read_tools=["get_user_details"], valid_tool_calls=1)
    low = {"kind": "after_read_tool", "turn_idx": 1, "tool_name": "list_all_airports", "features": {}}
    high = {"kind": "after_read_tool", "turn_idx": 1, "tool_name": "get_user_details", "features": {}}
    trace = {"flags": flags, "checkpoints": [low, high]}
    chosen = b_ndsr.select_best_checkpoint(trace)
    assert chosen is not None
    assert chosen["tool_name"] == "get_user_details"  # higher score


def test_select_best_checkpoint_filters_non_replayable():
    flags = _clean_flags(json_error=True)  # whole trace not replayable
    ckpt = {"kind": "before_write_tool", "turn_idx": 2, "tool_name": "book_reservation", "features": {}}
    assert b_ndsr.select_best_checkpoint({"flags": flags, "checkpoints": [ckpt]}) is None


def test_select_best_failed_prefix_picks_best_trace():
    flags_strong = _clean_flags(read_count=2, distinct_read_tools=["get_user_details", "search_direct_flight"], valid_tool_calls=2)
    flags_weak = _clean_flags(read_count=1, distinct_read_tools=["list_all_airports"], valid_tool_calls=1)
    weak_ckpt = {"kind": "after_read_tool", "turn_idx": 1, "tool_name": "list_all_airports", "features": {}}
    strong_ckpt = {"kind": "before_write_tool", "turn_idx": 3, "tool_name": "update_reservation_flights", "features": {}}
    traces = [
        {"flags": flags_weak, "checkpoints": [weak_ckpt]},
        None,
        {"flags": flags_strong, "checkpoints": [strong_ckpt]},
    ]
    result = b_ndsr.select_best_failed_prefix(traces)
    assert result is not None
    idx, ckpt = result
    assert idx == 2
    assert ckpt["kind"] == "before_write_tool"


def test_select_best_failed_prefix_none_when_no_checkpoints():
    assert b_ndsr.select_best_failed_prefix([]) is None
    assert b_ndsr.select_best_failed_prefix([None, {}]) is None


# --------------------------------------------------------------------------- #
# make_checkpoint / make_replay_action  (structure + deepcopy)
# --------------------------------------------------------------------------- #
def test_make_replay_action_shape():
    action = b_ndsr.make_replay_action("tool", tool_name="get_user_details", parameters={"id": 1})
    assert action["kind"] == "tool"
    assert action["tool_name"] == "get_user_details"
    assert action["parameters"] == {"id": 1}


def test_make_checkpoint_deepcopies_messages():
    original = [{"role": "user", "content": "hi"}]
    ckpt = b_ndsr.make_checkpoint(
        kind="before_write_tool",
        messages=original,
        replay_actions=[],
        turn_idx=2,
        tool_name="book_reservation",
    )
    # Mutating the source after the fact must not corrupt the checkpoint.
    original.append({"role": "assistant", "content": "mutated"})
    assert ckpt["messages"] == [{"role": "user", "content": "hi"}]
    assert ckpt["kind"] == "before_write_tool"
    assert ckpt["turn_idx"] == 2


# --------------------------------------------------------------------------- #
# config from env
# --------------------------------------------------------------------------- #
def test_bndsr_config_defaults_and_env_override(monkeypatch=None):
    # Defaults when env absent.
    for key in (
        "B_NDSR_ROOT_MIN_SAMPLES",
        "B_NDSR_ROOT_MAX_SAMPLES",
        "B_NDSR_ROOT_INCREMENT",
        "B_NDSR_TOTAL_BUDGET_PER_TASK",
        "B_NDSR_SUFFIX_MIN_SAMPLES",
    ):
        os.environ.pop(key, None)
    cfg = b_ndsr.BNDSRConfig.from_env()
    assert (cfg.root_min_samples, cfg.root_max_samples, cfg.root_increment) == (4, 8, 2)
    assert cfg.total_budget_per_task == 12 and cfg.suffix_min_samples == 4

    os.environ["B_NDSR_ROOT_MAX_SAMPLES"] = "6"
    os.environ["B_NDSR_TOTAL_BUDGET_PER_TASK"] = "10"
    cfg2 = b_ndsr.BNDSRConfig.from_env()
    assert cfg2.root_max_samples == 6 and cfg2.total_budget_per_task == 10
    os.environ.pop("B_NDSR_ROOT_MAX_SAMPLES", None)
    os.environ.pop("B_NDSR_TOTAL_BUDGET_PER_TASK", None)


def test_is_enabled_env_flag():
    os.environ.pop("B_NDSR_ENABLED", None)
    assert b_ndsr.is_enabled() is False
    os.environ["B_NDSR_ENABLED"] = "true"
    assert b_ndsr.is_enabled() is True
    os.environ["B_NDSR_ENABLED"] = "0"
    assert b_ndsr.is_enabled() is False
    os.environ.pop("B_NDSR_ENABLED", None)



def test_env_tool_set_empty_means_no_tools():
    os.environ["B_NDSR_WRITE_TOOLS"] = ""
    assert b_ndsr.env_tool_set("B_NDSR_WRITE_TOOLS", b_ndsr.WRITE_TOOLS) == frozenset()
    assert b_ndsr.is_write_tool("book_reservation") is False
    os.environ["B_NDSR_WRITE_TOOLS"] = "*"
    assert b_ndsr.is_write_tool("anything") is True
    os.environ.pop("B_NDSR_WRITE_TOOLS", None)

# --------------------------------------------------------------------------- #
# Standalone runner (no pytest required)
# --------------------------------------------------------------------------- #
def _collect_tests():
    return {name: fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)}


if __name__ == "__main__":
    tests = _collect_tests()
    failures = 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
