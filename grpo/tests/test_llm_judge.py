"""Standalone unit tests for llm_judge.py pure logic.

    python3 grpo/tests/test_llm_judge.py
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../grpo
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.training import llm_judge  # noqa: E402


# --------------------------------------------------------------------------- #
# blend_reward  (outcome-anchoring is the anti-hacking core)
# --------------------------------------------------------------------------- #
def test_blend_success_short_circuits():
    # success -> 1.0 even with empty/noisy steps (and must NOT depend on them).
    assert llm_judge.blend_reward(1, [], 0.2) == 1.0
    assert llm_judge.blend_reward(1, [0.0, 0.0], 0.2) == 1.0
    assert llm_judge.blend_reward(2, [0.5], 0.2) == 1.0  # any >=1 counts as success


def test_blend_failure_is_alpha_times_mean():
    assert llm_judge.blend_reward(0, [0.5, 0.7], 0.2) == 0.2 * 0.6


def test_blend_failure_capped_below_success():
    # Even a perfect step score for a failure cannot reach the success reward.
    assert llm_judge.blend_reward(0, [1.0, 1.0], 0.2) == 0.2
    assert llm_judge.blend_reward(0, [1.0], 0.2) < llm_judge.blend_reward(1, [0.0], 0.2)


def test_blend_clamps_step_scores():
    assert llm_judge.blend_reward(0, [1.5, -0.5], 0.2) == 0.2 * 0.5  # clamp to [1.0,0.0]


def test_blend_empty_failure_is_zero():
    assert llm_judge.blend_reward(0, [], 0.2) == 0.0


# --------------------------------------------------------------------------- #
# extract_steps / step_tool_names / select_key_steps
# --------------------------------------------------------------------------- #
def _msg(role, content, tool_calls=None):
    m = {"role": role, "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m


def test_extract_steps_splits_on_assistant():
    messages = [
        _msg("user", "book a flight"),
        _msg("assistant", "let me check", [{"name": "get_user_details"}]),
        _msg("tool", '{"id": 1}'),
        _msg("assistant", "booking now", [{"name": "book_reservation"}]),
        _msg("user", "done"),
    ]
    steps = llm_judge.extract_steps(messages)
    assert len(steps) == 2
    assert steps[0]["assistant"] == "let me check"
    assert steps[0]["observations"] == ['{"id": 1}']
    assert steps[1]["tool_calls"][0]["name"] == "book_reservation"


def test_extract_steps_no_assistant_returns_empty():
    assert llm_judge.extract_steps([_msg("user", "x"), _msg("tool", "y")]) == []


def test_step_tool_names_handles_two_shapes():
    step = {"tool_calls": [{"name": "get_user_details"}, {"function": {"name": "book_reservation"}}]}
    assert llm_judge.step_tool_names(step) == ["get_user_details", "book_reservation"]


def test_select_key_steps_picks_write_and_preceding():
    steps = [
        {"assistant": "read", "tool_calls": [{"name": "get_user_details"}], "observations": []},
        {"assistant": "read2", "tool_calls": [{"name": "search_direct_flight"}], "observations": []},
        {"assistant": "write", "tool_calls": [{"name": "book_reservation"}], "observations": []},
    ]
    key = llm_judge.select_key_steps(steps)
    # write step + the step right before it; first read not included.
    assert steps[2] in key and steps[1] in key and steps[0] not in key


def test_select_key_steps_fallback_when_no_write():
    steps = [{"assistant": "only read", "tool_calls": [{"name": "get_user_details"}], "observations": []}]
    assert llm_judge.select_key_steps(steps) == steps


# --------------------------------------------------------------------------- #
# parse_judge_response / build_step_prompt / cache key
# --------------------------------------------------------------------------- #
def test_parse_clean_json():
    assert llm_judge.parse_judge_response('{"score": 0.7, "reason": "ok"}') == 0.7


def test_parse_noisy_json():
    assert llm_judge.parse_judge_response('prefix {"score": 0.8} trailing') == 0.8


def test_parse_clamps_and_defaults():
    assert llm_judge.parse_judge_response('{"score": 1.5}') == 1.0
    assert llm_judge.parse_judge_response('{"reason": "no score"}') == 0.0
    assert llm_judge.parse_judge_response("total garbage") == 0.0


def test_build_step_prompt_shape():
    step = {"assistant": "do thing", "tool_calls": [{"name": "get_user_details"}], "observations": ["obs"]}
    prompt = llm_judge.build_step_prompt("task-7", step, llm_judge.RUBRIC, 600)
    assert len(prompt) == 2 and prompt[0]["role"] == "system" and prompt[1]["role"] == "user"
    assert "JSON" in prompt[0]["content"]
    assert "task-7" in prompt[1]["content"]


def test_step_cache_key_deterministic_and_sensitive():
    s1 = {"assistant": "abc", "tool_calls": [{"name": "get_user_details"}], "observations": []}
    s2 = {"assistant": "abd", "tool_calls": [{"name": "get_user_details"}], "observations": []}
    assert llm_judge.step_cache_key("t", s1, 600) == llm_judge.step_cache_key("t", s1, 600)
    assert llm_judge.step_cache_key("t", s1, 600) != llm_judge.step_cache_key("t", s2, 600)


# --------------------------------------------------------------------------- #
# score_steps  (ensemble + cache + error tolerance; judge_fn injected)
# --------------------------------------------------------------------------- #
def _const_judge(value):
    calls = []

    def fn(config, messages):
        calls.append(messages)
        return f'{{"score": {value}}}'

    return fn, calls


def test_score_steps_ensemble_mean():
    # temperature > 0 so the ensemble (n_samples=3) actually fires.
    cfg = llm_judge.JudgeConfig(n_samples=3, alpha=0.2, temperature=0.5)
    fn, calls = _const_judge(0.6)
    step = {"assistant": "x", "tool_calls": [{"name": "book_reservation"}], "observations": []}
    scores = llm_judge.score_steps("t", [step], cfg, judge_fn=fn)
    assert scores == [0.6]
    assert len(calls) == 3  # ensemble


def test_effective_n_samples_collapses_at_temp0():
    # temp=0 -> deterministic -> n_samples collapses to 1 (no wasted identical calls).
    assert llm_judge.effective_n_samples(llm_judge.JudgeConfig(n_samples=3, temperature=0.0)) == 1
    assert llm_judge.effective_n_samples(llm_judge.JudgeConfig(n_samples=1, temperature=0.0)) == 1
    # temp>0 -> ensemble honored.
    assert llm_judge.effective_n_samples(llm_judge.JudgeConfig(n_samples=3, temperature=0.5)) == 3


def test_score_steps_temp0_makes_one_call():
    cfg = llm_judge.JudgeConfig(n_samples=3, temperature=0.0)
    fn, calls = _const_judge(0.6)
    step = {"assistant": "x", "tool_calls": [{"name": "book_reservation"}], "observations": []}
    assert llm_judge.score_steps("t", [step], cfg, judge_fn=fn) == [0.6]
    assert len(calls) == 1  # n_samples=3 ignored at temp=0


def test_score_steps_cache_skips_judge():
    cfg = llm_judge.JudgeConfig(n_samples=3)
    fn, calls = _const_judge(0.9)
    step = {"assistant": "x", "tool_calls": [{"name": "book_reservation"}], "observations": []}
    key = llm_judge.step_cache_key("t", step, cfg.max_step_chars)
    cache = {key: 0.42}
    scores = llm_judge.score_steps("t", [step], cfg, judge_fn=fn, cache=cache)
    assert scores == [0.42]
    assert calls == []  # served from cache, judge not called


def test_score_steps_tolerates_judge_failure():
    cfg = llm_judge.JudgeConfig(n_samples=2)

    def fn(config, messages):
        raise OSError("endpoint down")

    step = {"assistant": "x", "tool_calls": [{"name": "book_reservation"}], "observations": []}
    assert llm_judge.score_steps("t", [step], cfg, judge_fn=fn) == [0.0]


# --------------------------------------------------------------------------- #
# densify_trajectory  (near-miss-only: success is never judged)
# --------------------------------------------------------------------------- #
def test_densify_success_never_calls_judge():
    cfg = llm_judge.JudgeConfig(alpha=0.2, n_samples=3)

    def boom(config, messages):
        raise AssertionError("success must not be judged")

    messages = [_msg("assistant", "x", [{"name": "book_reservation"}])]
    assert llm_judge.densify_trajectory(1, messages, "t", cfg, judge_fn=boom) == 1.0


def test_densify_failure_blends_key_steps_only():
    cfg = llm_judge.JudgeConfig(alpha=0.2, n_samples=1, only_key_steps=True)
    fn, calls = _const_judge(0.5)
    messages = [
        _msg("assistant", "read1", [{"name": "get_user_details"}]),
        _msg("tool", "obs1"),
        _msg("assistant", "read2", [{"name": "search_direct_flight"}]),
        _msg("tool", "obs2"),
        _msg("assistant", "write", [{"name": "book_reservation"}]),
    ]
    reward = llm_judge.densify_trajectory(0, messages, "t", cfg, judge_fn=fn)
    # key steps = read2 + write -> 2 judge calls; reward = 0.2 * mean([0.5,0.5]) = 0.1
    assert len(calls) == 2
    assert reward == 0.1


# --------------------------------------------------------------------------- #
# env toggle
# --------------------------------------------------------------------------- #
def test_is_enabled_env_toggle():
    os.environ.pop("LLM_JUDGE_ENABLED", None)
    assert llm_judge.is_enabled() is False
    os.environ["LLM_JUDGE_ENABLED"] = "true"
    assert llm_judge.is_enabled() is True
    os.environ["LLM_JUDGE_ENABLED"] = "false"
    assert llm_judge.is_enabled() is False
    os.environ.pop("LLM_JUDGE_ENABLED", None)



# --------------------------------------------------------------------------- #
# densify_rewards glue without torch/verl dependencies
# --------------------------------------------------------------------------- #
class _FakeVector:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _FakeRow:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(values),)

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, value):
        self.values[idx] = value


class _FakeScores:
    def __init__(self, rows):
        self.rows = [_FakeRow(row) for row in rows]

    def sum(self, dim=-1):
        assert dim == -1
        return _FakeVector([sum(row.values) for row in self.rows])

    def __getitem__(self, idx):
        return self.rows[idx]


class _FakeMaskRow:
    def __init__(self, values):
        self.values = values

    def nonzero(self, as_tuple=False):
        return [idx for idx, value in enumerate(self.values) if value]


class _FakeMask:
    def __init__(self, rows):
        self.rows = [_FakeMaskRow(row) for row in rows]

    def __getitem__(self, idx):
        return self.rows[idx]


class _FakeProto:
    def __init__(self, batch, non_tensor_batch=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}


def test_densify_rewards_reads_llm_judge_messages_and_only_scores_failures():
    rm = _FakeScores([[0.0, 1.0], [0.0, 0.0]])
    mask = _FakeMask([[0, 1], [0, 1]])
    batch = _FakeProto({'rm_scores': rm, 'response_mask': mask}, {'uid': ['ok', 'fail']})
    gen = _FakeProto(
        {'response_mask': mask},
        {
            'llm_judge_messages': [
                [_msg('assistant', 'done', [{'name': 'book_reservation'}])],
                [_msg('assistant', 'almost', [{'name': 'book_reservation'}])],
            ]
        },
    )
    cfg = llm_judge.JudgeConfig(alpha=0.2, n_samples=1)
    old_score_steps = llm_judge.score_steps
    try:
        llm_judge.score_steps = lambda task_id, steps, config, cache=None: [0.7]
        out, metrics = llm_judge.densify_rewards(batch, gen, config=cfg)
    finally:
        llm_judge.score_steps = old_score_steps

    assert out is batch
    assert rm.rows[0].values == [0.0, 1.0]
    assert rm.rows[1].values == [0.0, 0.13999999999999999]
    assert metrics['llm_judge/skipped_successes'] == 1.0
    assert metrics['llm_judge/judged_failures'] == 1.0
    assert metrics['llm_judge/missing_messages'] == 0.0


# --------------------------------------------------------------------------- #
# domain dispatch (BFCL support; tau_bench default must be unchanged)
# --------------------------------------------------------------------------- #
def test_get_rubric_dispatch():
    # tau_bench (default) -> unchanged airline rubric.
    assert llm_judge.get_rubric("tau_bench") is llm_judge.RUBRIC
    assert llm_judge.get_rubric("") is llm_judge.RUBRIC
    assert llm_judge.get_rubric("unknown") is llm_judge.RUBRIC  # unknown falls back to default
    # bfcl -> the dedicated BFCL rubric.
    bfcl = llm_judge.get_rubric("bfcl")
    assert bfcl is llm_judge.BFCL_RUBRIC
    assert {r["id"] for r in bfcl} == {"function_selection", "argument_correctness", "no_hallucination", "progress"}
    assert abs(sum(r["weight"] for r in bfcl) - 1.0) < 1e-9


def test_get_write_tools_dispatch():
    os.environ.pop("LLM_JUDGE_WRITE_TOOLS", None)
    # tau_bench (default) -> airline write set (unchanged).
    assert llm_judge.get_write_tools("tau_bench") == llm_judge.WRITE_TOOLS
    assert "book_reservation" in llm_judge.get_write_tools("tau_bench")
    # bfcl -> empty (no irreversible boundary).
    assert llm_judge.get_write_tools("bfcl") == frozenset()

    os.environ["LLM_JUDGE_WRITE_TOOLS"] = "foo,bar"
    assert llm_judge.get_write_tools("bfcl") == frozenset({"foo", "bar"})
    os.environ["LLM_JUDGE_WRITE_TOOLS"] = "*"
    assert llm_judge.get_write_tools("bfcl") == frozenset({"*"})
    os.environ["LLM_JUDGE_WRITE_TOOLS"] = ""
    assert llm_judge.get_write_tools("tau_bench") == frozenset()
    os.environ.pop("LLM_JUDGE_WRITE_TOOLS", None)


def test_select_key_steps_bfcl_judges_all_or_env_key_tools():
    steps = [
        {"assistant": "call A", "tool_calls": [{"name": "foo"}], "observations": []},
        {"assistant": "call B", "tool_calls": [{"name": "bar"}], "observations": []},
    ]
    # BFCL default -> empty write set -> no writes -> fallback to judging every step.
    os.environ.pop("LLM_JUDGE_WRITE_TOOLS", None)
    assert llm_judge.select_key_steps(steps, llm_judge.get_write_tools("bfcl")) == steps

    # Optional BFCL key-tool focusing: preceding step + configured key tool.
    os.environ["LLM_JUDGE_WRITE_TOOLS"] = "bar"
    assert llm_judge.select_key_steps(steps, llm_judge.get_write_tools("bfcl")) == steps
    os.environ["LLM_JUDGE_WRITE_TOOLS"] = "foo"
    assert llm_judge.select_key_steps(steps, llm_judge.get_write_tools("bfcl")) == [steps[0]]
    os.environ.pop("LLM_JUDGE_WRITE_TOOLS", None)


def test_build_step_prompt_uses_bfcl_wording():
    cfg = llm_judge.JudgeConfig(domain="bfcl")
    step = {"assistant": "x", "tool_calls": [{"name": "foo"}], "observations": ["o"]}
    prompt = llm_judge.build_step_prompt("t1", step, llm_judge.get_rubric(cfg.domain), 600)
    assert "BFCL" in prompt[0]["content"]
    assert "function_selection" in prompt[0]["content"]


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #
def _collect_tests():
    return {n: f for n, f in globals().items() if n.startswith("test_") and callable(f)}


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
