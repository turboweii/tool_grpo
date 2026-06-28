"""LLM-as-judge reward densifier for tau-bench GRPO.

Design (grounded in Step-GRPO / DS-GRPO / "GRPO is secretly a PRM"):
  * judge scores key steps of a trajectory (pointwise rubric);
  * step scores densify the trajectory reward fed to standard GRPO advantage;
  * NO change to the advantage estimator / core_algos -> non-invasive.

Non-invasive vs B-NDSR:
  B-NDSR grouping still uses the tau-bench 0/1 outcome (untouched). This module
  only rewrites rm_scores AFTER B-NDSR has selected groups.

Outcome-anchored (anti reward-hacking):
  success  -> reward 1.0  (never judged)
  failure  -> reward alpha * mean(step_scores) in [0, alpha]
  => a true success always outscores any failure; the judge can only separate
     near-miss failures from garbage ones, never flip an outcome into a win.

Opt-in via LLM_JUDGE_ENABLED. Off -> zero effect.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in TRUE_VALUES


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def is_enabled() -> bool:
    return env_flag("LLM_JUDGE_ENABLED", False)


# tau-bench airline rubric. Weights sum to 1.0. Reuses the same read/write
# semantics as b_ndsr (no coupling -- just aligned vocabulary).
RUBRIC = (
    {"id": "tool_correctness", "weight": 0.25, "desc": "Right tool, valid arguments, no invalid/unknown tool calls."},
    {"id": "info_gathering", "weight": 0.25, "desc": "Read enough (user/reservation/flights) before any irreversible write."},
    {"id": "safe_irreversible", "weight": 0.20, "desc": "Did not execute a wrong/cancel/update/book; writes match the task."},
    {"id": "task_progress", "weight": 0.20, "desc": "Net movement toward task completion."},
    {"id": "no_hallucination", "weight": 0.10, "desc": "Did not fabricate tool results or entity values."},
)

# Write tools = the irreversible decisions where credit lives (mirrors b_ndsr).
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

# BFCL multi-turn rubric: function-calling criteria (no write/irreversible boundary).
BFCL_RUBRIC = (
    {"id": "function_selection", "weight": 0.30, "desc": "Called the function that matches the user's intent and the available toolset."},
    {"id": "argument_correctness", "weight": 0.30, "desc": "Arguments have correct names, types, and values for the chosen function."},
    {"id": "no_hallucination", "weight": 0.20, "desc": "Did not invent functions/arguments outside the toolset or fabricate observations."},
    {"id": "progress", "weight": 0.20, "desc": "The call advances toward resolving the user's request."},
)

_RUBRICS = {"tau_bench": RUBRIC, "bfcl": BFCL_RUBRIC}


def get_rubric(domain: str) -> tuple[dict, ...]:
    """Rubric by domain. tau_bench (default) -> the unchanged airline rubric."""
    return _RUBRICS.get((domain or "tau_bench").strip().lower(), RUBRIC)


def env_tool_set(name: str, default: frozenset[str]) -> frozenset[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.strip() == "":
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def get_write_tools(domain: str) -> frozenset[str]:
    """Write-tool set used for key-step selection.

    Defaults preserve tau-bench behavior. BFCL defaults to no write boundary,
    but LLM_JUDGE_WRITE_TOOLS can override either domain; "*" means every tool
    call is a key step, "" means no write boundary -> judge all steps fallback.
    """
    default = frozenset() if (domain or "tau_bench").strip().lower() == "bfcl" else WRITE_TOOLS
    return env_tool_set("LLM_JUDGE_WRITE_TOOLS", default)


def is_write_tool(tool_name: str, write_tools: frozenset[str]) -> bool:
    return "*" in write_tools or tool_name in write_tools


@dataclass(frozen=True)
class JudgeConfig:
    model: str = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    base_url: str = "http://localhost:8001/v1"
    api_key: str = "dummy"
    alpha: float = 0.2            # failure reward ceiling (success=1.0)
    n_samples: int = 3            # ensemble for consistency
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_seconds: float = 20.0
    max_step_chars: int = 600
    only_key_steps: bool = True   # judge write-steps + preceding read, not every turn
    domain: str = "tau_bench"
    max_workers: int = 16         # concurrent judge calls across trajectories

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            model=os.getenv("LLM_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ"),
            base_url=os.getenv("LLM_JUDGE_BASE_URL", "http://localhost:8001/v1"),
            api_key=os.getenv("LLM_JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "dummy")),
            alpha=env_float("LLM_JUDGE_ALPHA", 0.2),
            n_samples=env_int("LLM_JUDGE_N_SAMPLES", 3),
            max_tokens=env_int("LLM_JUDGE_MAX_TOKENS", 256),
            temperature=env_float("LLM_JUDGE_TEMPERATURE", 0.0),
            top_p=env_float("LLM_JUDGE_TOP_P", 1.0),
            timeout_seconds=env_float("LLM_JUDGE_TIMEOUT_SECONDS", 20.0),
            max_step_chars=env_int("LLM_JUDGE_MAX_STEP_CHARS", 600),
            only_key_steps=env_flag("LLM_JUDGE_ONLY_KEY_STEPS", True),
            domain=os.getenv("LLM_JUDGE_DOMAIN", "tau_bench"),
            max_workers=env_int("LLM_JUDGE_MAX_WORKERS", 16),
        )


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested, no LLM / no verl)
# --------------------------------------------------------------------------- #
def effective_n_samples(config: JudgeConfig) -> int:
    """At temperature 0 the judge is deterministic -> n_samples>1 just repeats
    identical calls. Collapse to 1 unless temperature > 0 asks for an ensemble."""
    return 1 if config.temperature == 0 else max(1, config.n_samples)


def blend_reward(outcome: int | float, step_scores: list[float], alpha: float) -> float:
    """Outcome-anchored dense reward. success->1.0; failure->alpha*mean(steps).

    Guarantees success (>=1) always outscores any failure: failure reward is
    capped at alpha (<1). step_scores are clamped to [0,1].
    """
    if float(outcome) >= 1.0:
        return 1.0
    if not step_scores:
        return 0.0
    clamped = [max(0.0, min(1.0, float(s))) for s in step_scores]
    return alpha * (sum(clamped) / len(clamped))


def extract_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split a message history into steps.

    A step = one assistant turn + the tool/user messages that follow it, up to
    the next assistant turn. Each step records the assistant text and any tool
    names it called.
    """
    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            if current is not None:
                steps.append(current)
            current = {
                "assistant": str(msg.get("content", "") or ""),
                "tool_calls": list(msg.get("tool_calls") or []),
                "observations": [],
            }
        elif current is not None:
            current["observations"].append(str(msg.get("content", "") or ""))
    if current is not None:
        steps.append(current)
    return steps


def step_tool_names(step: dict[str, Any]) -> list[str]:
    names = []
    for call in step.get("tool_calls", []):
        if isinstance(call, dict):
            name = call.get("name") or call.get("function", {}).get("name")
            if name:
                names.append(str(name))
    return names


def select_key_steps(steps: list[dict[str, Any]], write_tools: frozenset[str] = WRITE_TOOLS) -> list[dict[str, Any]]:
    """Key steps = writes + the read/reasoning step immediately before each write."""
    key: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        if any(is_write_tool(name, write_tools) for name in step_tool_names(step)):
            if idx > 0 and steps[idx - 1] not in key:
                key.append(steps[idx - 1])
            key.append(step)
    return key or steps  # fallback: judge everything if no write happened


def step_cache_key(task_id: str, step: dict[str, Any], max_chars: int) -> str:
    raw = f"{task_id}|{step.get('assistant', '')[:max_chars]}|{step.get('tool_calls')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 16] + "...[truncated]"


def build_step_prompt(task_id: str, step: dict[str, Any], rubric: tuple[dict, ...], max_chars: int) -> list[dict[str, str]]:
    criteria = "\n".join(f"- {r['id']} (weight {r['weight']}): {r['desc']}" for r in rubric)
    # Domain wording follows the rubric (which follows config.domain), not the env
    # directly, so wording and criteria never disagree.
    domain = "BFCL multi-turn function-calling" if any(r["id"] == "function_selection" for r in rubric) else "tau-bench airline tool-use"
    system = (
        f"You are a strict process judge for {domain} trajectories. "
        "You are NOT a reward model and must not consider whether the task ultimately succeeded. "
        "Score ONLY the quality of the single step shown, on [0,1]. "
        "Return ONLY JSON: {\"score\": float, \"reason\": string}.\n"
        f"Criteria:\n{criteria}"
    )
    user = json.dumps(
        {
            "task_id": task_id,
            "assistant": _truncate(step.get("assistant", ""), max_chars),
            "tool_calls": step.get("tool_calls", []),
            "observations": [_truncate(o, max_chars) for o in step.get("observations", [])],
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judge_response(text: str) -> float:
    """Extract a [0,1] score from the judge's JSON response. Robust to noise."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return 0.0
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return 0.0
    score = data.get("score", 0.0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Judge client (mirrors jass._call_openai_compatible style)
# --------------------------------------------------------------------------- #
def _call_judge(config: JudgeConfig, messages: list[dict[str, str]]) -> str:
    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        body = response.read().decode("utf-8")
    choices = json.loads(body).get("choices") or []
    if not choices:
        raise ValueError("judge response has no choices")
    return choices[0].get("message", {}).get("content", "") or ""


def score_steps(
    task_id: str,
    steps: list[dict[str, Any]],
    config: JudgeConfig,
    judge_fn: Callable[[JudgeConfig, list[dict[str, str]]], str] | None = None,
    cache: dict[str, float] | None = None,
) -> list[float]:
    """Score each step with n_samples ensemble + hash cache. judge_fn injectable for tests."""
    judge_fn = judge_fn or _call_judge
    cache = {} if cache is None else cache
    scores: list[float] = []
    for step in steps:
        key = step_cache_key(task_id, step, config.max_step_chars)
        if key in cache:
            scores.append(cache[key])
            continue
        prompt = build_step_prompt(task_id, step, get_rubric(config.domain), config.max_step_chars)
        samples = []
        for _ in range(effective_n_samples(config)):
            try:
                samples.append(parse_judge_response(judge_fn(config, prompt)))
            except (OSError, ValueError):
                # OSError covers urllib.error.URLError + TimeoutError (both subclasses).
                samples.append(0.0)
        value = sum(samples) / len(samples) if samples else 0.0
        cache[key] = value
        scores.append(value)
    return scores


# --------------------------------------------------------------------------- #
# Trajectory-level entry (pure, testable) + verl glue (runtime-pending)
# --------------------------------------------------------------------------- #
def densify_trajectory(
    outcome: int | float,
    messages: list[dict[str, Any]],
    task_id: str,
    config: JudgeConfig,
    judge_fn: Callable[[JudgeConfig, list[dict[str, str]]], str] | None = None,
    cache: dict[str, float] | None = None,
) -> float:
    """Return the dense reward for one trajectory. success->1.0 (no judge call)."""
    if float(outcome) >= 1.0:
        return 1.0  # near-miss-only: never judge successes
    steps = extract_steps(messages)
    if config.only_key_steps:
        steps = select_key_steps(steps, get_write_tools(config.domain))
    if not steps:
        return 0.0
    scores = score_steps(task_id, steps, config, judge_fn=judge_fn, cache=cache)
    return blend_reward(outcome, scores, config.alpha)


def densify_rewards(
    batch: Any,
    gen_batch_output: Any,
    config: JudgeConfig | None = None,
) -> tuple[Any, dict[str, float]]:
    """Overwrite rm_scores with dense judge rewards. Call AFTER B-NDSR grouping.

    Returns (batch, metrics). B-NDSR grouping already used the binary outcome
    (untouched); here we only densify the reward that feeds the GRPO advantage.

    Anti-hacking monitor: metrics expose outcome_rate (true pass rate) vs
    mean_failure_process (judge's score on failures). If process climbs while
    outcome_rate flats/drops over training, the policy is gaming the judge.

    Defensive: a per-trajectory judge failure leaves that row's reward unchanged.
    """
    config = config or JudgeConfig.from_env()
    rm = batch.batch["rm_scores"]
    outcomes = rm.sum(dim=-1).detach().cpu().tolist()

    batch_non_tensor = getattr(batch, "non_tensor_batch", {}) or {}
    gen_non_tensor = getattr(gen_batch_output, "non_tensor_batch", {}) or {}
    raw_prompts = gen_non_tensor.get("llm_judge_messages")
    if raw_prompts is None:
        raw_prompts = gen_non_tensor.get("raw_prompt")
    uids = batch_non_tensor.get("uid")
    if uids is None:
        uids = gen_non_tensor.get("uid")

    # Locate the reward token (last response token) robustly.
    response_mask = None
    for source in (gen_batch_output.batch, batch.batch):
        if getattr(source, "keys", lambda: [])() and "response_mask" in source.keys():
            response_mask = source["response_mask"]
            break

    cache: dict[str, float] = {}
    judged = 0
    skipped_success = 0
    fail_process: list[float] = []
    dense_rewards: list[float] = []
    missing_messages = 0

    # Pass 1: extract steps for each failure (no judge calls yet -- cheap).
    to_judge: dict[int, tuple[str, list[dict[str, Any]]]] = {}
    for i, outcome in enumerate(outcomes):
        if float(outcome) >= 1.0:
            skipped_success += 1
            continue
        messages = raw_prompts[i] if (raw_prompts is not None and i < len(raw_prompts)) else None
        if not isinstance(messages, list):
            missing_messages += 1
            continue
        task_id = str(uids[i]) if (uids is not None and i < len(uids)) else str(i)
        try:
            steps = extract_steps(messages)
            if config.only_key_steps:
                steps = select_key_steps(steps, get_write_tools(config.domain))
        except Exception:  # noqa: BLE001
            missing_messages += 1
            continue
        if steps:
            to_judge[i] = (task_id, steps)

    # Pass 2: judge all failure trajectories concurrently. score_steps stays the unit
    # (so tests can monkeypatch it); the shared cache dedupes identical steps across
    # trajectories. dict writes are GIL-safe; a rare duplicate call is harmless.
    scores_by_traj: dict[int, list[float]] = {}
    if to_judge:
        from concurrent.futures import ThreadPoolExecutor

        workers = max(1, min(config.max_workers, len(to_judge)))

        def _judge_one(idx: int) -> tuple[int, list[float]]:
            tid, steps = to_judge[idx]
            return idx, score_steps(tid, steps, config, cache=cache)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for idx, scores in ex.map(_judge_one, to_judge):
                scores_by_traj[idx] = scores

    # Pass 3: blend + write rm + metrics, in original order.
    for i, outcome in enumerate(outcomes):
        if float(outcome) >= 1.0:
            dense_rewards.append(1.0)
            continue
        dense = float(outcome)
        scores = scores_by_traj.get(i)
        if scores is not None:
            try:
                dense = blend_reward(outcome, scores, config.alpha)
                if scores:
                    fail_process.append(sum(scores) / len(scores))
                judged += 1
            except Exception:  # noqa: BLE001 -- never let one trajectory corrupt the batch
                dense = float(outcome)
        dense_rewards.append(dense)
        row = rm[i]
        if response_mask is not None:
            pos = response_mask[i].nonzero(as_tuple=False)
            idx = int(pos[-1]) if len(pos) > 0 else int(row.shape[0]) - 1
        else:
            nonzero = (row != 0).nonzero(as_tuple=False)
            idx = int(nonzero[-1]) if len(nonzero) > 0 else int(row.shape[0]) - 1
        row[idx] = float(dense)

    n = max(1, len(outcomes))
    metrics = {
        "llm_judge/enabled": 1.0,
        "llm_judge/trajectories": float(len(outcomes)),
        "llm_judge/judged_failures": float(judged),
        "llm_judge/skipped_successes": float(skipped_success),
        "llm_judge/missing_messages": float(missing_messages),
        "llm_judge/outcome_rate": float(sum(1 for o in outcomes if float(o) >= 1.0) / n),
        "llm_judge/mean_dense_reward": float(sum(dense_rewards) / max(1, len(dense_rewards))),
        "llm_judge/mean_failure_process": (
            float(sum(fail_process) / len(fail_process)) if fail_process else 0.0
        ),
    }
    return batch, metrics
