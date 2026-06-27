"""Judge-Assisted State Selection for B-NDSR.

JASS never produces reward labels.  It only ranks heuristic checkpoint
candidates and suggests a suffix sampling mode.  Training still depends solely
on tau-bench 0/1 outcomes.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.training import b_ndsr


SAMPLING_MODES = {"normal", "explore", "conservative"}


@dataclass(frozen=True)
class JASSConfig:
    model: str = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    base_url: str = "http://localhost:8001/v1"
    api_key: str = "dummy"
    max_candidates: int = 3
    timeout_seconds: float = 20.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 384
    max_message_chars: int = 800
    max_recent_messages: int = 8

    @classmethod
    def from_env(cls) -> "JASSConfig":
        return cls(
            model=os.getenv("JASS_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ"),
            base_url=os.getenv("JASS_JUDGE_BASE_URL", "http://localhost:8001/v1"),
            api_key=os.getenv("JASS_JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "dummy")),
            max_candidates=b_ndsr.env_int("JASS_MAX_CANDIDATES", 3),
            timeout_seconds=float(os.getenv("JASS_TIMEOUT_SECONDS", "20")),
            temperature=float(os.getenv("JASS_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("JASS_TOP_P", "1.0")),
            max_tokens=b_ndsr.env_int("JASS_MAX_TOKENS", 384),
            max_message_chars=b_ndsr.env_int("JASS_MAX_MESSAGE_CHARS", 800),
            max_recent_messages=b_ndsr.env_int("JASS_MAX_RECENT_MESSAGES", 8),
        )


def is_enabled() -> bool:
    return b_ndsr.env_flag("JASS_ENABLED", False)


def _truncate(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "...[truncated]"


def _recent_messages(messages: list[dict[str, Any]], config: JASSConfig) -> list[dict[str, str]]:
    recent = messages[-config.max_recent_messages :]
    compact = []
    for message in recent:
        if not isinstance(message, dict):
            continue
        compact.append(
            {
                "role": str(message.get("role", "")),
                "content": _truncate(message.get("content", ""), config.max_message_chars),
            }
        )
    return compact


def build_candidates(
    traces: list[dict[str, Any] | None],
    config: JASSConfig,
) -> list[dict[str, Any]]:
    candidates = []
    for trace_idx, trace in enumerate(traces):
        if not trace:
            continue
        checkpoint = b_ndsr.select_best_checkpoint(trace)
        if checkpoint is None:
            continue
        flags = trace.get("flags", {})
        score = b_ndsr.checkpoint_score(checkpoint, flags)
        candidates.append(
            {
                "candidate_id": len(candidates),
                "trace_idx": trace_idx,
                "checkpoint": checkpoint,
                "heuristic_score": score,
                "judge_payload": {
                    "candidate_id": len(candidates),
                    "checkpoint_kind": checkpoint.get("kind"),
                    "tool_name": checkpoint.get("tool_name"),
                    "turn_idx": checkpoint.get("turn_idx"),
                    "heuristic_score": round(score, 3),
                    "flags": flags,
                    "features": checkpoint.get("features", {}),
                    "recent_messages": _recent_messages(checkpoint.get("messages", []), config),
                },
            }
        )

    candidates.sort(key=lambda item: item["heuristic_score"], reverse=True)
    candidates = candidates[: config.max_candidates]
    for new_id, candidate in enumerate(candidates):
        candidate["candidate_id"] = new_id
        candidate["judge_payload"]["candidate_id"] = new_id
    return candidates


def heuristic_selection(
    traces: list[dict[str, Any] | None],
    config: JASSConfig | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    config = config or JASSConfig.from_env()
    candidates = build_candidates(traces, config)
    if not candidates:
        return None
    best = candidates[0]
    decision = {
        "used_judge": False,
        "fallback": True,
        "selected_candidate": 0,
        "sampling_mode": "normal",
        "confidence": 0.0,
        "reason": "heuristic fallback",
        "candidate_count": len(candidates),
        "model": None,
    }
    return best["trace_idx"], best["checkpoint"], decision


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("judge response did not contain a JSON object")
    return json.loads(match.group(0))


def _call_openai_compatible(
    *,
    config: JASSConfig,
    messages: list[dict[str, str]],
) -> str:
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
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        body = response.read().decode("utf-8")

    data = json.loads(body)
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("judge response has no choices")
    return choices[0].get("message", {}).get("content", "") or ""


def _judge_messages(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    candidate_payloads = [candidate["judge_payload"] for candidate in candidates]
    system = (
        "You are JASS, a checkpoint selector for tau-bench airline rollouts. "
        "You are not a reward model and must not assign success labels. "
        "Choose the checkpoint most worth suffix replay under limited rollout budget. "
        "Prefer clean prefixes with useful read/search information, no format/tool errors, "
        "and states right before a risky write when the prefix looks well-grounded. "
        "Return only JSON with keys: selected_candidate, sampling_mode, confidence, reason. "
        "sampling_mode must be one of normal, explore, conservative."
    )
    user = json.dumps(
        {
            "task": "Select one candidate checkpoint for suffix replay. The final training signal will still be tau-bench 0/1 only.",
            "candidates": candidate_payloads,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def select_checkpoint(
    traces: list[dict[str, Any] | None],
    config: JASSConfig | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    config = config or JASSConfig.from_env()
    candidates = build_candidates(traces, config)
    if not candidates:
        return None

    if not is_enabled():
        return heuristic_selection(traces, config)

    try:
        raw_response = _call_openai_compatible(config=config, messages=_judge_messages(candidates))
        parsed = _extract_json_object(raw_response)
        selected_candidate = int(parsed.get("selected_candidate", 0))
        if selected_candidate < 0 or selected_candidate >= len(candidates):
            raise ValueError(f"judge selected invalid candidate {selected_candidate}")
        sampling_mode = str(parsed.get("sampling_mode", "normal")).strip().lower()
        if sampling_mode not in SAMPLING_MODES:
            sampling_mode = "normal"
        selected = candidates[selected_candidate]
        decision = {
            "used_judge": True,
            "fallback": False,
            "selected_candidate": selected_candidate,
            "sampling_mode": sampling_mode,
            "confidence": float(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", ""))[:500],
            "candidate_count": len(candidates),
            "model": config.model,
        }
        return selected["trace_idx"], selected["checkpoint"], decision
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
        fallback = heuristic_selection(traces, config)
        if fallback is None:
            return None
        trace_idx, checkpoint, decision = fallback
        decision.update(
            {
                "used_judge": True,
                "fallback": True,
                "error": f"{type(exc).__name__}: {exc}",
                "model": config.model,
            }
        )
        return trace_idx, checkpoint, decision
