"""Per-trajectory BFCL execution context for veRL tool rollouts."""
from __future__ import annotations

import contextvars
from typing import Any


CURRENT_BFCL_STATE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "current_bfcl_state", default=None
)


def make_initial_state(test_entry: dict[str, Any], request_id: str) -> dict[str, Any]:
    question = test_entry.get("question") or []
    return {
        "request_id": request_id,
        "test_entry": test_entry,
        "test_entry_id": str(test_entry.get("id", request_id)),
        "test_category": str(test_entry.get("id", "multi_turn")).rsplit("_", 1)[0],
        "current_turn": 0,
        "num_tool_calls": 0,
        "num_user_turns": 0,
        "done": False,
        "model_name": f"verl_bfcl_{request_id}",
        "model_result": [[] for _ in question],
        "last_score_detail": None,
    }
