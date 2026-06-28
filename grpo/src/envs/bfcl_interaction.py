"""veRL Interaction adapter for BFCL v4 multi-turn."""
from __future__ import annotations

import copy
import logging
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

from src.envs.bfcl_context import CURRENT_BFCL_STATE, make_initial_state
from src.envs.bfcl_tools import execute_bfcl_calls, function_call_to_python
from src.envs.bfcl_wrapper import load_bfcl_entries, score_bfcl_state, turn_to_user_text
from src.training import b_ndsr

logger = logging.getLogger(__name__)


class BFCLInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        categories = config.get("categories") or ["multi_turn_base"]
        if isinstance(categories, str):
            categories = [part.strip() for part in categories.split(",") if part.strip()]
        self.categories = categories
        self.entries = load_bfcl_entries(self.categories)
        self.max_turns = int(config.get("max_turns", 30))
        self._instance_dict: dict[str, dict[str, Any]] = {}

    def _entry_for(self, task_id: int | str | None = None, entry_id: str | None = None) -> dict[str, Any]:
        if entry_id is not None:
            for entry in self.entries:
                if str(entry.get("id")) == str(entry_id):
                    return copy.deepcopy(entry)
            raise KeyError(f"BFCL entry_id not found: {entry_id}")
        return copy.deepcopy(self.entries[int(task_id or 0)])

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        task_id: int = 0,
        entry_id: str | None = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        entry = self._entry_for(task_id=task_id, entry_id=entry_id)
        state = make_initial_state(entry, instance_id)

        checkpoint_messages = kwargs.get("b_ndsr_checkpoint_messages") or []
        if checkpoint_messages:
            # The prompt contains all BFCL user turns that have already happened.
            state["current_turn"] = min(
                max(0, sum(1 for msg in checkpoint_messages if isinstance(msg, dict) and msg.get("role") == "user") - 1),
                max(0, len(entry.get("question", [])) - 1),
            )

        replay_actions = kwargs.get("b_ndsr_replay_actions") or []
        if replay_actions:
            if not b_ndsr.is_enabled():
                raise RuntimeError("Received BFCL B-NDSR replay actions while B_NDSR_ENABLED is false.")
            for action in replay_actions:
                if action.get("kind") != "tool":
                    continue
                call = function_call_to_python(action.get("tool_name", ""), action.get("parameters") or {})
                execute_bfcl_calls(state, [call])

        CURRENT_BFCL_STATE.set(state)
        self._instance_dict[instance_id] = state
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._instance_dict.get(instance_id)
        if state is None:
            raise RuntimeError(f"BFCLInteraction missing instance {instance_id}")
        CURRENT_BFCL_STATE.set(state)

        entry = state["test_entry"]
        question = entry.get("question") or []
        state["current_turn"] += 1
        state["num_user_turns"] += 1

        total_turns = state["num_user_turns"] + state["num_tool_calls"]
        if state["current_turn"] >= len(question) or total_turns >= self.max_turns:
            state["done"] = True
            reward, detail = score_bfcl_state(state)
            return (
                True,
                "",
                reward,
                {
                    "task_id": state["test_entry_id"],
                    "num_turns": total_turns,
                    "num_tool_calls": state["num_tool_calls"],
                    "reason": "done" if state["current_turn"] >= len(question) else "max_turns",
                    "bfcl_valid": bool(detail.get("valid", False)),
                    "bfcl_error_type": detail.get("error_type", ""),
                },
            )

        user_reply = turn_to_user_text(question[state["current_turn"]])
        return (
            False,
            user_reply,
            0.0,
            {
                "turn": state["current_turn"],
                "task_id": state["test_entry_id"],
                "num_tool_calls": state["num_tool_calls"],
            },
        )

    async def calculate_score(self, instance_id: str, **kwargs) -> dict[str, float]:
        state = self._instance_dict.get(instance_id)
        if state is None:
            return {"score": 0.0, "outcome_score": 0.0, "process_score": 0.0}
        reward, _detail = score_bfcl_state(state)
        return {"score": reward, "outcome_score": reward, "process_score": 0.0}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
