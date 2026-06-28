"""Generic BFCL tool adapter for veRL."""
from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from src.envs.bfcl_context import CURRENT_BFCL_STATE


def function_call_to_python(tool_name: str, parameters: dict[str, Any] | None) -> str:
    parameters = parameters or {}
    if not isinstance(parameters, dict):
        return f"{tool_name}({parameters!r})"
    args = ", ".join(f"{key}={value!r}" for key, value in parameters.items())
    return f"{tool_name}({args})"


def parse_tool_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def execute_bfcl_calls(state: dict[str, Any], calls: list[str]) -> list[str]:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

    test_entry = state["test_entry"]
    test_category = state["test_category"]
    results, _instances = execute_multi_turn_func_call(
        func_call_list=calls,
        initial_config=test_entry.get("initial_config", {}),
        involved_classes=test_entry.get("involved_classes", []),
        model_name=state["model_name"],
        test_entry_id=state["test_entry_id"],
        long_context=("long_context" in test_category or "composite" in test_category),
        is_evaL_run=False,
    )
    return [str(item) for item in results]


class BFCLTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        state = CURRENT_BFCL_STATE.get()
        if state is None:
            raise RuntimeError("BFCL state missing from context; BFCLInteraction.start_interaction was not called")

        call = function_call_to_python(self.name, parameters)
        state["num_tool_calls"] += 1
        state["model_result"][state["current_turn"]].append([call])
        try:
            result = execute_bfcl_calls(state, [call])
        except Exception as exc:  # noqa: BLE001
            text = f"Error during execution: {type(exc).__name__}: {exc}"
            return ToolResponse(text=text), 0.0, {"error": str(exc), "tool": self.name}

        text = result[0] if result else ""
        return ToolResponse(text=str(text)), 0.0, {"tool": self.name, "call": call}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass
