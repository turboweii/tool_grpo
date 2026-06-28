"""Generate veRL tool_config YAML for BFCL v4 multi-turn categories."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.bfcl_wrapper import load_bfcl_entries, normalize_tool_schema  # noqa: E402


def build_config(categories: list[str]) -> dict:
    entries = load_bfcl_entries(categories)
    by_name: dict[str, dict] = {}
    for entry in entries:
        for function_doc in entry.get("function", []):
            schema = normalize_tool_schema(function_doc)
            name = schema["function"]["name"]
            by_name.setdefault(name, schema)

    return {
        "tools": [
            {
                "class_name": "src.envs.bfcl_tools.BFCLTool",
                "config": {"type": "native"},
                "tool_schema": schema,
            }
            for _name, schema in sorted(by_name.items())
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--output", default="configs/tool_config/bfcl_v4_multi_turn_tools.yaml")
    args = parser.parse_args()

    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    config = build_config(categories)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Wrote {len(config['tools'])} BFCL tools for {categories} -> {out}")


if __name__ == "__main__":
    main()
