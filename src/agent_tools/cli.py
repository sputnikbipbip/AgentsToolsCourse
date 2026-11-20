from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import ollama

from agent_tools.agent import OllamaAgent, ToolCallRecord
from agent_tools.registry import ToolRegistry
from agent_tools.tools import calculate, get_current_time, make_file_tools


def build_default_registry(workspace: Path | None = None) -> ToolRegistry:
    registry = ToolRegistry([calculate, get_current_time])
    for file_tool in make_file_tools(workspace):
        registry.register(file_tool)
    return registry


def _print_tool_call(record: ToolCallRecord) -> None:
    print(f"  [tool] {record.name}({record.arguments}) -> {record.output}")


def _ask(agent: OllamaAgent, prompt: str) -> str | None:
    try:
        return agent.run(prompt).content
    except (httpx.HTTPError, ollama.ResponseError, ConnectionError, ValueError) as exc:
        print(f"Agent error: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Interactive CLI for running local Ollama models with Python agent tools.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:3b",
        help="Ollama model name (default: qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Optional Ollama host URL (e.g. http://localhost:11434)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace directory for file tools (default: current directory)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Run a single prompt and exit instead of starting an interactive loop",
    )
    args = parser.parse_args(argv)

    try:
        registry = build_default_registry(args.workspace)
    except ValueError as exc:
        parser.error(str(exc))
    agent = OllamaAgent(
        model=args.model,
        registry=registry,
        host=args.host,
        on_tool_call=_print_tool_call,
    )

    if args.prompt is not None:
        answer = _ask(agent, args.prompt)
        if answer is None:
            return 1
        print(answer)
        return 0

    tool_names = ", ".join(t.name for t in registry.list_tools())
    print(f"Ollama Agent ({args.model}) | Tools: {tool_names}")
    print("Type 'exit' or 'quit' to stop, or 'reset' to clear history.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("Conversation reset.\n")
            continue

        answer = _ask(agent, user_input)
        if answer is not None:
            print(f"assistant> {answer}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
