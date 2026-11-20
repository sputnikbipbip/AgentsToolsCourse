from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import ollama

from agent_tools.registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to local tools. "
    "Use tools when they help answer the user accurately, and summarize the results clearly."
)


class ChatClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any] | str
    output: str


@dataclass
class AgentResult:
    content: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0


def _extract_message(response: Any) -> dict[str, Any]:
    # The Ollama client returns Pydantic models; a dictionary also works in tests.
    if not isinstance(response, dict):
        response = response.model_dump(exclude_none=True)
    message = response.get("message")
    if message is None:
        raise ValueError("Ollama response did not contain a message.")
    return message


class OllamaAgent:
    """Multi-turn tool-calling agent backed by a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        registry: ToolRegistry | None = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        host: str | None = None,
        client: ChatClient | None = None,
        max_iterations: int = 8,
        on_tool_call: Callable[[ToolCallRecord], None] | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        self.model = model
        self.registry = registry if registry is not None else ToolRegistry()
        self.system_prompt = system_prompt
        self.client: ChatClient = client if client is not None else ollama.Client(host=host)
        self.max_iterations = max_iterations
        self.on_tool_call = on_tool_call
        self.messages: list[dict[str, Any]] = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def reset(self) -> None:
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def run(self, user_prompt: str) -> AgentResult:
        """Run one turn, retaining complete history and discarding a failed turn."""
        if not user_prompt.strip():
            raise ValueError("The prompt must not be empty.")
        start = len(self.messages)
        try:
            return self._run_turn(user_prompt)
        except Exception:
            del self.messages[start:]
            raise

    def _run_turn(self, user_prompt: str) -> AgentResult:
        # 1. Send the conversation and available tool schemas to the model.
        self.messages.append({"role": "user", "content": user_prompt})
        tool_schemas = self.registry.schemas() or None
        recorded_calls: list[ToolCallRecord] = []

        for iteration in range(1, self.max_iterations + 1):
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=tool_schemas,
            )
            message = _extract_message(response)
            content = message.get("content") or ""
            calls = message.get("tool_calls") or []

            # 2. Remember the model's response, including any tool calls.
            self.messages.append(message)
            if not calls:
                return AgentResult(
                    content=content,
                    tool_calls=recorded_calls,
                    iterations=iteration,
                )

            # 3. Run each requested tool and send its result back on the next loop.
            for call in calls:
                function = call["function"]
                name = function["name"]
                args = function.get("arguments", {})
                output = self.registry.execute(name, args)
                record = ToolCallRecord(name=name, arguments=args, output=output)
                recorded_calls.append(record)
                if self.on_tool_call is not None:
                    self.on_tool_call(record)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": output,
                    }
                )

        content = (
            f"Stopped after reaching the maximum number of tool iterations ({self.max_iterations})."
        )
        self.messages.append({"role": "assistant", "content": content})
        return AgentResult(
            content=content,
            tool_calls=recorded_calls,
            iterations=self.max_iterations,
        )
