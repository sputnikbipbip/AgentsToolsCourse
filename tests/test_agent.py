from typing import Any

import pytest
from ollama import ChatResponse

from agent_tools.agent import OllamaAgent
from agent_tools.registry import ToolRegistry
from agent_tools.tools import calculate


class FakeOllamaClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": tools,
            }
        )
        return self._responses.pop(0)


def test_agent_executes_tool_call_and_returns_final_answer() -> None:
    client = FakeOllamaClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculate",
                                "arguments": {"expression": "6 * 7"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "6 * 7 is 42.",
                }
            },
        ]
    )

    registry = ToolRegistry([calculate])
    agent = OllamaAgent(model="qwen2.5-coder:3b", registry=registry, client=client)

    result = agent.run("What is 6 * 7?")

    assert result.content == "6 * 7 is 42."
    assert result.iterations == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "calculate"
    assert result.tool_calls[0].output == "42"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_name": "calculate",
        "content": "42",
    }


def test_agent_accepts_ollama_response_objects() -> None:
    client = FakeOllamaClient([])
    client._responses = [
        ChatResponse.model_validate(
            {
                "model": "test",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "calculate", "arguments": {"expression": "6 * 7"}}}
                    ],
                },
            }
        ),
        ChatResponse.model_validate(
            {"model": "test", "message": {"role": "assistant", "content": "42"}}
        ),
    ]
    agent = OllamaAgent(client=client, registry=ToolRegistry([calculate]))
    assert agent.run("What is 6 * 7?").content == "42"
    assert client.calls[1]["messages"][-1]["content"] == "42"


def test_agent_stops_at_max_iterations() -> None:
    looping_response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculate",
                        "arguments": {"expression": "1 + 1"},
                    }
                }
            ],
        }
    }
    client = FakeOllamaClient([looping_response, looping_response])
    agent = OllamaAgent(
        model="qwen2.5-coder:3b",
        registry=ToolRegistry([calculate]),
        client=client,
        max_iterations=2,
    )

    result = agent.run("Loop forever")
    assert result.iterations == 2
    assert "maximum number of tool iterations" in result.content
    assert agent.messages[-1] == {"role": "assistant", "content": result.content}


def test_agent_rolls_back_failed_turn_and_can_retry() -> None:
    class FailingClient(FakeOllamaClient):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            if not self._responses:
                raise ConnectionError("server down")
            return super().chat(**kwargs)

    client = FailingClient([])
    agent = OllamaAgent(client=client)
    initial_history = list(agent.messages)
    with pytest.raises(ConnectionError):
        agent.run("Hello")
    assert agent.messages == initial_history

    client._responses.append({"message": {"role": "assistant", "content": "Hi"}})
    assert agent.run("Hello again").content == "Hi"
    assert [message["role"] for message in agent.messages] == ["system", "user", "assistant"]


def test_agent_validates_input_and_missing_response() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        OllamaAgent(max_iterations=0)
    agent = OllamaAgent(client=FakeOllamaClient([{}]))
    with pytest.raises(ValueError, match="empty"):
        agent.run("  ")
    with pytest.raises(ValueError, match="did not contain a message"):
        agent.run("Hello")
    assert len(agent.messages) == 1
