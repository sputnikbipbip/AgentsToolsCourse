from typing import Annotated

import pytest
from pydantic import Field

from agent_tools.registry import ToolRegistry, tool


def test_tool_schema_generation() -> None:
    @tool
    def add(
        a: Annotated[int, Field(description="First integer")],
        b: Annotated[int, Field(description="Second integer")] = 10,
    ) -> int:
        """Add two integers together."""
        return a + b

    schema = add.to_ollama_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "add"
    assert fn["description"] == "Add two integers together."
    assert fn["parameters"]["required"] == ["a"]
    assert fn["parameters"]["properties"]["a"]["type"] == "integer"
    assert fn["parameters"]["properties"]["a"]["description"] == "First integer"
    assert fn["parameters"]["properties"]["b"]["default"] == 10


def test_tool_execution_and_validation() -> None:
    @tool(name="greet_user", description="Greet someone by name.")
    def greet(name: str, excited: bool = False) -> str:
        return f"Hello, {name}{'!' if excited else '.'}"

    registry = ToolRegistry([greet])

    assert registry.execute("greet_user", {"name": "Ada"}) == "Hello, Ada."
    assert registry.execute("greet_user", '{"name": "Ada", "excited": true}') == "Hello, Ada!"

    bad_json = registry.execute("greet_user", "{not valid json")
    assert bad_json.startswith("Error: invalid JSON arguments")

    missing_arg = registry.execute("greet_user", {})
    assert "invalid arguments" in missing_arg

    unknown = registry.execute("missing_tool", {})
    assert "unknown tool 'missing_tool'" in unknown


def test_duplicate_tool_raises() -> None:
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    registry = ToolRegistry([ping])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ping)


def test_tool_exception_is_captured_as_error_string() -> None:
    @tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    registry = ToolRegistry([boom])
    output = registry.execute("boom", {})
    assert output == "Error executing 'boom': kaboom"


def test_extra_arguments_are_rejected() -> None:
    @tool
    def greet(name: str) -> str:
        return f"Hello, {name}"

    assert "invalid arguments" in greet.run({"name": "Ada", "unexpected": 1})
    assert greet.to_ollama_schema()["function"]["parameters"]["additionalProperties"] is False


def test_unsupported_signatures_fail_at_registration() -> None:
    def positional(name: str, /) -> str:
        return name

    async def asynchronous(name: str) -> str:
        return name

    with pytest.raises(TypeError, match="positional-only"):
        tool(positional)
    with pytest.raises(TypeError, match="synchronous"):
        tool(asynchronous)
