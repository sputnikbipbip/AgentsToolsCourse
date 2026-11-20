from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model


@dataclass(frozen=True)
class Tool:
    """A callable wrapped with metadata and a Pydantic argument validator."""

    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]

    def to_ollama_schema(self) -> dict[str, Any]:
        """Return the tool definition in Ollama's function-calling schema format."""
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def run(self, arguments: dict[str, Any] | str) -> str:
        """Validate arguments and execute the underlying function."""
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return f"Error: invalid JSON arguments: {exc}"
        else:
            parsed_args = arguments

        if not isinstance(parsed_args, dict):
            return "Error: tool arguments must be a JSON object."

        try:
            validated = self.args_model(**parsed_args)
        except ValidationError as exc:
            return f"Error: invalid arguments for '{self.name}': {exc}"

        try:
            result = self.fn(**{k: getattr(validated, k) for k in type(validated).model_fields})
        except Exception as exc:  # noqa: BLE001 - surface tool errors back to the model
            return f"Error executing '{self.name}': {exc}"

        return _serialize_result(result)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def _serialize_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result, default=str)


def _build_args_model(fn: Callable[..., Any], model_name: str) -> type[BaseModel]:
    if inspect.iscoroutinefunction(fn):
        raise TypeError(f"Tool '{fn.__name__}' must be a synchronous function.")
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"Tool '{fn.__name__}' cannot use positional-only or variadic parameters "
                f"(parameter '{param_name}')."
            )
        annotation = hints.get(param_name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    return create_model(f"{model_name}Args", __config__=ConfigDict(extra="forbid"), **fields)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator that turns a typed Python function into an Ollama-compatible Tool."""

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        doc = inspect.getdoc(func) or ""
        tool_desc = description or (doc.splitlines()[0] if doc else f"Call {tool_name}")
        args_model = _build_args_model(func, tool_name)
        return Tool(
            name=tool_name,
            description=tool_desc,
            fn=func,
            args_model=args_model,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


class ToolRegistry:
    """Holds registered tools, exports Ollama schemas, and dispatches tool calls."""

    def __init__(self, tools: list[Tool | Callable[..., Any]] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for item in tools:
                self.register(item)

    def register(self, item: Tool | Callable[..., Any]) -> Tool:
        wrapped = item if isinstance(item, Tool) else tool(item)
        if wrapped.name in self._tools:
            raise ValueError(f"Tool '{wrapped.name}' is already registered.")
        self._tools[wrapped.name] = wrapped
        return wrapped

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_ollama_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any] | str) -> str:
        target = self._tools.get(name)
        if target is None:
            available = ", ".join(sorted(self._tools)) or "none"
            return f"Error: unknown tool '{name}'. Available tools: {available}"
        return target.run(arguments)
