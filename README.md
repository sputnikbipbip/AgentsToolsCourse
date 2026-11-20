# Agent Tools for Ollama

A small Python framework for defining functions as tools and letting a local
Ollama model call them. The included CLI offers a conversational agent with a
calculator, a clock, and read-only workspace inspection.

## How it works

1. Define Python functions and register them as tools. The registry creates a
   JSON schema from each function's type hints and docstring.
2. The agent sends your message and the schemas to Ollama.
3. If the model requests a tool, the agent validates its arguments, runs the
   function, and sends the result back to the model.
4. The model can request another tool or answer the user. Conversation history
   lets it remember earlier messages.

Start reading at `src/agent_tools/agent.py` for the agent loop, then
`src/agent_tools/registry.py` for tool registration and validation. The
functions themselves are in `src/agent_tools/tools/`; `cli.py` connects them
to the command line. The tests show example calls without requiring Ollama.

## Getting started

Install Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com/). Start Ollama and pull a model that supports
tool calling:

```bash
ollama serve                # if Ollama is not already running
ollama pull qwen2.5-coder:3b
uv sync
uv run agent
```

Enter prompts interactively; use `reset` to clear the conversation or `quit`
to exit. The CLI shows tool calls and their results. For a single prompt:

```bash
uv run agent --model qwen2.5-coder:3b --prompt 'What is (42 * 17) + 9?'
```

Use `--host http://localhost:11434` to choose an Ollama server and
`--workspace /path/to/project` to select the directory accessible to file
tools. Run `uv run agent --help` for all options. Models differ in how reliably
they use tools; choose a model with tool-calling support.

## Define your own agent and tools

Use type hints to define arguments. `Annotated` and Pydantic's `Field` add
descriptions and constraints to the schema sent to the model. The function's
first docstring line describes the tool.

```python
from typing import Annotated
from pydantic import Field
from agent_tools import OllamaAgent, ToolRegistry, tool


@tool
def greet(name: Annotated[str, Field(description="Name of the person")]) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"


registry = ToolRegistry([greet])
agent = OllamaAgent(model="qwen2.5-coder:3b", registry=registry)
result = agent.run("Greet Ada")
print(result.content)
print(result.tool_calls)
```

You can also use `@tool(name="...", description="...")` or register a plain
typed function with `registry.register(function)`. Tools must be synchronous
and accept named arguments; invalid arguments and tool exceptions are returned
to the model as errors. Calls to `agent.run()` share conversation history;
`agent.reset()` clears it. `max_iterations` limits model calls per turn and
`on_tool_call` can observe completed calls. Failed turns are removed from
history, but tool side effects cannot be undone.

## Included tools

- `calculate`: evaluates basic arithmetic without executing Python code.
- `get_current_time`: returns the time in an IANA timezone (UTC by default).
- `list_files` and `read_file`: inspect files inside the selected workspace.
  Listings omit hidden entries; file reads are limited to 50,000 characters.

The file tools resolve paths and reject paths outside the workspace, including
symlinks pointing outside it. Choose the workspace carefully: the model can
request any readable file inside it, including hidden files by name. This is
not a sandbox against concurrent changes to the filesystem.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```
