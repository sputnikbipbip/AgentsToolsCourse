from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from agent_tools.registry import Tool, tool


@tool
def get_current_time(
    timezone: Annotated[
        str,
        Field(description="IANA timezone name, e.g. 'UTC' or 'America/New_York'"),
    ] = "UTC",
) -> str:
    """Get the current date and time in ISO-8601 format for a given timezone."""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown timezone '{timezone}'.") from exc
    return datetime.now(tz).isoformat()


def _resolve_safe_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Access denied: '{relative_path}' is outside the workspace.")
    return candidate


def make_file_tools(workspace_root: Path | None = None) -> list[Tool]:
    """Create workspace-scoped file inspection tools."""
    root = (workspace_root or Path.cwd()).resolve()
    if not root.is_dir():
        raise ValueError(f"Workspace is not a directory: {root}")

    @tool
    def list_files(
        directory: Annotated[
            str,
            Field(description="Relative path to directory within the workspace"),
        ] = ".",
    ) -> list[str]:
        """List files and subdirectories in a workspace directory."""
        target = _resolve_safe_path(root, directory)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name
            for p in target.iterdir()
            if not p.name.startswith(".")
        )
        return entries

    @tool
    def read_file(
        path: Annotated[
            str,
            Field(description="Relative path to the file within the workspace"),
        ],
        max_chars: Annotated[
            int,
            Field(description="Maximum number of characters to return", ge=1, le=50000),
        ] = 8000,
    ) -> str:
        """Read the text contents of a file inside the workspace."""
        target = _resolve_safe_path(root, path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"Path is a directory: {path}")

        with target.open(encoding="utf-8") as file:
            content = file.read(max_chars + 1)
        if len(content) > max_chars:
            return f"{content[:max_chars]}\n... [truncated]"
        return content

    return [list_files, read_file]
