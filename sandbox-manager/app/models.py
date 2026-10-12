"""Small data models returned by the sandbox manager."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContainerInfo:
    """Useful container identity and runtime information."""

    id: str
    name: str
    status: str
    runtime: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class CommandResult:
    """Result of a command executed inside a container."""

    exit_code: int
    output: str
    execution_id: str | None = None
    capture_status: str | None = None
