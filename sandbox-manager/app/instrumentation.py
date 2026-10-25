"""Category validation and safe normalization for runtime instrumentation."""

from dataclasses import dataclass, replace
from enum import StrEnum
import re

from .telemetry import ExecutionTrace, SyscallEvent


class ModelCategory(StrEnum):
    """The only model task categories supported by ModelCage instrumentation."""

    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"


class InstrumentationInputError(ValueError):
    """Raised when an unsupported category is requested."""


@dataclass(frozen=True)
class ModelMetadata:
    """Validated model identity kept separate from behavioral observations."""

    model_sha256: str | None
    filename: str
    artifact_format: str
    category: ModelCategory


def require_category(value: str | ModelCategory) -> ModelCategory:
    """Validate an explicitly supplied category; ModelCage never infers it here."""
    try:
        return ModelCategory(value)
    except ValueError as error:
        supported = ", ".join(category.value for category in ModelCategory)
        raise InstrumentationInputError(f"Unsupported model category. Use one of: {supported}.") from error


def runner_only_syscalls(trace: ExecutionTrace, runner_pid: int | None) -> list[SyscallEvent]:
    """Attach the known runner PID to syscalls produced by its direct strace target.

    DockerRuntime deliberately invokes strace without ``-f``. Therefore its log
    contains the Python runner process only, not subprocesses created by a
    deserialized artifact or unrelated container processes.
    """
    return [
        replace(
            event,
            pid=runner_pid if runner_pid is not None else event.pid,
            execution_id=trace.execution_id,
        )
        for event in trace.syscalls
    ]


_WRITE_BUFFER = re.compile(r'("(?:\\.|[^"\\])*")')


def sanitize_arguments(name: str, arguments: str | None, limit: int = 512) -> str | None:
    """Keep path/flag context while excluding potentially sensitive data buffers."""
    if arguments is None:
        return None
    sanitized = arguments
    if name in {"read", "write", "sendto", "recvfrom"}:
        sanitized = _WRITE_BUFFER.sub('"<redacted>"', sanitized, count=1)
    if len(sanitized) > limit:
        sanitized = f"{sanitized[:limit]}…"
    return sanitized
