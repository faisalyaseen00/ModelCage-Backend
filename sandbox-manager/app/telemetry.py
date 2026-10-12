"""Structured, observation-only telemetry for ModelCage executions."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
from uuid import uuid4


def utc_timestamp() -> str:
    """Return a consistently formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def epoch_timestamp(value: str) -> str:
    """Normalize an epoch timestamp emitted by strace to the trace format."""
    return (
        datetime.fromtimestamp(float(value), timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class SyscallEvent:
    """A normalized syscall record collected from an in-sandbox tracer."""

    order: int
    timestamp: str
    name: str
    pid: int | None = None
    arguments: str | None = None
    return_value: str | None = None
    execution_id: str | None = None


@dataclass(frozen=True)
class FilesystemEvent:
    """A filesystem-related syscall observation."""

    order: int
    timestamp: str
    operation: str
    path: str | None
    pid: int | None = None


@dataclass(frozen=True)
class ProcessEvent:
    """An observable process lifecycle event for an execution."""

    timestamp: str
    event: str
    pid: int | None = None
    parent_pid: int | None = None
    command: str | None = None


@dataclass
class ExecutionTrace:
    """Stable behavioral trace retained independently of container lifetime."""

    execution_id: str
    sandbox_id: str
    model_id: str | None
    start_time: str
    end_time: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    termination_reason: str | None = None
    timed_out: bool = False
    error: str | None = None
    event_source: str | None = None
    capture_status: str = "not_requested"
    capture_reason: str | None = None
    syscalls: list[SyscallEvent] = field(default_factory=list)
    filesystem_events: list[FilesystemEvent] = field(default_factory=list)
    process_events: list[ProcessEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return JSON-native, deterministically named trace fields."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the trace deterministically for later pipeline consumers."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


_STRACE_LINE = re.compile(
    r"^(?:\[pid\s+(?P<bracket_pid>\d+)\]\s+)?(?P<timestamp>\d+(?:\.\d+)?)\s+"
    r"(?:(?P<pid>\d+)\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<arguments>.*)\)\s+=\s+"
    r"(?P<return_value>.*)$"
)
_QUOTED_PATH = re.compile(r'"(?P<path>(?:\\.|[^"\\])*)"')
_FILESYSTEM_SYSCALLS = {
    "access",
    "chmod",
    "chdir",
    "creat",
    "mkdir",
    "mkdirat",
    "open",
    "openat",
    "read",
    "rename",
    "renameat",
    "rmdir",
    "stat",
    "unlink",
    "unlinkat",
    "write",
}
_PROCESS_SYSCALLS = {"clone", "clone3", "execve", "exit", "exit_group", "fork", "vfork"}
_PATH_SYSCALLS = _FILESYSTEM_SYSCALLS - {"read", "write"}


class TelemetryCollector:
    """Builds traces and normalizes optional strace output without scoring it."""

    def start(
        self,
        sandbox_id: str,
        model_id: str | None = None,
        execution_id: str | None = None,
    ) -> ExecutionTrace:
        return ExecutionTrace(
            execution_id=execution_id or str(uuid4()),
            sandbox_id=sandbox_id,
            model_id=model_id,
            start_time=utc_timestamp(),
        )

    def finish(
        self,
        trace: ExecutionTrace,
        *,
        exit_code: int | None,
        termination_reason: str,
        timed_out: bool = False,
        error: Exception | None = None,
    ) -> None:
        end_time = utc_timestamp()
        trace.end_time = end_time
        trace.duration_seconds = round(
            datetime.fromisoformat(end_time.replace("Z", "+00:00")).timestamp()
            - datetime.fromisoformat(trace.start_time.replace("Z", "+00:00")).timestamp(),
            6,
        )
        trace.exit_code = exit_code
        trace.termination_reason = termination_reason
        trace.timed_out = timed_out
        trace.error = str(error) if error is not None else None

    def add_process_lifecycle(
        self,
        trace: ExecutionTrace,
        *,
        pid: int | None,
        command: str | None,
        exit_code: int | None,
    ) -> None:
        trace.process_events.extend(
            [
                ProcessEvent(timestamp=trace.start_time, event="started", pid=pid, command=command),
                ProcessEvent(
                    timestamp=trace.end_time or utc_timestamp(),
                    event="exited" if exit_code is not None else "failed",
                    pid=pid,
                    command=command,
                ),
            ]
        )

    def add_strace_output(
        self, trace: ExecutionTrace, output: str | None, *, default_pid: int | None = None
    ) -> None:
        """Normalize strace records into syscall, filesystem, and process events."""
        if not output:
            return

        for line in output.splitlines():
            match = _STRACE_LINE.match(line)
            if match is None:
                continue
            try:
                timestamp = epoch_timestamp(match.group("timestamp"))
            except (ValueError, OverflowError, OSError):
                continue
            order = len(trace.syscalls) + 1
            raw_pid = match.group("bracket_pid") or match.group("pid")
            pid = int(raw_pid) if raw_pid else default_pid
            name = match.group("name")
            arguments = match.group("arguments")
            trace.syscalls.append(
                SyscallEvent(
                    order,
                    timestamp,
                    name,
                    pid=pid,
                    arguments=arguments,
                    return_value=match.group("return_value"),
                    execution_id=trace.execution_id,
                )
            )
            if name in _FILESYSTEM_SYSCALLS:
                path_match = _QUOTED_PATH.search(arguments) if name in _PATH_SYSCALLS else None
                path = path_match.group("path") if path_match else None
                trace.filesystem_events.append(
                    FilesystemEvent(order, timestamp, name, path, pid=pid)
                )
            if name in _PROCESS_SYSCALLS:
                event = "created" if name in {"clone", "clone3", "fork", "vfork"} else "executed"
                if name in {"exit", "exit_group"}:
                    event = "exited"
                trace.process_events.append(
                    ProcessEvent(timestamp, event, pid=pid, command=name)
                )
