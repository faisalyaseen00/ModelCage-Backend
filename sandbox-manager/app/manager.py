"""High-level ModelCage sandbox orchestration."""

from typing import Any, Sequence

from .docker_runtime import DockerExecutionTimeout, DockerRuntime, DockerRuntimeError
from .models import CommandResult, ContainerInfo
from .policy import (
    CAP_DROP,
    CPU_LIMIT_NANO_CPUS,
    EXECUTION_TIMEOUT_SECONDS,
    FORBIDDEN_MOUNT_OPTIONS,
    FORBIDDEN_NETWORK_OPTIONS,
    FORBIDDEN_ISOLATION_OPTIONS,
    FORBIDDEN_PRIVILEGE_OPTIONS,
    FORBIDDEN_RESOURCE_OPTIONS,
    MEMORY_LIMIT_BYTES,
    MODELCAGE_RUNTIME,
    NETWORK_MODE,
    PID_LIMIT,
    PRIVILEGED,
    READ_ONLY_ROOT_FILESYSTEM,
    WRITABLE_TMPFS,
)
from .telemetry import ExecutionTrace, TelemetryCollector


class SandboxRuntimePolicyError(RuntimeError):
    """Raised when a container does not satisfy the ModelCage runtime policy."""


class SandboxFilesystemPolicyError(RuntimeError):
    """Raised when a container request violates filesystem isolation policy."""


class SandboxResourcePolicyError(RuntimeError):
    """Raised when a container does not satisfy the ModelCage resource policy."""


class SandboxPrivilegePolicyError(RuntimeError):
    """Raised when a container violates the ModelCage privilege policy."""


class SandboxNetworkPolicyError(RuntimeError):
    """Raised when a container violates the ModelCage network policy."""


class SandboxExecutionPolicyError(RuntimeError):
    """Raised when a caller attempts to override execution policy."""


class SandboxExecutionTimeout(SandboxExecutionPolicyError):
    """Raised after a timed-out execution has been terminated and cleaned up."""


class SandboxManager:
    """Orchestrates sandbox lifecycle without exposing Docker details."""

    def __init__(
        self, runtime: DockerRuntime | None = None, telemetry: TelemetryCollector | None = None
    ):
        self.runtime = runtime or DockerRuntime()
        self.telemetry = telemetry or TelemetryCollector()
        self._traces: dict[str, ExecutionTrace] = {}
        self._last_trace_id: str | None = None

    def create(
        self,
        image: str,
        command: str | Sequence[str] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Create a disposable, runsc-backed sandbox and return its Docker ID."""
        requested_runtime = kwargs.pop("runtime", MODELCAGE_RUNTIME)
        if requested_runtime != MODELCAGE_RUNTIME:
            raise SandboxRuntimePolicyError(
                f"ModelCage containers must use runtime '{MODELCAGE_RUNTIME}', "
                f"not '{requested_runtime}'."
            )

        forbidden_mounts = FORBIDDEN_MOUNT_OPTIONS.intersection(kwargs)
        if forbidden_mounts:
            options = ", ".join(sorted(forbidden_mounts))
            raise SandboxFilesystemPolicyError(
                f"ModelCage sandboxes do not permit host filesystem mounts ({options})."
            )

        requested_read_only = kwargs.pop("read_only", READ_ONLY_ROOT_FILESYSTEM)
        if requested_read_only is not READ_ONLY_ROOT_FILESYSTEM:
            raise SandboxFilesystemPolicyError("ModelCage sandboxes require a read-only root filesystem.")

        if "tmpfs" in kwargs:
            raise SandboxFilesystemPolicyError(
                "ModelCage manages the sandbox's writable /tmp filesystem."
            )

        forbidden_resources = FORBIDDEN_RESOURCE_OPTIONS.intersection(kwargs)
        if forbidden_resources:
            options = ", ".join(sorted(forbidden_resources))
            raise SandboxResourcePolicyError(
                f"ModelCage sandboxes do not permit resource-limit overrides ({options})."
            )

        forbidden_privileges = FORBIDDEN_PRIVILEGE_OPTIONS.intersection(kwargs)
        if forbidden_privileges:
            options = ", ".join(sorted(forbidden_privileges))
            raise SandboxPrivilegePolicyError(
                f"ModelCage sandboxes do not permit privilege overrides ({options})."
            )

        forbidden_networks = FORBIDDEN_NETWORK_OPTIONS.intersection(kwargs)
        if forbidden_networks:
            options = ", ".join(sorted(forbidden_networks))
            raise SandboxNetworkPolicyError(
                f"ModelCage sandboxes do not permit network overrides ({options})."
            )

        forbidden_isolation = FORBIDDEN_ISOLATION_OPTIONS.intersection(kwargs)
        if forbidden_isolation:
            options = ", ".join(sorted(forbidden_isolation))
            raise SandboxExecutionPolicyError(
                f"ModelCage sandboxes do not permit isolation overrides ({options})."
            )

        if kwargs:
            options = ", ".join(sorted(kwargs))
            raise SandboxExecutionPolicyError(
                f"Unsupported container options: {options}. ModelCage owns the sandbox configuration."
            )

        result = self.runtime.create_container(
            image=image,
            command=command,
            name=name,
            runtime=MODELCAGE_RUNTIME,
            read_only=READ_ONLY_ROOT_FILESYSTEM,
            tmpfs=WRITABLE_TMPFS,
            mem_limit=MEMORY_LIMIT_BYTES,
            nano_cpus=CPU_LIMIT_NANO_CPUS,
            pids_limit=PID_LIMIT,
            cap_drop=CAP_DROP,
            privileged=PRIVILEGED,
            network_mode=NETWORK_MODE,
            **kwargs,
        )
        container_id = result["Id"]

        try:
            host_config = self.runtime.inspect(container_id).get("HostConfig", {})
            actual_runtime = host_config.get("Runtime")
            if actual_runtime != MODELCAGE_RUNTIME:
                raise SandboxRuntimePolicyError(
                    f"ModelCage container '{container_id}' failed runtime policy validation: "
                    f"expected '{MODELCAGE_RUNTIME}', Docker reported {actual_runtime!r}."
                )
            actual_limits = {
                "memory": host_config.get("Memory"),
                "CPU": host_config.get("NanoCpus"),
                "PIDs": host_config.get("PidsLimit"),
            }
            expected_limits = {
                "memory": MEMORY_LIMIT_BYTES,
                "CPU": CPU_LIMIT_NANO_CPUS,
                "PIDs": PID_LIMIT,
            }
            if actual_limits != expected_limits:
                raise SandboxResourcePolicyError(
                    f"ModelCage container '{container_id}' failed resource policy validation: "
                    f"expected {expected_limits}, Docker reported {actual_limits}."
                )
            actual_privileges = {
                "CapAdd": host_config.get("CapAdd") or [],
                "CapDrop": host_config.get("CapDrop") or [],
                "Privileged": host_config.get("Privileged"),
            }
            expected_privileges = {
                "CapAdd": [],
                "CapDrop": CAP_DROP,
                "Privileged": PRIVILEGED,
            }
            if actual_privileges != expected_privileges:
                raise SandboxPrivilegePolicyError(
                    f"ModelCage container '{container_id}' failed privilege policy validation: "
                    f"expected {expected_privileges}, Docker reported {actual_privileges}."
                )
            actual_network_mode = host_config.get("NetworkMode")
            if actual_network_mode != NETWORK_MODE:
                raise SandboxNetworkPolicyError(
                    f"ModelCage container '{container_id}' failed network policy validation: "
                    f"expected '{NETWORK_MODE}', Docker reported {actual_network_mode!r}."
                )
        except Exception:
            self.runtime.remove_container(container_id)
            raise

        return container_id

    def start(self, container: str) -> None:
        """Start a previously created sandbox."""
        self.runtime.start_container(container)

    def execute(
        self,
        container: str,
        command: str | Sequence[str],
        *,
        model_id: str | None = None,
        analysis_id: str | None = None,
        progress=None,
        cleanup_on_error: bool = True,
        instrumentation=None,
        **kwargs: Any,
    ) -> CommandResult:
        """Execute a command in a sandbox."""
        if "timeout" in kwargs:
            raise SandboxExecutionPolicyError(
                "ModelCage controls the execution timeout; callers cannot override it."
            )
        if kwargs:
            options = ", ".join(sorted(kwargs))
            raise SandboxExecutionPolicyError(f"Unsupported execution options: {options}.")

        trace = self.telemetry.start(container, model_id=model_id, execution_id=analysis_id)
        command_name = command if isinstance(command, str) else " ".join(command)
        try:
            options = {"progress": progress} if progress is not None else {}
            if instrumentation is None:
                execution = self.runtime.exec(container, command, timeout=EXECUTION_TIMEOUT_SECONDS, **options)
            else:
                execution = self.runtime.exec_monitored(container, command, timeout=EXECUTION_TIMEOUT_SECONDS,
                                                       instrumentation=instrumentation, **options)
            if isinstance(execution, tuple):
                exit_code, output = execution
                pid = None
                syscall_output = None
                capture_status = "unavailable"
                capture_reason = "runtime did not provide instrumentation metadata"
            else:
                exit_code = execution.exit_code
                output = execution.output
                pid = execution.pid
                command_name = execution.command or command_name
                syscall_output = execution.syscall_output
                capture_status = execution.capture_status
                capture_reason = execution.capture_reason
            trace.event_source = "gvisor_seccheck" if instrumentation is not None else "strace_in_sandbox_runner"
            trace.capture_status = capture_status
            trace.capture_reason = capture_reason
            self.telemetry.finish(
                trace,
                exit_code=exit_code,
                termination_reason="completed" if exit_code == 0 else "non_zero_exit",
            )
            self.telemetry.add_process_lifecycle(
                trace, pid=pid, command=command_name, exit_code=exit_code
            )
            self.telemetry.add_strace_output(trace, syscall_output, default_pid=pid)
            if instrumentation is None and trace.capture_status == "captured" and not trace.syscalls:
                trace.capture_status = "empty"
                trace.capture_reason = "trace contained no parseable syscall records"
            self._store_trace(trace)
        except DockerExecutionTimeout as error:
            trace.event_source = "gvisor_seccheck" if instrumentation is not None else "strace_in_sandbox_runner"
            trace.capture_status = "failed"
            trace.capture_reason = "capture interrupted by execution timeout"
            self.telemetry.finish(
                trace, exit_code=None, termination_reason="timeout", timed_out=True, error=error
            )
            self.telemetry.add_process_lifecycle(
                trace, pid=None, command=command_name, exit_code=None
            )
            self.telemetry.add_strace_output(trace, getattr(error, "syscall_output", None))
            if trace.syscalls:
                trace.capture_status = "partial"
            self._store_trace(trace)
            if cleanup_on_error:
                self.destroy(container)
            raise SandboxExecutionTimeout(str(error)) from error
        except Exception as error:
            trace.event_source = "gvisor_seccheck" if instrumentation is not None else "strace_in_sandbox_runner"
            trace.capture_status = "failed"
            trace.capture_reason = str(error)
            self.telemetry.finish(
                trace, exit_code=None, termination_reason="execution_error", error=error
            )
            self.telemetry.add_process_lifecycle(
                trace, pid=None, command=command_name, exit_code=None
            )
            self.telemetry.add_strace_output(trace, getattr(error, "syscall_output", None))
            if trace.syscalls:
                trace.capture_status = "partial"
            self._store_trace(trace)
            if cleanup_on_error:
                self.destroy(container)
            raise
        return CommandResult(
            exit_code=exit_code,
            output=output,
            execution_id=trace.execution_id,
            capture_status=trace.capture_status,
        )

    def get_trace(self, execution_id: str) -> ExecutionTrace:
        """Return a trace retained after the corresponding sandbox is removed."""
        return self._traces[execution_id]

    def latest_trace(self) -> ExecutionTrace:
        """Return the most recently retained trace, including a failed execution."""
        if self._last_trace_id is None:
            raise LookupError("No ModelCage executions have produced telemetry yet.")
        return self._traces[self._last_trace_id]

    def export_trace(self, execution_id: str) -> str:
        """Serialize a retained trace as deterministic JSON."""
        return self.get_trace(execution_id).to_json()

    def _store_trace(self, trace: ExecutionTrace) -> None:
        self._traces[trace.execution_id] = trace
        self._last_trace_id = trace.execution_id

    def inspect(self, container: str) -> ContainerInfo:
        """Return simplified container information, including its runtime."""
        raw = self.runtime.inspect(container)
        host_config = raw.get("HostConfig", {})
        return ContainerInfo(
            id=raw["Id"],
            name=raw.get("Name", "").lstrip("/"),
            status=raw.get("State", {}).get("Status", "unknown"),
            runtime=host_config.get("Runtime"),
            raw=raw,
        )

    def destroy(self, container: str) -> None:
        """Remove a sandbox; removal is safe if it already disappeared."""
        self.runtime.remove_container(container)
