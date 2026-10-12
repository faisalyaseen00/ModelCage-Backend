"""ModelCage Sandbox Manager application package."""

from .manager import (
    SandboxExecutionPolicyError,
    SandboxExecutionTimeout,
    SandboxFilesystemPolicyError,
    SandboxManager,
    SandboxNetworkPolicyError,
    SandboxPrivilegePolicyError,
    SandboxResourcePolicyError,
    SandboxRuntimePolicyError,
)
from .models import CommandResult, ContainerInfo
from .modelcage import (
    ModelCage,
    ModelInputError,
    execute_model,
    export_csv,
    export_xlsx,
    get_trace,
)
from .quarantine import QuarantinedModel, QuarantineError, QuarantineStore
from .instrumentation import ModelCategory, ModelMetadata
from .telemetry import ExecutionTrace, FilesystemEvent, ProcessEvent, SyscallEvent, TelemetryCollector
from .policy import (
    CAP_DROP,
    CPU_LIMIT_NANO_CPUS,
    EXECUTION_TIMEOUT_SECONDS,
    MEMORY_LIMIT_BYTES,
    MODELCAGE_RUNTIME,
    NETWORK_MODE,
    PID_LIMIT,
    PRIVILEGED,
    READ_ONLY_ROOT_FILESYSTEM,
    WRITABLE_TMPFS,
)

__all__ = [
    "SandboxManager",
    "SandboxRuntimePolicyError",
    "SandboxFilesystemPolicyError",
    "SandboxResourcePolicyError",
    "SandboxPrivilegePolicyError",
    "SandboxNetworkPolicyError",
    "SandboxExecutionPolicyError",
    "SandboxExecutionTimeout",
    "MODELCAGE_RUNTIME",
    "MEMORY_LIMIT_BYTES",
    "CPU_LIMIT_NANO_CPUS",
    "PID_LIMIT",
    "CAP_DROP",
    "PRIVILEGED",
    "NETWORK_MODE",
    "EXECUTION_TIMEOUT_SECONDS",
    "READ_ONLY_ROOT_FILESYSTEM",
    "WRITABLE_TMPFS",
    "CommandResult",
    "ContainerInfo",
    "ModelCage",
    "ModelInputError",
    "QuarantinedModel",
    "QuarantineError",
    "QuarantineStore",
    "ModelCategory",
    "ModelMetadata",
    "execute_model",
    "get_trace",
    "export_csv",
    "export_xlsx",
    "ExecutionTrace",
    "FilesystemEvent",
    "ProcessEvent",
    "SyscallEvent",
    "TelemetryCollector",
]
