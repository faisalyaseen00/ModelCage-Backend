"""Unit and live integration tests for the ModelCage sandbox policies."""

from unittest.mock import Mock
import json

import pytest

import app.manager as manager_module
from app.docker_runtime import DockerExecutionTimeout, DockerRuntimeError
from app.manager import (
    SandboxExecutionPolicyError,
    SandboxExecutionTimeout,
    SandboxFilesystemPolicyError,
    SandboxManager,
    SandboxNetworkPolicyError,
    SandboxPrivilegePolicyError,
    SandboxResourcePolicyError,
)
from app.policy import (
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
from app.telemetry import TelemetryCollector
from app.manager import SandboxRuntimePolicyError


def policy_host_config(**overrides):
    """Return an inspection HostConfig that satisfies all ModelCage policies."""
    host_config = {
        "Runtime": MODELCAGE_RUNTIME,
        "Memory": MEMORY_LIMIT_BYTES,
        "NanoCpus": CPU_LIMIT_NANO_CPUS,
        "PidsLimit": PID_LIMIT,
        "CapAdd": [],
        "CapDrop": CAP_DROP,
        "Privileged": PRIVILEGED,
        "NetworkMode": NETWORK_MODE,
    }
    host_config.update(overrides)
    return host_config


def test_create_enforces_and_validates_runsc_runtime():
    runtime = Mock()
    runtime.create_container.return_value = {"Id": "runsc-container"}
    runtime.inspect.return_value = {"HostConfig": policy_host_config()}
    sandbox = SandboxManager(runtime=runtime)

    container = sandbox.create(image="alpine:latest")

    assert container == "runsc-container"
    runtime.create_container.assert_called_once_with(
        image="alpine:latest",
        command=None,
        name=None,
        runtime=MODELCAGE_RUNTIME,
        read_only=READ_ONLY_ROOT_FILESYSTEM,
        tmpfs=WRITABLE_TMPFS,
        mem_limit=MEMORY_LIMIT_BYTES,
        nano_cpus=CPU_LIMIT_NANO_CPUS,
        pids_limit=PID_LIMIT,
        cap_drop=CAP_DROP,
        privileged=PRIVILEGED,
        network_mode=NETWORK_MODE,
    )


def test_create_rejects_non_runsc_runtime_and_removes_container():
    runtime = Mock()
    runtime.create_container.return_value = {"Id": "invalid-runtime-container"}
    runtime.inspect.return_value = {"HostConfig": policy_host_config(Runtime="runc")}
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxRuntimePolicyError, match="expected 'runsc'.*'runc'"):
        sandbox.create(image="alpine:latest")

    runtime.remove_container.assert_called_once_with("invalid-runtime-container")


def test_create_rejects_runtime_override():
    sandbox = SandboxManager(runtime=Mock())

    with pytest.raises(SandboxRuntimePolicyError, match="must use runtime 'runsc'"):
        sandbox.create(image="alpine:latest", runtime="runc")


@pytest.mark.parametrize("mount_option", ["binds", "mounts", "volumes", "volumes_from"])
def test_create_rejects_filesystem_mounts(mount_option):
    runtime = Mock()
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxFilesystemPolicyError, match="do not permit host filesystem mounts"):
        sandbox.create(image="alpine:latest", **{mount_option: object()})

    runtime.create_container.assert_not_called()


def test_create_rejects_filesystem_policy_overrides():
    sandbox = SandboxManager(runtime=Mock())

    with pytest.raises(SandboxFilesystemPolicyError, match="read-only root"):
        sandbox.create(image="alpine:latest", read_only=False)

    with pytest.raises(SandboxFilesystemPolicyError, match="writable /tmp"):
        sandbox.create(image="alpine:latest", tmpfs={"/tmp": "rw"})


@pytest.mark.parametrize("resource_option", ["mem_limit", "nano_cpus", "pids_limit"])
def test_create_rejects_resource_limit_overrides(resource_option):
    runtime = Mock()
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxResourcePolicyError, match="do not permit resource-limit overrides"):
        sandbox.create(image="alpine:latest", **{resource_option: 1})

    runtime.create_container.assert_not_called()


def test_create_rejects_invalid_resource_limits_and_removes_container():
    runtime = Mock()
    runtime.create_container.return_value = {"Id": "invalid-limits-container"}
    runtime.inspect.return_value = {"HostConfig": policy_host_config(Memory=0)}
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxResourcePolicyError, match="failed resource policy validation"):
        sandbox.create(image="alpine:latest")

    runtime.remove_container.assert_called_once_with("invalid-limits-container")


@pytest.mark.parametrize("privilege_option", ["cap_add", "cap_drop", "privileged"])
def test_create_rejects_privilege_overrides(privilege_option):
    runtime = Mock()
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxPrivilegePolicyError, match="do not permit privilege overrides"):
        sandbox.create(image="alpine:latest", **{privilege_option: True})

    runtime.create_container.assert_not_called()


def test_create_rejects_invalid_privilege_policy_and_removes_container():
    runtime = Mock()
    runtime.create_container.return_value = {"Id": "privileged-container"}
    runtime.inspect.return_value = {"HostConfig": policy_host_config(Privileged=True)}
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxPrivilegePolicyError, match="failed privilege policy validation"):
        sandbox.create(image="alpine:latest")

    runtime.remove_container.assert_called_once_with("privileged-container")


@pytest.mark.parametrize(
    "network_option", ["network_mode", "network", "networking_config", "ports", "port_bindings"]
)
def test_create_rejects_network_overrides(network_option):
    runtime = Mock()
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxNetworkPolicyError, match="do not permit network overrides"):
        sandbox.create(image="alpine:latest", **{network_option: object()})

    runtime.create_container.assert_not_called()


def test_create_rejects_invalid_network_policy_and_removes_container():
    runtime = Mock()
    runtime.create_container.return_value = {"Id": "networked-container"}
    runtime.inspect.return_value = {"HostConfig": policy_host_config(NetworkMode="bridge")}
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxNetworkPolicyError, match="failed network policy validation"):
        sandbox.create(image="alpine:latest")

    runtime.remove_container.assert_called_once_with("networked-container")


def test_execute_returns_normal_and_nonzero_results_with_policy_timeout():
    runtime = Mock()
    runtime.exec.side_effect = [(0, "complete"), (7, "failed")]
    sandbox = SandboxManager(runtime=runtime)

    assert sandbox.execute("sandbox", "true").exit_code == 0
    assert sandbox.execute("sandbox", "false").exit_code == 7
    assert runtime.exec.call_args_list[0].kwargs["timeout"] == EXECUTION_TIMEOUT_SECONDS
    assert runtime.remove_container.call_count == 0


def test_execute_timeout_removes_container():
    runtime = Mock()
    runtime.exec.side_effect = DockerExecutionTimeout("execution timed out")
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxExecutionTimeout, match="execution timed out"):
        sandbox.execute("timed-out-container", "sleep 60")

    runtime.remove_container.assert_called_once_with("timed-out-container")


def test_execute_exception_removes_container():
    runtime = Mock()
    runtime.exec.side_effect = DockerRuntimeError("exec failed")
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(DockerRuntimeError, match="exec failed"):
        sandbox.execute("failed-container", "false")

    runtime.remove_container.assert_called_once_with("failed-container")


def test_execute_rejects_timeout_override():
    runtime = Mock()
    sandbox = SandboxManager(runtime=runtime)

    with pytest.raises(SandboxExecutionPolicyError, match="cannot override"):
        sandbox.execute("sandbox", "true", timeout=3600)

    runtime.exec.assert_not_called()


def test_successful_execution_produces_serializable_trace():
    runtime = Mock()
    runtime.exec.return_value = (0, "complete")
    sandbox = SandboxManager(runtime=runtime)

    result = sandbox.execute("sandbox-1", "true", model_id="models/example.py")
    trace = sandbox.get_trace(result.execution_id)

    assert trace.sandbox_id == "sandbox-1"
    assert trace.model_id == "models/example.py"
    assert trace.exit_code == 0
    assert trace.termination_reason == "completed"
    assert trace.timed_out is False
    assert trace.duration_seconds is not None
    assert len(trace.process_events) == 2
    assert json.loads(sandbox.export_trace(result.execution_id))["execution_id"] == result.execution_id


def test_nonzero_execution_produces_trace():
    runtime = Mock()
    runtime.exec.return_value = (13, "failed")
    sandbox = SandboxManager(runtime=runtime)

    result = sandbox.execute("sandbox-2", "false")
    trace = sandbox.get_trace(result.execution_id)

    assert result.exit_code == 13
    assert trace.exit_code == 13
    assert trace.termination_reason == "non_zero_exit"


def test_timeout_and_execution_exception_retain_traces():
    timed_out_runtime = Mock()
    timed_out_runtime.exec.side_effect = DockerExecutionTimeout("execution timed out")
    timed_out = SandboxManager(runtime=timed_out_runtime)

    with pytest.raises(SandboxExecutionTimeout):
        timed_out.execute("timeout-sandbox", "sleep 60")

    timeout_trace = timed_out.latest_trace()
    assert timeout_trace.termination_reason == "timeout"
    assert timeout_trace.timed_out is True
    assert timeout_trace.error == "execution timed out"
    timed_out_runtime.remove_container.assert_called_once_with("timeout-sandbox")

    failed_runtime = Mock()
    failed_runtime.exec.side_effect = DockerRuntimeError("exec failed")
    failed = SandboxManager(runtime=failed_runtime)

    with pytest.raises(DockerRuntimeError):
        failed.execute("failed-sandbox", "false")

    assert failed.latest_trace().termination_reason == "execution_error"
    assert failed.latest_trace().error == "exec failed"
    failed_runtime.remove_container.assert_called_once_with("failed-sandbox")


def test_strace_events_are_normalized_for_syscalls_filesystem_and_processes():
    collector = TelemetryCollector()
    trace = collector.start("sandbox-3")
    collector.add_strace_output(
        trace,
        "1700000000.001 execve(\"/bin/sh\", [\"sh\"], 0x0) = 0\n"
        "1700000000.002 openat(AT_FDCWD, \"/tmp/model.py\", O_RDONLY) = 3\n"
        "1700000000.003 write(1, \"ok\", 2) = 2\n"
        "1700000000.004 exit_group(0) = ?\n",
    )

    assert [event.name for event in trace.syscalls] == ["execve", "openat", "write", "exit_group"]
    assert [(event.operation, event.path) for event in trace.filesystem_events] == [
        ("openat", "/tmp/model.py"),
        ("write", None),
    ]
    assert [event.event for event in trace.process_events] == ["executed", "exited"]


def test_empty_telemetry_categories_serialize_consistently():
    trace = TelemetryCollector().start("sandbox-4")
    payload = json.loads(trace.to_json())

    assert payload["syscalls"] == []
    assert payload["filesystem_events"] == []
    assert payload["process_events"] == []


@pytest.mark.integration
def test_sandbox_uses_runsc_and_cleans_up():
    try:
        sandbox = SandboxManager()
    except DockerRuntimeError as error:
        pytest.skip(f"Docker daemon unavailable: {error}")

    container = None
    try:
        container = sandbox.create(image="alpine:latest", command=["sleep", "30"])
        sandbox.start(container)

        info = sandbox.inspect(container)
        assert info.runtime == "runsc"
        assert info.raw["HostConfig"]["ReadonlyRootfs"] is True
        assert info.raw["HostConfig"]["Tmpfs"] == WRITABLE_TMPFS
        assert info.raw["HostConfig"]["Memory"] == MEMORY_LIMIT_BYTES
        assert info.raw["HostConfig"]["NanoCpus"] == CPU_LIMIT_NANO_CPUS
        assert info.raw["HostConfig"]["PidsLimit"] == PID_LIMIT
        assert info.raw["HostConfig"].get("CapAdd") in (None, [])
        assert info.raw["HostConfig"]["CapDrop"] == CAP_DROP
        assert info.raw["HostConfig"]["Privileged"] is False
        assert info.raw["HostConfig"]["NetworkMode"] == NETWORK_MODE
        assert all(
            mount.get("Type") == "tmpfs" and mount.get("Destination") == "/tmp"
            for mount in info.raw.get("Mounts", [])
        )
        assert all(
            "ModelCage-Backend" not in mount.get("Source", "")
            for mount in info.raw.get("Mounts", [])
        )

        result = sandbox.execute(container, "uname -a")
        assert result.exit_code == 0
        assert "gvisor" in result.output.lower()
        trace_id = result.execution_id
        trace = sandbox.get_trace(trace_id)
        assert trace.termination_reason == "completed"
        assert len(trace.process_events) == 2

        model_result = sandbox.execute(
            container,
            [
                "sh",
                "-c",
                "echo model-ran > /tmp/model-output && test -f /tmp/model-output "
                "&& ! touch /modelcage-root-write-test "
                "&& test ! -e /home/faisal/ModelCage/ModelCage-Backend",
            ],
        )
        assert model_result.exit_code == 0

        bounded_workload = sandbox.execute(
            container,
            [
                "sh",
                "-c",
                "i=0; while [ \"$i\" -lt 100000 ]; do i=$((i + 1)); done; "
                "for _ in 1 2 3 4 5 6 7 8; do true & done; wait; echo workload-complete",
            ],
        )
        assert bounded_workload.exit_code == 0
        assert "workload-complete" in bounded_workload.output

        network_result = sandbox.execute(
            container,
            [
                "sh",
                "-c",
                "wget -q -T 2 -O /dev/null https://example.com && exit 1 || exit 0",
            ],
        )
        assert network_result.exit_code == 0

    finally:
        if container is not None:
            sandbox.destroy(container)

    with pytest.raises(DockerRuntimeError):
        sandbox.inspect(container)
    assert sandbox.get_trace(trace_id).sandbox_id == container
    assert json.loads(sandbox.export_trace(trace_id))["syscalls"] == []


@pytest.mark.integration
def test_timed_out_execution_is_removed_live(monkeypatch):
    try:
        sandbox = SandboxManager()
    except DockerRuntimeError as error:
        pytest.skip(f"Docker daemon unavailable: {error}")

    container = sandbox.create(image="alpine:latest", command=["sleep", "30"])
    sandbox.start(container)
    monkeypatch.setattr(manager_module, "EXECUTION_TIMEOUT_SECONDS", 0.2)

    with pytest.raises(SandboxExecutionTimeout, match="timeout"):
        sandbox.execute(container, ["sh", "-c", "sleep 10"])

    with pytest.raises(DockerRuntimeError):
        sandbox.inspect(container)
