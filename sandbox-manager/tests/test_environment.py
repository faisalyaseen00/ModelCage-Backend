"""Reproducible Phase 5 end-to-end validation of the ModelCage environment."""

from concurrent.futures import ThreadPoolExecutor
import json
from unittest.mock import Mock

import pytest
from app.docker_runtime import DockerExecutionTimeout, DockerRuntimeError
from app.manager import (
    SandboxExecutionPolicyError,
    SandboxExecutionTimeout,
    SandboxFilesystemPolicyError,
    SandboxManager,
    SandboxNetworkPolicyError,
    SandboxPrivilegePolicyError,
    SandboxResourcePolicyError,
    SandboxRuntimePolicyError,
)
from app.policy import (
    CAP_DROP,
    CPU_LIMIT_NANO_CPUS,
    MEMORY_LIMIT_BYTES,
    MODELCAGE_RUNTIME,
    NETWORK_MODE,
    PID_LIMIT,
    PRIVILEGED,
    READ_ONLY_ROOT_FILESYSTEM,
    WRITABLE_TMPFS,
)
from app.telemetry import TelemetryCollector


@pytest.fixture
def live_sandbox():
    """Provide one disposable, live gVisor sandbox for a validation case."""
    try:
        sandbox = SandboxManager()
    except DockerRuntimeError as error:
        pytest.skip(f"Docker daemon unavailable: {error}")

    container = sandbox.create(image="alpine:latest", command=["sleep", "30"])
    sandbox.start(container)
    try:
        yield sandbox, container
    finally:
        sandbox.destroy(container)


@pytest.mark.integration
def test_phase5_live_runtime_filesystem_privilege_resource_network_and_telemetry(live_sandbox):
    sandbox, container = live_sandbox
    info = sandbox.inspect(container)
    host_config = info.raw["HostConfig"]

    assert sandbox.runtime.client.info()["DefaultRuntime"] == "runc"
    assert info.runtime == MODELCAGE_RUNTIME
    assert host_config["ReadonlyRootfs"] is READ_ONLY_ROOT_FILESYSTEM
    assert host_config["Tmpfs"] == WRITABLE_TMPFS
    assert host_config["Memory"] == MEMORY_LIMIT_BYTES
    assert host_config["NanoCpus"] == CPU_LIMIT_NANO_CPUS
    assert host_config["PidsLimit"] == PID_LIMIT
    assert host_config.get("CapAdd") in (None, [])
    assert host_config["CapDrop"] == CAP_DROP
    assert host_config["Privileged"] is PRIVILEGED
    assert host_config["NetworkMode"] == NETWORK_MODE
    assert all(mount.get("Type") == "tmpfs" for mount in info.raw.get("Mounts", []))

    verification = sandbox.execute(
        container,
        [
            "sh",
            "-c",
            "uname -a; echo phase5 > /tmp/phase5; test -f /tmp/phase5 "
            "&& ! touch /phase5-root-write "
            "&& test ! -e /home/faisal/ModelCage/ModelCage-Backend",
        ],
        model_id="phase5-model",
    )
    assert verification.exit_code == 0
    assert "gvisor" in verification.output.lower()

    # Bounded workloads exercise each constrained resource without approaching
    # host-level exhaustion. Exact cgroup values above are the enforcement proof.
    assert sandbox.execute(
        container,
        ["sh", "-c", "dd if=/dev/zero of=/tmp/phase5-memory bs=1M count=64 2>/dev/null; rm /tmp/phase5-memory"],
    ).exit_code == 0
    assert sandbox.execute(
        container,
        ["sh", "-c", "i=0; while [ \"$i\" -lt 100000 ]; do i=$((i + 1)); done"],
    ).exit_code == 0
    assert sandbox.execute(
        container,
        ["sh", "-c", "for _ in 1 2 3 4 5 6 7 8; do sleep 0.1 & done; wait"],
    ).exit_code == 0
    assert sandbox.execute(
        container,
        ["sh", "-c", "wget -q -T 2 -O /dev/null https://example.com && exit 1 || exit 0"],
    ).exit_code == 0

    trace = sandbox.get_trace(verification.execution_id)
    payload = json.loads(sandbox.export_trace(verification.execution_id))
    assert trace.execution_id == verification.execution_id
    assert trace.sandbox_id == container
    assert trace.model_id == "phase5-model"
    assert trace.exit_code == 0
    assert trace.termination_reason == "completed"
    assert trace.start_time and trace.end_time and trace.duration_seconds is not None
    assert len(trace.process_events) >= 2
    assert payload["execution_id"] == verification.execution_id
    assert payload["syscalls"] == []  # Alpine has no in-sandbox strace.
    assert payload["filesystem_events"] == []


def test_phase5_execution_failure_paths_retain_traces_and_cleanup():
    timeout_runtime = Mock()
    timeout_runtime.exec.side_effect = DockerExecutionTimeout("timeout")
    timed_out = SandboxManager(runtime=timeout_runtime)
    with pytest.raises(SandboxExecutionTimeout):
        timed_out.execute("timeout-container", "sleep 60")
    assert timed_out.latest_trace().termination_reason == "timeout"
    timeout_runtime.remove_container.assert_called_once_with("timeout-container")

    failure_runtime = Mock()
    failure_runtime.exec.side_effect = DockerRuntimeError("unexpected termination")
    failed = SandboxManager(runtime=failure_runtime)
    with pytest.raises(DockerRuntimeError):
        failed.execute("failed-container", "false")
    assert failed.latest_trace().termination_reason == "execution_error"
    failure_runtime.remove_container.assert_called_once_with("failed-container")

    nonzero_runtime = Mock()
    nonzero_runtime.exec.return_value = (9, "failed")
    nonzero = SandboxManager(runtime=nonzero_runtime)
    result = nonzero.execute("nonzero-container", "false")
    assert result.exit_code == 9
    assert nonzero.get_trace(result.execution_id).termination_reason == "non_zero_exit"


def test_phase5_trace_event_schema_with_tracer_fixture():
    collector = TelemetryCollector()
    trace = collector.start("trace-container", model_id="trace-model")
    collector.add_strace_output(
        trace,
        "1700000000.001 execve(\"/bin/sh\", [\"sh\"], 0x0) = 0\n"
        "1700000000.002 openat(AT_FDCWD, \"/tmp/model.py\", O_RDONLY) = 3\n"
        "1700000000.003 clone(child_stack=NULL, flags=0) = 10\n",
    )

    assert trace.syscalls[0].name == "execve"
    assert trace.filesystem_events[0].path == "/tmp/model.py"
    assert trace.process_events[-1].event == "created"
    assert json.loads(trace.to_json())["syscalls"][0]["name"] == "execve"


@pytest.mark.integration
def test_phase5_concurrent_sandboxes_are_isolated_and_cleaned_up():
    def run(index: int):
        try:
            sandbox = SandboxManager()
        except DockerRuntimeError as error:
            pytest.skip(f"Docker daemon unavailable: {error}")
        container = sandbox.create(image="alpine:latest", command=["sleep", "30"])
        sandbox.start(container)
        try:
            result = sandbox.execute(
                container,
                ["sh", "-c", f"echo sandbox-{index} > /tmp/isolated; cat /tmp/isolated"],
                model_id=f"concurrent-{index}",
            )
            return sandbox, container, result
        except Exception:
            sandbox.destroy(container)
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = list(executor.map(run, range(2)))

    try:
        containers = {container for _, container, _ in completed}
        execution_ids = {result.execution_id for _, _, result in completed}
        assert len(containers) == len(execution_ids) == 2
        for index, (sandbox, container, result) in enumerate(completed):
            assert result.output.strip() == f"sandbox-{index}"
            assert sandbox.get_trace(result.execution_id).sandbox_id == container
            assert sandbox.inspect(container).raw["HostConfig"]["Memory"] == MEMORY_LIMIT_BYTES

        completed[0][0].destroy(completed[0][1])
        with pytest.raises(DockerRuntimeError):
            completed[0][0].inspect(completed[0][1])
        assert completed[1][0].execute(completed[1][1], "true").exit_code == 0
    finally:
        for sandbox, container, _ in completed:
            sandbox.destroy(container)


@pytest.mark.parametrize(
    ("option", "value", "error_type"),
    [
        ("runtime", "runc", SandboxRuntimePolicyError),
        ("read_only", False, SandboxFilesystemPolicyError),
        ("tmpfs", {"/tmp": "rw"}, SandboxFilesystemPolicyError),
        ("volumes", {"/host": {"bind": "/host", "mode": "ro"}}, SandboxFilesystemPolicyError),
        ("mem_limit", 1, SandboxResourcePolicyError),
        ("nano_cpus", 1, SandboxResourcePolicyError),
        ("pids_limit", 1, SandboxResourcePolicyError),
        ("cap_add", ["NET_ADMIN"], SandboxPrivilegePolicyError),
        ("privileged", True, SandboxPrivilegePolicyError),
        ("network_mode", "host", SandboxNetworkPolicyError),
        ("ports", {"80/tcp": 8080}, SandboxNetworkPolicyError),
    ],
)
def test_phase5_security_regression_rejects_every_caller_override(option, value, error_type):
    runtime = Mock()
    sandbox = SandboxManager(runtime=runtime)
    with pytest.raises(error_type):
        sandbox.create(image="alpine:latest", **{option: value})
    runtime.create_container.assert_not_called()


def test_phase5_security_regression_rejects_timeout_override():
    runtime = Mock()
    sandbox = SandboxManager(runtime=runtime)
    with pytest.raises(SandboxExecutionPolicyError):
        sandbox.execute("container", "true", timeout=3600)
    runtime.exec.assert_not_called()
