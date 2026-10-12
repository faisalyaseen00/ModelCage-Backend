"""Docker SDK adapter for ModelCage sandboxes."""

from queue import Empty, Queue
from threading import Thread
from typing import Any, Sequence
import posixpath
from uuid import uuid4

import docker
from docker.errors import APIError, DockerException, NotFound


class DockerRuntimeError(RuntimeError):
    """Raised when Docker cannot complete a runtime operation."""


class DockerExecutionTimeout(DockerRuntimeError):
    """Raised when Docker exec does not complete within its allowed duration."""


class DockerExecutionResult:
    """Docker execution result plus optional in-sandbox syscall trace output."""

    def __init__(
        self,
        exit_code: int,
        output: str,
        *,
        exec_id: str | None = None,
        pid: int | None = None,
        command: str | None = None,
        syscall_output: str | None = None,
        capture_status: str = "not_requested",
        capture_reason: str | None = None,
    ):
        self.exit_code = exit_code
        self.output = output
        self.exec_id = exec_id
        self.pid = pid
        self.command = command
        self.syscall_output = syscall_output
        self.capture_status = capture_status
        self.capture_reason = capture_reason


class DockerRuntime:
    """Keeps Docker-specific communication out of the sandbox manager."""

    def __init__(self, client: Any | None = None):
        if client is not None:
            self.client = client
            self._strace_available: dict[str, bool] = {}
            return

        try:
            self.client = docker.from_env()
            self.client.ping()
            self._strace_available: dict[str, bool] = {}
        except DockerException as error:
            raise DockerRuntimeError(f"Unable to connect to Docker daemon: {error}") from error

    def create_container(
        self,
        image: str,
        command: str | Sequence[str] | None = None,
        name: str | None = None,
        *,
        runtime: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a container using the runtime selected by the caller."""
        try:
            container = self.client.containers.create(
                image=image,
                command=command,
                name=name,
                runtime=runtime,
                **kwargs,
            )
            return {"Id": container.id, "Warnings": []}
        except (APIError, DockerException) as error:
            raise DockerRuntimeError(f"Unable to create container: {error}") from error

    def start_container(self, container: str) -> None:
        """Start a container by ID or name."""
        try:
            self.client.api.start(container=container)
        except (APIError, DockerException) as error:
            raise DockerRuntimeError(f"Unable to start container '{container}': {error}") from error

    def _exec_once(
        self,
        container: str,
        command: str | Sequence[str],
        *,
        timeout: float,
        on_tick=None,
    ) -> DockerExecutionResult:
        """Execute one Docker command without adding telemetry instrumentation."""
        try:
            result = self.client.api.exec_create(container=container, cmd=command)
        except (APIError, DockerException) as error:
            raise DockerRuntimeError(f"Unable to execute command in '{container}': {error}") from error

        outcome: Queue[tuple[Any, Any, Exception | None]] = Queue(maxsize=1)

        def wait_for_execution() -> None:
            try:
                output = self.client.api.exec_start(result["Id"])
                inspection = self.client.api.exec_inspect(result["Id"])
                outcome.put((output, inspection, None))
            except (APIError, DockerException) as error:
                outcome.put((None, None, error))
            except Exception as error:
                outcome.put((None, None, error))

        Thread(target=wait_for_execution, daemon=True).start()
        from time import monotonic
        deadline = monotonic() + timeout
        while True:
            try:
                remaining = self._remaining_timeout(deadline)
                output, inspection, error = outcome.get(timeout=min(1, remaining) if on_tick else remaining)
                break
            except Empty:
                self._remaining_timeout(deadline)
                if on_tick:
                    on_tick()

        if error is not None:
            raise DockerRuntimeError(f"Unable to execute command in '{container}': {error}") from error
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        process_config = inspection.get("ProcessConfig", {})
        command_name = process_config.get("entrypoint") or process_config.get("arguments")
        if isinstance(command_name, list):
            command_name = " ".join(str(part) for part in command_name)
        return DockerExecutionResult(
            inspection.get("ExitCode", 1),
            output,
            exec_id=result["Id"],
            pid=inspection.get("Pid"),
            command=command_name,
        )

    def _remaining_timeout(self, deadline: float) -> float:
        from time import monotonic

        remaining = deadline - monotonic()
        if remaining <= 0:
            raise DockerExecutionTimeout("Execution exceeded the configured timeout.")
        return remaining

    def _has_strace(self, container: str, deadline: float) -> bool:
        if container not in self._strace_available:
            probe = self._exec_once(
                container,
                ["sh", "-c", "command -v strace >/dev/null 2>&1"],
                timeout=self._remaining_timeout(deadline),
            )
            self._strace_available[container] = probe.exit_code == 0
        return self._strace_available[container]

    def _strace_command(
        self, command: str | Sequence[str], trace_prefix: str
    ) -> list[str]:
        tracer = [
            "strace",
            "-ttt",
            "-s",
            "128",
            "-e",
            "trace=%file,%process",
            "-o",
            trace_prefix,
            "--",
        ]
        if isinstance(command, str):
            return [*tracer, "sh", "-c", command]
        return [*tracer, *command]

    def exec(
        self,
        container: str,
        command: str | Sequence[str],
        *,
        timeout: float,
        progress=None,
    ) -> DockerExecutionResult:
        """Trace the direct target inside gVisor; never grant extra privileges.

        Progress counts parsed records retrieved from the trace file. Timeout/error
        exceptions retain the last readable partial capture before manager cleanup.
        """
        from time import monotonic
        from .telemetry import TelemetryCollector

        deadline = monotonic() + timeout

        def emit(stage, message, **fields):
            if progress:
                progress(stage, message, **fields)

        emit("instrumentation_starting", "Checking in-sandbox tracing capability")
        if not self._has_strace(container, deadline):
            emit("instrumentation_unavailable", "strace unavailable in sandbox", level="WARNING",
                 capture_status="unavailable")
            result = self._exec_once(container, command, timeout=self._remaining_timeout(deadline))
            result.capture_status = "unavailable"
            result.capture_reason = "strace is unavailable in the controlled sandbox image"
            return result

        trace_path = f"/tmp/modelcage-strace-{uuid4()}"

        def read_trace(path):
            # Fixed, generated paths; bound host memory use on hostile trace output.
            result = self._exec_once(container, ["head", "-c", "8388608", path], timeout=2)
            if result.exit_code != 0:
                raise DockerRuntimeError("Unable to retrieve in-sandbox strace output")
            return result.output

        probe = self._exec_once(container, self._strace_command(["true"], trace_path + ".probe"),
                                timeout=self._remaining_timeout(deadline))
        probe_text = read_trace(trace_path + ".probe")
        collector = TelemetryCollector()
        probe_trace = collector.start(container)
        collector.add_strace_output(probe_trace, probe_text)
        if probe.exit_code != 0 or not probe_trace.syscalls:
            emit("instrumentation_failed", "In-sandbox strace capability probe failed", level="ERROR",
                 capture_status="failed")
            raise DockerRuntimeError("In-sandbox strace capability probe failed; workload not started")
        emit("instrumentation_initialized", "In-sandbox tracing capability verified",
             capture_status="initialized")
        last_output = ""
        last_count = -1

        def capture():
            nonlocal last_output, last_count
            last_output = read_trace(trace_path)
            trace = collector.start(container)
            collector.add_strace_output(trace, last_output)
            count = len(trace.syscalls)
            if count != last_count:
                emit("capturing", f"Runtime events captured: {count}", event_count=count)
                last_count = count

        def tick():
            try:
                capture()
            except DockerRuntimeError:
                # Final retrieval explicitly reports a capture failure if this persists.
                pass

        emit("executing", "Workload execution requested inside sandbox")
        try:
            result = self._exec_once(container, self._strace_command(command, trace_path),
                                     timeout=self._remaining_timeout(deadline),
                                     on_tick=tick if progress else None)
            try:
                capture()
            except DockerRuntimeError:
                result.capture_status = "failed"
                result.capture_reason = "unable to retrieve in-sandbox strace output"
                return result
            result.syscall_output = last_output
            result.capture_status = "captured" if last_count > 0 else "empty"
            if not last_count:
                result.capture_reason = "strace completed without parseable syscall output"
            if len(last_output.encode("utf-8")) >= 8388608:
                result.capture_status = "partial"
                result.capture_reason = "trace exceeded the 8 MiB retrieval limit"
            return result
        except Exception as error:
            # Terminate exec processes while keeping PID 1 and tmpfs alive for retrieval.
            # kill(-1) excludes the sender and namespace init; no capabilities added.
            try:
                stopped = self._exec_once(container, ["sh", "-c", "kill -KILL -1"], timeout=2)
                if stopped.exit_code != 0:
                    error.capture_stop_failed = True
            except DockerRuntimeError:
                error.capture_stop_failed = True
            tick()
            error.syscall_output = last_output
            raise

    def exec_monitored(self, container, command, *, timeout, instrumentation, progress=None):
        """Launch through an explicit guest execve boundary observed by SecCheck."""
        from time import monotonic, sleep
        from .seccheck import CaptureFailure
        if isinstance(command, str) or not command:
            raise DockerRuntimeError("Monitored execution requires an argument vector")
        deadline = monotonic() + timeout
        last_count = -1
        started = False
        instrumentation.arm()

        def update():
            nonlocal last_count, started
            count = instrumentation.progress()
            if instrumentation.decoder.root_group and not started:
                started = True
                if progress:
                    progress("workload_started", "Controlled harness process observed by SecCheck")
            if count != last_count:
                if progress:
                    progress("capturing", f"Syscall entries received: {count}", event_count=count)
                last_count = count

        # Docker exec creates a guest task through runsc control; this need not
        # produce sentry/execve. A fixed launcher performs an actual guest execve.
        # Positional arguments avoid interpreting the command as shell source.
        launch = ["sh", "-c", 'exec "$@"', "modelcage-launch", *command]
        result = self._exec_once(container, launch, timeout=self._remaining_timeout(deadline), on_tick=update)
        attribution_deadline = min(deadline, monotonic() + 1)
        # Include descendants for the same execution budget, not just the parent process.
        while True:
            update()
            if instrumentation.decoder.root_group and not instrumentation.decoder.live_threads:
                break
            if not instrumentation.decoder.root_group and monotonic() >= attribution_deadline:
                raise CaptureFailure("Workload exited without a verified SecCheck execve boundary")
            self._remaining_timeout(deadline)
            sleep(0.05)
        result.capture_status = "captured" if last_count > 0 else "empty"
        return result

    def terminate_workload(self, container):
        """Kill namespace exec processes, retaining init/tmpfs for collector shutdown."""
        result = self._exec_once(container, ["sh", "-c", "kill -KILL -1"], timeout=3)
        if result.exit_code != 0:
            raise DockerRuntimeError("Unable to terminate sandbox workload processes")

    def inspect(self, container: str) -> dict[str, Any]:
        """Return Docker's inspection payload for a container."""
        try:
            return self.client.api.inspect_container(container)
        except NotFound as error:
            raise DockerRuntimeError(f"Container '{container}' was not found") from error
        except (APIError, DockerException) as error:
            raise DockerRuntimeError(f"Unable to inspect container '{container}': {error}") from error

    def stage_files(self, container: str, files: dict[str, bytes]) -> None:
        """Stream files into /tmp without putting model bytes in exec arguments.

        Docker archive uploads reject a read-only root even for writable tmpfs.
        A fixed Python writer consumes an exact byte count over attached stdin;
        the model is never deserialized during staging.
        """
        from time import monotonic, sleep

        writer = (
            "import os,sys\n"
            "path=sys.argv[1]; remaining=int(sys.argv[2])\n"
            "os.makedirs(os.path.dirname(path),mode=0o700,exist_ok=True)\n"
            "with open(path,'xb') as output:\n"
            " os.chmod(path,0o600)\n"
            " while remaining:\n"
            "  chunk=sys.stdin.buffer.read(min(remaining,1048576))\n"
            "  if not chunk: raise RuntimeError('Incomplete staging input')\n"
            "  output.write(chunk); remaining-=len(chunk)\n"
        )
        for path, content in files.items():
            if not path.startswith("modelcage/") or posixpath.isabs(path) or ".." in path.split("/"):
                raise DockerRuntimeError("ModelCage staging paths must remain under /tmp/modelcage")
            connection = None
            deadline = monotonic() + 30
            try:
                execution = self.client.api.exec_create(
                    container=container,
                    cmd=["python3", "-c", writer, f"/tmp/{path}", str(len(content))],
                    stdin=True,
                )
                connection = self.client.api.exec_start(execution["Id"], socket=True)
                transport = getattr(connection, "_sock", connection)
                transport.settimeout(self._remaining_timeout(deadline))
                transport.sendall(content)
                # The writer reads the exact length, so no EOF/half-close is needed.
                # Drain output to allow Docker to finish its attached stream.
                while True:
                    transport.settimeout(self._remaining_timeout(deadline))
                    if not transport.recv(65536):
                        break
                while True:
                    inspection = self.client.api.exec_inspect(execution["Id"])
                    if not inspection.get("Running"):
                        break
                    self._remaining_timeout(deadline)
                    sleep(0.01)
                if inspection.get("ExitCode") != 0:
                    raise DockerRuntimeError(f"Staging writer failed for '{path}' (exit {inspection.get('ExitCode')})")
            except (DockerException, OSError) as error:
                raise DockerRuntimeError(f"Unable to stream staging file '{path}': {type(error).__name__}") from error
            finally:
                if connection is not None:
                    connection.close()

    def read_staged_text(self, container: str, path: str) -> str:
        """Read a ModelCage-owned staging file without tracing a second workload."""
        if not path.startswith("modelcage/") or posixpath.isabs(path) or ".." in path.split("/"):
            raise DockerRuntimeError("ModelCage staging paths must remain under /tmp/modelcage")
        result = self._exec_once(container, ["cat", f"/tmp/{path}"], timeout=5)
        if result.exit_code != 0:
            raise DockerRuntimeError(f"Unable to read staged file '{path}': {result.output}")
        return result.output.strip()

    def remove_container(self, container: str) -> None:
        """Remove a container, including one that has already stopped."""
        self._strace_available.pop(container, None)
        try:
            self.client.api.remove_container(container, force=True)
        except NotFound:
            return
        except (APIError, DockerException) as error:
            raise DockerRuntimeError(f"Unable to remove container '{container}': {error}") from error
