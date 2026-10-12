from unittest.mock import Mock

import pytest

from app.docker_runtime import DockerRuntime, DockerExecutionResult
from app.seccheck import CaptureFailure
from tests.seccheck_fixtures import ProtocolSession


def test_monitored_launch_creates_explicit_exec_boundary():
    command = ["python3", "/tmp/modelcage/runner.py", "/tmp/modelcage/model.pkl"]
    session = ProtocolSession("analysis", "sandbox", command)
    runtime = DockerRuntime(Mock())

    def execute(container, launch, **kwargs):
        assert session.decoder.armed
        assert launch == ["sh", "-c", 'exec "$@"', "modelcage-launch", *command]
        session.capture()
        return DockerExecutionResult(0, "done")

    runtime._exec_once = Mock(side_effect=execute)
    result = runtime.exec_monitored("sandbox", command, timeout=30, instrumentation=session)
    assert result.capture_status == "captured"


def test_missing_root_after_exit_is_capture_error_not_model_timeout():
    command = ["python3", "/tmp/modelcage/runner.py"]
    session = ProtocolSession("analysis", "sandbox", command)
    runtime = DockerRuntime(Mock())
    runtime._exec_once = Mock(return_value=DockerExecutionResult(1, ""))
    with pytest.raises(CaptureFailure, match="without a verified"):
        runtime.exec_monitored("sandbox", command, timeout=30, instrumentation=session)
