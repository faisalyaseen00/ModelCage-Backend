from unittest.mock import Mock

import pytest

from app.docker_runtime import DockerRuntime, DockerRuntimeError


def test_large_model_uses_stdin_and_closes_connection():
    client = Mock()
    client.api.exec_create.return_value = {"Id": "staging"}
    connection = client.api.exec_start.return_value
    connection._sock.recv.side_effect = [b"", b""]
    client.api.exec_inspect.return_value = {"Running": False, "ExitCode": 0}
    payload = b"x" * 30_062_053
    DockerRuntime(client).stage_files("sandbox", {"modelcage/model.pkl": payload})
    options = client.api.exec_create.call_args.kwargs
    assert options["stdin"] is True
    assert sum(len(arg) for arg in options["cmd"]) < 2048
    assert options["cmd"][-1] == str(len(payload))
    connection._sock.sendall.assert_called_once_with(payload)
    connection.close.assert_called_once()


@pytest.mark.parametrize("socket_error", [False, True])
def test_staging_failure_closes_connection(socket_error):
    client = Mock()
    client.api.exec_create.return_value = {"Id": "staging"}
    connection = client.api.exec_start.return_value
    connection._sock.recv.return_value = b""
    client.api.exec_inspect.return_value = {"Running": False, "ExitCode": 1}
    if socket_error:
        connection._sock.sendall.side_effect = BrokenPipeError()
    with pytest.raises(DockerRuntimeError):
        DockerRuntime(client).stage_files("sandbox", {"modelcage/model.pkl": b"test"})
    connection.close.assert_called_once()
