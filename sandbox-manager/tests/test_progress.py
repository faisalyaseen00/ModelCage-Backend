"""Lifecycle/API unit tests; these never require Docker."""
import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from urllib.request import urlopen
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.docker_runtime import DockerExecutionResult, DockerExecutionTimeout, DockerRuntime, DockerRuntimeError
from app.execution_events import ExecutionEvents
from app.manager import SandboxManager, SandboxExecutionTimeout, SandboxRuntimePolicyError
from app.modelcage import ModelCage
from app.web import ModelCageWebServer
from tests.test_sandbox import policy_host_config
from tests.seccheck_fixtures import ProtocolSession, manager_fixture
from app.seccheck import InstrumentationUnavailable

TRACE = '1700000000.001 openat(AT_FDCWD, "/tmp/model.pkl", O_RDONLY) = 3\n'


@pytest.fixture
def setup_cage(tmp_path):
    manager, runtime = manager_fixture()
    cage = ModelCage(tmp_path / "traces.sqlite", manager=manager, capture_factory=ProtocolSession)
    artifact = tmp_path / "input.pkl"
    artifact.write_bytes(pickle.dumps({"harmless": True}))
    return cage, runtime, artifact


def test_terminal_journal_and_reconnect_snapshot_match(setup_cage, caplog):
    cage, runtime, artifact = setup_cage
    with caplog.at_level(logging.INFO, logger="modelcage.lifecycle"):
        result = cage.execute_model(artifact, "text_generation")
    snapshot = cage.events.snapshot(result.execution_id)
    assert snapshot["status"] == "succeeded"
    assert snapshot["event_count"] == 1
    assert snapshot["cleanup_status"] == "destroyed"
    assert snapshot["storage_status"] == "saved"
    assert snapshot["hashing_status"] == "paused"
    assert snapshot["model_sha256"] is None
    assert [r.message.split("] ", 1)[1] for r in caplog.records] == [e["message"] for e in snapshot["events"]]
    reloaded = ExecutionEvents(cage.events.root)
    assert reloaded.snapshot(result.execution_id) == snapshot
    stages = [e["stage"] for e in snapshot["events"]]
    assert stages.index("runtime_verified") < stages.index("sandbox_started")
    assert stages.index("capture_stopped") < stages.index("destroyed") < stages.index("stored") < stages.index("finalized")
    runtime.remove_container.assert_called_once()


def test_failed_verification_prevents_execution(setup_cage):
    cage, runtime, artifact = setup_cage
    runtime.inspect.return_value = {"HostConfig": policy_host_config(Runtime="runc")}
    analysis = cage.events.begin(category="text_generation")
    with pytest.raises(SandboxRuntimePolicyError):
        cage.execute_model(artifact, "text_generation", analysis_id=analysis)
    runtime.exec_monitored.assert_not_called()
    runtime.remove_container.assert_called_once()
    snapshot = cage.events.snapshot(analysis)
    assert snapshot["status"] == "failed"
    assert "runtime_verified" not in [e["stage"] for e in snapshot["events"]]


def test_timeout_flushes_partial_capture_before_cleanup(setup_cage):
    cage, runtime, artifact = setup_cage
    error = DockerExecutionTimeout("timeout")
    def timeout(*args, instrumentation, **kwargs):
        instrumentation.capture()
        raise error
    runtime.exec_monitored.side_effect = timeout
    analysis = cage.events.begin(category="text_generation")
    with pytest.raises(SandboxExecutionTimeout):
        cage.execute_model(artifact, "text_generation", analysis_id=analysis)
    snapshot = cage.events.snapshot(analysis)
    assert snapshot["status"] == "timed_out"
    assert snapshot["capture_status"] == "partial"
    records = (cage.policy_store.root / "raw" / "text_generation" / f"{analysis}.jsonl").read_text().splitlines()
    assert len(records) == 2
    assert json.loads(records[0])["analysis_id"] == analysis
    assert snapshot["cleanup_status"] == "destroyed"
    runtime.remove_container.assert_called_once()


@pytest.mark.parametrize("failure", ["start", "stage", "storage", "cleanup"])
def test_failures_finalize_truthfully_and_attempt_cleanup(setup_cage, failure):
    cage, runtime, artifact = setup_cage
    if failure == "start":
        runtime.start_container.side_effect = DockerRuntimeError("private contents")
    elif failure == "stage":
        runtime.stage_files.side_effect = DockerRuntimeError("private contents")
    elif failure == "storage":
        cage.policy_store.finalize = Mock(side_effect=OSError("private contents"))
    else:
        runtime.remove_container.side_effect = DockerRuntimeError("private contents")
    analysis = cage.events.begin(category="text_generation")
    with pytest.raises(Exception):
        cage.execute_model(artifact, "text_generation", analysis_id=analysis)
    snapshot = cage.events.snapshot(analysis)
    assert snapshot["status"] == "failed"
    assert snapshot["cleanup_status"] == ("failed" if failure == "cleanup" else "destroyed")
    assert "private contents" not in json.dumps(snapshot)
    runtime.remove_container.assert_called_once()


def test_concurrent_analyses_use_fresh_containers_and_never_mix_events(setup_cage):
    cage, runtime, artifact = setup_cage
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: cage.execute_model(artifact, "image_classification"), range(2)))
    assert len({r.execution_id for r in results}) == 2
    assert len({cage.get_trace(r.execution_id).sandbox_id for r in results}) == 2
    for result in results:
        snapshot = cage.events.snapshot(result.execution_id)
        assert {event["analysis_id"] for event in snapshot["events"]} == {result.execution_id}
        assert snapshot["event_count"] == 1
    assert runtime.remove_container.call_count == 2


@pytest.mark.network
def test_http_status_recovers_events_for_late_client(setup_cage, tmp_path):
    cage, _, artifact = setup_cage
    result = cage.execute_model(artifact, "text_classification")
    server = ModelCageWebServer(("127.0.0.1", 0), cage, tmp_path)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/analyses/{result.execution_id}") as response:
            assert json.load(response) == cage.events.snapshot(result.execution_id)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_tracer_capability_failure_does_not_launch_model():
    runtime = DockerRuntime(client=Mock())
    runtime._has_strace = Mock(return_value=True)
    runtime._exec_once = Mock(side_effect=[DockerExecutionResult(1, "denied"), DockerExecutionResult(0, "")])
    progress = Mock()
    with pytest.raises(DockerRuntimeError, match="probe failed"):
        runtime.exec("sandbox", ["python3", "model.py"], timeout=30, progress=progress)
    assert all("model.py" not in call.args[1] for call in runtime._exec_once.call_args_list)
    assert any(call.args[0] == "instrumentation_failed" for call in progress.call_args_list)


def test_empty_final_trace_does_not_claim_capture_success():
    runtime = DockerRuntime(client=Mock())
    runtime._has_strace = Mock(return_value=True)
    runtime._exec_once = Mock(side_effect=[DockerExecutionResult(0, ""), DockerExecutionResult(0, TRACE),
                                          DockerExecutionResult(0, ""), DockerExecutionResult(0, "")])
    result = runtime.exec("sandbox", ["true"], timeout=30)
    assert result.capture_status == "empty"


def test_instrumentation_unavailable_prevents_workload_and_preserves_failure_metadata(setup_cage):
    cage, runtime, artifact = setup_cage
    session = Mock()
    session.start.side_effect = InstrumentationUnavailable("runsc control directory permission denied")
    session.summary.return_value = {}
    session.decoder = None
    cage.capture_factory = Mock(return_value=session)
    analysis = cage.events.begin(category="text_generation")
    with pytest.raises(InstrumentationUnavailable):
        cage.execute_model(artifact, "text_generation", analysis_id=analysis)
    runtime.exec_monitored.assert_not_called()
    runtime.exec.assert_not_called()
    runtime.remove_container.assert_called_once()
    state = cage.events.snapshot(analysis)
    assert state["status"] == "instrumentation_unavailable"
    metadata = json.loads((cage.policy_store.root / 'metadata' / f'{analysis}.json').read_text())
    assert metadata['failure_reason'] == 'runsc control directory permission denied'
    assert metadata['cleanup_status'] == 'destroyed'


def test_journal_disk_failure_cannot_prevent_cleanup(setup_cage, monkeypatch):
    cage, runtime, artifact = setup_cage
    import app.execution_events as event_module
    def execute(*args, instrumentation, **kwargs):
        instrumentation.capture()
        monkeypatch.setattr(event_module.os, "fsync", Mock(side_effect=OSError("disk error")))
        return DockerExecutionResult(0, "", syscall_output=TRACE, capture_status="captured")
    runtime.exec_monitored.side_effect = execute
    analysis = cage.events.begin(category="text_generation")
    with pytest.raises(OSError):
        cage.execute_model(artifact, "text_generation", analysis_id=analysis)
    runtime.remove_container.assert_called_once()
    snapshot = cage.events.snapshot(analysis)
    assert snapshot["stage"] == "finalized"
    assert snapshot["status"] == "failed"
    assert snapshot["cleanup_status"] == "destroyed"
    assert snapshot["progress_storage_status"] == "failed"


@pytest.mark.parametrize('category,manifest',[(None,None),('text_generation',{'task_category':'image_classification'})])
def test_category_review_stops_before_sandbox_creation(setup_cage, category, manifest):
    cage, runtime, artifact = setup_cage
    analysis=cage.events.begin(category=category)
    assert cage.execute_model(artifact,category,analysis_id=analysis,manifest=manifest) is None
    runtime.create_container.assert_not_called()
    state=cage.events.snapshot(analysis)
    assert state['status']=='pending_review'
    assert not (cage.policy_store.root/'raw'/'text_generation'/f'{analysis}.jsonl').exists()
