"""Phase 6A pickle POC tests, including safe generated model artifacts."""

import pickle
import pickletools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from app.modelcage import ModelCage, ModelInputError
from app.models import CommandResult
from app.telemetry import TelemetryCollector
from app.docker_runtime import DockerRuntime, DockerRuntimeError
from app.seccheck import InstrumentationUnavailable
from tests.seccheck_fixtures import ProtocolSession, manager_fixture


class HarmlessMarker:
    """A test-only pickle payload that creates an in-sandbox /tmp marker."""

    def __reduce__(self):
        return (Path.touch, (Path("/tmp/modelcage-malicious-marker"),))


@pytest.fixture
def model_artifacts(tmp_path):
    benign = tmp_path / "benign_model.pkl"
    malicious = tmp_path / "malicious_test_model.pkl"
    alternate = tmp_path / "benign_model.pickle"
    for path, value in ((benign, {"model": "benign"}), (alternate, [1, 2, 3])):
        path.write_bytes(pickle.dumps(value))
    malicious.write_bytes(pickle.dumps(HarmlessMarker()))
    return benign, malicious, alternate


def poc_manager():
    manager, runtime = manager_fixture()
    manager.create = Mock(wraps=manager.create)
    manager.execute = Mock(wraps=manager.execute)
    return manager, None


def test_phase6a_accepts_pkl_and_pickle_and_stages_only_controlled_paths(model_artifacts, tmp_path):
    manager, trace = poc_manager()
    poc = ModelCage(tmp_path / "traces.sqlite", manager=manager, capture_factory=ProtocolSession)
    for artifact in (model_artifacts[0], model_artifacts[2]):
        result = poc.execute_model(artifact, category="text_generation")
        assert result.execution_id is not None
    staged = manager.runtime.stage_files.call_args.args[1]
    assert all(path.startswith("modelcage/") and ".." not in path for path in staged)
    command = manager.execute.call_args.args[1]
    assert command[0] == "python3"
    assert command[-1].endswith("/model.pkl")


def test_phase6a_rejects_unsupported_and_traversal_paths(tmp_path):
    poc = ModelCage(tmp_path / "traces.sqlite", manager=Mock())
    unsupported = tmp_path / "model.pt"
    unsupported.write_bytes(b"not a pickle")
    with pytest.raises(ModelInputError, match="only .pkl and .pickle"):
        poc.execute_model(unsupported)
    with pytest.raises(ModelInputError, match="existing regular file"):
        poc.execute_model(tmp_path / "../missing.pkl")


def test_phase6a_persists_runner_only_syscalls_and_exports(model_artifacts, tmp_path):
    manager, trace = poc_manager()
    poc = ModelCage(tmp_path / "traces.sqlite", manager=manager, capture_factory=ProtocolSession)
    result = poc.execute_model(model_artifacts[0], category="text_generation")
    rows = poc.store.syscall_events(result.execution_id)
    assert [row["sequence"] for row in rows] == [1]
    assert {row["pid"] for row in rows} == {42}
    assert {row["return_value"] for row in rows} == {"3"}
    assert "Sequence,PID,Timestamp,Syscall,Arguments,Return Value" in poc.export_csv(result.execution_id)
    with ZipFile(__import__("io").BytesIO(poc.export_xlsx(result.execution_id))) as workbook:
        assert "xl/worksheets/sheet1.xml" in workbook.namelist()
    with sqlite3.connect(tmp_path / "traces.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM executions").fetchone()[0] == 1


def test_phase6a_trace_reloads_after_manager_and_sandbox_are_gone(model_artifacts, tmp_path):
    manager, _ = poc_manager()
    database = tmp_path / "persistent.sqlite"
    poc = ModelCage(database, manager=manager, capture_factory=ProtocolSession)
    result = poc.execute_model(model_artifacts[0], category="text_generation")

    reloaded = ModelCage(database, manager=Mock())
    trace = reloaded.get_trace(result.execution_id)
    assert trace.execution_id == result.execution_id
    assert trace.syscalls[0].execution_id == result.execution_id


def test_phase6a_staging_rejects_traversal():
    runtime = DockerRuntime(client=Mock())
    with pytest.raises(DockerRuntimeError, match="under /tmp/modelcage"):
        runtime.stage_files("container", {"modelcage/../escape": b"x"})


def test_phase6a_syscall_schema_preserves_order_and_runner_pid():
    collector = TelemetryCollector()
    trace = collector.start("sandbox", "model.pkl")
    collector.add_strace_output(
        trace,
        "1700000000.001 execve(\"/tmp/model_runner.py\", [\"python3\"], 0x0) = 0\n"
        "1700000000.002 openat(AT_FDCWD, \"/tmp/model.pkl\", O_RDONLY) = 3\n",
        default_pid=77,
    )
    assert [event.order for event in trace.syscalls] == [1, 2]
    assert {event.pid for event in trace.syscalls} == {77}
    assert [event.return_value for event in trace.syscalls] == ["0", "3"]
    assert all(event.execution_id == trace.execution_id for event in trace.syscalls)


@pytest.mark.integration
def test_phase6a_simultaneous_executions_have_separate_traces(model_artifacts, tmp_path):
    def execute(index):
        try:
            poc = ModelCage(tmp_path / f"trace-{index}.sqlite")
        except DockerRuntimeError as error:
            pytest.skip(f"Docker daemon unavailable: {error}")
        try:
            result = poc.execute_model(model_artifacts[0], category="text_classification")
        except InstrumentationUnavailable as error:
            pytest.skip(str(error))
        return poc, result

    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = list(executor.map(execute, range(2)))

    execution_ids = {result.execution_id for _, result in completed}
    sandbox_ids = {poc.get_trace(result.execution_id).sandbox_id for poc, result in completed}
    assert len(execution_ids) == 2
    assert len(sandbox_ids) == 2
    for poc, result in completed:
        rows = poc.store.syscall_events(result.execution_id)
        assert rows
        assert {row["execution_id"] for row in rows} == {result.execution_id}


def test_phase6a_malicious_test_artifact_is_pickle_based(model_artifacts):
    assert any(op.name == "REDUCE" for op, _, _ in pickletools.genops(model_artifacts[1].read_bytes()))


@pytest.mark.integration
def test_phase6a_real_runsc_pickle_poc(model_artifacts, tmp_path):
    """Execute both approved test pickles in the controlled Python/strace image."""
    try:
        poc = ModelCage(tmp_path / "live-traces.sqlite")
    except DockerRuntimeError as error:
        pytest.skip(f"Docker daemon unavailable: {error}")

    try:
        benign_result = poc.execute_model(model_artifacts[0], category="text_generation")
    except InstrumentationUnavailable as error:
        pytest.skip(str(error))
    malicious_result = poc.execute_model(model_artifacts[1], category="image_classification")
    classification_result = poc.execute_model(model_artifacts[2], category="text_classification")
    benign_trace = poc.get_trace(benign_result.execution_id)
    malicious_trace = poc.get_trace(malicious_result.execution_id)

    assert benign_result.exit_code == malicious_result.exit_code == 0
    assert benign_trace.syscalls and malicious_trace.syscalls
    assert all(event.pid is not None for event in benign_trace.syscalls)
    assert [event.order for event in benign_trace.syscalls] == sorted(event.order for event in benign_trace.syscalls)
    assert len({event.pid for event in benign_trace.syscalls}) == 1
    assert len({event.pid for event in malicious_trace.syscalls}) == 1
    assert poc.store.syscall_events(benign_result.execution_id)
    assert poc.store.syscall_events(malicious_result.execution_id)
    assert poc.store.syscall_events(classification_result.execution_id)
    for result, category in (
        (benign_result, "text_generation"),
        (malicious_result, "image_classification"),
        (classification_result, "text_classification"),
    ):
        trace = poc.get_trace(result.execution_id)
        raw_events = poc.policy_store.root / "raw" / category / f"{result.execution_id}.jsonl"
        raw_metadata = poc.policy_store.root / "metadata" / f"{result.execution_id}.json"
        assert trace.capture_status == "captured"
        assert all(event.execution_id == result.execution_id for event in trace.syscalls)
        assert all(json.loads(line)["analysis_id"] == result.execution_id for line in raw_events.read_text().splitlines())
        assert json.loads(raw_metadata.read_text())["task_category"] == category
