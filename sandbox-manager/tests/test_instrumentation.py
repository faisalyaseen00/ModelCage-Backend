"""Unit tests for Phase 2 raw runtime instrumentation artifacts."""

import json

import pytest

from app.instrumentation import (
    InstrumentationInputError,
    ModelCategory,
    ModelMetadata,
    require_category,
    runner_only_syscalls,
    sanitize_arguments,
)
from app.raw_events import RawEventStore
from app.storage import TraceStore
from app.telemetry import TelemetryCollector


def completed_trace(analysis_id: str):
    collector = TelemetryCollector()
    trace = collector.start("sandbox-a", "sha256", execution_id=analysis_id)
    trace.event_source = "strace_in_sandbox_runner"
    trace.capture_status = "captured"
    collector.add_strace_output(
        trace,
        "1700000000.001 openat(AT_FDCWD, \"/tmp/model.pkl\", O_RDONLY) = 3\n"
        "1700000000.002 write(1, \"model output\", 12) = 12\n",
        default_pid=55,
    )
    collector.finish(trace, exit_code=0, termination_reason="completed")
    return trace


@pytest.mark.parametrize("category", list(ModelCategory))
def test_supported_categories_are_explicit_and_preserved(category):
    assert require_category(category.value) is category


def test_unrelated_category_is_rejected():
    with pytest.raises(InstrumentationInputError, match="Unsupported model category"):
        require_category("speech_recognition")


def test_jsonl_events_are_valid_and_bound_to_one_analysis(tmp_path):
    trace = completed_trace("analysis-one")
    metadata = ModelMetadata("sha256", "model.pkl", ".pkl", ModelCategory.TEXT_GENERATION)
    store = RawEventStore(tmp_path / "raw")
    event_path, metadata_path = store.save(trace, model=metadata)

    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    saved_metadata = json.loads(metadata_path.read_text())
    assert [event["sequence"] for event in events] == [1, 2]
    assert {event["analysis_id"] for event in events} == {"analysis-one"}
    assert {event["pid"] for event in events} == {55}
    assert events[1]["arguments"] == '1, "<redacted>", 12'
    assert saved_metadata["category"] == "text_generation"
    assert saved_metadata["capture_status"] == "captured"


def test_empty_capture_is_reported_without_fabricated_events(tmp_path):
    collector = TelemetryCollector()
    trace = collector.start("sandbox-empty", "sha", execution_id="analysis-empty")
    trace.event_source = "strace_in_sandbox_runner"
    trace.capture_status = "unavailable"
    trace.capture_reason = "strace missing"
    collector.finish(trace, exit_code=0, termination_reason="completed")
    metadata = ModelMetadata("sha", "model.pkl", ".pkl", ModelCategory.IMAGE_CLASSIFICATION)
    events, metadata_file = RawEventStore(tmp_path / "raw").save(trace, model=metadata)

    assert events.read_text() == ""
    assert json.loads(metadata_file.read_text())["capture_status"] == "unavailable"


def test_runner_pid_is_applied_without_reordering_events():
    trace = completed_trace("analysis-pid")
    events = runner_only_syscalls(trace, runner_pid=912)

    assert [event.order for event in events] == [1, 2]
    assert {event.pid for event in events} == {912}
    assert {event.execution_id for event in events} == {"analysis-pid"}


def test_sqlite_retains_capture_metadata(tmp_path):
    trace = completed_trace("analysis-sqlite")
    metadata = ModelMetadata("sha", "model.pickle", ".pickle", ModelCategory.TEXT_CLASSIFICATION)
    store = TraceStore(tmp_path / "trace.sqlite")
    store.save(trace, model=metadata)

    loaded = store.load("analysis-sqlite")
    assert loaded.capture_status == "captured"
    assert loaded.event_source == "strace_in_sandbox_runner"
    assert [event.order for event in loaded.syscalls] == [1, 2]
