"""Phase 6A controlled pickle-execution facade."""

from pathlib import Path
from dataclasses import replace
import json

from .categories import validate_category, CategoryAssignment
from .seccheck import SecCheckSession, InstrumentationUnavailable, CaptureFailure
from .observations import observe
from .raw_events import PolicyEventStore
from .telemetry import SyscallEvent, utc_timestamp

from .docker_runtime import DockerRuntimeError
from .manager import SandboxManager, SandboxExecutionTimeout
from .execution_events import ExecutionEvents
from .instrumentation import (
    InstrumentationInputError,
    ModelCategory,
    ModelMetadata,
    require_category,
)
from .model_runner import RUNNER_SOURCE, loading_failure
from .policy import MODELCAGE_RUNNER_IMAGE
from .quarantine import QuarantineError, QuarantineStore
from .raw_events import RawEventStore
from .storage import TraceStore


class ModelInputError(ValueError):
    """Raised for unsupported or unsafe model input paths."""


class ModelCage:
    """Execute a pickle only through the ModelCage-owned runner command."""

    def __init__(
        self,
        database_path: str = "modelcage.sqlite",
        manager: SandboxManager | None = None,
        quarantine_path: str | Path | None = None,
        raw_events_path: str | Path | None = None,
        capture_factory=None,
        observation_policy=None,
    ):
        self.manager = manager or SandboxManager()
        self.store = TraceStore(database_path)
        database = Path(database_path)
        self.events = ExecutionEvents(database.parent / "data" / "progress")
        self.capture_factory = capture_factory or SecCheckSession
        self.observation_policy = observation_policy
        self.policy_store = PolicyEventStore(Path(raw_events_path).parent if raw_events_path else database.parent / "data")
        self.quarantine = QuarantineStore(
            quarantine_path if quarantine_path is not None else database.parent / "quarantine"
        )
        self.raw_events = RawEventStore(
            raw_events_path if raw_events_path is not None else database.parent / "data" / "raw"
        )

    def execute_model(self, model_path, category=None, *, analysis_id=None, manifest=None,
                      collection_context=None, original_filename=None):
        analysis_id = analysis_id or self.events.begin(category=category)
        if self.events.snapshot(analysis_id)["stage"] != "requested":
            raise ValueError("Analysis has already started")

        def emit(stage, message, **fields):
            return self.events.publish(analysis_id, stage, message, **fields)

        container = trace = session = result = model_metadata = quarantined = None
        failure = None
        status = "failed"
        capture_status = "not_started"
        cleanup_status = "not_needed"
        assignment = CategoryAssignment(None, "none", "unclassified")
        started = utc_timestamp()
        records = []
        summary = {}
        reason = None
        try:
            source = Path(model_path)
            if source.suffix.lower() not in {".pkl", ".pickle"}:
                raise ModelInputError("ModelCage accepts only .pkl and .pickle files.")
            hashing_status = "enabled" if self.quarantine.hashing_enabled else "paused"
            emit("identifying",
                 "Calculating model SHA-256 and quarantining input" if self.quarantine.hashing_enabled
                 else "Hashing paused; quarantining a fresh upload copy",
                 hashing_status=hashing_status)
            try:
                quarantined = self.quarantine.quarantine(source)
            except QuarantineError as error:
                raise ModelInputError(str(error)) from error
            assignment = validate_category(category, manifest)
            emit("category_validated", "Category declarations checked", **assignment.to_dict(),
                 model_sha256=quarantined.sha256,
                 hashing_status="calculated" if quarantined.sha256 is not None else "paused")
            if assignment.validation_status in {"unclassified", "conflict"}:
                status = "pending_review"
                reason = "Select a category" if assignment.task_category is None else "Category conflicts with manifest; review required"
                emit("category_review", reason, level="WARNING")
            else:
                model_metadata = ModelMetadata(quarantined.sha256, quarantined.original_name,
                                               source.suffix.lower(), require_category(assignment.task_category))
                root = f"modelcage/{analysis_id}"
                command = ["python3", f"/tmp/{root}/model_runner.py", f"/tmp/{root}/model.pkl"]
                emit("creating", "Creating disposable sandbox and verifying runsc policy")
                container = self.manager.create(image=MODELCAGE_RUNNER_IMAGE, command=["sleep", "infinity"])
                cleanup_status = "pending"
                emit("runtime_verified", "Docker runsc runtime and isolation policy verified", sandbox_id=container,
                     runtime="runsc", cleanup_status=cleanup_status)
                self.manager.start(container)
                emit("sandbox_started", "Disposable sandbox started")
                self.manager.runtime.stage_files(container,
                    {f"{root}/model.pkl": quarantined.path.read_bytes(), f"{root}/model_runner.py": RUNNER_SOURCE.encode()})
                emit("staged", "Setup finished; starting SecCheck outside the workload")
                session = self.capture_factory(analysis_id, container, command)
                session.start()
                capture_status = "initialized"
                emit("instrumentation_initialized", "SecCheck session and sandbox identity verified",
                     capture_status=capture_status)
                result = self.manager.execute(container, command, model_id=quarantined.sha256,
                    analysis_id=analysis_id, progress=emit, cleanup_on_error=False, instrumentation=session)
                trace = self.manager.get_trace(result.execution_id)
                status = "succeeded" if result.exit_code == 0 else "failed"
                if result.exit_code != 0:
                    reason = loading_failure(result.output, result.exit_code)
                emit("workload_completed" if result.exit_code == 0 else "workload_failed",
                     "Workload process tree finished" if result.exit_code == 0 else reason,
                     exit_code=result.exit_code, failure_reason=reason,
                     level="INFO" if result.exit_code == 0 else "WARNING")
        except Exception as error:
            failure = error
            if isinstance(error, InstrumentationUnavailable):
                status = capture_status = "instrumentation_unavailable"
                reason = str(error)
            else:
                status = "timed_out" if isinstance(error, SandboxExecutionTimeout) else "failed"
                reason = str(error) if isinstance(error, CaptureFailure) else type(error).__name__
            emit("operation_failed", reason, level="ERROR", capture_status=capture_status, failure_reason=reason)
            if container is not None:
                try:
                    trace = self.manager.get_trace(analysis_id)
                except (KeyError, LookupError):
                    pass
                try:
                    self.manager.runtime.terminate_workload(container)
                except Exception:
                    emit("termination_failed", "Unable to stop workload via exec; force removal follows", level="WARNING")
        finally:
            try:
                if session is not None:
                    session.stop()
                    summary = session.summary()
                    records = list(session.decoder.events) if session.decoder else []
                    if capture_status != "instrumentation_unavailable":
                        capture_status = "captured" if records and summary.get("workload_root") else "empty"
                        if (summary.get("capture_errors") or summary.get("reported_losses") or summary.get("unmatched_exits")
                                or summary.get("unexpected_unmatched_entries") or status == "timed_out"):
                            capture_status = "partial" if records else "failed"
                        if capture_status != "captured" and status == "succeeded":
                            status = "incomplete"
                    emit("capture_stopped", "SecCheck collection stopped", capture_status=capture_status,
                         event_count=summary.get("event_count", 0), record_count=len(records),
                         reported_losses=summary.get("reported_losses"), capture_errors=summary.get("capture_errors", []))
            except Exception as error:
                failure = failure or error
                reason = "collector_shutdown_failed"
                status, capture_status = "failed", "failed"
                emit("collector_failed", "Collector shutdown failed", level="ERROR", capture_status=capture_status)
            finally:
                if container is not None:
                    emit("destroying", "Destroying disposable sandbox")
                    try:
                        self.manager.destroy(container)
                        cleanup_status = "destroyed"
                        emit("destroyed", "Disposable sandbox destroyed", cleanup_status=cleanup_status)
                    except Exception as error:
                        failure = failure or error
                        status, cleanup_status = "failed", "failed"
                        reason = "sandbox_removal_failed"
                        emit("cleanup_failed", "Sandbox removal failed; manual cleanup required", level="ERROR", cleanup_status=cleanup_status)
            try:
                observations = observe(records, self.observation_policy)
                for observation in observations[:100]:
                    emit("observation", observation["reason_code"], observation=observation)
                if len(observations) > 100:
                    emit("observations_batched", "Additional observations retained in the observation artifact")
                emit("storing", "Flushing immutable syscall, observation and feature artifacts")
                summary.setdefault("event_count", 0)
                summary.setdefault("record_count", 0)
                summary.setdefault("reported_losses", None)
                metadata = dict(analysis_id=analysis_id, sandbox_id=container, runtime="runsc",
                    filename=original_filename or (quarantined.original_name if quarantined else source.name),
                    artifact_format=source.suffix.lower(), execution_profile="pickle-load-v1",
                    runner_image=MODELCAGE_RUNNER_IMAGE,
                    collection=collection_context or {},
                    runtime_verified=container is not None, seccheck_verified=bool(summary.get("workload_root")),
                    model_sha256=quarantined.sha256 if quarantined else None,
                    hashing_status="calculated" if quarantined and quarantined.sha256 is not None else "paused" if not self.quarantine.hashing_enabled else "not_calculated",
                    start_time=started, end_time=utc_timestamp(), status=status, capture_status=capture_status,
                    exit_code=result.exit_code if result is not None else None,
                    cleanup_status=cleanup_status, failure_reason=reason, observation_count=len(observations), **summary)
                if trace is not None and model_metadata is not None:
                    trace.event_source = "gvisor_seccheck"
                    trace.capture_status = capture_status
                    exits = {e['entry_sequence']: e for e in records if e['entry_sequence'] is not None}
                    trace.syscalls = [SyscallEvent(e['sequence'], str(e['runtime_time_ns']), e['syscall_name'],
                        pid=e['process_id'], arguments=json.dumps(e['arguments'], sort_keys=True),
                        return_value=str(exits[e['sequence']]['return_value']) if e['sequence'] in exits else None,
                        execution_id=analysis_id) for e in records if e['phase'] == 'enter']
                    self.store.save(trace, model=model_metadata)
                self.policy_store.finalize(analysis_id, assignment, records, metadata, observations)
                from .datasets import DatasetBuilder
                DatasetBuilder(self.policy_store.root).build()
                emit("features_stored", "Category syscall feature CSVs and metadata CSV saved",
                     feature_directory=str(self.policy_store.root / 'features'),
                     frequency_directory=str(self.policy_store.root / 'frequencies'),
                     metadata_csv=str(self.policy_store.root / 'metadata.csv'))
                emit("stored", "Immutable artifacts and metadata saved", storage_status="saved",
                     observation_count=len(observations))
            except Exception as error:
                failure = failure or error
                status = "failed"
                emit("storage_failed", "Artifact publication failed", level="ERROR", storage_status="failed")
            emit("finalized", f"Analysis finalized: {status}", status=status, capture_status=capture_status,
                 level="INFO" if status == "succeeded" else "WARNING")
        if failure:
            raise failure
        return replace(result, capture_status=capture_status) if result is not None else None

    def get_trace(self, execution_id: str):
        try:
            return self.store.load(execution_id)
        except KeyError:
            return self.manager.get_trace(execution_id)

    def export_csv(self, execution_id: str) -> str:
        return self.store.export_csv(execution_id)

    def export_xlsx(self, execution_id: str) -> bytes:
        return self.store.export_xlsx(execution_id)


_default_modelcage: ModelCage | None = None


def _default() -> ModelCage:
    global _default_modelcage
    if _default_modelcage is None:
        _default_modelcage = ModelCage()
    return _default_modelcage


def execute_model(model_path: str | Path, category: str | ModelCategory | None = None):
    """Execute one approved pickle through the process-wide default facade."""
    return _default().execute_model(model_path, category)


def get_trace(execution_id: str):
    return _default().get_trace(execution_id)


def export_csv(execution_id: str) -> str:
    return _default().export_csv(execution_id)


def export_xlsx(execution_id: str) -> bytes:
    return _default().export_xlsx(execution_id)
