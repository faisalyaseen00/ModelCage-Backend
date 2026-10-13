"""Durable JSONL storage for raw, normalized instrumentation events."""

import json
import os
from pathlib import Path
import tempfile

from .instrumentation import ModelMetadata, sanitize_arguments
from .telemetry import ExecutionTrace


class RawEventStore:
    """Persist one analysis's syscall stream and metadata without model binaries."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def save(self, trace: ExecutionTrace, *, model: ModelMetadata) -> tuple[Path, Path]:
        """Atomically save events as JSONL and execution metadata as JSON."""
        events_path = self.root / f"{trace.execution_id}.jsonl"
        metadata_path = self.root / f"{trace.execution_id}.metadata.json"
        event_lines = []
        for event in trace.syscalls:
            event_lines.append(
                json.dumps(
                    {
                        "schema_version": 1,
                        "analysis_id": trace.execution_id,
                        "sandbox_id": trace.sandbox_id,
                        "sequence": event.order,
                        "timestamp": event.timestamp,
                        "event_type": "syscall",
                        "syscall_name": event.name,
                        "arguments": sanitize_arguments(event.name, event.arguments),
                        "return_value": event.return_value,
                        "pid": event.pid,
                        "event_source": trace.event_source,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        self._atomic_write(events_path, "\n".join(event_lines) + ("\n" if event_lines else ""))
        metadata = {
            "analysis_id": trace.execution_id,
            "model_sha256": model.model_sha256,
            "model_filename": model.filename,
            "artifact_format": model.artifact_format,
            "category": model.category.value,
            "sandbox_id": trace.sandbox_id,
            "runtime": "runsc",
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "duration_seconds": trace.duration_seconds,
            "exit_code": trace.exit_code,
            "termination_reason": trace.termination_reason,
            "timed_out": trace.timed_out,
            "capture_status": trace.capture_status,
            "capture_reason": trace.capture_reason,
            "failure_reason": trace.error,
            "scope": "direct runner only; file/process syscalls; no descendants",
            "arguments_policy": "strace strings limited to 128 bytes; selected buffers redacted",
            "event_source": trace.event_source,
            "event_count": len(trace.syscalls),
        }
        self._atomic_write(metadata_path, json.dumps(metadata, sort_keys=True, indent=2) + "\n")
        return events_path, metadata_path

    def _atomic_write(self, destination: Path, content: str) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.root, prefix=".pending-", delete=False
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_name = temporary_file.name
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, destination)


class PolicyEventStore:
    """Write-once category artifacts; metadata is the final commit record."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _publish(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', encoding='utf-8', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o400)
            os.link(temporary, path)  # atomic no-overwrite publication, unlike replace()
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def finalize(self, analysis_id, assignment, events, metadata, observations):
        from uuid import UUID
        from .features import ngrams
        from .instrumentation import require_category
        from .categories import VALID_CATEGORY_STATUSES
        if str(UUID(analysis_id)) != analysis_id:
            raise ValueError('Invalid analysis ID')
        valid = assignment.validation_status in VALID_CATEGORY_STATUSES
        category = require_category(assignment.task_category).value if valid else None
        if events and not valid:
            raise ValueError('Unclassified events cannot enter category datasets')
        paths = {}
        if valid:
            previous = 0
            for event in events:
                if event['analysis_id'] != analysis_id or event['sequence'] <= previous or event['event_source'] != 'gvisor_seccheck' or not event['tracepoint'].startswith('syscall/'):
                    raise ValueError('Invalid, mixed or unordered syscall record')
                if event.get('phase') not in {'enter', 'exit'}:
                    raise ValueError('Unknown syscall phase')
                if event['container_id'] != metadata['sandbox_id']:
                    raise ValueError('Container identity mismatch in persisted records')
                previous = event['sequence']
            # Do not create empty dataset entries when instrumentation never initialized.
            if events:
                raw = self.root / 'raw' / category / f'{analysis_id}.jsonl'
                self._publish(raw, ''.join(json.dumps(e, sort_keys=True) + '\n' for e in events))
                features = self.root / 'features' / category / f'{analysis_id}.json'
                self._publish(features, json.dumps({'analysis_id': analysis_id, 'task_category': category,
                    'capture_status': metadata['capture_status'],
                    'global': ngrams(events), 'per_thread': ngrams(events, per_thread=True)}, sort_keys=True))
                paths.update(raw_events=str(raw), features=str(features))
        observation_file = self.root / 'observations' / f'{analysis_id}.jsonl'
        self._publish(observation_file, ''.join(json.dumps(o, sort_keys=True)+'\n' for o in observations))
        paths['observations'] = str(observation_file)
        metadata = dict(metadata, **assignment.to_dict(), artifacts=paths, immutable=True)
        destination = self.root / 'metadata' / f'{analysis_id}.json'
        self._publish(destination, json.dumps(metadata, sort_keys=True, indent=2) + '\n')
        return destination
