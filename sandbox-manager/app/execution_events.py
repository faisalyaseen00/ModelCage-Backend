"""Durable lifecycle events shared by terminal logging and the local interface."""

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from .telemetry import utc_timestamp


logger = logging.getLogger("modelcage.lifecycle")
_TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "pending_review"}


class ExecutionEvents:
    """One ordered journal per analysis; snapshots survive browser/server reconnects."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = RLock()
        self._cached = {}

    def _path(self, analysis_id):
        if str(UUID(analysis_id)) != analysis_id:
            raise ValueError("Invalid analysis ID")
        return self.root / f"{analysis_id}.progress.jsonl"

    def begin(self, *, category=None, analysis_id=None):
        analysis_id = analysis_id or str(uuid4())
        if self._path(analysis_id).exists() or analysis_id in self._cached:
            raise ValueError("Analysis has already started")
        self.publish(analysis_id, "requested", "Analysis requested", category=category,
                     status="running", event_count=0, cleanup_status="not_needed",
                     storage_status="pending", capture_status="not_started")
        return analysis_id

    def publish(self, analysis_id, stage, message, *, level="INFO", **fields):
        with self._lock:
            path = self._path(analysis_id)
            previous = self.snapshot(analysis_id) if path.exists() or analysis_id in self._cached else {"events": []}
            event = dict(analysis_id=analysis_id, sequence=len(previous["events"]) + 1,
                         timestamp=utc_timestamp(), stage=stage, message=message,
                         level=level, **fields)
            try:
                with path.open("a", encoding="utf-8") as stream:
                    path.chmod(0o600)
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError:
                # Observability failure must never bypass sandbox cleanup.
                event["progress_storage_status"] = "failed"
                logger.error("%s [%s] Progress journal unavailable; retaining status in memory",
                             event["timestamp"], analysis_id)
            self._cached[analysis_id] = [*previous["events"], event]
            logger.log(getattr(logging, level), "%s [%s] %s", event["timestamp"], analysis_id, message)
            return event

    def snapshot(self, analysis_id):
        with self._lock:
            events = self._cached.get(analysis_id)
            if events is None:
                events = [json.loads(line) for line in self._path(analysis_id).read_text().splitlines()]
            snapshot = {}
            for event in events:
                snapshot.update(event)
            snapshot["events"] = events
            return deepcopy(snapshot)

    def active_count(self) -> int:
        """Return the number of analyses that have not reached a terminal state."""
        with self._lock:
            analysis_ids = set(self._cached)
            analysis_ids.update(path.name.removesuffix(".progress.jsonl") for path in self.root.glob("*.progress.jsonl"))
            return sum(
                1
                for analysis_id in analysis_ids
                if self.snapshot(analysis_id).get("status") not in _TERMINAL_STATUSES
            )
