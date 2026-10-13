"""Minimal SQLite persistence for Phase 6A execution traces."""

import csv
from io import BytesIO, StringIO
import sqlite3
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

from .telemetry import ExecutionTrace, SyscallEvent

if TYPE_CHECKING:
    from .instrumentation import ModelMetadata


class TraceStore:
    """Persist execution summaries and runner-only syscall events in SQLite."""

    def __init__(self, database_path: str):
        self.database_path = database_path
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    sandbox_id TEXT NOT NULL,
                    model_id TEXT,
                    model_filename TEXT,
                    artifact_format TEXT,
                    category TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    exit_code INTEGER,
                    termination_reason TEXT,
                    timed_out INTEGER NOT NULL,
                    error TEXT,
                    event_source TEXT,
                    capture_status TEXT,
                    capture_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS syscall_events (
                    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
                    sequence INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    pid INTEGER,
                    syscall TEXT NOT NULL,
                    arguments TEXT,
                    return_value TEXT,
                    PRIMARY KEY (execution_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    sha256 TEXT PRIMARY KEY,
                    size_bytes INTEGER NOT NULL,
                    quarantined_file TEXT NOT NULL,
                    first_seen TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_submissions (
                    analysis_id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL REFERENCES model_artifacts(sha256),
                    filename TEXT NOT NULL,
                    model_id TEXT,
                    category TEXT,
                    family TEXT,
                    architecture TEXT,
                    source TEXT,
                    status TEXT NOT NULL,
                    capture_status TEXT,
                    submitted_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS model_submissions_sha256 ON model_submissions(sha256);
                """
            )
            existing_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(executions)")
            }
            for name in (
                "model_filename",
                "artifact_format",
                "category",
                "event_source",
                "capture_status",
                "capture_reason",
            ):
                if name not in existing_columns:
                    connection.execute(f"ALTER TABLE executions ADD COLUMN {name} TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def identify(self, model, analysis_id, filename, submitted_at):
        """Persist content identity separately from names, categories and executions."""
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO model_artifacts VALUES (?,?,?,?)",
                (model.sha256, model.size_bytes, str(model.path), submitted_at),
            )
            existing = connection.execute(
                "SELECT size_bytes FROM model_artifacts WHERE sha256=?", (model.sha256,)
            ).fetchone()
            if existing['size_bytes'] != model.size_bytes:
                raise ValueError("SHA-256 registry size mismatch")
            connection.execute(
                """INSERT INTO model_submissions
                   (analysis_id,sha256,filename,status,submitted_at) VALUES (?,?,?,?,?)""",
                (analysis_id, model.sha256, filename, 'quarantined', submitted_at),
            )

    def finish_identification(self, analysis_id, *, assignment, details, status, capture_status, completed_at):
        with self._connect() as connection:
            connection.execute(
                """UPDATE model_submissions SET model_id=?,category=?,family=?,architecture=?,source=?,
                   status=?,capture_status=?,completed_at=? WHERE analysis_id=?""",
                (details.get('model_id'), assignment.task_category, details.get('family'),
                 details.get('architecture'), details.get('source', 'manual_upload'), status,
                 capture_status, completed_at, analysis_id),
            )

    def save(self, trace: ExecutionTrace, *, model: "ModelMetadata | None" = None) -> None:
        with self._connect() as connection:
            connection.execute(
                     """INSERT OR REPLACE INTO executions
                         (execution_id, sandbox_id, model_id, model_filename, artifact_format, category,
                          start_time, end_time, duration_seconds, exit_code, termination_reason, timed_out, error,
                          event_source, capture_status, capture_reason)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace.execution_id,
                    trace.sandbox_id,
                    trace.model_id,
                    model.filename if model else None,
                    model.artifact_format if model else None,
                    model.category.value if model else None,
                    trace.start_time,
                    trace.end_time,
                    trace.duration_seconds,
                    trace.exit_code,
                    trace.termination_reason,
                    int(trace.timed_out),
                    trace.error,
                    trace.event_source,
                    trace.capture_status,
                    trace.capture_reason,
                ),
            )
            connection.execute("DELETE FROM syscall_events WHERE execution_id = ?", (trace.execution_id,))
            connection.executemany(
                     """INSERT INTO syscall_events
                         (execution_id, sequence, timestamp, pid, syscall, arguments, return_value)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        trace.execution_id,
                        event.order,
                        event.timestamp,
                        event.pid,
                        event.name,
                        event.arguments,
                        event.return_value,
                    )
                    for event in trace.syscalls
                ],
            )

    def load(self, execution_id: str) -> ExecutionTrace:
        with self._connect() as connection:
            execution = connection.execute(
                "SELECT * FROM executions WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if execution is None:
                raise KeyError(execution_id)
            events = connection.execute(
                "SELECT * FROM syscall_events WHERE execution_id = ? ORDER BY sequence",
                (execution_id,),
            ).fetchall()
        trace = ExecutionTrace(
            execution_id=execution["execution_id"],
            sandbox_id=execution["sandbox_id"],
            model_id=execution["model_id"],
            start_time=execution["start_time"],
            end_time=execution["end_time"],
            duration_seconds=execution["duration_seconds"],
            exit_code=execution["exit_code"],
            termination_reason=execution["termination_reason"],
            timed_out=bool(execution["timed_out"]),
            error=execution["error"],
            event_source=execution["event_source"],
            capture_status=execution["capture_status"] or "not_requested",
            capture_reason=execution["capture_reason"],
        )
        trace.syscalls = [
            SyscallEvent(
                row["sequence"],
                row["timestamp"],
                row["syscall"],
                pid=row["pid"],
                arguments=row["arguments"],
                return_value=row["return_value"],
                execution_id=execution_id,
            )
            for row in events
        ]
        return trace

    def syscall_events(self, execution_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM syscall_events WHERE execution_id = ? ORDER BY sequence", (execution_id,)
            ).fetchall()

    def export_csv(self, execution_id: str) -> str:
        output = StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["Sequence", "PID", "Timestamp", "Syscall", "Arguments", "Return Value"])
        for event in self.syscall_events(execution_id):
            writer.writerow(
                [event["sequence"], event["pid"], event["timestamp"], event["syscall"], event["arguments"], event["return_value"]]
            )
        return output.getvalue()

    def export_xlsx(self, execution_id: str) -> bytes:
        """Create a minimal standards-compliant XLSX workbook without an ORM/library."""
        rows = [["Sequence", "PID", "Timestamp", "Syscall", "Arguments", "Return Value"]]
        rows.extend([list(row)[1:] for row in self.syscall_events(execution_id)])
        sheet_rows = "".join(
            "<row r=\"%d\">%s</row>"
            % (
                row_index,
                "".join(
                    "<c t=\"inlineStr\"><is><t>%s</t></is></c>" % _xml(value)
                    for value in row
                ),
            )
            for row_index, row in enumerate(rows, start=1)
        )
        output = BytesIO()
        with ZipFile(output, "w", ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
            archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
            archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheets><sheet name="Syscalls" sheetId="1" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></sheets></workbook>')
            archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
            archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>%s</sheetData></worksheet>' % sheet_rows)
        return output.getvalue()


def _xml(value: object) -> str:
    return str("" if value is None else value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
