"""Host-side quarantine for model bytes that have not yet entered a sandbox."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from uuid import uuid4

from . import policy


class QuarantineError(ValueError):
    """Raised when a submitted model cannot be safely quarantined."""


def normalize_sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise QuarantineError("Expected SHA-256 must contain exactly 64 hexadecimal characters.")
    return value.lower()


def open_regular(path):
    """Open bytes without following a final symlink or blocking on a FIFO."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise QuarantineError("Model input must be an existing regular file and not a symlink.")
        stream = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    return stream


@dataclass(frozen=True)
class QuarantinedModel:
    """Metadata for an immutable ModelCage-owned copy of a submitted model."""

    sha256: str | None
    path: Path
    original_name: str
    size_bytes: int
    reused: bool = False

    def read_verified_bytes(self) -> bytes:
        """Return exactly the bytes to stage, failing closed on quarantine changes."""
        try:
            with open_regular(self.path) as stream:
                payload = stream.read()
            if len(payload) != self.size_bytes or (
                self.sha256 is not None and hashlib.sha256(payload).hexdigest() != self.sha256
            ):
                raise QuarantineError("Quarantined model SHA-256 or size verification failed.")
            return payload
        except OSError as error:
            raise QuarantineError("Unable to verify quarantined model bytes.") from error


class QuarantineStore:
    """Copies submitted bytes into private quarantine, optionally addressed by hash."""

    def __init__(self, root: str | Path, *, hashing_enabled: bool | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.hashing_enabled = policy.MODEL_HASHING_ENABLED if hashing_enabled is None else hashing_enabled

    def quarantine(self, submitted_path: str | Path, *, expected_sha256: str | None = None) -> QuarantinedModel:
        """Copy and optionally hash a regular file without deserializing it on the host."""
        source = Path(submitted_path)
        if source.is_symlink() or not source.is_file():
            raise QuarantineError("Model input must be an existing regular file and not a symlink.")
        if expected_sha256 is not None:
            expected_sha256 = normalize_sha256(expected_sha256)
            if not self.hashing_enabled:
                raise QuarantineError("Expected SHA-256 cannot be verified while hashing is disabled.")

        digest = hashlib.sha256() if self.hashing_enabled else None
        size_bytes = 0
        temporary_name: str | None = None
        try:
            with open_regular(source) as input_file, tempfile.NamedTemporaryFile(
                mode="wb", dir=self.root, prefix=".incoming-", delete=False
            ) as temporary_file:
                temporary_name = temporary_file.name
                while chunk := input_file.read(1024 * 1024):
                    if digest is not None:
                        digest.update(chunk)
                    size_bytes += len(chunk)
                    temporary_file.write(chunk)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            sha256 = digest.hexdigest() if digest is not None else None
            if expected_sha256 is not None and sha256 != expected_sha256:
                raise QuarantineError("Model SHA-256 does not match the supplied expected digest.")
            storage_id = sha256 or f"unhashed-{uuid4()}"
            # The same bytes have one location regardless of .pkl/.pickle alias.
            suffix = ".pkl" if sha256 is not None else source.suffix.lower()
            destination = self.root / f"{storage_id}{suffix}"
            os.chmod(temporary_name, 0o400 if sha256 is not None else 0o600)
            reused = False
            try:
                os.link(temporary_name, destination)  # atomic, never overwrites another submission
            except FileExistsError:
                reused = True
            result = QuarantinedModel(sha256, destination, source.name, size_bytes, reused)
            result.read_verified_bytes()
            self._write_metadata(sha256, source.name, size_bytes, destination.name)
            directory = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return result
        except OSError as error:
            raise QuarantineError(f"Unable to quarantine submitted model: {error}") from error
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _write_metadata(
        self, sha256: str | None, original_name: str, size_bytes: int, quarantined_name: str
    ) -> None:
        metadata = {
            "sha256": sha256,
            "hashing_status": "calculated" if sha256 is not None else "paused",
            "original_name": original_name,
            "size_bytes": size_bytes,
            "quarantined_file": quarantined_name,
            "quarantined_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        destination = self.root / f"{Path(quarantined_name).stem}.json"
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.root, prefix=".metadata-", delete=False
            ) as temporary_file:
                temporary_name = temporary_file.name
                json.dump(metadata, temporary_file, sort_keys=True)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.chmod(temporary_name, 0o400)
            try:
                os.link(temporary_name, destination)
            except FileExistsError:
                pass  # First-copy provenance remains immutable; submissions retain aliases.
        finally:
            if temporary_name is not None:
                os.unlink(temporary_name)
