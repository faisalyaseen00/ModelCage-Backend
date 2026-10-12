"""Host-side quarantine for model bytes that have not yet entered a sandbox."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from . import policy


class QuarantineError(ValueError):
    """Raised when a submitted model cannot be safely quarantined."""


@dataclass(frozen=True)
class QuarantinedModel:
    """Metadata for an immutable ModelCage-owned copy of a submitted model."""

    sha256: str | None
    path: Path
    original_name: str
    size_bytes: int


class QuarantineStore:
    """Copies submitted bytes into private quarantine, optionally addressed by hash."""

    def __init__(self, root: str | Path, *, hashing_enabled: bool | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.hashing_enabled = policy.MODEL_HASHING_ENABLED if hashing_enabled is None else hashing_enabled

    def quarantine(self, submitted_path: str | Path) -> QuarantinedModel:
        """Copy and optionally hash a regular file without deserializing it on the host."""
        source = Path(submitted_path)
        if source.is_symlink() or not source.is_file():
            raise QuarantineError("Model input must be an existing regular file and not a symlink.")

        digest = hashlib.sha256() if self.hashing_enabled else None
        size_bytes = 0
        temporary_name: str | None = None
        try:
            with source.open("rb") as input_file, tempfile.NamedTemporaryFile(
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
            storage_id = sha256 or f"unhashed-{uuid4()}"
            destination = self.root / f"{storage_id}{source.suffix.lower()}"
            os.chmod(temporary_name, 0o600)
            if destination.exists():
                os.unlink(temporary_name)
            else:
                os.replace(temporary_name, destination)
            self._write_metadata(sha256, source.name, size_bytes, destination.name)
            return QuarantinedModel(sha256, destination, source.name, size_bytes)
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
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.root, prefix=".metadata-", delete=False
        ) as temporary_file:
            json.dump(metadata, temporary_file, sort_keys=True)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_name = temporary_file.name
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, destination)
