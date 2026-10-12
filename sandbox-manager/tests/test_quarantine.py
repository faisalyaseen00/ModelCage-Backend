"""Regression coverage for the reversible development hashing pause."""
import hashlib
import json
import stat

from app.quarantine import QuarantineStore


def test_paused_hashing_copies_without_hashing_or_reusing_old_file(tmp_path, monkeypatch):
    source = tmp_path / "input.pkl"
    source.write_bytes(b"model bytes")
    root = tmp_path / "quarantine"
    store = QuarantineStore(root, hashing_enabled=False)
    old = root / f"{hashlib.sha256(source.read_bytes()).hexdigest()}.pkl"
    old.write_bytes(source.read_bytes())
    old.chmod(0o000)

    def forbidden_hash(*args, **kwargs):
        raise AssertionError("Paused hashing must not calculate a digest")

    monkeypatch.setattr("app.quarantine.hashlib.sha256", forbidden_hash)
    try:
        first = store.quarantine(source)
        second = store.quarantine(source)
        assert first.sha256 is second.sha256 is None
        assert first.path != second.path
        assert first.path.name.startswith("unhashed-")
        assert first.path.read_bytes() == second.path.read_bytes() == source.read_bytes()
        assert stat.S_IMODE(first.path.stat().st_mode) == 0o600
        assert old.exists() and stat.S_IMODE(old.stat().st_mode) == 0
        metadata = json.loads(first.path.with_suffix(".json").read_text())
        assert metadata["sha256"] is None
        assert metadata["hashing_status"] == "paused"
    finally:
        old.chmod(0o600)
    assert old.read_bytes() == source.read_bytes()


def test_hashing_can_be_reenabled(tmp_path):
    source = tmp_path / "input.pkl"
    source.write_bytes(b"model bytes")
    store = QuarantineStore(tmp_path / "quarantine", hashing_enabled=True)
    first = store.quarantine(source)
    second = store.quarantine(source)
    assert first.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert first.path == second.path
    assert first.path.name == f"{first.sha256}.pkl"
    assert json.loads(first.path.with_suffix(".json").read_text())["hashing_status"] == "calculated"
