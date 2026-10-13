"""Byte identity, concurrent publication, integrity and legacy-pause coverage."""
import hashlib
import json
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.quarantine import QuarantineError, QuarantineStore


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


@pytest.mark.parametrize('payload,expected', [
    (b'', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    (b'abc', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'),
])
def test_default_hash_matches_known_sha256_vectors(tmp_path, payload, expected):
    source = tmp_path / 'model.pkl'; source.write_bytes(payload)
    result = QuarantineStore(tmp_path / 'quarantine').quarantine(source)
    assert result.sha256 == expected
    assert result.size_bytes == len(payload)
    assert result.read_verified_bytes() == payload
    assert stat.S_IMODE(result.path.stat().st_mode) == 0o400


def test_concurrent_renamed_artifacts_share_verified_bytes_and_first_metadata(tmp_path):
    payload = b'abc123' * 400000  # spans multiple streaming hash chunks
    sources = [tmp_path / 'one.pkl', tmp_path / 'two.pickle']
    for source in sources:
        source.write_bytes(payload)
    store = QuarantineStore(tmp_path / 'quarantine')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(store.quarantine, sources))
    assert {r.sha256 for r in results} == {hashlib.sha256(payload).hexdigest()}
    assert len({r.path for r in results}) == 1
    assert sorted(r.reused for r in results) == [False, True]
    metadata_path = results[0].path.with_suffix('.json')
    metadata_before = metadata_path.read_bytes()
    assert store.quarantine(sources[0]).read_verified_bytes() == payload
    assert metadata_path.read_bytes() == metadata_before
    assert not list(store.root.glob('.incoming-*'))
    assert not list(store.root.glob('.metadata-*'))


def test_same_filename_changed_content_has_new_identity(tmp_path):
    source = tmp_path / 'model.pkl'; source.write_bytes(b'first')
    store = QuarantineStore(tmp_path / 'quarantine')
    first = store.quarantine(source)
    source.write_bytes(b'second')
    second = store.quarantine(source)
    assert first.sha256 != second.sha256
    assert first.read_verified_bytes() == b'first'
    assert second.read_verified_bytes() == b'second'


@pytest.mark.parametrize('replacement', ['corrupted', 'symlink'])
def test_existing_hash_named_file_is_verified_not_trusted(tmp_path, replacement):
    source = tmp_path / 'model.pkl'; source.write_bytes(b'original')
    store = QuarantineStore(tmp_path / 'quarantine')
    first = store.quarantine(source)
    first.path.unlink()
    if replacement == 'corrupted':
        first.path.write_bytes(b'modified')
    else:
        first.path.symlink_to(source)
    with pytest.raises(QuarantineError):
        store.quarantine(source)
    with pytest.raises(QuarantineError):
        first.read_verified_bytes()
    assert not list(store.root.glob('.incoming-*'))


@pytest.mark.parametrize('expected', ['0' * 64, 'invalid', 123])
def test_expected_digest_failure_leaves_no_quarantine_artifact(tmp_path, expected):
    source = tmp_path / 'model.pkl'; source.write_bytes(b'abc')
    store = QuarantineStore(tmp_path / 'quarantine')
    with pytest.raises(QuarantineError):
        store.quarantine(source, expected_sha256=expected)
    assert list(store.root.iterdir()) == []


def test_expected_digest_accepts_uppercase_hex(tmp_path):
    source = tmp_path / 'model.pkl'; source.write_bytes(b'abc')
    digest = hashlib.sha256(b'abc').hexdigest()
    result = QuarantineStore(tmp_path / 'quarantine').quarantine(source, expected_sha256=digest.upper())
    assert result.sha256 == digest
