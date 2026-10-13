"""Identity must describe staged bytes across persistence, failure and resubmission."""
import csv
import hashlib
import json
import pickle
import sqlite3

import pytest

from app.modelcage import ModelCage, ModelInputError
from app.quarantine import QuarantineError
from tests.seccheck_fixtures import ProtocolSession, manager_fixture


def setup(tmp_path):
    manager, runtime = manager_fixture()
    cage = ModelCage(tmp_path / 'traces.sqlite', manager=manager, capture_factory=ProtocolSession)
    source = tmp_path / 'original.pkl'
    source.write_bytes(pickle.dumps({'fixture': True}))
    return cage, runtime, source


def test_identity_matches_staged_bytes_registry_trace_and_csv_after_restart(tmp_path):
    cage, runtime, source = setup(tmp_path)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    result = cage.execute_model(source, 'text_generation', original_filename='user-name.pkl',
                                manifest={'model_id': 'model-one', 'family': 'test-family',
                                          'architecture': 'fixture', 'source': 'local test'})
    staged = runtime.stage_files.call_args.args[1]
    payload = next(value for name, value in staged.items() if name.endswith('/model.pkl'))
    assert hashlib.sha256(payload).hexdigest() == expected
    restarted = ModelCage(cage.store.database_path, manager=cage.manager, capture_factory=ProtocolSession)
    assert restarted.get_trace(result.execution_id).model_id == expected
    meta = json.loads((cage.policy_store.root / 'metadata' / f'{result.execution_id}.json').read_text())
    assert meta['model_sha256'] == expected and meta['identity_verified'] is True
    assert meta['hash_algorithm'] == 'sha256' and meta['size_bytes'] == len(payload)
    with (cage.policy_store.root / 'metadata.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert row['model_sha256'] == expected and row['hashing_status'] == 'calculated'
    assert row['identity_verified'] == 'True'
    assert row['family'] == 'test-family' and row['architecture'] == 'fixture' and row['source'] == 'local test'
    with sqlite3.connect(cage.store.database_path) as db:
        assert db.execute('SELECT sha256,size_bytes FROM model_artifacts').fetchall() == [(expected, len(payload))]
        assert db.execute('SELECT sha256,filename,model_id,family,status FROM model_submissions').fetchall() == [
            (expected, 'user-name.pkl', 'model-one', 'test-family', 'succeeded')]
        assert db.execute('SELECT model_id,model_filename FROM executions').fetchone() == (expected, 'user-name.pkl')


def test_manual_rescan_reuses_identity_but_has_fresh_execution_and_alias_history(tmp_path):
    cage, runtime, source = setup(tmp_path)
    first = cage.execute_model(source, 'text_generation')
    renamed = tmp_path / 'renamed.pickle'; renamed.write_bytes(source.read_bytes())
    second = cage.execute_model(renamed, 'text_generation')
    assert first.execution_id != second.execution_id
    assert cage.get_trace(first.execution_id).sandbox_id != cage.get_trace(second.execution_id).sandbox_id
    assert cage.events.snapshot(second.execution_id)['quarantine_reused'] is True
    with sqlite3.connect(cage.store.database_path) as db:
        assert db.execute('SELECT count(*) FROM model_artifacts').fetchone()[0] == 1
        assert {r[0] for r in db.execute('SELECT filename FROM model_submissions')} == {'original.pkl', 'renamed.pickle'}
    assert runtime.exec_monitored.call_count == 2


@pytest.mark.parametrize('outcome', ['failed', 'pending_review'])
def test_identity_survives_failure_and_category_review(tmp_path, outcome):
    cage, runtime, source = setup(tmp_path)
    aid = cage.events.begin(category='text_generation')
    if outcome == 'failed':
        runtime.stage_files.side_effect = RuntimeError('staging failed')
        with pytest.raises(RuntimeError):
            cage.execute_model(source, 'text_generation', analysis_id=aid)
    else:
        cage.execute_model(source, analysis_id=aid)
        runtime.create_container.assert_not_called()
    with sqlite3.connect(cage.store.database_path) as db:
        assert db.execute('SELECT sha256,status FROM model_submissions WHERE analysis_id=?', (aid,)).fetchone() == (
            hashlib.sha256(source.read_bytes()).hexdigest(), outcome)


def test_manifest_hash_mismatch_prevents_sandbox_creation(tmp_path):
    cage, runtime, source = setup(tmp_path)
    with pytest.raises(ModelInputError, match='does not match'):
        cage.execute_model(source, 'text_generation', manifest={'sha256': '0' * 64})
    runtime.create_container.assert_not_called()


def test_quarantine_change_before_staging_fails_closed(tmp_path, monkeypatch):
    cage, runtime, source = setup(tmp_path)
    identify = cage.store.identify
    def corrupt(model, *args):
        identify(model, *args)
        model.path.chmod(0o600)
        model.path.write_bytes(b'changed after quarantine')
    monkeypatch.setattr(cage.store, 'identify', corrupt)
    with pytest.raises(QuarantineError, match='verification failed'):
        cage.execute_model(source, 'text_generation')
    runtime.create_container.assert_not_called()
    with sqlite3.connect(cage.store.database_path) as db:
        assert db.execute('SELECT status FROM model_submissions').fetchone()[0] == 'failed'
