import csv
from io import StringIO
import json
import pickle
from uuid import uuid4

import pytest

from app.collection import Collector, validate_candidate
from app.datasets import DatasetBuilder, CATEGORIES, SYSCALLS, PROFILE, DATASET_FIELDS
from app.modelcage import ModelCage
from tests.seccheck_fixtures import manager_fixture, ProtocolSession


def candidate(path, **extra):
    return dict(model_id=path.name, path=str(path), category='text_generation', family='test-family',
        architecture='fixture', source='local-reviewed-fixture', benign_review='Test fixture authored in this test',
        category_evidence={'task': 'text_generation', 'reference': 'fixture manifest', 'reviewed_by': 'test'}, **extra)


def setup_collector(tmp_path, family_limit=20):
    workspace = tmp_path / 'collection'
    workspace.mkdir()
    manager, runtime = manager_fixture()
    cage = ModelCage(str(workspace / 'modelcage.sqlite'), manager=manager, capture_factory=ProtocolSession)
    return Collector(workspace, cage=cage, family_limit=family_limit), runtime


def test_collection_exports_fixed_counts_normalization_and_metadata(tmp_path):
    collector, runtime = setup_collector(tmp_path)
    source = tmp_path / 'model.pkl'
    source.write_bytes(pickle.dumps({'fixture': True}))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([candidate(source)]))
    result, output = collector.run(manifest)
    assert result[0]['status'] == 'succeeded', result
    counts = list(csv.DictReader((output / 'text_generation.csv').open()))
    normalized = list(csv.DictReader((output / 'text_generation.normalized.csv').open()))
    assert len(counts) == 1 and counts[0]['openat'] == '1'
    assert counts[0]['read'] == '0'
    assert sum(float(normalized[0][name]) for name in SYSCALLS) == 1
    assert list(counts[0]) == DATASET_FIELDS
    meta = list(csv.DictReader((output / 'metadata.csv').open()))[0]
    assert meta['filename'] == 'model.pkl'
    assert meta['family'] == 'test-family'
    assert meta['execution_profile'] == PROFILE
    assert meta['dataset_eligible'] == 'True'
    assert len(meta['model_sha256']) == 64
    before = (output / 'text_generation.csv').read_bytes()
    DatasetBuilder(collector.workspace / 'data').build()
    assert before == (output / 'text_generation.csv').read_bytes()
    result, _ = collector.run(manifest)
    assert result[0]['status'] == 'duplicate'
    runtime.exec_monitored.assert_called_once()


def test_reject_unreviewed_and_continue_then_apply_quota(tmp_path):
    collector, runtime = setup_collector(tmp_path, family_limit=1)
    sources = []
    for i in range(3):
        source = tmp_path / f'{i}.pkl'
        source.write_bytes(pickle.dumps(i))
        sources.append(candidate(source))
    sources[0]['benign_review'] = ''
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(sources))
    results, _ = collector.run(manifest)
    assert [r['status'] for r in results] == ['rejected', 'succeeded', 'quota_skipped']
    runtime.exec_monitored.assert_called_once()


@pytest.mark.parametrize('field,value,reason', [
    ('status', 'failed', 'execution_not_successful'),
    ('capture_status', 'partial', 'capture_incomplete'),
    ('reported_losses', None, 'losses_unknown_or_nonzero'),
    ('reported_losses', 1, 'losses_unknown_or_nonzero'),
    ('model_sha256', None, 'hashing_paused_or_missing'),
    ('execution_profile', 'inference-v1', 'unsupported_profile'),
    ('cleanup_status', 'failed', 'cleanup_incomplete'),
])
def test_ineligible_runs_remain_only_in_metadata(tmp_path, field, value, reason):
    collector, _ = setup_collector(tmp_path)
    source = tmp_path / 'model.pkl'; source.write_bytes(pickle.dumps(1))
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps([candidate(source)]))
    results, _ = collector.run(manifest)
    metadata_path = collector.workspace / 'data' / 'metadata' / f'{results[0]["analysis_id"]}.json'
    m = json.loads(metadata_path.read_text()); m[field] = value
    metadata_path.chmod(0o600); metadata_path.write_text(json.dumps(m))
    outputs = DatasetBuilder(collector.workspace / 'data').render()
    assert list(csv.DictReader(StringIO(outputs['text_generation.csv']))) == []
    row = list(csv.DictReader(StringIO(outputs['metadata.csv'])))[0]
    assert reason in row['exclusion_reasons']


def test_category_conflict_and_hash_mismatch_never_execute(tmp_path):
    collector, runtime = setup_collector(tmp_path)
    source = tmp_path / 'model.pkl'; source.write_bytes(pickle.dumps(1))
    bad = candidate(source); bad['category_evidence']['task'] = 'image_classification'
    mismatched = candidate(source, sha256='0' * 64)
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps([bad, mismatched]))
    results, _ = collector.run(manifest)
    assert [r['status'] for r in results] == ['rejected', 'rejected']
    runtime.exec_monitored.assert_not_called()


def test_remote_candidates_are_rejected(tmp_path):
    c = candidate(tmp_path / 'model.pkl', repo_id='owner/model')
    with pytest.raises(ValueError, match='downloads are disabled'):
        validate_candidate(c)


def test_failed_model_recorded_and_retry_is_explicit(tmp_path):
    collector, runtime = setup_collector(tmp_path)
    source = tmp_path / 'model.pkl'; source.write_bytes(pickle.dumps(1))
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps([candidate(source)]))
    runtime.stage_files.side_effect = RuntimeError('staging failed')
    results, _ = collector.run(manifest)
    assert results[0]['status'] == 'failed'
    results, _ = collector.run(manifest)
    assert results[0]['status'] == 'duplicate'
    runtime.stage_files.side_effect = None
    results, output = collector.run(manifest, retry_failed=True)
    assert results[0]['status'] == 'succeeded'
    assert len(list(csv.DictReader((output / 'metadata.csv').open()))) == 2


def test_empty_dataset_has_all_three_headers(tmp_path):
    outputs = DatasetBuilder(tmp_path).render()
    for category in CATEGORIES:
        assert outputs[f'{category}.csv'].splitlines() == [','.join(DATASET_FIELDS)]


def test_manual_upload_with_hashing_paused_still_exports_features(tmp_path):
    manager, runtime = manager_fixture()
    cage = ModelCage(str(tmp_path / 'manual.sqlite'), manager=manager, capture_factory=ProtocolSession)
    source = tmp_path / 'uploaded.pkl'; source.write_bytes(pickle.dumps(1))
    result = cage.execute_model(source, 'text_classification', original_filename='user-model.pkl')
    builder = DatasetBuilder(cage.policy_store.root)
    rows = list(csv.DictReader(StringIO(builder.analysis_csv(result.execution_id))))
    assert rows[0]['openat'] == '1' and rows[0]['connect'] == '0'
    assert list(rows[0]) == DATASET_FIELDS
    normalized = list(csv.DictReader(StringIO(builder.analysis_csv(result.execution_id, normalized=True))))
    assert normalized[0]['openat'] == '1.0'
    outputs = builder.render()
    assert list(csv.DictReader(StringIO(outputs['text_classification.csv']))) == []
    meta = list(csv.DictReader(StringIO(outputs['metadata.csv'])))[0]
    assert meta['filename'] == 'user-model.pkl' and meta['model_sha256'] == ''
    assert 'hashing_paused_or_missing' in meta['exclusion_reasons']


def test_backend_auto_routes_all_categories_and_retains_prior_rows(tmp_path):
    manager, runtime = manager_fixture()
    cage = ModelCage(str(tmp_path / 'manual.sqlite'), manager=manager, capture_factory=ProtocolSession)
    source = tmp_path / 'upload.pkl'; source.write_bytes(pickle.dumps(1))
    for category in CATEGORIES:
        cage.execute_model(source, category)
    cage.execute_model(source, 'text_generation')
    root = cage.policy_store.root
    for category in CATEGORIES:
        rows = list(csv.DictReader((root / 'features' / f'{category}.csv').open()))
        assert len(rows) == (2 if category == 'text_generation' else 1)
        assert len({r['analysis_id'] for r in rows}) == len(rows)
        assert all(row['openat'] == '1' and row['read'] == '0' for row in rows)
    metadata = list(csv.DictReader((root / 'metadata.csv').open()))
    assert len(metadata) == 4
    assert all(m['feature_row_included'] == 'True' for m in metadata)
    assert all(m['dataset_eligible'] == 'False' for m in metadata)
    before = (root / 'features' / 'text_generation.csv').read_bytes()
    DatasetBuilder(root).build()
    assert before == (root / 'features' / 'text_generation.csv').read_bytes()


def test_unreadable_metadata_does_not_overwrite_existing_csv(tmp_path, monkeypatch):
    from pathlib import Path
    root = tmp_path / 'data'
    (root / 'metadata').mkdir(parents=True)
    (root / 'datasets').mkdir()
    target = root / 'datasets' / 'text_generation.csv'
    target.write_text('existing capture frequencies\n')
    real_iterdir = Path.iterdir

    def unreadable(path):
        if path == root / 'metadata':
            raise PermissionError('root-owned directory')
        return real_iterdir(path)

    monkeypatch.setattr(Path, 'iterdir', unreadable)
    with pytest.raises(PermissionError, match='Existing CSVs are unchanged'):
        DatasetBuilder(root).build()
    assert target.read_text() == 'existing capture frequencies\n'


def test_datasets_contains_real_measurement_frequencies_while_hashing_paused(tmp_path):
    manager, runtime = manager_fixture()
    cage = ModelCage(str(tmp_path / 'manual.sqlite'), manager=manager, capture_factory=ProtocolSession)
    source = tmp_path / 'model.pkl'; source.write_bytes(pickle.dumps(1))
    cage.execute_model(source, 'text_generation')
    root = cage.policy_store.root
    rows = list(csv.DictReader((root / 'datasets' / 'text_generation.csv').open()))
    normalized = list(csv.DictReader((root / 'datasets' / 'text_generation.normalized.csv').open()))
    assert len(rows) == len(normalized) == 1
    assert rows[0]['openat'] == '1' and normalized[0]['openat'] == '1.0'
    assert rows[0]['connect'] == '0' and normalized[0]['connect'] == '0.0'
    assert list(csv.DictReader((root / 'training' / 'text_generation.csv').open())) == []


def test_uploaded_family_and_model_columns_match_dataset_layout(tmp_path):
    manager, _ = manager_fixture()
    cage = ModelCage(str(tmp_path / 'manual.sqlite'), manager=manager, capture_factory=ProtocolSession)
    source = tmp_path / 'model.pkl'; source.write_bytes(pickle.dumps(1))
    cage.execute_model(source, 'text_generation', original_filename='example.pkl', manifest={'family': 'GPT'})
    with (cage.policy_store.root / 'datasets' / 'text_generation.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0])[:8] == ['Model', 'Category', 'Family', 'read', 'write', 'openat', 'execve', 'connect']
    assert rows[0]['Model'] == 'example.pkl'
    assert rows[0]['Family'] == 'GPT'
    assert rows[0]['Category'] == 'text_generation'
