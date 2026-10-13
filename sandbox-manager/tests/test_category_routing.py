"""Automatic classification feeds the correct datasets and preserves review cases."""
import csv
import json
import pickle
import sqlite3

import pytest

from app.datasets import CATEGORIES, DatasetBuilder
from app.docker_runtime import DockerExecutionResult
from app.modelcage import ModelCage
from tests.seccheck_fixtures import manager_fixture, ProtocolSession
from tests.test_categories import HEADS, instance
from tests.test_datasets import candidate, setup_collector


def setup(tmp_path):
    manager, runtime = manager_fixture()
    return ModelCage(tmp_path / 'models.sqlite', manager=manager, capture_factory=ProtocolSession), runtime


def test_no_selection_routes_all_three_categories_and_ngram_outputs(tmp_path):
    cage, runtime = setup(tmp_path)
    source = tmp_path / 'misleading-name-image-classification.pkl'
    for architecture, category in HEADS:
        source.write_bytes(instance('transformers', architecture))
        result = cage.execute_model(source)
        root = cage.policy_store.root
        assert (root / 'raw' / category / f'{result.execution_id}.jsonl').is_file()
        features = json.loads((root / 'features' / category / f'{result.execution_id}.json').read_text())
        assert set(features['global']['streams']['collector_order']) == {'1', '2', '3'}
        with sqlite3.connect(cage.store.database_path) as db:
            assert db.execute('SELECT category FROM model_submissions WHERE analysis_id=?',
                              (result.execution_id,)).fetchone()[0] == category
    for category in CATEGORIES:
        with (root / 'features' / f'{category}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and rows[0]['Category'] == category and rows[0]['openat'] == '1'
        with (root / 'training' / f'{category}.csv').open() as stream:
            assert list(csv.DictReader(stream)) == []  # identification does not certify benignness
    runtime.exec_monitored.assert_called()
    assert runtime.exec_monitored.call_count == 3


@pytest.mark.parametrize('payload,manifest,status', [
    (pickle.dumps({'weights': [1]}), None, 'unclassified'),
    (instance('transformers', 'GPT2LMHeadModel'), {'pipeline_tag': 'image-classification'}, 'conflict'),
    (pickle.dumps({'pipeline_tag': 'audio-classification'}), None, 'unsupported'),
])
def test_unresolved_category_preserves_metadata_but_creates_no_sandbox_or_dataset_row(tmp_path, payload, manifest, status):
    cage, runtime = setup(tmp_path)
    source = tmp_path / 'input.pkl'; source.write_bytes(payload)
    aid = cage.events.begin()
    assert cage.execute_model(source, analysis_id=aid, manifest=manifest) is None
    state = cage.events.snapshot(aid)
    assert state['status'] == 'pending_review' and state['validation_status'] == status
    assert state['task_category'] is None
    runtime.create_container.assert_not_called()
    metadata = json.loads((cage.policy_store.root / 'metadata' / f'{aid}.json').read_text())
    assert metadata['metadata']['category_reason']
    for category in CATEGORIES:
        with (cage.policy_store.root / 'features' / f'{category}.csv').open() as stream:
            assert list(csv.DictReader(stream)) == []


def test_review_then_metadata_resubmission_does_not_create_false_hash_category_conflict(tmp_path):
    cage, runtime = setup(tmp_path)
    source = tmp_path / 'custom.pkl'; source.write_bytes(pickle.dumps({'weights': [1]}))
    assert cage.execute_model(source) is None
    result = cage.execute_model(source, manifest={'config': {'architectures': ['GPT2LMHeadModel']}})
    assert result is not None
    rows = DatasetBuilder(cage.policy_store.root).rows()
    assert all('duplicate_category_conflict' not in reasons for _, _, reasons in rows)
    with (cage.policy_store.root / 'features' / 'text_generation.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_identified_model_loading_failure_keeps_real_capture_in_correct_measurement_dataset(tmp_path):
    cage, runtime = setup(tmp_path)
    source = tmp_path / 'custom.pkl'; source.write_bytes(instance('transformers', 'GPT2LMHeadModel'))
    def execute(*args, instrumentation, **kwargs):
        instrumentation.capture()
        return DockerExecutionResult(1, "ModuleNotFoundError: No module named 'transformers'")
    runtime.exec_monitored.side_effect = execute
    result = cage.execute_model(source)
    state = cage.events.snapshot(result.execution_id)
    assert state['status'] == 'failed' and state['task_category'] == 'text_generation'
    with (cage.policy_store.root / 'features' / 'text_generation.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_batch_collection_accepts_automatic_category_and_retains_dedup(tmp_path):
    collector, runtime = setup_collector(tmp_path)
    source = tmp_path / 'model.pkl'; source.write_bytes(instance('transformers', 'BertForSequenceClassification'))
    item = candidate(source)
    del item['category']
    del item['category_evidence']['task']
    item['architecture'] = 'BertForSequenceClassification'
    manifest = tmp_path / 'batch.json'; manifest.write_text(json.dumps([item]))
    results, output = collector.run(manifest)
    assert results[0]['status'] == 'succeeded'
    with (output / 'text_classification.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    assert collector.run(manifest)[0][0]['status'] == 'duplicate'
    runtime.exec_monitored.assert_called_once()


def test_batch_reviewed_category_cannot_contradict_detected_head(tmp_path):
    collector, runtime = setup_collector(tmp_path)
    source = tmp_path / 'model.pkl'; source.write_bytes(instance('transformers', 'ViTForImageClassification'))
    manifest = tmp_path / 'batch.json'; manifest.write_text(json.dumps([candidate(source)]))
    results, _ = collector.run(manifest)
    assert results[0]['status'] == 'rejected'
    runtime.exec_monitored.assert_not_called()
