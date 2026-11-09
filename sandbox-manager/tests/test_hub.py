"""Hugging Face collection tests with a fake hub client; no network access."""
from collections import OrderedDict
import hashlib
from io import BytesIO
import json
import pickle
import sqlite3
from zipfile import ZipFile

import pytest

from app.artifact_metadata import inspect_pickle
from app.hub import HubCollector, select_checkpoint, validate_hub_model
from app.modelcage import ModelCage
from app.policy import MAX_MODEL_BYTES
from tests.seccheck_fixtures import manager_fixture, ProtocolSession


def torch_archive(seed=0):
    """A torch.save-shaped zip: one top-level <name>/data.pkl plus a storage member."""
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr('archive/data.pkl', pickle.dumps(OrderedDict(weight=[float(seed)])))
        archive.writestr('archive/data/0', b'\0' * 8)
        archive.writestr('archive/version', b'3\n')
    return buffer.getvalue()


def info(repo_id, payload, *, task='text-classification', architecture='BertForSequenceClassification',
         model_type='bert', library='transformers', **extra):
    return {
        'id': repo_id, 'sha': 'a' * 40, 'author': repo_id.split('/')[0], 'private': False, 'gated': False,
        'pipeline_tag': task, 'library_name': library, 'tags': [library, task],
        'config': {'architectures': [architecture], 'model_type': model_type},
        'siblings': [{'rfilename': 'config.json', 'size': 10},
                     {'rfilename': 'pytorch_model.bin', 'size': len(payload),
                      'lfs': {'sha256': hashlib.sha256(payload).hexdigest(), 'size': len(payload)}}],
        **extra,
    }


class FakeHub:
    def __init__(self, models):
        self.models = models  # repo_id -> (info, payload)
        self.downloads = []
        self.info_calls = []

    def list_models(self, category, limit):
        return [{'id': repo_id, 'sha': self.models[repo_id][0]['sha']} for repo_id in self.models][:limit]

    def model_info(self, repo_id):
        self.info_calls.append(repo_id)
        return self.models[repo_id][0]

    def download(self, repo_id, revision, filename, destination, expected_sha256, expected_size):
        self.downloads.append(repo_id)
        payload = self.models[repo_id][1]
        destination.write_bytes(payload)
        value = hashlib.sha256(payload).hexdigest()
        if value != expected_sha256 or len(payload) != expected_size:
            raise ValueError('Downloaded bytes do not match the hub SHA-256/size')
        return value


def hub_collector(tmp_path, models, family_limit=20):
    workspace = tmp_path / 'hub'
    workspace.mkdir()
    manager, runtime = manager_fixture()
    cage = ModelCage(str(workspace / 'modelcage.sqlite'), manager=manager, capture_factory=ProtocolSession)
    client = FakeHub(models)
    return HubCollector(workspace, cage=cage, client=client, family_limit=family_limit), client, runtime


def test_inspector_reads_torch_archive_pickle_without_loading():
    root, diagnostic = inspect_pickle(torch_archive())
    assert diagnostic is None and root == {'weight': [0.0]}
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr('a/data.pkl', b'')
        archive.writestr('b/data.pkl', b'')
    assert inspect_pickle(buffer.getvalue()) == (None, 'torch_archive_without_single_data_pkl')


def test_accepts_matching_metadata_and_records_why():
    payload = torch_archive()
    details, reason = validate_hub_model(info('org/bert-sst2', payload), 'text_classification')
    assert reason is None
    assert details['family'] == 'bert' and details['architecture'] == 'BertForSequenceClassification'
    assert details['source'] == 'huggingface:org/bert-sst2@' + 'a' * 40
    assert 'pipeline_tag=text-classification' in details['category_acceptance']
    assert details['category_evidence']['reviewed_by'] == 'automatic:huggingface-metadata'


@pytest.mark.parametrize('overrides, message', [
    (dict(task='image-classification', architecture='ViTForImageClassification'), 'pipeline tag'),
    (dict(architecture='BertModel'), 'no recognised'),
    (dict(architecture='ViTForImageClassification'), 'Conflicting'),
    (dict(library='timm'), 'not transformers'),
    (dict(gated='auto'), 'gated'),
])
def test_rejects_unrelated_or_inconsistent_models(overrides, message):
    details, reason = validate_hub_model(info('org/model', b'x', **overrides), 'text_classification')
    assert details is None and message in reason


def test_checkpoint_selection_requires_single_bounded_lfs_file():
    assert select_checkpoint(info('o/m', b'abc'))[0]['size'] == 3
    model = info('o/m', b'abc')
    model['siblings'] = [{'rfilename': 'model.safetensors', 'lfs': {'sha256': '0' * 64, 'size': 3}}]
    assert 'no single-file' in select_checkpoint(model)[1]
    model['siblings'].append({'rfilename': 'pytorch_model-00001-of-00002.bin'})
    assert 'sharded' in select_checkpoint(model)[1]
    model['siblings'] = [{'rfilename': 'pytorch_model.bin', 'lfs': {'sha256': '0' * 64, 'size': MAX_MODEL_BYTES + 1}}]
    assert 'above' in select_checkpoint(model)[1]


def test_pipeline_runs_checkpoint_in_sandbox_and_registers_it(tmp_path):
    payload = torch_archive()
    collector, client, runtime = hub_collector(tmp_path, {'org/bert-sst2': (info('org/bert-sst2', payload), payload)})
    results, _ = collector.run_hub(['text_classification'])
    assert [r['status'] for r in results] == ['succeeded']
    command = runtime.exec_monitored.call_args.args[1]
    assert command[-1].endswith('/model.bin')
    with sqlite3.connect(collector.workspace / 'registry.sqlite') as db:
        sha, category, family, status, details = db.execute(
            'SELECT sha256, category, family, status, details FROM models').fetchone()
        hub_row = db.execute('SELECT repo_id, status FROM hub_candidates').fetchone()
    assert (sha, category, family, status) == (hashlib.sha256(payload).hexdigest(), 'text_classification', 'bert', 'succeeded')
    assert json.loads(details)['category_acceptance'].startswith('pipeline_tag=text-classification')
    assert hub_row == ('org/bert-sst2', 'succeeded')


def test_known_sha256_is_skipped_before_download(tmp_path):
    payload = torch_archive()
    models = {'org/original': (info('org/original', payload), payload),
              'org/renamed-copy': (info('org/renamed-copy', payload), payload)}
    collector, client, _ = hub_collector(tmp_path, models)
    results, _ = collector.run_hub(['text_classification'])
    assert [r['status'] for r in results] == ['succeeded', 'duplicate']
    assert client.downloads == ['org/original']


def test_family_quota_is_applied_before_download(tmp_path):
    first, second = torch_archive(1), torch_archive(2)
    models = {'org/bert-a': (info('org/bert-a', first), first), 'org/bert-b': (info('org/bert-b', second), second)}
    collector, client, _ = hub_collector(tmp_path, models, family_limit=1)
    results, _ = collector.run_hub(['text_classification'])
    assert [r['status'] for r in results] == ['succeeded', 'quota_skipped']
    assert client.downloads == ['org/bert-a']


def test_rejected_listing_is_recorded_and_not_downloaded(tmp_path):
    payload = torch_archive()
    vit = info('org/vit', payload, task='image-classification', architecture='ViTForImageClassification', model_type='vit')
    collector, client, _ = hub_collector(tmp_path, {'org/vit': (vit, payload)})
    results, _ = collector.run_hub(['text_classification'])
    assert results[0]['status'] == 'rejected' and 'pipeline tag' in results[0]['reason']
    assert client.downloads == []


def test_tampered_download_fails_closed(tmp_path):
    payload = torch_archive()
    listed = info('org/bert', payload)
    collector, client, runtime = hub_collector(tmp_path, {'org/bert': (listed, torch_archive(9))})
    results, _ = collector.run_hub(['text_classification'])
    assert results[0]['status'] == 'rejected' and 'do not match' in results[0]['reason']
    runtime.exec_monitored.assert_not_called()


def test_stops_after_per_category_successes(tmp_path):
    models = {f'org/m{i}': (info(f'org/m{i}', torch_archive(i), model_type=f'f{i}'), torch_archive(i)) for i in range(3)}
    collector, client, _ = hub_collector(tmp_path, models)
    results, _ = collector.run_hub(['text_classification'], per_category=2)
    assert [r['status'] for r in results] == ['succeeded', 'succeeded']
    assert len(client.downloads) == 2


def test_downloaded_model_is_kept_in_category_folder(tmp_path):
    payload = torch_archive()
    collector, _, _ = hub_collector(tmp_path, {'org/bert-sst2': (info('org/bert-sst2', payload), payload)})
    results, _ = collector.run_hub(['text_classification'])
    folder = collector.workspace / 'models' / 'text_classification' / 'org__bert-sst2'
    assert results[0]['model_folder'] == 'models/text_classification/org__bert-sst2'
    assert (folder / 'pytorch_model.bin').read_bytes() == payload
    assert not (folder / 'pytorch_model.bin').stat().st_mode & 0o222
    record = json.loads((folder / 'model.json').read_text())
    assert record['status'] == 'succeeded' and record['sha256'] == hashlib.sha256(payload).hexdigest()
    assert record['raw_events'] == f"data/raw/text_classification/{results[0]['analysis_id']}.jsonl"


def test_rejected_model_gets_no_folder(tmp_path):
    payload = torch_archive()
    collector, _, _ = hub_collector(tmp_path, {'org/bert': (info('org/bert', payload), torch_archive(9))})
    results, _ = collector.run_hub(['text_classification'])
    assert 'model_folder' not in results[0]
    assert not (collector.workspace / 'models').exists()


def test_rerun_skips_processed_revisions_without_hub_calls(tmp_path):
    payload = torch_archive()
    collector, client, _ = hub_collector(tmp_path, {'org/bert': (info('org/bert', payload), payload)})
    collector.run_hub(['text_classification'])
    client.info_calls.clear()
    results, _ = collector.run_hub(['text_classification'])
    assert results == [] and client.info_calls == [] and client.downloads == ['org/bert']
