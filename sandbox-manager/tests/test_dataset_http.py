import csv
import hashlib
from io import StringIO
import pickle
from threading import Thread
from urllib.request import urlopen

import pytest
import requests

from app.modelcage import ModelCage
from app.web import ModelCageWebServer
from tests.seccheck_fixtures import manager_fixture, ProtocolSession
from tests.test_categories import HEADS


@pytest.mark.network
@pytest.mark.parametrize('architecture,category', HEADS)
def test_upload_automatically_saves_backend_csvs(tmp_path, architecture, category):
    manager, _ = manager_fixture()
    cage = ModelCage(str(tmp_path / 'manual.sqlite'), manager=manager, capture_factory=ProtocolSession)
    uploads = tmp_path / 'uploads'; uploads.mkdir()
    server = ModelCageWebServer(('127.0.0.1', 0), cage, uploads)
    thread = Thread(target=server.serve_forever); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    payload = pickle.dumps({'config': {'architectures': [architecture]}})
    try:
        result = requests.post(base + '/execute', files={'model': ('chosen-name.pkl', payload)},
                               allow_redirects=False, timeout=5)
        assert result.status_code == 303
        aid = result.headers['Location'].split('analysis=')[1]
        from time import monotonic, sleep
        deadline = monotonic() + 5
        while monotonic() < deadline:
            state = requests.get(base + '/analyses/' + aid, timeout=2).json()
            if state['stage'] == 'finalized':
                break
            sleep(.02)
        assert state['status'] == 'succeeded'
        assert state['model_sha256'] == hashlib.sha256(payload).hexdigest()
        assert state['identity_verified'] is True
        assert state['task_category'] == category
        assert state['validation_status'] == 'auto_identified'
        rows = list(csv.DictReader((cage.policy_store.root / 'features' / f'{category}.csv').open()))
        assert len(rows) == 1 and rows[0]['openat'] == '1'
        normalized = list(csv.DictReader((cage.policy_store.root / 'features' / f'{category}.normalized.csv').open()))
        assert normalized[0]['openat'] == '1.0'
        rows = list(csv.DictReader((cage.policy_store.root / 'metadata.csv').open()))
        assert rows[0]['filename'] == 'chosen-name.pkl'
        assert rows[0]['model_sha256'] == state['model_sha256']
        assert rows[0]['dataset_eligible'] == 'False'
        assert rows[0]['category_source'] == 'artifact_metadata'
        page = requests.get(base, timeout=5).text
        assert 'saved automatically in the backend' in page
        assert 'name="category"' not in page
    finally:
        server.shutdown(); server.server_close(); thread.join()
