import csv
from io import StringIO
import pickle
from threading import Thread
from urllib.request import urlopen

import pytest
import requests

from app.modelcage import ModelCage
from app.web import ModelCageWebServer
from tests.seccheck_fixtures import manager_fixture, ProtocolSession


@pytest.mark.network
def test_upload_automatically_saves_backend_csvs(tmp_path):
    manager, _ = manager_fixture()
    cage = ModelCage(str(tmp_path / 'manual.sqlite'), manager=manager, capture_factory=ProtocolSession)
    uploads = tmp_path / 'uploads'; uploads.mkdir()
    server = ModelCageWebServer(('127.0.0.1', 0), cage, uploads)
    thread = Thread(target=server.serve_forever); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        result = requests.post(base + '/execute', files={'model': ('chosen-name.pkl', pickle.dumps(1))},
                               data={'category': 'text_generation'}, allow_redirects=False, timeout=5)
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
        assert (cage.policy_store.root / 'features' / 'text_generation.csv').is_file()
        rows = list(csv.DictReader((cage.policy_store.root / 'features' / 'text_generation.csv').open()))
        assert len(rows) == 1 and rows[0]['openat'] == '1'
        rows = list(csv.DictReader((cage.policy_store.root / 'metadata.csv').open()))
        assert rows[0]['filename'] == 'chosen-name.pkl'
        assert rows[0]['dataset_eligible'] == 'False'
        page = requests.get(base, timeout=5).text
        assert 'saved automatically in the backend' in page
    finally:
        server.shutdown(); server.server_close(); thread.join()
