"""Real harmless Docker/gVisor and HTTP validation, never synthetic syscall data."""
import json
import logging
import pickle
from threading import Thread
from time import monotonic, sleep

import pytest
import requests
from docker.errors import NotFound

from app.docker_runtime import DockerRuntimeError
from app.modelcage import ModelCage
from app.web import ModelCageWebServer
import app.manager as manager_module

pytestmark = pytest.mark.integration


class HarmlessDelay:
    def __reduce__(self):
        return sleep, (10,)


def test_live_minimal_seccheck_capture(tmp_path):
    """Run explicitly with authorized runsc access; do not silently skip failure."""
    cage = ModelCage(tmp_path / "probe.sqlite")
    artifact = tmp_path / "probe.pkl"
    artifact.write_bytes(pickle.dumps({"probe": True}))
    result = cage.execute_model(artifact, "text_generation")
    state = cage.events.snapshot(result.execution_id)
    assert state["status"] == "succeeded", state
    assert state["event_count"] > 0, state
    assert state["capture_status"] == "captured", state
    assert state["cleanup_status"] == "destroyed", state


def test_live_collection_dataset(tmp_path):
    """Verify real capture, CSV export and exact dedup as one collection flow."""
    import csv
    from app.collection import Collector
    source = tmp_path / 'benign.pkl'
    source.write_bytes(pickle.dumps({'harmless': True}))
    manifest = tmp_path / 'reviewed.json'
    manifest.write_text(json.dumps([dict(model_id='controlled-fixture', path=str(source),
        category='text_generation', family='test-fixture', architecture='stdlib-dict',
        source='locally generated test fixture', benign_review='Authored harmless dictionary fixture',
        category_evidence={'task': 'text_generation', 'reference': 'test routing declaration only', 'reviewed_by': 'test'})]))
    collector = Collector(tmp_path / 'collection')
    results, output = collector.run(manifest)
    assert results[0]['status'] == 'succeeded', results
    rows = list(csv.DictReader((output / 'text_generation.csv').open()))
    assert len(rows) == 1
    assert sum(int(v) for k, v in rows[0].items() if k not in {'Model', 'Category', 'Family', 'analysis_id'}) > 0
    assert collector.run(manifest)[0][0]['status'] == 'duplicate'


@pytest.mark.parametrize("case", ["timeout", "nonzero"])
def test_live_failure_progress_capture_and_removal(tmp_path, monkeypatch, case, caplog):
    try:
        cage = ModelCage(tmp_path / "live.sqlite")
    except DockerRuntimeError as error:
        pytest.skip(f"Docker unavailable: {error}")
    monkeypatch.setattr(manager_module, "EXECUTION_TIMEOUT_SECONDS", 4)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    server = ModelCageWebServer(("127.0.0.1", 0), cage, uploads)
    thread = Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    container = None
    try:
        with caplog.at_level(logging.INFO, logger="modelcage.lifecycle"):
            payload = pickle.dumps(HarmlessDelay()) if case == "timeout" else b"invalid pickle"
            started = monotonic()
            response = requests.post(base + "/execute", files={"model": ("harmless.pkl", payload)},
                                     data={"category": "text_generation"}, allow_redirects=False, timeout=3)
            assert response.status_code == 303
            analysis = response.headers["Location"].split("analysis=")[1]
            assert monotonic() - started < 3
            observed_running_count = False
            deadline = monotonic() + 25
            while monotonic() < deadline:
                snapshot = requests.get(base + "/analyses/" + analysis, timeout=3).json()
                container = snapshot.get("sandbox_id")
                if snapshot["stage"] != "finalized" and snapshot["event_count"] > 0:
                    observed_running_count = True
                if snapshot["stage"] == "finalized":
                    break
                sleep(.1)
            assert snapshot["stage"] == "finalized"
            if snapshot["status"] == "instrumentation_unavailable":
                assert snapshot["cleanup_status"] == "destroyed"
                assert snapshot["event_count"] == 0
                pytest.skip(snapshot["failure_reason"])
            assert snapshot["status"] == ("timed_out" if case == "timeout" else "failed")
            assert snapshot["cleanup_status"] == "destroyed"
            assert snapshot["storage_status"] == "saved"
            assert snapshot["event_count"] > 0
            if case == "timeout":
                assert observed_running_count
                assert snapshot["capture_status"] == "partial"
            records = [json.loads(line) for line in (cage.policy_store.root / "raw" / "text_generation" / f"{analysis}.jsonl").read_text().splitlines()]
            assert sum(record['phase']=='enter' for record in records) == snapshot["event_count"]
            assert {record["analysis_id"] for record in records} == {analysis}
            assert len({record["process_id"] for record in records}) == 1
            assert records[0]["process_id"] is not None
            with pytest.raises(NotFound):
                cage.manager.runtime.client.containers.get(container)
            # A new connection gets the identical full state, including final cleanup.
            assert requests.get(base + "/analyses/" + analysis, timeout=3).json() == snapshot
            messages = [record.message for record in caplog.records]
            assert all(any(event["message"] in message for message in messages) for event in snapshot["events"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        if container:
            cage.manager.destroy(container)


def test_live_seccheck_unavailable_is_fail_closed_and_cleaned_up(tmp_path, monkeypatch):
    from app.seccheck import InstrumentationUnavailable
    try:
        cage = ModelCage(tmp_path / 'closed.sqlite')
    except DockerRuntimeError as error:
        pytest.skip(str(error))
    artifact = tmp_path / 'benign.pkl'
    artifact.write_bytes(pickle.dumps({'benign': True}))
    analysis = cage.events.begin(category='image_classification')
    # Prove there is no legacy strace fallback on the model execution path.
    def forbidden(*args, **kwargs):
        raise AssertionError('Legacy tracer must never run')
    monkeypatch.setattr(cage.manager.runtime, 'exec', forbidden)
    try:
        cage.execute_model(artifact, 'image_classification', analysis_id=analysis)
    except InstrumentationUnavailable:
        state = cage.events.snapshot(analysis)
        assert state['status'] == 'instrumentation_unavailable'
        assert state['event_count'] == 0
        assert state['cleanup_status'] == 'destroyed'
        metadata = json.loads((cage.policy_store.root/'metadata'/f'{analysis}.json').read_text())
        assert metadata['capture_status'] == 'instrumentation_unavailable'
        assert not (cage.policy_store.root/'raw'/'image_classification'/f'{analysis}.jsonl').exists()
    else:
        assert cage.events.snapshot(analysis)['capture_status'] == 'captured'
