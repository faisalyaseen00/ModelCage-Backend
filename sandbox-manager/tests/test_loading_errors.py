import json
import csv
import pickle

from app.docker_runtime import DockerExecutionResult
from app.model_runner import loading_failure
from app.modelcage import ModelCage
from tests.seccheck_fixtures import manager_fixture, ProtocolSession


ERROR = "AttributeError: Can't get attribute 'GPTLanguageModel' on <module '__main__' from '/tmp/runner.py'>"


def test_missing_class_reason_is_saved_without_discarding_capture(tmp_path):
    manager, runtime = manager_fixture()
    def execute(*args, instrumentation, **kwargs):
        instrumentation.capture()
        return DockerExecutionResult(1, ERROR)
    runtime.exec_monitored.side_effect = execute
    cage = ModelCage(tmp_path / 'modelcage.sqlite', manager=manager, capture_factory=ProtocolSession)
    source = tmp_path / 'model.pkl'; source.write_bytes(pickle.dumps(1))
    result = cage.execute_model(source, 'text_generation')
    state = cage.events.snapshot(result.execution_id)
    assert state['status'] == 'failed'
    assert state['capture_status'] == 'captured'
    assert state['event_count'] > 0
    assert 'GPTLanguageModel' in state['failure_reason']
    metadata = json.loads((cage.policy_store.root / 'metadata' / f'{result.execution_id}.json').read_text())
    assert metadata['exit_code'] == 1
    assert metadata['failure_reason'] == state['failure_reason']
    report = cage.policy_store.root / 'frequencies' / f'{result.execution_id}.csv'
    with report.open() as stream:
        rows = list(csv.DictReader(stream))
    from app.datasets import SYSCALLS
    assert len(rows) == 1
    assert sum(int(rows[0][name]) for name in SYSCALLS) == state['event_count']
    assert rows[0]['Model'] == 'model.pkl'
    assert rows[0]['Family'] == 'Unknown'
    with (cage.policy_store.root / 'datasets' / 'text_generation.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    with (cage.policy_store.root / 'training' / 'text_generation.csv').open() as stream:
        assert list(csv.DictReader(stream)) == []
    runtime.remove_container.assert_called_once()


def test_loading_diagnostics_are_bounded_and_do_not_echo_arbitrary_output():
    assert "missing dependency 'torch'" in loading_failure("ModuleNotFoundError: No module named 'torch'", 1)
    assert 'SECRET' not in loading_failure('SECRET arbitrary model output', 1)
    assert 'invalid, incompatible' in loading_failure('EOFError: Ran out of input', 1)
