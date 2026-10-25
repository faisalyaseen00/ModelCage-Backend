"""Strict live SecCheck checks against known, bounded sandbox-only syscalls.

No mock capture and no skips: missing Docker/runsc access is a failed validation.
Fixtures execute only under the normal disposable ModelCage sandbox policy.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import pickle

import pytest
from docker.errors import NotFound

from app.datasets import SYSCALLS
from app.modelcage import ModelCage
from app.seccheck import SecCheckSession, InstrumentationUnavailable

pytestmark = pytest.mark.integration


class SandboxWorkload:
    def __init__(self, source):
        self.source = source

    def __reduce__(self):
        # Serialization only on the host; exec occurs in the sandbox pickle loader.
        return exec, (self.source,)


PRELUDE = '''
import ctypes, json, os, socket, threading
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
def counted_calls(n):
    for _ in range(n):
        assert libc.syscall(121, 0) >= 0  # x86-64 getpgid(0)
'''


def run_workload(directory, body, *, category='text_generation', cage=None):
    directory.mkdir(parents=True, exist_ok=True)
    cage = cage or ModelCage(directory / 'trace.sqlite')
    source = directory / 'fixture.pkl'
    source.write_bytes(pickle.dumps(SandboxWorkload(PRELUDE + body)))
    result = cage.execute_model(source, category)
    state = cage.events.snapshot(result.execution_id)
    metadata = json.loads((cage.policy_store.root / 'metadata' / f'{result.execution_id}.json').read_text())
    records = [json.loads(line) for line in
               (cage.policy_store.root / 'raw' / category / f'{result.execution_id}.jsonl').read_text().splitlines()]
    return cage, result, state, metadata, records


def assert_complete(cage, result, state, metadata, records):
    assert result.exit_code == 0, result.output
    assert state['status'] == 'succeeded', state
    assert state['capture_status'] == 'captured', state
    assert state['cleanup_status'] == 'destroyed'
    assert metadata['reported_losses'] == 0
    assert metadata['capture_errors'] == []
    assert metadata['unmatched_exits'] == metadata['unexpected_unmatched_entries'] == 0
    assert metadata['seccheck_verified'] is True
    assert len(records) == metadata['record_count']
    assert {r['analysis_id'] for r in records} == {result.execution_id}
    assert {r['container_id'] for r in records} == {metadata['sandbox_id']}
    assert all(r['event_source'] == 'gvisor_seccheck' and r['tracepoint'].startswith('syscall/') for r in records)
    assert [r['sequence'] for r in records] == list(range(1, len(records) + 1))
    entries = {r['sequence']: r for r in records if r['phase'] == 'enter'}
    assert len(entries) == state['event_count'] == metadata['event_count']
    for record in records:
        if record['phase'] == 'exit':
            entry = entries[record['entry_sequence']]
            assert (entry['syscall_number'], entry['thread_id'], entry['thread_start_time_ns']) == (
                record['syscall_number'], record['thread_id'], record['thread_start_time_ns'])
    with pytest.raises(NotFound):
        cage.manager.runtime.client.containers.get(metadata['sandbox_id'])
    category = metadata['task_category']
    with (cage.policy_store.root / 'features' / f'{category}.csv').open() as stream:
        row = next(r for r in csv.DictReader(stream) if r['analysis_id'] == result.execution_id)
    counts = Counter(e['syscall_name'] for e in entries.values())
    assert all(int(row[name]) == counts[name] for name in SYSCALLS)
    with (cage.policy_store.root / 'features' / f'{category}.normalized.csv').open() as stream:
        normalized = next(r for r in csv.DictReader(stream) if r['analysis_id'] == result.execution_id)
    assert sum(float(normalized[name]) for name in SYSCALLS) == pytest.approx(1)


@pytest.mark.parametrize('count', [1, 37, 1000])
def test_live_exact_entry_exit_counts_and_repeated_capture(tmp_path, count):
    run = run_workload(tmp_path, f'counted_calls({count})\n')
    assert_complete(*run)
    calls = [r for r in run[-1] if r['syscall_name'] == 'getpgid']
    assert Counter(r['phase'] for r in calls) == {'enter': count, 'exit': count}
    assert all(r['return_value'] >= 0 and r['error_status'] == 0 for r in calls if r['phase'] == 'exit')


def test_live_file_operations_and_denied_network_are_observed(tmp_path):
    run = run_workload(tmp_path, '''
path = b'/tmp/modelcage-live-coverage'
fd = libc.syscall(257, -100, ctypes.c_char_p(path), os.O_CREAT | os.O_RDWR, 0o600)
assert fd >= 0
assert libc.syscall(1, fd, ctypes.c_char_p(b'probe'), 5) == 5
assert libc.syscall(8, fd, 0, 0) == 0
buffer = ctypes.create_string_buffer(5)
assert libc.syscall(0, fd, buffer, 5) == 5
assert buffer.raw == b'probe'
assert libc.syscall(3, fd) == 0
assert libc.syscall(257, -100, ctypes.c_char_p(b'/modelcage-denied-write'), os.O_CREAT | os.O_WRONLY, 0o600) == -1
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.2)
try:
    sock.connect(('127.0.0.1', 9))
except OSError:
    pass
else:
    raise AssertionError('Unexpected listener inside offline disposable sandbox')
sock.close()
''')
    assert_complete(*run)
    records = run[-1]
    exits = {r['entry_sequence']: r for r in records if r['phase'] == 'exit'}
    opened = [r for r in records if r['phase'] == 'enter' and r['syscall_name'] == 'openat'
              and r['arguments'].get('pathname') == '/tmp/modelcage-live-coverage']
    assert len(opened) == 1 and exits[opened[0]['sequence']]['return_value'] >= 0
    fd = exits[opened[0]['sequence']]['return_value']
    for name, result in [('write', 5), ('read', 5), ('lseek', 0), ('close', 0)]:
        matching = [r for r in records if r['phase'] == 'enter' and r['syscall_name'] == name
                    and r['arguments'].get('fd', r['arguments'].get('arg1')) == fd
                    and r['sequence'] > opened[0]['sequence']
                    and exits.get(r['sequence'], {}).get('return_value') == result]
        assert matching, name
    denied = [r for r in records if r['phase'] == 'enter'
              and r['arguments'].get('pathname') == '/modelcage-denied-write']
    assert denied and all(exits[r['sequence']]['error_status'] != 0 for r in denied)
    assert any(r['syscall_name'] == 'connect' and r['phase'] == 'exit' and r['error_status'] != 0 for r in records)


def test_live_fork_and_thread_attribution(tmp_path):
    run = run_workload(tmp_path, '''
counted_calls(7)
thread = threading.Thread(target=counted_calls, args=(11,))
thread.start()
thread.join()
child = os.fork()
if child == 0:
    counted_calls(13)
    os._exit(0)
_, status = os.waitpid(child, 0)
assert status == 0
''')
    assert_complete(*run)
    metadata, records = run[-2:]
    counts = Counter((r['process_id'], r['thread_id']) for r in records
                     if r['phase'] == 'enter' and r['syscall_name'] == 'getpgid')
    assert sorted(counts.values()) == [7, 11, 13]
    assert len({pid for pid, _ in counts}) == 2
    assert len({tid for _, tid in counts}) == 3
    assert len(metadata['process_groups']) == 2


def test_live_parallel_sandboxes_never_mix_records(tmp_path):
    def execute(index):
        return run_workload(tmp_path / str(index), f'counted_calls({17 + index})\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(execute, range(2)))
    assert len({run[3]['sandbox_id'] for run in runs}) == 2
    for index, run in enumerate(runs):
        assert_complete(*run)
        assert sum(r['phase'] == 'enter' and r['syscall_name'] == 'getpgid' for r in run[-1]) == 17 + index


@pytest.mark.parametrize('category', ['text_generation', 'text_classification', 'image_classification'])
def test_live_automatic_category_dataset_routing(tmp_path, category):
    # Controlled metadata fixture, not a claim of representative ML model coverage.
    cage = ModelCage(tmp_path / 'trace.sqlite')
    source = tmp_path / 'fixture.pkl'
    source.write_bytes(pickle.dumps({'pipeline_tag': category.replace('_', '-'), 'weights': [0.1, 0.2]}))
    result = cage.execute_model(source)
    state = cage.events.snapshot(result.execution_id)
    metadata = json.loads((cage.policy_store.root / 'metadata' / f'{result.execution_id}.json').read_text())
    records = [json.loads(line) for line in (cage.policy_store.root / 'raw' / category / f'{result.execution_id}.jsonl').read_text().splitlines()]
    assert_complete(cage, result, state, metadata, records)
    assert state['validation_status'] == 'auto_identified'


def test_live_capture_limit_is_partial_and_excluded(tmp_path):
    class LimitedSession(SecCheckSession):
        def start(self):
            super().start()
            self.decoder.max_records = 100  # exercise a real receiver overflow with bounded work
    cage = ModelCage(tmp_path / 'trace.sqlite', capture_factory=LimitedSession)
    source = tmp_path / 'fixture.pkl'
    source.write_bytes(pickle.dumps(SandboxWorkload(PRELUDE + 'counted_calls(1000)\n')))
    aid = cage.events.begin(category='text_generation')
    try:
        cage.execute_model(source, 'text_generation', analysis_id=aid)
    except InstrumentationUnavailable:
        raise  # Missing live access must not accidentally satisfy a failure-path test.
    except Exception:
        pass
    state = cage.events.snapshot(aid)
    assert state['capture_status'] in {'partial', 'failed'}, state
    assert state['cleanup_status'] == 'destroyed'
    assert state['capture_errors']
    with (cage.policy_store.root / 'features' / 'text_generation.csv').open() as stream:
        assert list(csv.DictReader(stream)) == []
    with pytest.raises(NotFound):
        cage.manager.runtime.client.containers.get(state['sandbox_id'])
