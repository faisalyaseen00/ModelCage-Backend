"""Source policy, process attribution, category/feature and evidence tests."""
import json
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.categories import validate_category
from app.features import ngrams
from app.observations import observe, ObservationPolicy
from app.raw_events import PolicyEventStore
from app.seccheck import EventDecoder, select_points, normalized_arguments, InstrumentationUnavailable, CONTEXT
from app.seccheck_proto import syscall_pb2 as syscall, sentry_pb2 as sentry
from tests.seccheck_fixtures import context, packet


@pytest.fixture
def decoder():
    selected = {}
    for num, name in [(257,'openat'), (42,'connect'), (56,'clone'), (59,'execve'), (10,'sysno/10'), (165,'sysno/165')]:
        selected.update({(num,p):f'syscall/{name}/{p}' for p in ['enter','exit']})
    decoder = EventDecoder(str(uuid4()), 'sandbox', ['python3','/tmp/runner.py','/tmp/model.pkl'], selected)
    decoder.armed = True
    decoder.consume(packet(3,sentry.ExecveInfo(context_data=context(),argv=decoder.command)))
    return decoder


def call(decoder, kind, cls, number, *, ctx=None, error=0, result=0, **args):
    ctx = ctx or context()
    decoder.consume(packet(kind,cls(context_data=ctx,sysno=number,**args)))
    decoder.consume(packet(kind,cls(context_data=ctx,sysno=number,exit=syscall.Exit(result=result,errorno=error),**args)))


def test_selection_prefers_schema_but_preserves_other_raw_points():
    names = ['syscall/sysno/257/enter','syscall/sysno/257/exit','syscall/openat/enter','syscall/openat/exit',
             'syscall/sysno/10/enter','syscall/sysno/10/exit','sentry/execve','sentry/clone','sentry/task_exit']
    metadata = '\n'.join(f'Name: {name}, optional fields: [fd_path], context fields: [{"|".join(CONTEXT)}]' for name in names)
    selected, points, available = select_points(metadata+'\nName: remote\n')
    assert available == 6 and len(selected) == 4
    assert selected[257,'enter'] == 'syscall/openat/enter'
    assert selected[10,'enter'] == 'syscall/sysno/10/enter'
    assert all('envv' not in p['optional_fields'] for p in points)


def test_entries_exits_and_duplicate_representations(decoder):
    call(decoder,7,syscall.Open,257,pathname='/tmp/a',result=0)
    decoder.consume(packet(6,syscall.Syscall(context_data=context(),sysno=257)))
    assert len(decoder.events) == 2 and decoder.count == 1
    assert decoder.events[0]['return_value'] is None
    assert decoder.events[1]['return_value'] == 0
    assert decoder.events[1]['entry_sequence'] == 1
    assert decoder.duplicates == 1
    assert decoder.events[1]['duration_ns'] is None


def test_infrastructure_other_sandbox_and_unknown_tree_excluded(decoder):
    call(decoder,7,syscall.Open,257,ctx=context(container='other'))
    call(decoder,7,syscall.Open,257,ctx=context(tid=1,group=1))
    decoder.consume(packet(1,sentry.TaskExit(context_data=context())))
    assert decoder.events == []
    assert 'container_identity_mismatch' in decoder.errors


def test_descendant_processes_and_threads_are_included_with_identity(decoder):
    decoder.consume(packet(2,sentry.CloneInfo(context_data=context(),created_thread_id=43,
         created_thread_group_id=43,created_thread_start_time_ns=200,flags=17)))
    call(decoder,7,syscall.Open,257,ctx=context(tid=43,group=43,parent=42,start=200),pathname='/tmp/child')
    decoder.consume(packet(2,sentry.CloneInfo(context_data=context(),created_thread_id=44,
         created_thread_group_id=42,created_thread_start_time_ns=300,flags=0x10000)))
    ctx=context(tid=44)
    ctx.thread_start_time_ns=300
    call(decoder,7,syscall.Open,257,ctx=ctx,pathname='/tmp/thread')
    # Same numeric PID with a different generation is not the same workload.
    call(decoder,7,syscall.Open,257,ctx=context(tid=43,group=43,start=999),pathname='/tmp/reused')
    assert decoder.count == 2
    assert {e['thread_id'] for e in decoder.events} == {43,44}


def test_loss_invalid_packet_and_unmatched_exit_are_explicit(decoder):
    decoder.consume(packet(7,syscall.Open(context_data=context(),sysno=257,exit=syscall.Exit(errorno=13)),drops=2))
    assert decoder.dropped == 2
    assert decoder.events[0]['correlation'] == 'unmatched_exit'
    with pytest.raises(ValueError): decoder.consume(b'bad')


def test_sensitive_arguments_are_removed_and_raw_pointers_never_decoded():
    message=syscall.Execve(argv=['secret model input'], envv=['TOKEN=secret'],pathname='/tmp/token=secret')
    args=normalized_arguments(message)
    assert 'argv' not in args and 'envv' not in args
    assert 'secret' not in args['pathname']
    raw=normalized_arguments(syscall.Syscall(sysno=165,arg1=123456))
    assert raw['arg1'] == 123456 and 'pathname' not in raw


def test_observations_preserve_denied_attempts_and_evidence(decoder):
    call(decoder,10,syscall.Connect,42,error=101,address=b'\x02\x00\x01\xbb\x08\x08\x08\x08'+b'\x00'*8)
    call(decoder,7,syscall.Open,257,pathname='/etc/shadow',error=13)
    call(decoder,6,syscall.Syscall,165,error=1,arg1=123)
    observations=observe(decoder.events)
    assert {o['reason_code'] for o in observations} >= {'network_attempt','sensitive_path_access','isolation_change_attempt'}
    assert all(o['outcome']=='failed' for o in observations)
    assert observations[0]['priority']=='high'
    assert all(set(o['source_sequences']) <= {e['sequence'] for e in decoder.events} for o in observations)
    assert observe(decoder.events,ObservationPolicy(enabled_groups=())) == []


def test_clone_thread_is_not_process_alert_and_executable_memory_not_intrinsically_bad(decoder):
    call(decoder,23,syscall.Clone,56,flags=0x10000)
    call(decoder,6,syscall.Syscall,10,arg1=100,arg2=4096,arg3=5)
    assert observe(decoder.events) == []
    call(decoder,6,syscall.Syscall,10,arg1=100,arg2=4096,arg3=7)
    assert observe(decoder.events)[0]['reason_code']=='executable_memory_pattern'


@pytest.mark.parametrize('category',['text_generation','text_classification','image_classification'])
def test_category_storage_is_immutable_and_features_repeatable(tmp_path,decoder,category):
    call(decoder,7,syscall.Open,257,pathname='/tmp/file')
    assignment=validate_category(category)
    store=PolicyEventStore(tmp_path)
    metadata={'sandbox_id':'sandbox','capture_status':'captured'}
    store.finalize(decoder.analysis_id,assignment,decoder.events,metadata,observe(decoder.events))
    path=tmp_path/'raw'/category/f'{decoder.analysis_id}.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()]
    assert records == decoder.events
    assert ngrams(records) == ngrams(records)
    assert ngrams(records)['streams']['collector_order']['1'][0]['frequency']==1
    with pytest.raises(FileExistsError):
        store.finalize(decoder.analysis_id,assignment,records,metadata,[])
    assert path.read_text().count('\n') == 2


def test_missing_and_conflicting_category_never_guessed(tmp_path):
    assert validate_category(None).validation_status=='unclassified'
    assignment=validate_category('text_generation',{'task_category':'image_classification'})
    assert assignment.validation_status=='conflict'
    store=PolicyEventStore(tmp_path)
    store.finalize(str(uuid4()),assignment,[],{'sandbox_id':None},[])
    assert not (tmp_path/'raw').exists()


def test_all_ngram_orders_and_thread_separation(decoder):
    for _ in range(5): call(decoder,7,syscall.Open,257)
    for per_thread in (False, True):
        result = ngrams(decoder.events, per_thread=per_thread)
        assert result['schema_version'] == 2
        assert result['experimental'] == [2, 3]
        assert len(result['streams']) == 1
        features = next(iter(result['streams'].values()))
        assert set(features) == {'1', '2', '3'}
        assert [features[str(n)][0]['frequency'] for n in (1, 2, 3)] == [5, 4, 3]
    with pytest.raises(ValueError): ngrams(list(reversed(decoder.events)))


@pytest.mark.network
def test_remote_protocol_handshake_and_shutdown(tmp_path):
    import socket
    import threading
    from app.seccheck import SecCheckSession, RELEASE
    from app.seccheck_proto import common_pb2 as common
    names = ['syscall/openat/enter','syscall/openat/exit','sentry/execve','sentry/clone','sentry/task_exit']
    metadata = '\n'.join(f'Name: {name}, optional fields: [fd_path], context fields: [{"|".join(CONTEXT)}]' for name in names)+'\nName: remote\n'
    session=SecCheckSession(str(uuid4()), 'sandbox', ['python3','/tmp/runner.py'])
    release=threading.Event()
    sent=threading.Event()
    errors=[]
    sender=None
    def control(args):
        nonlocal sender
        if args==['--version']: return 'runsc version '+RELEASE
        if args==['trace','metadata']: return metadata
        if 'state' in args: return json.dumps({'id':'sandbox','status':'running'})
        if 'create' in args:
            config=json.loads(Path(args[args.index('--config')+1]).read_text())
            endpoint=config['trace_session']['sinks'][0]['config']['endpoint']
            def send():
                try:
                    with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as peer:
                        peer.connect(endpoint)
                        peer.sendall(common.Handshake(version=1).SerializeToString())
                        handshake=common.Handshake.FromString(peer.recv(1024))
                        assert handshake.version==1
                        assert release.wait(3)
                        peer.sendall(packet(3,sentry.ExecveInfo(context_data=context(),argv=session.command)))
                        peer.sendall(packet(7,syscall.Open(context_data=context(),sysno=257,pathname='/tmp/file')))
                        peer.sendall(packet(7,syscall.Open(context_data=context(),sysno=257,exit=syscall.Exit(result=3))))
                        sent.set()
                        session.finished.wait(3)
                except Exception as error:
                    errors.append(error)
            sender=threading.Thread(target=send)
            sender.start()
            return ''
        if 'list' in args: return 'Sink: "remote", dropped: 0'
        if 'delete' in args: return ''
        raise AssertionError(args)
    session._run=control
    try:
        session.start()
        session.arm()
        release.set()
        assert sent.wait(3)
        # stop drains queued packets; no artificial events enter the decoder.
        session.stop()
        sender.join(4)
        assert errors==[]
        assert session.summary()['event_count']==1
        assert session.summary()['record_count']==2
        assert session.summary()['capture_errors']==[]
        assert not session.worker.is_alive()
    finally:
        session.stop()


def test_identity_verification_failure_prevents_session_creation():
    from app.seccheck import SecCheckSession, RELEASE
    names=['syscall/openat/enter','syscall/openat/exit','sentry/execve','sentry/clone','sentry/task_exit']
    metadata='\n'.join(f'Name: {n}, optional fields: [], context fields: [{"|".join(CONTEXT)}]' for n in names)+'\nName: remote\n'
    session=SecCheckSession(str(uuid4()),'sandbox',['python3'])
    session._run=Mock(side_effect=[RELEASE,metadata,json.dumps({'id':'wrong','status':'running'})])
    with pytest.raises(InstrumentationUnavailable,match='identity'):
        session.start()
    assert session.listener is None
    session.stop()
