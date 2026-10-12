"""Synthetic protobuf fixtures for unit tests; never integration evidence."""
import struct
from unittest.mock import Mock
from uuid import uuid4

from app.seccheck import EventDecoder, SOURCE
from app.seccheck_proto import common_pb2 as common, syscall_pb2 as syscall, sentry_pb2 as sentry
from app.docker_runtime import DockerExecutionResult
from app.manager import SandboxManager
from tests.test_sandbox import policy_host_config


def context(container='sandbox', tid=42, group=42, parent=1, start=100):
    return common.ContextData(container_id=container, time_ns=1700000000000000000,
        thread_id=tid, thread_start_time_ns=start, thread_group_id=group,
        thread_group_start_time_ns=start, parent_thread_group_id=parent,
        process_name='python3', is_exec_session=True)


def packet(kind, message, drops=0):
    return struct.pack('<HHI', 8, kind, drops) + message.SerializeToString()


class ProtocolSession:
    def __init__(self, analysis_id, container, command):
        points = {(257, phase): f'syscall/openat/{phase}' for phase in ['enter','exit']}
        self.decoder = EventDecoder(analysis_id, container, command, points)
        self.closed = False

    def start(self):
        pass

    def arm(self):
        self.decoder.armed = True

    def capture(self):
        self.arm()
        ctx = context(container=self.decoder.container)
        self.decoder.consume(packet(3, sentry.ExecveInfo(context_data=ctx, argv=self.decoder.command)))
        self.decoder.consume(packet(7, syscall.Open(context_data=ctx, sysno=257, pathname='/tmp/model.pkl', flags=0)))
        self.decoder.consume(packet(7, syscall.Open(context_data=ctx, sysno=257, pathname='/tmp/model.pkl', flags=0,
                                                   exit=syscall.Exit(result=3))))
        self.decoder.consume(packet(5, sentry.TaskExit(context_data=ctx)))

    def progress(self):
        return self.decoder.count

    def stop(self):
        self.closed = True

    def summary(self):
        return dict(event_count=self.decoder.count, record_count=len(self.decoder.events),
                    reported_losses=self.decoder.dropped, capture_errors=list(self.decoder.errors),
                    workload_root=list(self.decoder.root_group) if self.decoder.root_group else None)


def manager_fixture():
    runtime = Mock()
    runtime.create_container.side_effect = lambda **kw: {'Id': str(uuid4())}
    runtime.inspect.return_value = {'HostConfig': policy_host_config()}
    def execute(*args, instrumentation, progress=None, **kwargs):
        instrumentation.capture()
        if progress:
            progress('workload_started', 'Controlled harness process observed by SecCheck')
            progress('capturing', 'Syscall entries received: 1', event_count=1)
        return DockerExecutionResult(0, '', capture_status='captured')
    runtime.exec_monitored.side_effect = execute
    return SandboxManager(runtime=runtime), runtime
