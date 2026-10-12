"""Per-sandbox gVisor SecCheck remote receiver. No host tracing or strace fallback."""
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import socket
import struct
import subprocess
import tempfile
import threading
import time

from .seccheck_proto import common_pb2 as common, syscall_pb2 as syscall, sentry_pb2 as sentry
from .syscall_names import NAMES

RELEASE = "release-20260907.0"
SOURCE = "gvisor_seccheck"
CONTEXT = ["time", "thread_id", "task_start_time", "group_id", "thread_group_start_time",
           "container_id", "process_name", "parent_thread_group_id", "is_exec_session"]
TYPES = {6: syscall.Syscall, 7: syscall.Open, 8: syscall.Close, 9: syscall.Read,
         10: syscall.Connect, 11: syscall.Execve, 12: syscall.Socket, 13: syscall.Chdir,
         14: syscall.Setid, 15: syscall.Setresid, 16: syscall.Prlimit, 17: syscall.Pipe,
         18: syscall.Fcntl, 19: syscall.Dup, 20: syscall.Signalfd, 21: syscall.Chroot,
         22: syscall.Eventfd, 23: syscall.Clone, 24: syscall.Bind, 25: syscall.Accept,
         26: syscall.TimerfdCreate, 27: syscall.TimerfdSetTime, 28: syscall.TimerfdGetTime,
         29: syscall.Fork, 30: syscall.InotifyInit, 31: syscall.InotifyAddWatch,
         32: syscall.InotifyRmWatch, 33: syscall.SocketPair, 34: syscall.Write,
         36: syscall.Mmap, 37: syscall.Listen, 38: syscall.Ptrace}
LIFECYCLE = {2: sentry.CloneInfo, 3: sentry.ExecveInfo, 5: sentry.TaskExit}


class InstrumentationUnavailable(RuntimeError):
    """Required SecCheck capability or authorized control access is unavailable."""


class CaptureFailure(RuntimeError):
    """Collector failed during a workload; terminate the analysis."""


def select_points(metadata):
    available = {}
    for line in metadata.splitlines():
        match = re.fullmatch(r"Name: ([^,]+), optional fields: \[(.*)\], context fields: \[(.*)\]", line)
        if match:
            available[match[1]] = (set(filter(None, match[2].split('|'))), set(match[3].split('|')))
    if not re.search(r"^Name: remote$", metadata, re.M):
        raise InstrumentationUnavailable("Installed runsc does not expose the remote sink")
    selected = {}
    name_to_number = {name: number for number, name in NAMES.items()}
    for point in sorted(available):
        parts = point.split('/')
        if parts[0] != 'syscall' or parts[-1] not in {'enter', 'exit'}:
            continue
        number = int(parts[2]) if len(parts) == 4 and parts[1] == 'sysno' else name_to_number.get(parts[1])
        if number is None:
            raise InstrumentationUnavailable(f"No verified x86-64 syscall number for {point}")
        key = (number, parts[-1])
        if key not in selected or len(parts) == 3:
            selected[key] = point
    if not selected or not any(phase == 'exit' for _, phase in selected):
        raise InstrumentationUnavailable("Application syscall entry/exit tracepoints unavailable")
    points = []
    for point in [*selected.values(), 'sentry/execve', 'sentry/clone', 'sentry/task_exit']:
        if point not in available or not set(CONTEXT) <= available[point][1]:
            raise InstrumentationUnavailable(f"Required process attribution context missing: {point}")
        points.append({"name": point, "context_fields": CONTEXT,
                       "optional_fields": ["fd_path"] if 'fd_path' in available[point][0] else []})
    return selected, points, len([name for name in available if name.startswith('syscall/')])


def normalized_arguments(message):
    """Allowlist metadata only. Never persist argv/env, buffers or dereference pointers."""
    arguments = {}
    for descriptor in message.DESCRIPTOR.fields:
        name = descriptor.name
        if name in {'context_data', 'exit', 'sysno', 'argv', 'envv'}:
            continue
        value = getattr(message, name)
        if name == 'address':
            # sockaddr metadata only, never payload; native family, network-order port.
            if len(value) >= 8 and int.from_bytes(value[:2], 'little') == 2:
                arguments[name] = {"family": "AF_INET", "ip": str(ipaddress.ip_address(value[4:8])),
                                   "port": int.from_bytes(value[2:4], 'big')}
            elif len(value) >= 24 and int.from_bytes(value[:2], 'little') == 10:
                arguments[name] = {"family": "AF_INET6", "ip": str(ipaddress.ip_address(value[8:24])),
                                   "port": int.from_bytes(value[2:4], 'big')}
            else:
                arguments[name] = None
        elif descriptor.type == descriptor.TYPE_STRING:
            # Paths/process names can themselves contain secrets: redact common secret syntax.
            clean = re.sub(r'(?i)(token|password|secret|api[_-]?key)[=:][^/\s]*', r'\1=<redacted>', value)
            arguments[name] = clean[:256] if clean else None
        elif descriptor.type != descriptor.TYPE_MESSAGE and descriptor.type != descriptor.TYPE_BYTES:
            arguments[name] = value
    if isinstance(message, syscall.Syscall):
        # Preserve opaque scalar registers, never claim decoded pointer contents.
        arguments['encoding'] = 'raw_registers_no_pointer_dereference'
        if NAMES.get(message.sysno) == 'mprotect':
            arguments.update(addr=message.arg1, length=message.arg2, prot=message.arg3)
    return arguments


class EventDecoder:
    """Single receiver's ordered, sandbox-checked, process-tree-scoped event stream."""
    def __init__(self, analysis_id, container, command, selected):
        self.analysis_id, self.container, self.command = analysis_id, container, list(command)
        self.selected = selected
        self.events = []
        self.groups = set()
        self.live_threads = set()
        self.root_group = None
        self.pending = {}
        self.entry_count = 0
        self.dropped = 0
        self._drop_counter = 0
        self.errors = []
        self.excluded = 0
        self.duplicates = 0
        self.lifecycle_count = 0
        self.armed = False
        self.max_records = 250000

    @property
    def count(self):
        return self.entry_count

    def consume(self, packet, received_ns=None):
        if len(packet) < 8:
            raise ValueError('Truncated SecCheck header')
        size, kind, drops = struct.unpack('<HHI', packet[:8])
        if size < 8 or size > len(packet):
            raise ValueError('Invalid SecCheck header length')
        delta = (drops - self._drop_counter) % 2**32
        self.dropped += delta
        self._drop_counter = drops
        cls = TYPES.get(kind) or LIFECYCLE.get(kind)
        if cls is None:
            if kind == 1 or kind in {4, 35}:
                self.excluded += 1
                return
            raise ValueError('Unknown SecCheck message type')
        msg = cls()
        msg.ParseFromString(packet[size:])
        ctx = msg.context_data
        if ctx.container_id != self.container:
            self.excluded += 1
            self.errors.append('container_identity_mismatch')
            return
        if not self.armed:
            self.excluded += 1
            return
        group = (ctx.thread_group_id, ctx.thread_group_start_time_ns)
        thread = (ctx.thread_id, ctx.thread_start_time_ns)
        if kind in LIFECYCLE:
            self.lifecycle_count += 1
            if kind == 3 and self.root_group is None and list(msg.argv) == self.command and ctx.is_exec_session:
                self.root_group = group
                self.groups.add(group)
                self.live_threads.add(thread)
            elif kind == 2 and group in self.groups:
                child_group = (msg.created_thread_group_id, ctx.thread_group_start_time_ns if msg.flags & 0x10000 else msg.created_thread_start_time_ns)
                self.groups.add(child_group)
                self.live_threads.add((msg.created_thread_id, msg.created_thread_start_time_ns))
            elif kind == 5 and group in self.groups:
                self.live_threads.discard(thread)
            return  # lifecycle never enters model syscall sequence
        if group not in self.groups or ctx.thread_id <= 0:
            self.excluded += 1
            return
        phase = 'exit' if msg.HasField('exit') else 'enter'
        point = self.selected.get((msg.sysno, phase))
        if point is None:
            raise ValueError('Received unrequested syscall point')
        if (kind == 6) != ('/sysno/' in point):
            self.duplicates += 1
            return
        if len(self.events) >= self.max_records:
            raise CaptureFailure('Collector record limit reached; capture incomplete')
        sequence = len(self.events) + 1
        key = (*thread, msg.sysno)
        entry = self.pending.pop(key, None) if phase == 'exit' else None
        record = dict(schema_version=2, analysis_id=self.analysis_id, sandbox_id=self.container,
            container_id=ctx.container_id, sequence=sequence, event_type='syscall', event_source=SOURCE,
            tracepoint=point, syscall_name=NAMES.get(msg.sysno, f'syscall_{msg.sysno}'), syscall_number=msg.sysno,
            phase=phase, monotonic_timestamp_ns=received_ns if received_ns is not None else time.monotonic_ns(),
            monotonic_clock='collector_receive', runtime_time_ns=ctx.time_ns or None,
            thread_id=ctx.thread_id, thread_start_time_ns=ctx.thread_start_time_ns or None,
            process_id=ctx.thread_group_id, process_start_time_ns=ctx.thread_group_start_time_ns or None,
            parent_process_id=ctx.parent_thread_group_id or None,
            process_name=ctx.process_name[:64] or None, arguments=normalized_arguments(msg),
            return_value=msg.exit.result if phase == 'exit' else None,
            error_status=msg.exit.errorno if phase == 'exit' else None,
            entry_sequence=entry, duration_ns=None,
            correlation='matched' if entry is not None else 'unmatched_exit' if phase == 'exit' else 'entry',
            reported_dropped_events=self.dropped)
        if phase == 'enter':
            if key in self.pending:
                self.errors.append('overlapping_syscall_entries')
            self.pending[key] = sequence
            self.entry_count += 1
        self.events.append(record)


class SecCheckSession:
    """Attach to one existing sandbox using authorized runsc control access."""
    def __init__(self, analysis_id, container, command, *, runsc='runsc', root=None):
        self.analysis_id, self.container, self.command = analysis_id, container, command
        self.runsc = os.environ.get('MODELCAGE_RUNSC_COMMAND', runsc)
        self.root = root or os.environ.get('MODELCAGE_RUNSC_ROOT', '/run/docker/runtime-runc/moby')
        self.directory = None
        self.listener = self.connection = self.worker = None
        self.connected = threading.Event()
        self.finished = threading.Event()
        self.decoder = None
        self.attached = False
        self.stopping = False
        self.version = None
        self.available_count = None

    def _run(self, args):
        try:
            result = subprocess.run([self.runsc, *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise InstrumentationUnavailable('runsc control command unavailable or timed out') from error
        if result.returncode:
            # No model command/output in error text. Exact control failure saved separately bounded.
            reason = result.stderr.strip()[:400]
            raise InstrumentationUnavailable(f'runsc control failed: {reason}')
        return result.stdout

    def start(self):
        if platform.machine() != 'x86_64':
            raise InstrumentationUnavailable('Decoder ABI validated for x86_64 only')
        self.version = self._run(['--version']).strip()
        if RELEASE not in self.version:
            raise InstrumentationUnavailable(f'Decoder requires validated runsc {RELEASE}')
        selected, points, self.available_count = select_points(self._run(['trace', 'metadata']))
        self.decoder = EventDecoder(self.analysis_id, self.container, self.command, selected)
        # Confirm the full Docker container ID resolves to this exact running OCI state.
        state = json.loads(self._run(['--root', self.root, 'state', self.container]))
        if state.get('id') != self.container or state.get('status') != 'running':
            raise InstrumentationUnavailable('runsc sandbox identity/state could not be verified')
        self.directory = tempfile.TemporaryDirectory(prefix='mc-seccheck-')
        endpoint = str(Path(self.directory.name) / 'sink.sock')
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.bind(endpoint)
        os.chmod(endpoint, 0o600)
        self.listener.listen(1)
        self.listener.settimeout(0.2)
        self.worker = threading.Thread(target=self._receive, daemon=True)
        self.worker.start()
        config = {'trace_session': {'name': 'Default', 'ignore_missing': False, 'points': points,
            'sinks': [{'name': 'remote', 'ignore_setup_error': False,
                       'config': {'endpoint': endpoint, 'retries': 3}}]}}
        config_path = Path(self.directory.name) / 'session.json'
        config_path.write_text(json.dumps(config))
        self._run(['--root', self.root, 'trace', 'create', '--config', str(config_path), self.container])
        self.attached = True
        if not self.connected.wait(3):
            raise InstrumentationUnavailable('SecCheck remote handshake did not complete')
        self.check()

    def _receive(self):
        try:
            while not self.stopping:
                try:
                    self.connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if self.connection is None:
                return
            self.connection.settimeout(3)
            handshake = common.Handshake()
            handshake.ParseFromString(self.connection.recv(1024))
            if handshake.version != 1:
                raise ValueError('Unsupported SecCheck protocol version')
            self.connection.sendall(common.Handshake(version=1).SerializeToString())
            self.connection.settimeout(0.2)
            self.connected.set()
            while True:
                try:
                    packet, _, flags, _ = self.connection.recvmsg(1024*1024)
                except socket.timeout:
                    if self.stopping:
                        return
                    continue
                if not packet:
                    if not self.stopping:
                        self.decoder.errors.append('collector_disconnected')
                    return
                if flags & socket.MSG_TRUNC:
                    raise ValueError('Truncated SecCheck packet')
                self.decoder.consume(packet)
        except Exception as error:
            if self.decoder is not None:
                self.decoder.errors.append(type(error).__name__ + ': ' + str(error)[:120])
        finally:
            self.finished.set()

    def arm(self):
        self.check()
        self.decoder.armed = True

    def check(self):
        if self.decoder and (self.decoder.errors or self.decoder.dropped):
            raise CaptureFailure('SecCheck collector errors or event loss reported')
        if self.connected.is_set() and self.finished.is_set() and not self.stopping:
            raise CaptureFailure('SecCheck receiver disconnected')

    def progress(self):
        self.check()
        return self.decoder.count if self.decoder else 0

    def stop(self):
        try:
            if self.attached:
                # Query sink counters while the workload is quiescent before detaching.
                counters = self._run(['--root', self.root, 'trace', 'list', self.container])
                reported = re.findall(r'Sink: "remote", dropped: (\d+)', counters)
                if reported and self.decoder:
                    self.decoder.dropped = max(self.decoder.dropped, max(map(int, reported)))
                elif self.decoder:
                    self.decoder.errors.append('final_sink_loss_counter_unavailable')
                self.stopping = True
                self._run(['--root', self.root, 'trace', 'delete', '--name', 'Default', self.container])
                self.attached = False
        except InstrumentationUnavailable:
            if self.decoder:
                self.decoder.errors.append('trace_session_delete_failed')
        finally:
            self.stopping = True
            if self.worker:
                self.worker.join(timeout=3)
            if self.connection:
                self.connection.close()
            if self.listener:
                self.listener.close()
            if self.worker and self.worker.is_alive():
                self.worker.join(timeout=1)
                if self.decoder:
                    self.decoder.errors.append('collector_shutdown_incomplete')
            if self.directory:
                self.directory.cleanup()

    def summary(self):
        decoder = self.decoder
        events = decoder.events if decoder else []
        return dict(runtime_version=self.version, available_syscall_tracepoints=self.available_count,
            selected_syscall_tracepoints=sorted(decoder.selected.values()) if decoder else [],
            event_count=decoder.count if decoder else 0, record_count=len(events),
            reported_losses=decoder.dropped if decoder else None,
            capture_errors=list(decoder.errors) if decoder else [],
            workload_root=list(decoder.root_group) if decoder and decoder.root_group else None,
            process_groups=sorted([list(group) for group in decoder.groups]) if decoder else [],
            excluded_records=decoder.excluded if decoder else 0,
            lifecycle_records=decoder.lifecycle_count if decoder else 0,
            duplicate_representation_records=decoder.duplicates if decoder else 0,
            unmatched_entries=len(decoder.pending) if decoder else 0,
            unexpected_unmatched_entries=sum(NAMES.get(key[-1]) not in {'exit','exit_group'} for key in decoder.pending) if decoder else 0,
            unmatched_exits=sum(e['correlation']=='unmatched_exit' for e in events),
            duration_policy='unavailable: runtime monotonic event clock not supplied; receive time is not syscall duration',
            loss_visibility='wire dropped counter only; terminal losses after last packet may be unobservable')
