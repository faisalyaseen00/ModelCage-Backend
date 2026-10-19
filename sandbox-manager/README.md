# ModelCage — policy-driven runtime instrumentation

The model-analysis API now requires **gVisor SecCheck**. It does not fall back to
in-container strace, host tracing, or runc. The existing disposable sandbox manager,
controlled pickle harness, isolation settings and local polling interface remain.

After each upload, the backend automatically saves one-row-per-execution syscall
**feature CSVs** and normalized frequency CSVs, with metadata stored separately.
No download action is needed.
See [Phase 2 collection](docs/phase2-collection.md) for the format, reviewed batch
collection, category datasets and the loading/deserialization scope.

## Current deployment status

SHA-256 identification is enabled for normal uploads and reviewed batch collection.
The backend streams the submitted bytes into private quarantine while calculating
SHA-256; it never deserializes a model on the host. Identical `.pkl`/`.pickle` bytes
share `quarantine/<sha256>.pkl`, independent of filename. New copies are read-only,
publication never overwrites an existing copy, and reused copies are verified.
The exact bytes staged into a fresh sandbox are checked again against the digest
and size before container creation. A mismatch stops execution.

An optional upload manifest `sha256`  field (64 hexadecimal characters) supplies
an expected digest. A mismatch rejects the submission. The digest, byte size,
verification state and quarantine reuse appear in progress and execution metadata;
`data/metadata.csv` also records them. SQLite `model_artifacts` stores one identity
per digest, and `model_submissions` preserves each filename, declared model/category,
family, available architecture/source, analysis ID and final processing status.
Manual rescans retain independent executions and fresh sandboxes; the batch
collector skips processed hashes and supports explicit failed-run retries.
SHA-256 proves byte identity, not model safety or category correctness.

Restart the backend to activate the code. Existing unhashed analyses and quarantine
files remain untouched; resubmit their original bytes to create a verified identity.
No digest is invented for historical records. The explicit hashing-disable option
is retained for legacy compatibility tests; the default policy is enabled.

The installed `runsc release-20260907.0` on x86-64 advertises 992 syscall
tracepoints: 884 raw entry/exit points and 108 schematized points. ModelCage selects
884 non-duplicating syscall number/phase points, preferring schematized variants.
Three sentry lifecycle points are subscribed solely for process-tree attribution.
The installed sinks are `remote` and `null`; collection uses `remote` protocol 1.
The full inventory and binary SHA-256 are in `docs/runsc-capabilities.json`.

**The user has verified the minimal live SecCheck test under sudo on this host.**
The unprivileged user still cannot access the root-owned runsc control directory;
passwordless sudo is unavailable. Without authorized control access, analyses
finish as `instrumentation_unavailable`, with cleanup and failure metadata.
Synthetic protocol tests are not evidence of live coverage for every tracepoint.

An administrator must provide an authorized runsc control mechanism before live
capture can be tested. ModelCage does not alter Docker configuration, weaken
sandbox permissions, install sudo rules or mount host control sockets into the
model sandbox. `MODELCAGE_RUNSC_ROOT` selects the administrator-confirmed OCI
state directory (default `/run/docker/runtime-runc/moby`).
`MODELCAGE_RUNSC_COMMAND` can select an existing trusted executable wrapper in
place of `runsc`; it is a single executable path, not a shell command. The wrapper
must expose the runsc version, metadata, state, trace-create/list/delete operations
for the relevant sandbox. Its privilege policy needs administrator review.

## Run

Use Python 3.11 or 3.12; the existing `cgi` upload parser is unavailable in 3.13.

```bash
cd /home/faisal/Documents/GitHub/ModelCage-Backend/sandbox-manager
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
# Only if the existing runner image has not been built:
docker build -t modelcage-runner:phase6a -f images/model-runner.Dockerfile .
.venv/bin/python -m modelcage.web
```

Open <http://127.0.0.1:8080> and upload a `.pkl`/`.pickle`. **Category selection
is automatic**; the form has no category selector. Before execution, the backend
examines the SHA-256-verified bytes using bounded, symbolic pickle inspection.
It never calls `pickle.load`/`loads`, imports referenced model modules, executes
reducers, or constructs model objects on the host.

Supported evidence includes:

- Recognized Transformers task heads (for example `GPT2LMHeadModel`,
  `BertForSequenceClassification`, and `ViTForImageClassification`).
- Recognized torchvision classification architectures.
- Scikit-learn Pipelines with a text vectorizer and a recognized final classifier.
- Embedded root/model/config metadata, or optional model-config/model-card JSON
  with `architectures`, `pipeline_tag`, task tags or task declarations.

A model config can be supplied directly in the optional metadata box, without
selecting a category, for example:

```json
{"architectures":["BertForSequenceClassification"],"model_type":"bert"}
```

Evidence is reconciled: conflicting architectures/tasks produce `conflict`, an
unsupported task produces `unsupported`, and insufficient evidence produces
`unclassified`. These cases finish as `pending_review` before sandbox creation
and receive metadata but no category dataset row. The reason is shown in the UI.
The backend never guesses a task from the filename, generic BERT/GPT family name,
class labels, or a bare numerical classifier. A custom `GPTLanguageModel` class
without recognizable task/config metadata remains unresolved; provide its actual
config or task metadata. Loading dependencies/custom class definitions are still
required inside the sandbox even after the category has been identified.

The category routes raw traces, 1/2/3-gram JSON, syscall count CSVs and normalized
frequency CSVs to `text_generation`, `text_classification`, or
`image_classification`. `auto_identified` means evidence came from the artifact;
`metadata_identified` means it came from supplied metadata. Reasons, evidence and
detector version are retained with each analysis. Architecture/task metadata is
untrusted: identification does not prove model behavior or benignness, and it does
not automatically admit an upload to the reviewed training corpus.

Inspection is bounded to 64 MiB and 200,000 pickle operations. Unsupported or
opaque reconstruction, inspection limits and missing metadata may require review.
No downloads, model-hub lookup, `.pt` support or inference were added. Explicit
categories remain compatible for existing API/reviewed batch clients, but cannot
override conflicting extracted evidence. See [automatic category identification](docs/category-identification.md).

The UI redirects promptly to `/?analysis=<uuid>` and polls
`GET /analyses/<uuid>` once per second. It shows actual backend events, entry count,
record count, loss/capture/error state, category provenance, observation evidence,
storage and cleanup. Reconnecting to the same URL retrieves all progress.
Terminal logs and the interface share `ExecutionEvents`; timers only refresh
backend state, never synthesize progress. Restart an already-running backend to
load new code. The local server supports at most two simultaneous analyses.

## Source and workload boundary

After quarantine and staging, the manager verifies Docker's `runsc` setting and
existing resource/privilege/network policy. The collector additionally resolves
the **exact full Docker ID** with `runsc state` and requires a running state with
that same ID. A private, per-analysis host Unix `SOCK_SEQPACKET` receiver attaches
to that sandbox's `Default` trace session. It never force-replaces another session.

The harness is launched through the fixed shell command `exec "$@"`, passing
`python3 <unique-runner> <unique-model>` as positional arguments. This creates an
explicit guest execve boundary; Docker exec process creation alone cannot be
assumed to emit that event. Launcher events before the boundary are excluded.
A matching SecCheck `sentry/execve` argv/context anchors
the workload root; argv is used transiently for matching and never stored. Clone
metadata identifies child processes/threads and their start-time generations.
Only syscall records in this verified sandbox and workload tree enter raw JSONL.
PID reuse, other sandboxes, container init and setup commands are excluded.
Sentry/container lifecycle records are not syscalls and never enter counts or
n-grams. The collector and host Docker/containerd/runsc/API processes are not
inside this guest process tree and are not traced as model behavior.

The boundary includes interpreter startup, libraries, deserialization and children
created by the harness. These are workload events, not events exclusively caused
by the model file. The current harness deserializes a pickle; category-specific
inference inputs and `.pt` loading remain unimplemented.

The source protocol and schemas are pinned to the installed release:
[gVisor SecCheck](https://github.com/google/gvisor/tree/release-20260907.0/pkg/sentry/seccheck).
Generated protobuf code, source schemas, license and provenance are bundled under
`app/seccheck_proto`. A different runsc release/ABI fails closed pending validation.

## Record semantics and limitations

Schema v2 retains entry and exit records, an analysis-local receive-order sequence,
syscall name/number, selected tracepoint, process/thread/start-time/parent context,
normalized arguments, result/errno, dropped-event counter and source identity.
`event_count` counts entries; `record_count` includes both phases. Raw and
schematized duplicates are excluded by configuration and decoder checks.

- `monotonic_timestamp_ns` is explicitly the **collector receive clock**. The
  runtime's `time_ns` is preserved separately. Receive latency is not syscall
  duration. This protocol supplies no verified runtime monotonic call-duration
  clock, so `duration_ns` remains null. Exits reference matched entry sequences;
  unmatched/overlapping events are reported, never converted into successes.
- Rich decoded paths/flags are used when available; raw register values remain
  numbers. No pointers are dereferenced or interpreted as strings. Unknown syscall
  numbers retain `syscall_<number>` names. Proto3 scalar zero defaults are decoded
  according to the pinned schema; wire omission cannot distinguish unknown from
  explicit zero for every scalar field. Missing optional paths remain null.
- Environment and command arguments are not persisted; no file contents, model
  inputs or network payloads are requested/stored. Strings are bounded to 256
  characters and common secret assignments are redacted. Container paths may
  still be sensitive; comprehensive semantic path redaction is not guaranteed.
- Wire losses and final sink counters are recorded; collector exceptions, invalid
  packets, identity mismatches, unpaired calls and a 250,000-record memory limit
  make capture incomplete. Invalid packets are not fabricated as syscall records.
- Cross-thread ordering is collector arrival order, not proof of kernel execution
  order. Per-thread feature output is provided separately.
- On timeouts/errors, exec processes are terminated while PID 1 remains available
  for session shutdown; final force removal stops remaining processes. If control
  or collector shutdown fails, metadata reports failure/partial capture. Automatic
  recovery following an abrupt backend/host crash is not implemented.

The 30-second execution budget includes the harness and its descendants. Phase 1
restrictions remain: read-only root, private `/tmp` tmpfs, 512 MiB memory, one CPU,
64 PIDs, no capabilities, no privileged mode, no host mounts and offline networking.
Old low-level Phase 1/legacy trace helpers remain for compatibility; the
`ModelCage.execute_model` path exclusively requires SecCheck.

## Category artifacts and immutability

Paths are relative to the database parent (default this directory):

```text
data/raw/<task_category>/<analysis_id>.jsonl
data/features/<task_category>/<analysis_id>.json
data/metadata/<analysis_id>.json
data/observations/<analysis_id>.jsonl
data/progress/<analysis_id>.progress.jsonl
modelcage.sqlite
quarantine/
```

Raw records have the same schema in all three category directories. Metadata
contains category/source/validation status, hash, sandbox identity, verification,
coverage, capture status, counts/losses, timestamps, failure and cleanup results.
Unclassified or unavailable analyses get metadata but no invented dataset rows.
Old flat strace files are preserved in place and never silently relabeled or mixed
into SecCheck datasets. The existing SQLite/CSV/XLSX view projects syscall entries;
JSONL is authoritative for entry/exit, thread, loss and correlation information.

Artifacts use fsynced temporary files and atomic **no-overwrite** publication with
read-only permissions. Metadata is published last as the completion record.
A failed publication can leave completed individual artifacts without final
metadata; progress reports the failure. Existing finalized records are never
rewritten by the API. Filesystem administrators can still change permissions.

## Observations and features

`app/observations.py` is independent of capture and training. `ObservationPolicy`
lets library callers choose groups, executable allowlists, approved write roots
and sensitive paths. It emits evidence references and reasoning, not malware
verdicts. Groups cover unexpected process execution, network attempts (including
denied attempts), filesystem changes, sensitive container-path access, privilege
and isolation operations, and writable/executable memory combinations. Ordinary
thread clones and executable memory alone are not maliciousness rules. Raw
pointer-only paths never trigger path-based conclusions. External-address priority
requires a decoded address; otherwise the destination is reported as unknown.

Each observation has `reason_code`, `group`, `priority`, `outcome`, `reason`, and
`source_sequences`. The UI displays up to 100 individually; all observations are
preserved separately. Rules can be disabled/configured in the library:

```python
from modelcage.modelcage import ModelCage
from modelcage.observations import ObservationPolicy
cage = ModelCage(observation_policy=ObservationPolicy(enabled_groups=("network", "privilege")))
```

Deterministic features use **only syscall entries**. 1-gram presence/frequency is
the baseline; 2- and 3-grams are experimental. These are the only three supported
orders in new feature outputs (feature schema v2). Historical schema-v1 artifacts
may contain 4-grams and remain unchanged. Both global receive-order and per-thread
views are saved beside the category dataset. Incomplete-trace outputs carry their
capture status and are not evidence of complete behavior. Reproduce them offline:

```bash
.venv/bin/python -m modelcage.features data/raw/text_generation/ANALYSIS_ID.jsonl
.venv/bin/python -m modelcage.features data/raw/text_generation/ANALYSIS_ID.jsonl --per-thread
```

No cross-category benign baseline, claimed best n-gram order, copied paper scores,
OCSVM training, final risk scoring, model discovery/download or dataset crawler
is implemented. Image classification is a ModelCage extension requiring its own
future benign baseline.

## Tests

```bash
.venv/bin/python -m pytest -q -m 'not integration and not network'
.venv/bin/python -m pytest -q -m network
.venv/bin/python -m pytest -q -m integration -rs
.venv/bin/python -m pytest -q
```

Unit fixtures are synthetic protobuf records and explicitly not genuine capture
evidence. Network tests exercise the real socket protocol and HTTP snapshots.
Docker tests verify isolation and unavailable-instrumentation cleanup now;
positive SecCheck tests skip with the precise control-access failure until the
administrator provides access. Consult `IMPLEMENTATION_STATUS.md` for results.
