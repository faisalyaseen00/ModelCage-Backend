# Current Phase 2 upload workflow

The user verified the minimal live SecCheck test under sudo (1 passed). The
explicit guest execve launcher resolved the prior attribution timeout for that
test. The current turn adds automatic backend CSV generation after immutable raw
capture storage: `data/features/<category>.csv`, relative-frequency companions,
and separate `data/metadata.csv`. Counts are syscall entries only, one complete
execution per row. Failed/partial runs remain in metadata/raw artifacts.

Manual hashing stays paused. Complete measurements are stored with honest
category/hash provenance; reviewed training eligibility is a separate field/view.
The selected scope is loading/deserialization, not inference. No model downloader
is included. An optional local-manifest collector provides hashing, registry,
exact deduplication, retries and family quotas in a dedicated workspace.

Validation for these changes: 110 unit tests passed, plus one HTTP multipart-upload
test proving automatic CSV storage without a download request. Capture is mocked
in that HTTP test. The new privileged collection smoke test has not been run in
this session; it requires the user's sudo password. See
[backend workflow and schema](docs/phase2-collection.md).

---

# Historical implementation report (before live capture and backend CSV fixes)

# ModelCage policy-driven Phase 2 — implementation report

Validated 2026-10-12. This report supersedes the earlier strace-based Phase 2
report. Historical successful strace runs do **not** validate SecCheck capture.

## Outcome

The policy-driven implementation is present and unit/protocol tested. Real Docker
isolation and fail-closed cleanup are tested. **Positive live SecCheck collection
remains blocked by host runsc-control permissions.** The application refuses model
execution as `instrumentation_unavailable` rather than falling back to another
source. It saves the reason and confirms sandbox removal.

## Architecture

Submission and quarantine/hash still feed the existing sandbox orchestrator,
Docker/runsc boundary and controlled model harness. SecCheck replaces the model
path's strace stage. Existing progress events continue to feed the terminal and
polling page. Trace processing now routes schema-identical syscall records into
three category directories, with observations and reproducible features kept
separate. The old manager's generic Phase 1 API remains compatible.

The diagram's `.pt` path and category-specific inference inputs are still absent;
the controlled harness loads pickle artifacts. Task categories are explicit
validated declarations, not inferred claims about a pickle's behavior. No model
hub discovery/download, large-scale collection, training or scoring was added.

## Source and coverage

Source: installed `runsc release-20260907.0`, x86-64, SecCheck remote protocol 1.
`runsc trace metadata` actually reported 998 points: 992 syscall points and six
infrastructure points; sinks `remote` and `null`.

- 884 raw syscall entry/exit points and 108 schematized points are advertised.
- Configuration selects 884 syscall number/phase points without duplicate raw and
  schematized representations, preferring schemas where available.
- `sentry/execve`, `sentry/clone`, `sentry/task_exit` serve attribution only. They
  never enter syscall JSONL, counts or n-gram sequences.
- `docs/runsc-capabilities.json` records the entire enumerated inventory, selected
  configuration, runtime version and installed binary SHA-256.
- Pinned source schemas and generated Python protobuf decoder are included with
  provenance and the Apache license in `app/seccheck_proto`.

The receiver verifies the full container ID against runsc state, uses a private
socket for one sandbox, anchors the harness using exact exec argv, then tracks
children/threads with PID and start-time generations. Host Docker/containerd,
runsc and API syscalls are not subscribed sources. Setup/init and unrelated
processes are excluded. The stream includes interpreter/library/model-loading
activity within the workload; it is not exclusively attributable to artifact
bytes.

## Categories and storage

Only `text_generation`, `text_classification`, `image_classification` are accepted.
Optional JSON manifests provide non-executing declarations. Missing selections
and conflicting declarations stop before sandbox creation as `pending_review`.
Metadata records source, validation status and declaration evidence. No extension
or filename inference and no host deserialization occurs.

Under the database parent's `data` directory:

```text
raw/<category>/<id>.jsonl
features/<category>/<id>.json
metadata/<id>.json
observations/<id>.jsonl
progress/<id>.progress.jsonl
```

Dataset files are created only when genuine records exist. Publication uses
flush/fsync and atomic no-overwrite links; metadata is published last. Existing
flat strace data is preserved without migration or reinterpretation. SQLite and
CSV/XLSX remain an entry-only compatibility view of the authoritative raw stream.

## Observations and features

`ObservationPolicy` configures enabled groups, executable allowlists, approved
write roots and sensitive container paths. Observations retain reason codes,
reasoning, priority, attempted/succeeded/failed outcome and source-event sequence
references. Failed or denied network/privilege attempts are retained. Decoded
external network addresses receive high priority; unknown pointer-only addresses
are not guessed. Thread clones and executable memory alone are not suspiciousness
rules. Writable/executable combinations are observations, not malware verdicts.

N-grams are deterministic and entry-only. One-gram presence/frequency is the
baseline, with two-, three- and four-grams retained for experiments. Global output
uses collector receive order; per-thread output is separate. Category datasets
and future baselines stay separate. No DynaHug scores, feature-order superiority,
image-category validation claim or OCSVM model is copied or asserted.

## Changed components

New: `app/seccheck.py`, `app/seccheck_proto/`, `app/syscall_names.py`,
`app/categories.py`, `app/observations.py`, `app/features.py`,
`tests/seccheck_fixtures.py`, `tests/test_seccheck_policy.py`,
`docs/runsc-capabilities.json`, `PHASE2_POLICY_PLAN.md`.

Integrated: `app/modelcage.py`, `app/manager.py`, `app/docker_runtime.py`,
`app/raw_events.py`, `app/web.py`, model/progress/integration tests, dependencies,
package configuration, README and this report. Phase 1 isolation policy constants
were not changed.

## Executed validation

Final command, with local Docker and socket access:

```bash
.venv/bin/python -m pytest -q
```

Result: **95 passed, 4 skipped**, one existing `cgi` deprecation warning, 28.14 s.
The four skipped cases are positive SecCheck model/concurrency/timeout/failure
capture tests requiring runsc control access. They are explicitly not reported as
passing integrations. No root password or host policy change was used.

Also executed:

- Phase 1/environment/legacy normalization unit subset: **53 passed**, 4 Docker
  cases deselected in that invocation; real Phase 1 Docker cases passed in the
  complete suite above.
- SecCheck policy unit subset: **14 passed**, one socket test deselected.
- Real Unix socket handshake/decoder/drain/shutdown test: **1 passed**. Messages
  in this protocol test are synthetic fixtures, not application syscall evidence.
- `python -m compileall -q app tests`: passed.
- Installed `modelcage.seccheck` and `modelcage.features` imports: passed.
- `git diff --check`: passed.

Coverage includes other-sandbox/infrastructure exclusion, workload descendants,
PID-generation separation, entry/exit pairing, raw/schema deduplication, loss and
invalid-packet reporting, denied-operation evidence, category conflicts and
placement, immutable output, all n-gram orders, concurrent isolated analyses,
HTTP reconnects, terminal event agreement, failures and cleanup.

## Limitations and next validation step

The current account is in the Docker group but cannot access
`/run/docker/runtime-runc/moby`; `sudo -n` requires a password. An administrator
must provide an authorized runsc control executable/state root. Use the documented
`MODELCAGE_RUNSC_COMMAND` and `MODELCAGE_RUNSC_ROOT` only for an existing authorized
setup, then rerun:

```bash
.venv/bin/python -m pytest -q -m integration -rs
```

Until positive SecCheck tests pass, real syscall coverage, process-tree attribution
and observation output are **not claimed verified on the installed runtime**.

Additional limitations: x86-64 and the pinned release only; collector-arrival
ordering; raw pointers remain opaque; proto3 scalar-presence ambiguity; unknown
runtime monotonic duration (stored null rather than fabricated); 250,000-record
capture bound reported as failure/partial; incomplete results retain their capture
status; no automatic backend/host-crash reconciliation; existing local HTTP parser
requires Python 3.11–3.12. See README for full operation and privacy limitations.
