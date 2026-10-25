# Live SecCheck validation

Run from the editable source installation with Docker access and authorized
runsc control access. The default control root is `/run/docker/runtime-runc/moby`.
If it is root-owned, run locally with sudo (never supply passwords to the agent):

```bash
cd /home/faisal/Documents/GitHub/ModelCage-Backend/sandbox-manager
sudo .venv/bin/python -m modelcage.live_validation --output /tmp/modelcage-live-root-20261014
```

Choose a **new** output directory on each run. Alternatively use an existing,
authorized `MODELCAGE_RUNSC_COMMAND` wrapper and the appropriate
`MODELCAGE_RUNSC_ROOT`; the validator does not modify system permissions.

The command runs all integration tests, retaining `pytest.log`, `junit.xml`,
`summary.json`, and pytest workload artifacts. A skip, failure, missing report,
empty suite or interrupted pytest process prevents a successful result. Review
the individual assertions and traces as well as the summary.

The expanded suite checks exact entry/exit counts for controlled getpgid calls;
file operations and failed writes/connections; fork/thread attribution;
independent concurrent sandboxes; automatic routing to all three category CSVs;
entry-count and normalized CSV consistency; and real receiver record-limit
exclusion. Existing live tests cover timeout/nonzero exits, progress, collection,
deduplication, isolation and cleanup. Workload code deserializes only inside the
normal disposable sandbox. Missing instrumentation is a failure, not a fallback.

These are bounded controlled fixtures, **not representative trained ML models**.
The current runner loads pickles and lacks ML dependencies and inference
harnesses. Passing this suite cannot establish exhaustive tracepoint coverage,
arbitrary workload reliability, .pt support, or inference coverage. Protocol
tests with synthetic packets are separate from live evidence.

On 2026-10-14 the existing 11-test live baseline returned 5 passed, 2 failed and
4 skipped. Positive SecCheck runs failed at runsc container-state lookup
(`FetchSpec failed: loading container: file does not exist`). The account could
use Docker but could not traverse its default runsc control directory; sudo
required a password. A privileged rerun is necessary to distinguish access
issues from any remaining runtime/configuration or capture defects.

The expanded 21-test run then returned **5 passed, 12 failed, 4 skipped**;
`/tmp/modelcage-live-validation-20261014/summary.json` records `validated: false`.
The new positive cases were also blocked before capture. Non-integration tests
passed 188 cases across the ordinary run and an authorized socket-test rerun.
