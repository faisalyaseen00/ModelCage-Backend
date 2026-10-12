# Policy-driven instrumentation plan

Reference: the supplied ModelCage architecture and current Phase 2 implementation.
The existing upload, quarantine, manager, runsc isolation, controlled pickle harness,
SQLite exports, JSONL storage and polling UI remain in place.

Installed runtime: release-20260907.0, x86_64. `runsc trace metadata` reports 998
points: 992 syscall points (884 raw and 108 schematized), plus six infrastructure
points, and remote/null sinks. SecCheck remote protocol version 1 is supported.
The current user cannot read Docker's runsc control root; no passwordless sudo is
available. Fail closed and report this; do not change host runtime configuration.

Implementation:
- Add pinned protobuf schemas/decoder and SecCheck capabilities/session/collector.
  Select each available syscall number/phase once, preferring schematized points.
- Start a per-analysis host Unix SEQPACKET receiver, attach only to the verified
  sandbox after staging, and identify the direct harness exec using its exact argv.
  Track descendants from process lifecycle context; never persist those lifecycle
  records in syscall JSONL. No host tracing and no in-sandbox strace fallback.
- Add explicit category/manifest validation without host deserialization; conflicts
  and missing assignments stop before sandbox execution.
- Preserve schema-identical ordered entry/exit records in category directories;
  metadata, observations, features and progress stay separate. Publish immutable
  artifacts, record losses/invalid records/partial capture and preserve old data.
- Add configurable evidence-based observation rules and reproducible offline
  1/2/3/4-grams using only entries, with separate per-thread experiments.
- Integrate facade, runtime adapter, existing progress page and tests. Keep low-level
  Phase 1 sandbox APIs compatible; update model-level tests for the new strict policy.

No downloads, dataset crawling, training, scoring or unrelated UI redesign.
