# Phase 2 architecture alignment

The supplied diagram is the architectural reference: model submission feeds metadata
identification and the sandbox orchestrator; Docker/runsc bounds execution and security
controls; execution feeds strace and status/logging; trace processing feeds later category
datasets. Existing manager, runtime adapter, runner, quarantine, SQLite and JSONL stores
remain the implementation boundaries.

Current gaps: blocking web submission, no shared progress journal, no tracing capability
probe, no live counts, failure cleanup preceding raw storage, unsafe latest-trace lookup
under concurrency, and stale operating instructions.

Plan:
1. Add `app/execution_events.py`: durable, analysis-scoped lifecycle journal, snapshot and
   standard Python logging from the same records, separate from syscall JSONL.
2. Extend `docker_runtime.py` and `manager.py`: verify tracing with a harmless workload,
   publish actual count updates, retain partial timeout captures, and allow the facade to
   flush before final cleanup while preserving manager-only cleanup defaults.
3. Extend `modelcage.py`: one analysis identity, truthful transitions, exact trace lookup,
   independent persistence and cleanup error handling, final status after cleanup.
4. Extend `web.py`: asynchronous submission, current-status API and polling of the journal
   from the existing upload page. Reconnect/reload uses the analysis URL. No new framework.
5. Add unit and opt-in Docker integration checks; update README and status documentation.

Scope discrepancies: `.pt` in the diagram has no existing runner and remains unsupported;
category is explicitly selected, not inferred; metadata JSON replaces the diagram's
metadata CSV in this phase. Syscall frequencies, category datasets and Iteration 02 are
explicitly deferred by the brief. Existing per-execution CSV exports remain unchanged.
Strace captures file/process calls from the direct runner only, not descendants or host
calls. In-sandbox trace files and PID handshake are not tamper-proof against hostile code.
