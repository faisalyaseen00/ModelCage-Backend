# ModelCage Backend

ModelCage loads untrusted ML model files inside a disposable gVisor sandbox,
records every system call the load makes, and turns those calls into per-category
datasets. The goal is to train a One-Class SVM per category on benign models, so
that a malicious pickle (one that runs code when loaded) stands out.

All code lives in `sandbox-manager/`. Run every command from that folder.

---

## 1. How the automated pipeline works

```text
huggingface.co public API ─► pick model ─► download ─► quarantine ─► gVisor sandbox
                                                                    │  torch.load()
                                                                    ▼
             category CSVs ◄─ features ◄─ raw JSONL ◄─ SecCheck syscall stream
```

| Step | What happens | Code |
|---|---|---|
| 1. Discover | Ask the public Hugging Face API for the most-downloaded `transformers` models per task (`text-generation`, `text-classification`, `image-classification`) | `app/hub.py` `HubClient.list_models` |
| 2. Validate category | Accept a model only if library, pipeline tag and `config.architectures` all agree on one category; record the reason | `app/hub.py` `validate_hub_model`, `app/categories.py` |
| 3. Pick checkpoint | Need one `pytorch_model.bin` ≤ 1 GiB; safetensors-only, sharded, gated or oversized repos are rejected | `app/hub.py` `select_checkpoint` |
| 4. Skip early | The hub publishes each file's SHA-256. Already-registered hashes and full family quotas are skipped before downloading | `app/hub.py`, `app/collection.py` |
| 5. Download | Stream the file at the exact pinned revision; size and SHA-256 must match the hub | `HubClient.download` |
| 6. Quarantine | Read-only, content-addressed copy in `quarantine/<sha256>.pkl`, re-verified before use. The host never unpickles a model | `app/quarantine.py` |
| 7. Sandbox | New Docker container on the `runsc` (gVisor) runtime: no network, read-only root, 4 GiB, 1 CPU, 64 PIDs, 60 s, no capabilities | `app/docker_runtime.py`, `app/manager.py`, `app/policy.py` |
| 8. Load | A fixed runner script calls `torch.load(..., weights_only=False)` inside the sandbox, so any pickle payload runs there | `app/model_runner.py` |
| 9. Capture | gVisor SecCheck streams each syscall entry/exit over a private Unix socket to the host. Only the runner process tree is kept | `app/seccheck.py` |
| 10. Store | Immutable raw JSONL per run, plus metadata, observations and progress logs | `app/raw_events.py`, `app/storage.py`, `app/artifact_metadata.py` |
| 11. Features | Syscall counts, normalized frequencies, 1/2/3-grams; one CSV row per complete capture, per category | `app/features.py`, `app/datasets.py` |
| 12. Clean up | Container destroyed every time, including timeouts and errors | `app/manager.py` |
| 13. Registry | SQLite registry of every model, candidate and attempt; model kept under `models/<category>/` | `app/collection.py`, `app/hub.py` |

The orchestration entry point for a single model is `ModelCage.execute_model` in
`app/modelcage.py`. The web UI (`app/web.py`) and the Hugging Face collector both
go through it.

Full detail: `sandbox-manager/docs/automated-dataset-pipeline.md`.

---

## 2. Hugging Face access: why no credentials are needed

The collector only reads **public** data. Hugging Face lets anyone list public
models, read their metadata and download their files over plain HTTPS without an
account, the same as opening a model page in a browser. The client
(`app/hub.py` `HubClient`) uses Python's standard `urllib` and calls three URLs:

| Purpose | URL |
|---|---|
| List models | `https://huggingface.co/api/models?pipeline_tag=…&filter=transformers&sort=downloads&full=true` |
| Model metadata, file sizes and SHA-256 | `https://huggingface.co/api/models/<repo>?blobs=true` |
| Download | `https://huggingface.co/<repo>/resolve/<revision>/pytorch_model.bin` |

- No token is stored in the repo and none was used.
- `HF_TOKEN` is optional; it only helps if Hugging Face rate-limits anonymous requests.
- Gated or private models (for example Llama) need an account and are rejected, not bypassed.
- Nothing is uploaded to Hugging Face.

Anyone can check this with a browser or `curl` on the URLs above, or with the
dry run (`.venv/bin/python -m modelcage.hub --workspace hf-data --dry-run`).

---

## 3. Syscall capture: what is real and what has been verified

Capture uses gVisor's own SecCheck feature (`runsc trace`). gVisor implements the
Linux kernel interface in user space, so it sees every syscall the sandboxed
process makes and reports it. There is no strace fallback and no simulated data
in the real path; if SecCheck cannot attach, the run is recorded as
`instrumentation_unavailable` and the model is not executed.

The decoder only runs against the gVisor release it was checked against
(`RELEASE` in `app/seccheck.py`, currently `release-20261019.0`). If apt upgrades
`runsc`, every run becomes `instrumentation_unavailable` until the new release's
tracepoints (`runsc trace metadata`) and `pkg/sentry/seccheck/points/*.proto` are
compared and the pin is updated. `sudo apt-mark hold runsc` prevents surprise upgrades.

Root access is needed because runsc's control directory
(`/run/docker/runtime-runc/moby`) is root-owned. That is why collection runs under `sudo`.

**Verification status (as of 2026-11-18):**

| Claim | Status | Evidence |
|---|---|---|
| Real SecCheck capture works on this machine | Verified once, by hand | `data/raw/text_generation/f236d3bf-….jsonl` (2026-10-12): 2,980 records, 1,490 syscalls, 0 lost events, `event_source: gvisor_seccheck`, real container ID. Top calls: `mmap`, `munmap`, `stat`, `lseek`, `fstat`, `read`, `open`. That model failed to load (missing class), but capture itself completed. |
| Minimal live SecCheck test under sudo | Passed (run by the user) | Recorded in `sandbox-manager/IMPLEMENTATION_STATUS.md` |
| Full live validation suite | **Not yet passing** | 2026-10-14: 5 passed, 12 failed, 4 skipped (sudo/runsc permissions). 2026-11-18 under sudo: same counts, because apt had upgraded runsc to `release-20261019.0` (2026-10-29) and the decoder refuses unvalidated releases. The pin is now moved to that release (identical tracepoints and wire schema); rerun to confirm |
| Hugging Face discovery and category checks | Verified live | Dry run on 2026-11-18 returned real candidates in all three categories |
| End-to-end Hugging Face download → sandbox → capture → dataset | **Not yet run** | Needs `sudo`; unit tests cover it with a fake hub and simulated capture only |

Unit tests use synthetic SecCheck packets (`tests/seccheck_fixtures.py`). They prove
the decoding and bookkeeping logic, **not** live capture. Live proof comes only
from `modelcage.live_validation` and real collection runs under sudo.

### How to prove capture is real

```bash
cd sandbox-manager
sudo .venv/bin/python -m modelcage.live_validation --output /tmp/modelcage-live-$(date +%s)   # must say validated: true
sudo .venv/bin/python -m modelcage.hub --workspace hf-data --per-category 1
sudo head -c 600 hf-data/data/raw/*/*.jsonl          # real container_id, gvisor_seccheck, syscall names
sudo python3 -m json.tool hf-data/collection-report.json
```

Different models should give different syscall counts in
`hf-data/data/features/<category>.csv`; identical vectors are flagged in
`data/training/quality.json`.

---

## 4. Work done so far (from git history)

These were built across several Claude Code sessions with Faisal Yaseen, and
partly by Waseem Sajjad. Sessions do not share memory; this list is reconstructed
from commits and docs.

| Date | Commit | Delivered |
|---|---|---|
| 2026-09-06 | `69b94b3` | Initial sandbox manager |
| 2026-10-13 | `c08c53a` | Iteration 1: disposable Docker sandbox, pickle harness, web upload UI |
| 2026-10-14 | `e4f6fc3` | SHA-256 identification, quarantine, deduplication, local-manifest batch collector |
| 2026-10-19 | `9fac900`, `a23cb89`, `8c08716`, `33ccd95` | Web UI refresh, upload details, browser security headers, health endpoint |
| 2026-10-19 | `17247f2` | gVisor SecCheck instrumentation replacing strace; per-sandbox receiver; raw JSONL |
| 2026-10-26 | `a0ad698` | Automatic category identification (symbolic pickle inspection), features, datasets, live-validation command |
| 2026-11-10 | `c6fd417` | ML runner image (PyTorch, Transformers), `.bin/.pt/.pth` loading, Hugging Face collector |
| 2026-11-10 | `f535638` | Keep collected models in `models/<category>/`, `--dry-run`, resume fix, pipeline guide |

How the main requirements were met:

- **Isolation:** every model runs in a fresh gVisor container with no network and strict limits; never on the host (`app/policy.py`).
- **Real syscall capture:** gVisor SecCheck, all 884 syscall entry/exit points, filtered to the model's process tree (`app/seccheck.py`).
- **Three category datasets:** category is proven from hub metadata or symbolic inspection, never guessed from names (`app/categories.py`).
- **Automation:** one command discovers, validates, downloads, runs, captures and builds datasets (`python -m modelcage.hub`).
- **No duplicates, diverse data:** SHA-256 dedup before download, per-family quotas.
- **Traceability:** every accept/reject reason, hash, revision and outcome is stored in `registry.sqlite`, `metadata.csv` and `model.json`.

Not built yet: OCSVM training, detection rules, risk scoring, inference-time
capture, parallel workers.

---

## 5. Working on this repo

- Python 3.11 or 3.12 (`sandbox-manager/.venv`). 3.13 breaks the `cgi` upload parser.
- Tests: `.venv/bin/python run_tests.py`, or `.venv/bin/python -m pytest -q -m 'not integration and not network'`.
  As of 2026-11-18, 15 quarantine/hashing tests fail because `MODEL_HASHING_ENABLED = False` in `app/policy.py`.
- Web UI: `.venv/bin/python -m modelcage.web` → http://127.0.0.1:8080
- Never `torch.load` or `pickle.load` a collected model on the host; they are untrusted.
- Changing the runner image or its libraries changes syscall profiles: bump `EXECUTION_PROFILE` in `app/policy.py`.
- Collection workspaces must always be used with the same user (`sudo`).
- Docs: `sandbox-manager/README.md`, `IMPLEMENTATION_STATUS.md`, `docs/`.
