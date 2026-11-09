# Testing the Hugging Face pipeline yourself

Run every command from `sandbox-manager/`. The capture steps need `sudo`
because SecCheck reads runsc's root-owned control directory.

## 0. Prerequisites (once)

```bash
docker build -t modelcage-runner:ml-v1 -f images/model-runner.Dockerfile .
.venv/bin/pip install -e .
```

## 1. See what would be picked (no download)

```bash
.venv/bin/python -m modelcage.hub --workspace hf-test --dry-run --category text_classification --scan-limit 10
```

Each line shows `would_download` with the size and the metadata that
justified the category, or `rejected` with the reason (wrong library, wrong
pipeline tag, no `pytorch_model.bin`, too large, gated…). Nothing is written.

## 2. Check that syscall capture works

```bash
sudo .venv/bin/python -m modelcage.live_validation --output /tmp/modelcage-live-$(date +%s)
```

It must report `validated: true`. If capture fails here, step 3 will record
the model as `instrumentation_unavailable` instead of producing a dataset row.

## 3. Collect one real model

```bash
sudo .venv/bin/python -m modelcage.hub --workspace hf-test --category text_classification --per-category 1
```

This validates the category, checks the registry by SHA-256, downloads
`pytorch_model.bin`, verifies its hash, runs `torch.load` in a fresh offline
gVisor sandbox while SecCheck records syscalls, stores the results and
destroys the sandbox. The printed result should show `"status": "succeeded"`
and a `model_folder`.

## 4. Inspect what was stored

```text
hf-test/
  models/text_classification/<org>__<model>/
    pytorch_model.bin        downloaded checkpoint (read-only, same bytes as quarantine)
    model.json               repo, revision, SHA-256, category reason, status, analysis_id
  quarantine/<sha256>.pkl    the quarantined original the sandbox received
  data/raw/text_classification/<analysis_id>.jsonl   raw syscall events
  data/features/text_classification.csv              one row per model (syscall counts)
  data/metadata.csv                                  execution and identity metadata
  registry.sqlite                                    models / attempts / hub_candidates
  collection-report.json                             this run's results
```

```bash
sudo cat hf-test/models/text_classification/*/model.json
sudo head -c 600 hf-test/data/features/text_classification.csv
sudo sqlite3 hf-test/registry.sqlite 'SELECT repo_id, status, reason FROM hub_candidates'
```

## 5. Check deduplication

Run step 3 again. The model from the first run is skipped without
re-downloading, because its repository revision is already recorded, and the
next suitable model is collected instead (`--per-category` counts new
successes in each run). A different repository publishing identical bytes
is reported as `duplicate` before download.

The models folder holds untrusted checkpoints. Never `torch.load` them on the
host; only ModelCage's sandbox loads them.
