# ModelCage automated dataset pipeline

This guide explains how ModelCage automatically collects models from Hugging
Face, runs each one in a disposable gVisor sandbox, captures its system calls
and builds the three category datasets. It covers setup, commands, every
decision the pipeline makes, the files it produces, and how to check them.

Run every command from `sandbox-manager/`.

---

## 1. Overview

```text
Hugging Face listing (per category, ranked by downloads)
  │
  ├─ 1. Category check      hub metadata: pipeline tag, config architectures, tags, library
  ├─ 2. Checkpoint pick     single pytorch_model.bin ≤ 1 GiB with LFS SHA-256 and size
  ├─ 3. Already processed?  SHA-256 in registry → duplicate; family quota full → skip   (no download)
  ├─ 4. Download            pinned revision, size and SHA-256 must match the hub
  ├─ 5. Quarantine          private, read-only, content-addressed copy; re-verified before use
  ├─ 6. Sandbox             fresh Docker + runsc (gVisor) container, offline, limited resources
  ├─ 7. Load                torch.load(weights_only=False) inside the sandbox only
  ├─ 8. Capture             gVisor SecCheck streams every syscall entry/exit to the host
  ├─ 9. Store raw events    immutable JSONL, one file per execution
  ├─ 10. Features           syscall counts, normalized frequencies, 1/2/3-gram sequences
  ├─ 11. Dataset row        appended to the model's category CSV
  ├─ 12. Destroy sandbox    container removed, its writable state discarded
  └─ 13. Registry + folder  status recorded, model kept under models/<category>/
```

The three categories each get their own dataset, and later their own OCSVM:

| Category | Hugging Face pipeline tag | Example accepted architectures |
|---|---|---|
| `text_generation` | `text-generation` | `GPT2LMHeadModel`, `OPTForCausalLM`, `LlamaForCausalLM` |
| `text_classification` | `text-classification` | `BertForSequenceClassification`, `RobertaForSequenceClassification` |
| `image_classification` | `image-classification` | `ViTForImageClassification`, `ConvNextForImageClassification` |

The full list of accepted architectures is `ARCHITECTURES` in `app/categories.py`.

---

## 2. One-time setup

| Requirement | Check |
|---|---|
| Ubuntu with Docker | `docker info` |
| gVisor registered as Docker runtime `runsc` | `docker info --format '{{json .Runtimes}}'` lists `runsc` |
| Python 3.11 or 3.12 venv | `.venv/bin/python --version` |
| Root access (sudo) | needed so SecCheck can read runsc's control directory |
| Internet access on the host | only the host downloads; sandboxes stay offline |
| Disk space | about 1.5 GB for the image, plus the size of each model you collect |

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
docker build -t modelcage-runner:ml-v1 -f images/model-runner.Dockerfile .
```

The runner image `modelcage-runner:ml-v1` contains Python 3.12.7, CPU PyTorch
2.5.1, torchvision 0.20.1, Transformers 4.46.3, scikit-learn 1.5.2, NumPy 2.1.3
and safetensors 0.4.5. All versions are pinned, so every model is loaded in the
same environment.

Optional: set `HF_TOKEN` if Hugging Face rate-limits anonymous requests. The
token is only sent to huggingface.co.

---

## 3. Commands

### 3.1 Preview the selection (no download, no sudo)

```bash
.venv/bin/python -m modelcage.hub --workspace hf-data --dry-run --scan-limit 20
```

For each listed model this prints `would_download` (with size and the
metadata that justified the category) or `rejected` (with the reason). It
writes nothing.

### 3.2 Confirm that syscall capture works

```bash
sudo .venv/bin/python -m modelcage.live_validation --output /tmp/modelcage-live-$(date +%s)
```

This must report `validated: true`. If capture is unavailable, the pipeline
refuses to run models and records them as `instrumentation_unavailable`; it
never falls back to an untraced or non-gVisor run.

### 3.3 Collect

```bash
# All three categories, 5 new successful models each
sudo .venv/bin/python -m modelcage.hub --workspace hf-data --per-category 5

# One category
sudo .venv/bin/python -m modelcage.hub --workspace hf-data --category image_classification --per-category 10

# Several categories, search deeper in the listing
sudo .venv/bin/python -m modelcage.hub --workspace hf-data \
  --category text_generation --category text_classification --per-category 20 --scan-limit 500
```

| Option | Default | Meaning |
|---|---|---|
| `--workspace` | required | Folder that holds everything for this collection |
| `--category` | all three | Repeat to select several categories |
| `--per-category` | 5 | Stop a category after this many **new** successful models in this run |
| `--scan-limit` | 100 | How many hub models to examine per category (maximum 1000) |
| `--family-limit` | 20 | Maximum successful models per family (config `model_type`) per category |
| `--retry-failed` | off | Re-run models that previously failed or were interrupted |
| `--dry-run` | off | Only list and validate; no download, sandbox or writes |

Use the same OS user (always `sudo`) for one workspace; mixing users causes
permission errors on private files.

### 3.4 Resume or extend

Run the same command again. The pipeline:

- skips repositories already `succeeded`, `rejected` or `duplicate` at the same revision, without contacting the download URL;
- skips any checkpoint whose SHA-256 is already registered, before downloading it;
- marks runs left `running` by an interrupted process as `interrupted` (rerun them with `--retry-failed`);
- holds a lock, so two collectors cannot work on one workspace at once.

### 3.5 Rebuild the datasets offline

Features can be regenerated from the stored raw events without running any
model again, for example after changing feature code:

```bash
sudo .venv/bin/python -m modelcage.datasets --data hf-data/data
```

---

## 4. What happens to each model

### 4.1 Category check (before download)

A model is accepted for category *C* only if **all** of these hold:

1. The repository is public, not gated and not disabled.
2. `library_name` is `transformers`.
3. `pipeline_tag` equals *C*'s pipeline tag.
4. `config.architectures` contains a recognised head for *C*.
5. The combined hub evidence (pipeline tag, task tags, architectures) resolves
   to *C* alone. Any conflicting evidence, such as an image head on a
   text-classification repository, rejects the model.

The acceptance reason is saved, for example:

```text
pipeline_tag=text-classification, architectures=['BertForSequenceClassification'], library=transformers, model_type=bert
```

It is stored in the registry (`models.details.category_acceptance`), in the
run metadata, and in the model folder's `model.json`. Rejected models are
stored in `hub_candidates` with their reason.

Common rejection reasons:

| Reason | Meaning |
|---|---|
| `library is 'timm' / 'sentence-transformers', not transformers` | Not a Transformers model |
| `pipeline tag ... does not match` | Listed under a different task |
| `config architectures [...] contain no recognised <category> head` | Base model or an architecture not in the list |
| `no single-file pickle-based PyTorch checkpoint` | Only safetensors available (safetensors cannot run code when loaded) |
| `only a sharded PyTorch checkpoint is available` | Split `pytorch_model-0000x-of-0000y.bin` files |
| `... above the 1073741824 byte limit` | Larger than 1 GiB |
| `repository is private, gated or disabled` | Cannot or should not be downloaded |

### 4.2 Deduplication and diversity (before download)

- **Exact duplicates:** the SHA-256 comes from the hub's LFS metadata. If it
  is already in the registry, the model is recorded as `duplicate` and not
  downloaded, even under a different repository name or filename.
- **Category conflict:** if the same SHA-256 is registered under another
  category, the model is rejected.
- **Family quota:** once a family (config `model_type`, such as `bert` or
  `vit`) has `--family-limit` successful models in a category, further models
  of that family are recorded as `quota_skipped` and not downloaded. This stops
  one family from dominating a dataset.

### 4.3 Download and quarantine

- The file is streamed from `huggingface.co/<repo>/resolve/<revision>/pytorch_model.bin`
  at the exact revision that was validated.
- Size and SHA-256 must match the hub metadata, or the model is rejected and
  never reaches a sandbox.
- The bytes are copied into `quarantine/<sha256>.pkl` (read-only,
  content-addressed, never overwritten) and verified again just before they
  are staged into the sandbox.
- The host never imports, unpickles or loads a model. Category inspection
  reads the archive's `data.pkl` symbolically, without executing it.

### 4.4 Sandbox and loading

Each model gets a new container that is destroyed afterwards. Limits come from
`app/policy.py`:

| Setting | Value |
|---|---|
| Runtime | `runsc` (gVisor), verified before start; no fallback to `runc` |
| Network | `none` |
| Root filesystem | read-only; only `/tmp` is a writable tmpfs |
| Capabilities | all dropped; never privileged |
| Memory | 4 GiB (the staged model in `/tmp` counts towards it) |
| CPU | 1 |
| Processes/threads | 64 |
| Timeout | 60 seconds |
| Host mounts, Docker socket, host PID or network namespace | none |

Inside the sandbox the fixed runner calls
`torch.load(path, map_location="cpu", weights_only=False)`, the same way a
victim's `from_pretrained` or `torch.load` would, so any pickle code in the
checkpoint executes here. `.pkl`/`.pickle` files use `pickle.load` instead.

Cleanup always runs, including after a timeout, a crash, an exception or a
Docker error.

### 4.5 Syscall capture

- gVisor SecCheck streams events over a private per-sandbox socket to the host.
- It subscribes to every available syscall entry and exit point (884 points
  on runsc `release-20260907.0`).
- Only the runner process and its children and threads are recorded. Docker,
  runsc, container setup and other sandboxes are excluded.
- Lost events, invalid packets, unmatched entry/exit pairs and the 250,000
  record cap are all recorded. Any of them marks the capture incomplete, and
  incomplete captures are kept out of the datasets.

The capture covers interpreter start-up, `import torch` and deserialization.
It does **not** cover inference (running the model on an input).

---

## 5. Output files

```text
hf-data/
├── models/<category>/<org>__<model>/
│   ├── pytorch_model.bin          downloaded checkpoint (read-only hard link to quarantine)
│   └── model.json                 repo, revision, URL, SHA-256, category reason, status, analysis_id
├── quarantine/<sha256>.pkl        quarantined original (read-only)
├── registry.sqlite                model registry (section 6)
├── modelcage.sqlite               execution store
├── collection-report.json         results of the latest run
└── data/
    ├── raw/<category>/<analysis_id>.jsonl          raw syscall events (immutable)
    ├── observations/<analysis_id>.jsonl            notable behaviour (network, exec, writes…)
    ├── progress/<analysis_id>.progress.jsonl       lifecycle log of the run
    ├── metadata/<analysis_id>.json                 full metadata of one execution
    ├── metadata.csv                                one row per execution, every outcome
    ├── frequencies/<analysis_id>.csv               per-run syscall counts
    ├── features/
    │   ├── <category>.csv                          syscall counts, one row per complete capture
    │   ├── <category>.normalized.csv               same rows divided by total syscalls
    │   ├── <category>/<analysis_id>.json           1/2/3-gram features (global and per thread)
    │   └── schema.json                             column order and schema version
    ├── datasets/                                   published complete-capture CSVs + quality.json
    └── training/                                   strict training view (section 7)
```

The `models/` folder contains untrusted checkpoints. Never `torch.load` them on
the host.

### 5.1 Raw event (`data/raw/<category>/<analysis_id>.jsonl`)

One JSON object per syscall entry or exit, in collector order. Key fields:

| Field | Example | Meaning |
|---|---|---|
| `analysis_id` | `f236d3bf-…` | Links to metadata and dataset rows |
| `sequence` | `1`, `2`, … | Strictly increasing order |
| `event_source` | `gvisor_seccheck` | Only SecCheck events are accepted |
| `tracepoint` | `syscall/openat/enter` | gVisor tracepoint |
| `syscall_name` | `openat` | Syscall |
| `phase` | `enter` / `exit` | Entry or exit |
| `process_id`, `thread_id`, `process_name` | `4`, `4`, `python3` | Who made the call (sandbox PIDs) |
| `arguments` | raw registers | Pointers are never dereferenced |
| `return_value`, `error_status` | | Exit result |
| `container_id`, `sandbox_id` | | Must match the run's sandbox |
| `reported_dropped_events` | `0` | Losses reported by gVisor |

### 5.2 Category dataset (`data/features/<category>.csv`)

One row per successful, complete capture:

| Column | Meaning |
|---|---|
| `Model` | Hugging Face repo ID |
| `Category` | `text_generation`, `text_classification` or `image_classification` |
| `Family` | config `model_type` |
| `read`, `write`, `openat`, `execve`, `connect` | listed first for readability |
| … every other x86-64 syscall (451 columns) | number of **entries** of that syscall; absent means 0 |
| `analysis_id` | join key to `metadata.csv` and the raw events |

Example (abbreviated):

```csv
Model,Category,Family,read,write,openat,execve,connect,...,analysis_id
distilbert/distilbert-base-uncased-finetuned-sst-2-english,text_classification,distilbert,1432,24,411,1,0,...,9c1e…
```

- Each entry/exit pair counts once; `open` and `openat` are separate columns.
- `<category>.normalized.csv` divides each count by the row total, so each row sums to 1.
- `Model`, `Category`, `Family` and `analysis_id` are identifiers, not training inputs. Train the OCSVM on the syscall columns only, and fit any scaling on training data only.
- The files are rebuilt atomically from the raw events after each run, so rows are never duplicated.

### 5.3 N-gram features (`data/features/<category>/<analysis_id>.json`)

Ordered syscall sequences (entries only) as 1-, 2- and 3-grams with
frequencies, computed for the whole run (`global`, collector order) and
separately for each thread (`per_thread`). 1-grams are the baseline; 2- and
3-grams are for experiments.

### 5.4 Metadata (`data/metadata.csv`)

One row per execution, including failures. Columns include `analysis_id`,
`model_id`, `model_sha256`, `size_bytes`, `task_category`, `family`,
`architecture`, `provider`, `library`, `source` (`huggingface:<repo>@<revision>`),
`revision`, `category_source`, `validation_status`, `category_evidence`,
`category_reason`, `benign_review`, `execution_profile` (`pickle-load-ml-v1`),
`runner_image`, `runtime_version`, `status`, `capture_status`, `exit_code`,
`failure_reason`, `start_time`, `end_time`, `reported_losses`, `event_count`,
`raw_events`, `feature_row_included`, `dataset_eligible` and `exclusion_reasons`.

---

## 6. Model registry (`registry.sqlite`)

| Table | Columns | Purpose |
|---|---|---|
| `models` | `sha256` (primary key), `category`, `family`, `analysis_id`, `status`, `details` (JSON: model ID, architecture, provider, source, revision, library, checkpoint, category acceptance and evidence) | One row per distinct artifact; the deduplication key is the SHA-256, never the filename |
| `hub_candidates` | `repo_id`, `revision`, `category`, `filename`, `sha256`, `status`, `reason` | Every Hugging Face repository examined, including rejections |
| `attempts` | `id`, `model_id`, `status`, `reason`, `sha256`, `analysis_id` | Every attempt in order, for auditing |

Useful queries:

```bash
sudo sqlite3 -header -column hf-data/registry.sqlite \
  "SELECT category, family, count(*) FROM models WHERE status='succeeded' GROUP BY 1,2"
sudo sqlite3 -header -column hf-data/registry.sqlite \
  "SELECT status, count(*) FROM hub_candidates GROUP BY 1"
sudo sqlite3 -header -column hf-data/registry.sqlite \
  "SELECT repo_id, reason FROM hub_candidates WHERE status='rejected' LIMIT 20"
```

### Result statuses

| Status | Meaning | Downloaded? | Dataset row? |
|---|---|---|---|
| `succeeded` | Loaded and fully captured | yes | yes |
| `failed` | Model failed to load (for example a missing class), or a pipeline error | yes | no (kept in metadata and raw events) |
| `timed_out` | Exceeded 60 s; sandbox killed | yes | no |
| `incomplete` | Capture lost or invalid events | yes | no |
| `instrumentation_unavailable` | SecCheck could not attach; model not run | yes | no |
| `rejected` | Failed category, checkpoint or hash checks | no (or deleted on hash mismatch) | no |
| `duplicate` | SHA-256 already registered | no | no |
| `quota_skipped` | Family quota reached | no | no |
| `interrupted` | Collector stopped mid-run; rerun with `--retry-failed` | yes | no |

---

## 7. Dataset quality and the training view

Each rebuild writes `data/datasets/quality.json` and `data/training/quality.json` with:

- analyses per category (balance);
- counts per family and per architecture (diversity);
- exclusion counts by reason;
- `identical_vectors`: groups of runs with exactly the same syscall counts.

`data/training/` holds the stricter set intended for OCSVM training. A run is
included only if it passes every check:

| Exclusion reason | Rule |
|---|---|
| `execution_not_successful` | status must be `succeeded` |
| `capture_incomplete` | capture complete and SecCheck verified |
| `losses_unknown_or_nonzero` | gVisor reported zero dropped events |
| `capture_errors`, `unmatched_events`, `invalid_raw_events`, `event_count_mismatch` | raw stream consistent with metadata |
| `cleanup_incomplete` | sandbox confirmed destroyed |
| `hashing_paused_or_missing` | SHA-256 present |
| `category_not_validated` | category validated during collection |
| `benign_not_reviewed` | a benign-selection record is present |
| `unsupported_profile` | profile is `pickle-load-ml-v1` (older Alpine-image rows are excluded) |
| `unsupported_category` | one of the three categories |
| `duplicate_sha256`, `duplicate_category_conflict` | first copy of each SHA-256 only; conflicts excluded |

Models collected from Hugging Face carry the benign-selection record
`automated: public, ungated Hugging Face transformers model ranked by downloads;
not manually reviewed for benignness`. That makes them eligible for
`data/training/`. Review `collection-report.json` and the registry before you
train on them.

---

## 8. Checking a run

```bash
# Summary of the latest run
sudo python3 -m json.tool hf-data/collection-report.json | head -40

# Models kept, and why each was accepted
sudo cat hf-data/models/*/*/model.json

# Rows per category dataset (minus the header)
for f in hf-data/data/features/*_*.csv; do echo "$f $(( $(sudo wc -l < "$f") - 1 ))"; done

# Raw events of one run
sudo head -3 hf-data/data/raw/text_classification/<analysis_id>.jsonl

# Balance, diversity and exclusions
sudo python3 -m json.tool hf-data/data/training/quality.json
```

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Every model is `instrumentation_unavailable` | SecCheck cannot read runsc state. Run with `sudo`, or set `MODELCAGE_RUNSC_ROOT` / `MODELCAGE_RUNSC_COMMAND` to an authorised setup. Check with section 3.2. |
| `failed` with "missing class" or "missing dependency" | The checkpoint needs a package or custom code not in the runner image. The outcome is recorded; add the package to the Dockerfile only if the category needs it. |
| Many `timed_out` | Large checkpoints on one CPU. Raise `EXECUTION_TIMEOUT_SECONDS` in `app/policy.py`. |
| Many `rejected` for size | Raise `MAX_MODEL_BYTES`, and `MEMORY_LIMIT_BYTES` with it (memory must hold about twice the model plus around 0.5 GB). |
| Few `text_generation` candidates | Modern LLMs ship safetensors only or are too large. Increase `--scan-limit`. |
| HTTP 429 from Hugging Face | Set `HF_TOKEN` or retry later. |
| `BlockingIOError` on start | Another collector is using the workspace. |
| Permission denied on workspace files | The workspace was created by another user. Always use `sudo` for it. |

---

## 10. Configuration reference (`app/policy.py`)

| Constant | Value | Used for |
|---|---|---|
| `MODELCAGE_RUNNER_IMAGE` | `modelcage-runner:ml-v1` | Sandbox image |
| `EXECUTION_PROFILE` | `pickle-load-ml-v1` | Recorded per run; datasets only mix identical profiles |
| `MAX_MODEL_BYTES` | 1 GiB | Upload, collection and inspection limit |
| `MEMORY_LIMIT_BYTES` | 4 GiB | Sandbox memory |
| `CPU_LIMIT_NANO_CPUS` | 1 CPU | Sandbox CPU |
| `PID_LIMIT` | 64 | Sandbox processes and threads |
| `EXECUTION_TIMEOUT_SECONDS` | 60 | Per-model timeout |
| `NETWORK_MODE` | `none` | Sandbox network |

Changing the runner image or its libraries changes the syscall profile. When
you do, also change `EXECUTION_PROFILE`, so old and new measurements are not
mixed.

---

## 11. Current limitations

- **End-to-end live run not yet confirmed.** Unit tests cover the pipeline with
  a fake hub and simulated capture. Run section 3.2 and a small section 3.3
  collection to confirm it on your machine.
- **Loading only.** No inference, so behaviour that only appears when the model
  runs is not captured.
- **Formats.** Single-file `pytorch_model.bin` from Transformers repositories
  only; no sharded checkpoints, `timm`, sentence-transformers or other libraries.
- **One model at a time.** There are no parallel workers yet.
- **Similarity analysis.** Only exactly identical feature vectors are flagged;
  there is no clustering yet.
- **Benignness.** Automatically selected models are popular, ungated models,
  not manually reviewed ones.
- **Next stages not built.** OCSVM training, detection rules and risk assessment.

See also: [phase2-collection.md](phase2-collection.md) (backend formats),
[huggingface-self-test.md](huggingface-self-test.md) (a short manual test),
[live-validation.md](live-validation.md) (capture validation).
