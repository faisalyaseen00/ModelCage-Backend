# Phase 2 backend collection

## Automatic upload workflow

1. Upload a `.pkl`/`.pickle` and select one of the three supported categories.
2. The backend quarantines the file and runs the loading/deserialization harness
   in a disposable Docker/runsc sandbox, with the existing isolation limits.
3. SecCheck captures syscalls. Immutable raw entry/exit records are saved first.
4. After capture and cleanup, the backend counts syscall **entries**, separately
   calculates relative frequencies and saves the category CSVs automatically.
5. Model identity, category evidence, capture status and execution outcome are
   saved in a separate metadata CSV. No download action is necessary.

Restart the existing server to activate this change:

```bash
sudo .venv/bin/python -m modelcage.web --port 8081
```

## Backend files

Paths are relative to the backend database directory (normally sandbox-manager):

```text
data/
  raw/<category>/<analysis_id>.jsonl
  metadata/<analysis_id>.json
  metadata.csv
  features/
    text_generation.csv
    text_classification.csv
    image_classification.csv
    text_generation.normalized.csv
    text_classification.normalized.csv
    image_classification.normalized.csv
    schema.json
    <category>/<analysis_id>.json
```

Each category CSV has one row per complete capture, including model-loading
failures. The columns start with `Model`, `Category`, `Family`, then `read`,
`write`, `openat`, `execve`, `connect` and the remaining syscall counts. A trailing
`analysis_id` joins the row to separate execution metadata. Family is supplied
through the optional upload field or manifest; absent values are `Unknown`. A small example:

```csv
Model,Category,Family,read,write,openat,execve,connect,open
model-01.pkl,text_generation,Unknown,115,5,0,0,0,78
```

This example abbreviates the full vocabulary; `schema.json` records its order.
Absent syscalls are zero. `open` and `openat` stay separate. Entry/exit pairs count
once. The normalized companion divides each count by the total entry count;
nonempty rows sum to one within floating-point precision. Identifier columns are
not training inputs. Scaling for an OCSVM is a later operation fitted on training
data only.

The loading profile includes interpreter startup, imports, deserialization and
children. It does **not** measure inference, and it does not establish that a
model is benign. Only `.pkl`/`.pickle` up to the upload limit of 64 MiB is supported.
The current runner contains Python/stdlib, not PyTorch, Transformers or sklearn;
artifacts requiring unavailable dependencies fail and their outcome is recorded.
No model code is executed or imported on the host.

## Metadata and incomplete captures

Every valid, nonempty trace also gets `data/frequencies/<analysis_id>.csv` in
the same wide dataset format (one model execution per row). Execution and capture
status remain in metadata. Failed loading runs are measurements only and remain
excluded from `data/training/`. Partial captures can be inspected in the per-run
file but do not enter the complete-capture category datasets.

`metadata.csv` joins by `analysis_id` and includes original filename, hash (blank
while paused), category, category provenance, format, profile, runtime, timestamps,
status, raw trace path and feature/training inclusion fields. Family/architecture/
provider are recorded when supplied, not inferred from filenames.

The manual hashing pause remains enabled. Complete captures still enter the
category **measurement** CSVs with `feature_row_included=True`; metadata truthfully
records their operator-declared category and missing hash. No deduplication is
claimed when hashing is paused.

Timed-out, empty, lossy, mismatched or incomplete captures do not enter
those measurement CSVs. Their raw events and metadata remain available for review.
The CSVs are regenerated from immutable records under a process lock and replaced
atomically, so repeated rebuilds do not append duplicate execution rows. Unreadable metadata stops a rebuild without replacing existing CSVs. Each
build uses one metadata snapshot; each CSV is atomic, but the entire set is not a
filesystem transaction. Consumers should read after the backend reports storage
complete.

Offline rebuild without rerunning models:

```bash
sudo .venv/bin/python -m modelcage.datasets --data data
```

Historical captures with unknown execution profile are retained in metadata and
excluded from category measurements until reviewed, rather than assumed to match
the current profile. The detailed per-event trace export remains in the API for
compatibility, but the upload workflow saves backend CSVs automatically.

## Optional reviewed local corpus

The complete-capture CSVs are also published to `data/datasets/`. A stricter
derived view is written to `data/training/`: the same three category
count/normalized CSVs, plus metadata, schema and `quality.json`. It only includes
successful captures with SHA-256 identity, reviewed category evidence and an
explicit benign-corpus selection record. Exact hash duplicates are excluded;
cross-category hash conflicts require review. This view can be empty during
manual development even while `data/features/<category>.csv` has measurements.

For a reviewed batch of **local** files, create a JSON array such as:

```json
[
  {
    "model_id": "your-model-id",
    "path": "models/your-model.pkl",
    "category": "text_generation",
    "family": "documented-family",
    "architecture": "documented-architecture",
    "source": "local artifact provenance reference",
    "benign_review": "reviewer/date and reason for selecting this benign candidate",
    "category_evidence": {
      "task": "text_generation",
      "reference": "model card/configuration evidence",
      "reviewed_by": "reviewer"
    }
  }
]
```

Replace placeholders with real metadata. Paths are relative to the manifest.
The controller does not download models. It copies and hashes bytes without
loading them on the host, records exact duplicates in a SQLite registry, applies
per-family quotas and continues after a failed candidate:

```bash
sudo .venv/bin/python -m modelcage.collection reviewed-models.json \
  --workspace collection-loading --family-limit 20
```

This optional mode explicitly enables hashing only in its dedicated workspace;
it does not change the manual UI policy. Successful hashes are skipped on rerun.
Failed/interrupted records require `--retry-failed`. A single worker deliberately
matches the host's current resource budget. `quality.json` reports category balance,
family/architecture counts, exclusion reasons and identical feature vectors.
Discovery, downloading, inference, clustering and OCSVM training are outside this
upload/local-file workflow.

Use the same OS user consistently for each workspace. Mixing sudo and non-sudo
writers causes permission errors on private artifacts; no broad permission
changes are made by this implementation.

## Validation

Unit tests cover counts, normalization, empty category headers, stable rebuilding,
metadata separation, retries, deduplication, quotas, category conflicts, failed-run
exclusion and manual uploads with hashing paused. The HTTP test verifies a real
multipart upload saves backend CSVs without an export request; its sandbox capture
is mocked. The user already verified the minimal real SecCheck test under sudo.
The optional real collection smoke test is:

```bash
sudo .venv/bin/python -m pytest -q \
  tests/test_live_instrumentation.py::test_live_collection_dataset
```
