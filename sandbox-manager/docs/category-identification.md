# Automatic category identification

Upload a `.pkl` or `.pickle` without choosing a category. The backend hashes and
quarantines it, verifies the exact bytes, extracts task evidence, then executes a
recognized model through the existing disposable runsc/SecCheck pipeline.

| Evidence | Destination |
| --- | --- |
| Recognized causal/text-generation head, e.g. GPT2LMHeadModel | text_generation |
| Recognized text sequence-classification head, e.g. BertForSequenceClassification | text_classification |
| Recognized image-classification head, e.g. ViTForImageClassification or torchvision ResNet | image_classification |
| sklearn Pipeline combining text vectorization with a recognized final classifier | text_classification |

The explicit recognized-head/class catalog is in `app/categories.py`. Generic
class-name suffix matching is not used: audio sequence classifiers must not become
text classifiers, and audio conditional generators must not become text generators.
Known unsupported heads and declared unsupported tasks are held for review.

Task/config metadata can also supply `architectures`, `pipeline_tag`, `task`,
`task_category`, or supported task tags. Embedded root, model, estimator, config
and metadata fields are inspected with bounded depth. The optional upload JSON
box accepts the model's config/model-card metadata directly. For example:

```json
{"architectures":["BertForSequenceClassification"],"model_type":"bert"}
```

All usable evidence must agree. There is no precedence rule that silently lets a
supplied label override contradictory model structure. Filenames, labels, generic
model families and standalone numerical classifiers are insufficient evidence.
Custom classes such as `GPTLanguageModel` without recognizable config/task metadata
cannot be reliably classified from their names alone. Opaque reducers, pure weight
arrays and unsupported serialization may also require metadata.

## No host model execution

`app/artifact_metadata.py` uses `pickletools.genops` to interpret only a bounded
symbolic representation of the pickle. It records references and state; it never
imports model modules, invokes reducers/constructors, or calls a pickle unpickler.
Tensor/storage bytes remain opaque. Limits are 64 MiB of input, 200,000 operations,
512-character retained pickle strings, and bounded metadata depth/list lengths.
Malformed or unsupported structures do not produce guessed task evidence.

These are declarations extracted from untrusted content, not proof of actual
behavior. A malicious artifact can misrepresent its type/config. Automatic
identification does not establish benignness and does not automatically admit a
manual upload to the reviewed training corpus. The sandbox remains mandatory for
deserialization. Existing Python-only runner dependencies are unchanged; successful
identification does not imply successful model loading, `.pt` support, or inference.

## Outputs and review cases

`auto_identified` records artifact-derived evidence; `metadata_identified` records
supplied metadata evidence. Category source, reason, detector version and evidence
are retained in analysis JSON/progress; metadata CSV includes source, status,
reason and evidence. The UI shows the detected category and resolution reason.

After capture, the existing pipeline writes:

- Raw events: `data/raw/<category>/<analysis_id>.jsonl`
- 1/2/3-gram features: `data/features/<category>/<analysis_id>.json`
- Counts: `data/features/<category>.csv`
- Relative frequencies: `data/features/<category>.normalized.csv`
- Identification/execution metadata: `data/metadata.csv`

`unclassified`, `unsupported`, or `conflict` produces `pending_review` before
sandbox creation. Such submissions retain hash and review metadata but contribute
no category feature rows. Resubmitting the same bytes with resolving metadata is
supported; an earlier unclassified record does not create a false cross-category
hash conflict. Previously finalized artifacts are not relabeled.

Explicit categories remain compatible for existing library/HTTP clients and
reviewed batches. They cannot override conflicting extracted evidence. Batch
manifests may omit `category` and `category_evidence.task` when architecture/model
metadata resolves the category; reviewed provenance and benign review remain
required for collection into the reviewed corpus.

## Verification

Tests cover protocol 0–5 embedded config, GLOBAL/STACK_GLOBAL references, all three
categories, nested sklearn pipeline state, absent dependencies, unsupported audio
heads, conflicting declarations, malformed/recursive/budget-limited input, and a
malicious reducer that must not execute on the host. Routing tests inspect raw
traces, n-grams, count/normalized CSVs, registry outcomes, review cases and batch
deduplication. HTTP tests upload without a category for all three destinations;
their syscall capture is synthetic, not live integration evidence.

Latest verification: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q
-m 'not integration' -p no:cacheprovider` — **181 passed, 11 deselected**. Live
Docker/SecCheck integration was not rerun for this change.

Design references: [Python pickletools](https://docs.python.org/3/library/pickletools.html)
and [Transformers task-specific auto classes](https://huggingface.co/docs/transformers/model_doc/auto).
