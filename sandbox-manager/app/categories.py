"""Automatic task identification from bounded, non-executing model metadata."""
from dataclasses import asdict, dataclass, field
from .artifact_metadata import Object, inspect_pickle, state_dict
from .instrumentation import require_category

VALID_CATEGORY_STATUSES = frozenset({'user_declared', 'manifest_consistent', 'auto_identified', 'metadata_identified'})

# Explicit heads only: a base encoder, class labels or model family do not prove task.
ARCHITECTURES = {}
for category, names in {
    'text_generation': ('GPT2LMHeadModel', 'GPTNeoForCausalLM', 'GPTJForCausalLM',
        'GPTNeoXForCausalLM', 'LlamaForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM',
        'Qwen2ForCausalLM', 'Qwen3ForCausalLM', 'OPTForCausalLM', 'BloomForCausalLM',
        'FalconForCausalLM', 'GemmaForCausalLM', 'Gemma2ForCausalLM', 'PhiForCausalLM',
        'Phi3ForCausalLM', 'T5ForConditionalGeneration', 'BartForConditionalGeneration'),
    'text_classification': ('BertForSequenceClassification', 'DistilBertForSequenceClassification',
        'RobertaForSequenceClassification', 'XLMRobertaForSequenceClassification',
        'AlbertForSequenceClassification', 'ElectraForSequenceClassification',
        'DebertaForSequenceClassification', 'DebertaV2ForSequenceClassification',
        'XLNetForSequenceClassification', 'LlamaForSequenceClassification'),
    'image_classification': ('ViTForImageClassification', 'DeiTForImageClassification',
        'DeiTForImageClassificationWithTeacher', 'SwinForImageClassification',
        'Swinv2ForImageClassification', 'ConvNextForImageClassification',
        'ConvNextV2ForImageClassification', 'ResNetForImageClassification',
        'BeitForImageClassification', 'MobileNetV2ForImageClassification',
        'EfficientNetForImageClassification'),
}.items():
    ARCHITECTURES.update({name: category for name in names})

TASK_ALIASES = {name: name.replace('-', '_') for name in
                ('text-generation', 'text-classification', 'image-classification')}
TASK_ALIASES.update({value: value for value in TASK_ALIASES.values()})
TASK_ALIASES.update({'text2text-generation': 'text_generation', 'sentiment-analysis': 'text_classification'})
VISION_CLASSES = {
    'torchvision.models.resnet': {'ResNet'}, 'torchvision.models.vgg': {'VGG'},
    'torchvision.models.densenet': {'DenseNet'}, 'torchvision.models.mobilenetv2': {'MobileNetV2'},
    'torchvision.models.mobilenetv3': {'MobileNetV3'}, 'torchvision.models.efficientnet': {'EfficientNet'},
    'torchvision.models.vision_transformer': {'VisionTransformer'},
    'torchvision.models.swin_transformer': {'SwinTransformer'},
}
TEXT_VECTORIZERS = {'CountVectorizer', 'TfidfVectorizer', 'HashingVectorizer'}
CLASSIFIERS = {'LogisticRegression', 'LogisticRegressionCV', 'SGDClassifier', 'LinearSVC',
               'SVC', 'MultinomialNB', 'BernoulliNB', 'ComplementNB', 'RandomForestClassifier'}


def unsupported_head(name):
    return (name.endswith(('ForTokenClassification', 'ForQuestionAnswering', 'ForObjectDetection',
                           'ForSemanticSegmentation', 'ForAudioClassification', 'ForCTC', 'ForMaskedLM'))
            or name in {'Wav2Vec2ForSequenceClassification', 'HubertForSequenceClassification',
                        'WhisperForConditionalGeneration'})


@dataclass
class CategoryAssignment:
    task_category: str | None
    category_source: str
    validation_status: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def validate_category(category, manifest=None):
    """Compatibility for reviewed batch manifests and explicit library callers."""
    if category is None or category == "":
        return CategoryAssignment(None, "none", "unclassified")
    selected = require_category(category).value
    if manifest is None:
        return CategoryAssignment(selected, "explicit_selection", "user_declared")
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    # Only task declarations affect classification. Never guess from names/extensions.
    declared = manifest.get("task_category")
    tags = manifest.get("task_tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("Manifest task_tags must be a list of strings")
    known = {"text_generation", "text_classification", "image_classification"}
    candidates = set(tags) & known
    if declared is not None:
        candidates.add(require_category(declared).value)
    evidence = {"declared_categories": sorted(candidates), "trust": "operator_supplied_declaration"}
    for key in ('model_id', 'family', 'architecture', 'provider', 'source', 'library', 'revision'):
        value = manifest.get(key)
        if value is not None:
            if not isinstance(value, str) or len(value) > 200:
                raise ValueError(f'Manifest {key} must be a string of at most 200 characters')
            evidence[key] = value.strip()
    status = "conflict" if candidates and candidates != {selected} else "manifest_consistent" if candidates else "user_declared"
    return CategoryAssignment(selected, "explicit_selection", status, evidence)


def identify_category(payload, category=None, manifest=None):
    """Resolve artifact/config evidence, rejecting disagreement instead of guessing.

Category evidence is untrusted structural metadata, never a safety verdict.
Explicit categories remain compatible for existing reviewed collection clients,
but cannot override contradictory automatically extracted evidence.
"""
    if manifest is not None and not isinstance(manifest, dict):
        raise ValueError('Manifest must be a JSON object')
    manifest = manifest or {}
    selected = require_category(category).value if category else None
    evidence, unsupported = [], []
    details = {}
    for key in ('model_id', 'family', 'architecture', 'provider', 'source', 'library', 'revision'):
        value = manifest.get(key)
        if value is not None:
            if not isinstance(value, str) or len(value) > 200:
                raise ValueError(f'Manifest {key} must be a string of at most 200 characters')
            details[key] = value.strip()

    def add(task, source, field, value):
        evidence.append(dict(category=task, source=source, field=field, value=value))

    def metadata(mapping, source):
        family = mapping.get('model_type')
        if isinstance(family, str) and len(family) <= 200:
            details.setdefault('family', family)
        for key in ('task_category', 'pipeline_tag', 'task'):
            value = mapping.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or len(value) > 200:
                raise ValueError('Task metadata must be a string of at most 200 characters')
            task = TASK_ALIASES.get(value)
            if task:
                add(task, source, key, value)
            else:
                unsupported.append(dict(source=source, field=key, value=value))
        tags = mapping.get('task_tags', mapping.get('tags', []))
        if tags is None:
            tags = []
        if not isinstance(tags, list) or len(tags) > 64 or any(not isinstance(t, str) or len(t) > 200 for t in tags):
            raise ValueError('Task tags must be a list of at most 64 short strings')
        for tag in tags:
            if tag in TASK_ALIASES:
                add(TASK_ALIASES[tag], source, 'tags', tag)
        architectures = mapping.get('architectures', [])
        if architectures is None:
            architectures = []
        if not isinstance(architectures, list) or len(architectures) > 64 or any(not isinstance(a, str) or len(a) > 200 for a in architectures):
            raise ValueError('Architectures must be a list of at most 64 short strings')
        architecture = mapping.get('architecture')
        if isinstance(architecture, str):
            architectures = [*architectures, architecture]
        for name in architectures:
            if name in ARCHITECTURES:
                add(ARCHITECTURES[name], source, 'architectures', name)
                details.setdefault('architecture', name)
            elif unsupported_head(name):
                unsupported.append(dict(source=source, field='architectures', value=name))

    root, diagnostic = inspect_pickle(payload)
    # Only root/model/config/metadata structure is consulted, not arbitrary strings
    # in weights, filenames, labels or unrelated helper objects.
    visited = set()
    def inspect(value, source, depth=0):
        if depth > 4 or id(value) in visited:
            return
        visited.add(id(value))
        mapping = state_dict(value)
        if isinstance(value, Object) and value.reference:
            ref = value.reference
            if (ref.module == 'transformers' or ref.module.startswith('transformers.')) and ref.name in ARCHITECTURES:
                add(ARCHITECTURES[ref.name], source, 'model_class', ref.name)
                details.setdefault('architecture', ref.name)
                details.setdefault('library', 'transformers')
            elif (ref.module == 'transformers' or ref.module.startswith('transformers.')) and unsupported_head(ref.name):
                unsupported.append(dict(source=source, field='model_class', value=ref.name))
            elif ref.name in VISION_CLASSES.get(ref.module, set()):
                add('image_classification', source, 'model_class', ref.name)
                details.setdefault('architecture', ref.name)
                details.setdefault('library', 'torchvision')
            elif ref.module == 'sklearn.pipeline' and ref.name == 'Pipeline':
                steps = mapping.get('steps')
                if isinstance(steps, list) and 2 <= len(steps) <= 32:
                    estimators = [step[1] for step in steps if isinstance(step, list) and len(step) == 2]
                    if len(estimators) == len(steps):
                        refs = [e.reference if isinstance(e, Object) else None for e in estimators]
                        has_text = any(r and r.module == 'sklearn.feature_extraction.text' and r.name in TEXT_VECTORIZERS for r in refs[:-1])
                        final = refs[-1]
                        if has_text and final and final.module.startswith('sklearn.') and final.name in CLASSIFIERS:
                            add('text_classification', source, 'pipeline', 'text vectorizer + ' + final.name)
                            details.setdefault('architecture', 'Pipeline/' + final.name)
                            details.setdefault('library', 'scikit-learn')
        metadata(mapping, source)
        for key in ('config', 'metadata', 'model', 'estimator'):
            if key in mapping:
                inspect(mapping[key], source + '.' + key, depth + 1)

    inspect(root, 'artifact')
    inspect(manifest, 'manifest')
    if selected:
        add(selected, 'explicit_selection', 'task_category', selected)
    candidates = {item['category'] for item in evidence}
    details.update(category_evidence=evidence, unsupported_tasks=unsupported,
                   inspection_status=diagnostic or 'inspected',
                   trust='untrusted_structural_metadata_not_behavioral_proof', detector_version=1)
    if len(candidates) > 1 or unsupported and candidates:
        details['category_reason'] = 'Conflicting task or architecture evidence; no dataset selected.'
        return CategoryAssignment(None, 'automatic_metadata', 'conflict', details)
    if unsupported:
        details['category_reason'] = 'Model metadata describes a task outside the three supported categories.'
        return CategoryAssignment(None, 'automatic_metadata', 'unsupported', details)
    if not candidates:
        details['category_reason'] = ('Automatic identification found no supported task-specific architecture or task metadata. '
                                      'Provide the model config/model card metadata for this custom or ambiguous artifact.')
        return CategoryAssignment(None, 'automatic_metadata', 'unclassified', details)
    task = next(iter(candidates))
    artifact_evidence = any(e['source'].startswith('artifact') for e in evidence)
    external_evidence = any(e['source'].startswith('manifest') for e in evidence)
    if artifact_evidence:
        status, source = 'auto_identified', 'artifact_metadata'
    elif external_evidence and not selected:
        status, source = 'metadata_identified', 'model_metadata'
    else:
        status, source = ('manifest_consistent' if external_evidence else 'user_declared'), 'explicit_selection'
    details['category_reason'] = f'Identified {task} from consistent task/architecture evidence.'
    return CategoryAssignment(task, source, status, details)
