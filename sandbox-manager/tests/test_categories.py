"""Automatic task detection without importing or deserializing model code."""
import pickle

import pytest

from app.artifact_metadata import inspect_pickle, state_dict
from app.categories import identify_category


HEADS = [
    ('GPT2LMHeadModel', 'text_generation'),
    ('BertForSequenceClassification', 'text_classification'),
    ('ViTForImageClassification', 'image_classification'),
]


def instance(module, name, state=None, *, stack_global=False):
    """A serialized class reference; producing it never imports that class."""
    if stack_global:
        def string(value):
            raw = value.encode()
            return b'\x8c' + bytes([len(raw)]) + raw
        prefix = b'\x80\x04' + string(module) + string(name) + b'\x93'
    else:
        prefix = b'\x80\x02c' + module.encode() + b'\n' + name.encode() + b'\n'
    return prefix + b')\x81' + pickle.dumps(state or {}, protocol=2)[2:-1] + b'b.'


class CustomModel:
    def __init__(self):
        self.config = {'architectures': ['BertForSequenceClassification'], 'model_type': 'bert'}


@pytest.mark.parametrize('name,category', HEADS)
@pytest.mark.parametrize('stack_global', [False, True])
def test_known_model_head_is_identified_from_pickle(name, category, stack_global):
    result = identify_category(instance('transformers.models.fixture.modeling_fixture', name,
                                       stack_global=stack_global))
    assert result.task_category == category
    assert result.validation_status == 'auto_identified'
    assert result.category_source == 'artifact_metadata'
    assert result.metadata['architecture'] == name
    assert result.metadata['category_evidence'][0]['field'] == 'model_class'


@pytest.mark.parametrize('protocol', range(6))
def test_embedded_config_is_read_without_model_dependencies(protocol):
    result = identify_category(pickle.dumps(CustomModel(), protocol=protocol))
    assert result.task_category == 'text_classification'
    assert result.metadata['family'] == 'bert'


def test_torchvision_image_classifier_is_recognized():
    result = identify_category(instance('torchvision.models.resnet', 'ResNet'))
    assert result.task_category == 'image_classification'


def test_sklearn_text_pipeline_needs_vectorizer_and_classifier():
    # A protocol-2 Pipeline state with two estimator references; no sklearn import.
    def string(value):
        return pickle.dumps(value, protocol=2)[2:-1]
    vectorizer = instance('sklearn.feature_extraction.text', 'TfidfVectorizer')[2:-1]
    classifier = instance('sklearn.linear_model._logistic', 'LogisticRegression')[2:-1]
    payload = (b'\x80\x02csklearn.pipeline\nPipeline\n)\x81}(' + string('steps') + b']('
               + string('text') + vectorizer + b'\x86'
               + string('classifier') + classifier + b'\x86eub.')
    assert identify_category(payload).task_category == 'text_classification'
    assert identify_category(b'\x80\x02' + classifier + b'.').validation_status == 'unclassified'


@pytest.mark.parametrize('metadata,category', [
    ({'architectures': ['GPT2LMHeadModel']}, 'text_generation'),
    ({'config': {'architectures': ['BertForSequenceClassification']}}, 'text_classification'),
    ({'pipeline_tag': 'image-classification'}, 'image_classification'),
])
def test_optional_model_metadata_resolves_custom_artifact(metadata, category):
    result = identify_category(pickle.dumps({'opaque_weights': [1, 2]}), manifest=metadata)
    assert result.task_category == category
    assert result.validation_status == 'metadata_identified'


@pytest.mark.parametrize('payload', [
    pickle.dumps({'labels': ['positive', 'negative'], 'model_type': 'bert'}),
    pickle.dumps({'note': 'GPT2LMHeadModel', 'filename': 'text-generation.pkl'}),
    pickle.dumps([{'task_category': 'text_generation'}, {}]),
    instance('__main__', 'GPTLanguageModel'),
    instance('untrusted_package', 'BertForSequenceClassification'),
    instance('transformers.models.bert.modeling_bert', 'BertModel'),
    b'not a pickle',
])
def test_ambiguous_and_malformed_artifacts_are_not_guessed(payload):
    result = identify_category(payload)
    assert result.task_category is None and result.validation_status == 'unclassified'


@pytest.mark.parametrize('manifest', [
    {'pipeline_tag': 'image-classification'},
    {'config': {'architectures': ['ViTForImageClassification']}},
    {'pipeline_tag': 'audio-classification'},
])
def test_conflicting_metadata_does_not_override_artifact(manifest):
    result = identify_category(instance('transformers', 'BertForSequenceClassification'), manifest=manifest)
    assert result.validation_status == 'conflict' and result.task_category is None


def test_explicit_legacy_category_cannot_override_detected_task():
    result = identify_category(instance('transformers', 'GPT2LMHeadModel'), 'image_classification')
    assert result.validation_status == 'conflict'


def test_unsupported_task_stays_outside_supported_categories():
    result = identify_category(pickle.dumps({'pipeline_tag': 'object-detection'}))
    assert result.validation_status == 'unsupported' and result.task_category is None


@pytest.mark.parametrize('name', ['WhisperForConditionalGeneration', 'BertForTokenClassification',
                                 'Wav2Vec2ForSequenceClassification'])
def test_unsupported_heads_are_not_mistaken_for_text_tasks(name):
    result = identify_category(instance('transformers', name))
    assert result.validation_status == 'unsupported' and result.task_category is None


def test_host_never_executes_pickle_reducer(tmp_path, monkeypatch):
    import os
    marker = tmp_path / 'must-not-exist'
    class Payload:
        def __reduce__(self):
            return os.system, (f'touch {marker}',)
    payload = pickle.dumps(Payload())
    def forbidden(*args, **kwargs):
        raise AssertionError('Host deserialization or command execution is forbidden')
    monkeypatch.setattr(pickle, 'loads', forbidden)
    monkeypatch.setattr(pickle, 'load', forbidden)
    monkeypatch.setattr(os, 'system', forbidden)
    result = identify_category(payload)
    assert result.validation_status == 'unclassified'
    assert not marker.exists()


def test_parser_handles_shared_and_recursive_metadata_without_recursing_forever():
    config = {'architectures': ['GPT2LMHeadModel']}
    model = {'config': config, 'metadata': config}
    model['model'] = model
    result = identify_category(pickle.dumps(model))
    assert result.task_category == 'text_generation'


def test_parser_budget_fails_closed(monkeypatch):
    monkeypatch.setattr('app.artifact_metadata.MAX_OPS', 2)
    result = identify_category(instance('transformers', 'GPT2LMHeadModel'))
    assert result.validation_status == 'unclassified'
    assert result.metadata['inspection_status'] == 'pickle_operation_limit'


def test_null_config_architectures_do_not_invalidate_known_head():
    result = identify_category(instance('transformers', 'GPT2LMHeadModel', {'config': {'architectures': None}}))
    assert result.task_category == 'text_generation'
