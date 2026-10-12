"""Non-executing category validation; declarations are not model inference."""
from dataclasses import asdict, dataclass, field
from .instrumentation import require_category


@dataclass
class CategoryAssignment:
    task_category: str | None
    category_source: str
    validation_status: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def validate_category(category, manifest=None):
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
    for key in ('model_id', 'family'):
        value = manifest.get(key)
        if value is not None:
            if not isinstance(value, str) or len(value) > 200:
                raise ValueError(f'Manifest {key} must be a string of at most 200 characters')
            evidence[key] = value.strip()
    status = "conflict" if candidates and candidates != {selected} else "manifest_consistent" if candidates else "user_declared"
    return CategoryAssignment(selected, "explicit_selection", status, evidence)
