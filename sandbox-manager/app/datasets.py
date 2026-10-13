"""Rebuild category datasets from immutable events; no model execution required."""
import argparse
from collections import Counter
import csv
import fcntl
from io import StringIO
import json
import os
from pathlib import Path
import tempfile

from .instrumentation import ModelCategory
from .categories import VALID_CATEGORY_STATUSES
from .syscall_names import NAMES

CATEGORIES = tuple(c.value for c in ModelCategory)
# Frozen ABI vocabulary includes unknown/unassigned numbers, never drops events.
SYSCALLS = tuple(NAMES.get(i, f'syscall_{i}') for i in range(451))
PROFILE = 'pickle-load-v1'
SCHEMA = 'syscall-counts-x86_64-v2'
LEADING_SYSCALLS = ('read', 'write', 'openat', 'execve', 'connect')
DATASET_SYSCALLS = (*LEADING_SYSCALLS, *(name for name in SYSCALLS if name not in LEADING_SYSCALLS))
DATASET_FIELDS = ['Model', 'Category', 'Family', *DATASET_SYSCALLS, 'analysis_id']

def model_columns(metadata):
    details = metadata.get('collection', {})
    declared = metadata.get('metadata', {})
    return {'Model': details.get('model_id') or declared.get('model_id') or metadata.get('filename') or metadata['analysis_id'],
            'Category': metadata.get('task_category', 'unknown'),
            'Family': details.get('family') or declared.get('family') or 'Unknown',
            'analysis_id': metadata['analysis_id']}

META_FIELDS = ['analysis_id', 'model_id', 'filename', 'model_sha256', 'task_category',
    'hash_algorithm', 'hashing_status', 'size_bytes', 'quarantined_file', 'quarantine_reused', 'identity_verified',
    'family', 'architecture', 'provider', 'library', 'source', 'revision',
    'category_source', 'validation_status', 'category_evidence', 'category_reason', 'benign_review',
    'execution_profile', 'runner_image', 'runtime_version', 'status', 'capture_status', 'exit_code', 'failure_reason',
    'artifact_format', 'start_time', 'end_time', 'reported_losses', 'event_count', 'raw_events',
    'feature_row_included', 'dataset_eligible', 'exclusion_reasons']


def csv_text(fields, rows):
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        # Escape spreadsheet formulas in metadata, not numeric feature values.
        writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')) else v)
                         for k, v in row.items() if k in fields})
    return stream.getvalue()


def frequencies(events, analysis_id):
    counts = Counter()
    previous = 0
    for e in events:
        if (e['analysis_id'] != analysis_id or e['sequence'] <= previous
                or e.get('event_source') != 'gvisor_seccheck'
                or not e.get('tracepoint', '').startswith('syscall/')
                or e.get('phase') not in {'enter', 'exit'}):
            raise ValueError('Invalid or mixed raw event stream')
        previous = e['sequence']
        if e['phase'] == 'enter':
            name = e['syscall_name']
            if name not in SYSCALLS:
                raise ValueError('Unknown syscall outside dataset schema; upgrade schema explicitly')
            counts[name] += 1
    return {name: counts[name] for name in SYSCALLS}


def frequency_report(analysis_id, counts, status, capture_status):
    """Inspectable per-syscall counts, including zeroes, without implying success."""
    total = sum(counts.values())
    return csv_text(['analysis_id', 'execution_status', 'capture_status', 'syscall', 'count', 'relative_frequency'],
        [dict(analysis_id=analysis_id, execution_status=status, capture_status=capture_status,
              syscall=name, count=counts[name], relative_frequency=counts[name] / total if total else 0)
         for name in SYSCALLS])


def eligibility(m):
    reasons = []
    checks = {
        'execution_not_successful': m.get('status') == 'succeeded',
        'capture_incomplete': m.get('capture_status') == 'captured' and m.get('seccheck_verified') is True,
        'losses_unknown_or_nonzero': m.get('reported_losses') == 0,
        'capture_errors': not m.get('capture_errors'),
        'unmatched_events': m.get('unmatched_exits', 0) == 0 and m.get('unexpected_unmatched_entries', 0) == 0,
        'cleanup_incomplete': m.get('cleanup_status') == 'destroyed',
        'hashing_paused_or_missing': bool(m.get('model_sha256')),
        'category_not_validated': m.get('collection', {}).get('category_validated') is True,
        'benign_not_reviewed': bool(m.get('collection', {}).get('benign_review')),
        'unsupported_profile': m.get('execution_profile') == PROFILE,
        'unsupported_category': m.get('task_category') in CATEGORIES,
    }
    return [name for name, passed in checks.items() if not passed]


TRAINING_ONLY_REASONS = {'hashing_paused_or_missing', 'category_not_validated', 'benign_not_reviewed',
                         'duplicate_sha256', 'execution_not_successful'}


def measurement_eligible(metadata, reasons):
    """A complete measurement is not automatically an approved training sample."""
    return (metadata.get('validation_status') in VALID_CATEGORY_STATUSES
            and not (set(reasons) - TRAINING_ONLY_REASONS))


class DatasetBuilder:
    def __init__(self, root):
        self.root = Path(root)

    def rows(self):
        result = []
        metadata_directory = self.root / 'metadata'
        # Path.glob can silently suppress PermissionError and look like an empty
        # corpus. iterdir fails loudly, before any existing CSV is replaced.
        try:
            metadata_files = sorted(p for p in metadata_directory.iterdir() if p.suffix == '.json')
        except FileNotFoundError:
            metadata_files = []
        except PermissionError as error:
            raise PermissionError(
                f'Cannot read {metadata_directory}; rebuild as the same user that captured the traces. Existing CSVs are unchanged.'
            ) from error
        for path in metadata_files:
            m = json.loads(path.read_text())
            aid = m['analysis_id']
            if path.stem != aid:
                raise ValueError('Metadata identity mismatch')
            reasons = eligibility(m)
            counts = None
            category = m.get('task_category')
            raw = self.root / 'raw' / category / f'{aid}.jsonl' if category in CATEGORIES else None
            if raw is None or not raw.is_file():
                reasons.append('raw_events_missing')
            else:
                try:
                    events = [json.loads(line) for line in raw.read_text().splitlines()]
                    if any(e.get('container_id') != m.get('sandbox_id') for e in events):
                        raise ValueError('Sandbox identity mismatch')
                    counts = frequencies(events, aid)
                    if not sum(counts.values()) or sum(counts.values()) != m.get('event_count') or len(events) != m.get('record_count'):
                        reasons.append('event_count_mismatch')
                except (ValueError, KeyError, TypeError):
                    reasons.append('invalid_raw_events')
            result.append((m, counts, reasons))
        # Stable exact dedup across categories, with conflicting category hashes excluded.
        categories = {}
        for m, _, _ in result:
            if (m.get('model_sha256') and m.get('task_category') in CATEGORIES
                    and m.get('validation_status') in VALID_CATEGORY_STATUSES):
                categories.setdefault(m['model_sha256'], set()).add(m.get('task_category'))
        seen = set()
        for m, counts, reasons in result:
            digest = m.get('model_sha256')
            if digest and len(categories.get(digest, set())) > 1:
                reasons.append('duplicate_category_conflict')
            if not reasons:
                if digest in seen:
                    reasons.append('duplicate_sha256')
                seen.add(digest)
        return result

    def render(self, rows=None, *, training_only=True):
        rows = self.rows() if rows is None else rows
        outputs = {}
        metadata = []
        quality = {'schema': SCHEMA, 'profile': PROFILE,
                   'scope': 'reviewed_training' if training_only else 'complete_capture_measurements',
                   'analyses': len(rows), 'eligible': 0,
                   'excluded': {}, 'categories': {}, 'families': {}, 'architectures': {}, 'identical_vectors': []}
        vectors = {}
        for category in CATEGORIES:
            category_rows, normalized = [], []
            for m, counts, reasons in rows:
                included = not reasons if training_only else measurement_eligible(m, reasons)
                if m.get('task_category') != category or not included:
                    continue
                category_rows.append({**model_columns(m), **counts})
                total = sum(counts.values())
                normalized.append({**model_columns(m), **{k: v / total for k, v in counts.items()}})
                signature = (category, tuple(counts.values()))
                vectors.setdefault(signature, []).append(m['analysis_id'])
            outputs[f'{category}.csv'] = csv_text(DATASET_FIELDS, category_rows)
            outputs[f'{category}.normalized.csv'] = csv_text(DATASET_FIELDS, normalized)
            quality['categories'][category] = len(category_rows)
        for m, counts, reasons in rows:
            details = {**m.get('metadata', {}), **m.get('collection', {})}
            metadata.append({**{k: m.get(k, '') for k in META_FIELDS},
                **{k: details.get(k, '') for k in ['model_id', 'family', 'architecture', 'provider', 'library', 'source', 'revision', 'benign_review']},
                'category_evidence': json.dumps(details.get('category_evidence', {}), sort_keys=True),
                'category_reason': m.get('metadata', {}).get('category_reason', ''),
                'raw_events': m.get('artifacts', {}).get('raw_events', ''),
                'feature_row_included': measurement_eligible(m, reasons),
                'dataset_eligible': not reasons, 'exclusion_reasons': ';'.join(reasons)})
            quality_reasons = reasons if training_only else [r for r in reasons if r not in TRAINING_ONLY_REASONS]
            if quality_reasons:
                for reason in quality_reasons:
                    quality['excluded'][reason] = quality['excluded'].get(reason, 0) + 1
            else:
                quality['eligible'] += 1
                for key, field in [('families', 'family'), ('architectures', 'architecture')]:
                    value = str(details.get(field) or 'unknown')
                    quality[key][value] = quality[key].get(value, 0) + 1
        quality['identical_vectors'] = [ids for ids in vectors.values() if len(ids) > 1]
        outputs['metadata.csv'] = csv_text(META_FIELDS, metadata)
        outputs['quality.json'] = json.dumps(quality, indent=2, sort_keys=True) + '\n'
        outputs['schema.json'] = json.dumps({'version': SCHEMA, 'execution_profile': PROFILE,
            'feature_columns': DATASET_SYSCALLS, 'identifier_columns': ['Model', 'Category', 'Family', 'analysis_id'],
            'counts': 'syscall entries only', 'normalized': 'count / total syscall entries',
            'training': 'Use only feature_columns for training; fit any scaler on the training split only.'}, indent=2) + '\n'
        return outputs

    def analysis_csv(self, analysis_id, *, normalized=False):
        for metadata, counts, reasons in self.rows():
            if metadata['analysis_id'] == analysis_id:
                if counts is None or not sum(counts.values()) or 'invalid_raw_events' in reasons:
                    raise ValueError('No valid syscall features for this analysis')
                total = sum(counts.values())
                values = {k: v / total for k, v in counts.items()} if normalized else counts
                # This is an inspection export. Eligibility stays in metadata.csv;
                # only render() selects rows for the training category datasets.
                return csv_text(DATASET_FIELDS, [{**model_columns(metadata), **values}])
        raise KeyError(analysis_id)

    def build(self):
        destination = self.root / 'datasets'
        destination.mkdir(parents=True, exist_ok=True)
        # Serialize writers; each output is atomically replaced, readers never see torn CSV.
        with (destination / '.build.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            measurements = self.rows()
            # Diagnostic measurements are useful even if model loading failed.
            # Keep them separate from category datasets and preserve their status.
            for metadata, counts, reasons in measurements:
                if counts is None or not sum(counts.values()) or any(
                        r in reasons for r in ('invalid_raw_events', 'event_count_mismatch')):
                    continue
                self._write(self.root / 'frequencies' / f'{metadata["analysis_id"]}.csv',
                    csv_text(DATASET_FIELDS, [{**model_columns(metadata), **counts}]))
            outputs = self.render(measurements, training_only=False)
            for name, content in outputs.items():
                self._write(destination / name, content)
            for name, content in self.render(measurements, training_only=True).items():
                self._write(self.root / 'training' / name, content)
            # Primary Phase 2 output: one complete capture per row, routed by
            # declared/consistent category. Hashing pause does not suppress it.
            for category in CATEGORIES:
                rows, normalized = [], []
                for m, counts, reasons in measurements:
                    if m.get('task_category') != category or not measurement_eligible(m, reasons):
                        continue
                    rows.append({**model_columns(m), **counts})
                    total = sum(counts.values())
                    normalized.append({**model_columns(m), **{k: v / total for k, v in counts.items()}})
                self._write(self.root / 'features' / f'{category}.csv', csv_text(DATASET_FIELDS, rows))
                self._write(self.root / 'features' / f'{category}.normalized.csv', csv_text(DATASET_FIELDS, normalized))
            self._write(self.root / 'metadata.csv', outputs['metadata.csv'])
            self._write(self.root / 'features' / 'schema.json', outputs['schema.json'])
        return destination

    @staticmethod
    def _write(destination, content):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    args = parser.parse_args()
    try:
        destination = DatasetBuilder(args.data).build()
    except OSError as error:
        parser.exit(2, f'CSV rebuild failed: {error}\nUse the same OS user that captured the traces (sudo for your sudo-run backend).\n')
    print(destination)
    quality = json.loads((destination / 'quality.json').read_text())
    print(json.dumps({'capture_rows': quality['categories'], 'excluded': quality['excluded']}, indent=2))


if __name__ == '__main__':
    main()
