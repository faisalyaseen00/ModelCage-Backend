"""Resumable, sequential collection of reviewed pickle artifacts inside runsc."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from uuid import uuid4

from .datasets import CATEGORIES, DatasetBuilder, PROFILE
from .modelcage import ModelCage
from .quarantine import QuarantineStore, normalize_sha256, open_regular
from .categories import identify_category, VALID_CATEGORY_STATUSES

MAX_BYTES = 64 * 1024 * 1024
def validate_candidate(candidate):
    if not isinstance(candidate, dict):
        raise ValueError('Each candidate must be a JSON object')
    category = candidate.get('category')
    if category is not None and category not in CATEGORIES:
        raise ValueError('Unsupported category')
    for field in ('model_id', 'family', 'architecture', 'source', 'benign_review'):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            raise ValueError(f'Missing reviewed {field}')
    if candidate.get('repo_id') or candidate.get('url'):
        raise ValueError('Collection accepts uploaded/local files only; downloads are disabled')
    evidence = candidate.get('category_evidence', {})
    if (not isinstance(evidence, dict) or (category is not None and evidence.get('task') != category)
            or not evidence.get('reference') or not evidence.get('reviewed_by')):
        raise ValueError('Local artifact requires reviewed task metadata and reference')
    filename = str(candidate.get('path', ''))
    if Path(filename).suffix.lower() not in {'.pkl', '.pickle'}:
        raise ValueError('Loading profile supports .pkl/.pickle only; no format conversion')
    return {**{k: candidate.get(k, '') for k in ('model_id', 'family', 'architecture', 'provider',
        'source', 'library', 'revision', 'benign_review')},
        'category_validated': False, 'category_evidence': evidence}


def obtain(candidate, base, destination):
    """Copy local bytes with a hard bound; never import or deserialize artifacts."""
    source = base / candidate['path']
    if source.is_symlink() or not source.is_file():
        raise ValueError('Local input must be a regular file, not a symlink')
    stream = open_regular(source)
    digest = hashlib.sha256()
    size = 0
    with stream, destination.open('xb') as target:
        destination.chmod(0o600)
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError('Artifact exceeds 64 MiB loading profile limit')
            digest.update(chunk)
            target.write(chunk)
    value = digest.hexdigest()
    if candidate.get('sha256') is not None and normalize_sha256(candidate['sha256']) != value:
        raise ValueError('Artifact SHA-256 does not match manifest')
    return value


class Collector:
    def __init__(self, workspace, *, family_limit=20, cage=None):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.family_limit = family_limit
        if family_limit < 1:
            raise ValueError('Family limit must be positive')
        self.cage = cage
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS models (sha256 TEXT PRIMARY KEY, category TEXT, family TEXT, analysis_id TEXT, status TEXT, details TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, model_id TEXT, status TEXT, reason TEXT, sha256 TEXT, analysis_id TEXT)')

    def db(self):
        return sqlite3.connect(self.workspace / 'registry.sqlite')

    def run(self, manifest_path, *, retry_failed=False):
        manifest_path = Path(manifest_path).resolve()
        candidates = json.loads(manifest_path.read_text())
        if not isinstance(candidates, list):
            raise ValueError('Manifest must be a JSON array')
        results = []
        # One controller per workspace prevents duplicate runs and races in quotas.
        with (self.workspace / '.collection.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.db() as db:
                db.execute("UPDATE models SET status='interrupted' WHERE status='running'")
            for candidate in candidates:
                aid = digest = None
                status, reason = 'rejected', ''
                claimed = False
                try:
                    details = validate_candidate(candidate)
                    with tempfile.TemporaryDirectory(dir=self.workspace, prefix='incoming-') as directory:
                        artifact = Path(directory) / 'model.pkl'
                        digest = obtain(candidate, manifest_path.parent, artifact)
                        model_metadata = candidate.get('metadata', {})
                        if not isinstance(model_metadata, dict):
                            raise ValueError('Model metadata must be a JSON object')
                        model_metadata = {**model_metadata, 'architecture': details['architecture'],
                                          'model_id': details['model_id'], 'family': details['family']}
                        reviewed_task = details['category_evidence'].get('task')
                        if reviewed_task:
                            model_metadata['task_category'] = reviewed_task
                        assignment = identify_category(artifact.read_bytes(), candidate.get('category'), model_metadata)
                        if assignment.validation_status not in VALID_CATEGORY_STATUSES:
                            raise ValueError(assignment.metadata['category_reason'])
                        category = assignment.task_category
                        details['category_validated'] = True
                        details['detected_category'] = category
                        with self.db() as db:
                            old = db.execute('SELECT category,status,analysis_id FROM models WHERE sha256=?', (digest,)).fetchone()
                            if old and old[0] != category:
                                raise ValueError('Exact duplicate assigned to a different category')
                            if old and (old[1] == 'succeeded' or not retry_failed):
                                status, aid = 'duplicate', old[2]
                            else:
                                count = db.execute("SELECT count(*) FROM models WHERE category=? AND family=? AND status='succeeded'", (category, details['family'])).fetchone()[0]
                                if count >= self.family_limit:
                                    status = 'quota_skipped'
                                else:
                                    aid = str(uuid4())
                                    db.execute('INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?)',
                                        (digest, category, details['family'], aid, 'running', json.dumps(details)))
                                    claimed = True
                        if claimed:
                            if self.cage is None:
                                self.cage = ModelCage(str(self.workspace / 'modelcage.sqlite'))
                            # Collection always requires exact byte identity for deduplication.
                            self.cage.quarantine = QuarantineStore(self.workspace / 'quarantine', hashing_enabled=True)
                            self.cage.events.begin(category=category, analysis_id=aid)
                            self.cage.execute_model(artifact, candidate.get('category'), analysis_id=aid,
                                manifest=model_metadata, collection_context=details,
                                expected_sha256=digest,
                                original_filename=Path(candidate.get('filename') or candidate.get('path')).name)
                            state = self.cage.events.snapshot(aid)
                            status = state['status']
                            reason = state.get('failure_reason') or ''
                except Exception as error:
                    status = 'failed' if claimed else 'rejected'
                    reason = str(error)[:500] if isinstance(error, ValueError) else type(error).__name__
                finally:
                    with self.db() as db:
                        if claimed:
                            db.execute('UPDATE models SET status=? WHERE sha256=?', (status, digest))
                        db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?)',
                            (str(uuid4()), candidate.get('model_id', '') if isinstance(candidate, dict) else '', status, reason, digest, aid))
                    results.append(dict(model_id=candidate.get('model_id', '') if isinstance(candidate, dict) else '',
                        status=status, reason=reason, sha256=digest, analysis_id=aid))
            output = DatasetBuilder(self.workspace / 'data').build()
            (self.workspace / 'collection-report.json').write_text(json.dumps(results, indent=2) + '\n')
        return results, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--family-limit', type=int, default=20)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    results, path = Collector(args.workspace, family_limit=args.family_limit).run(args.manifest, retry_failed=args.retry_failed)
    print(json.dumps({'results': results, 'datasets': str(path)}, indent=2))
    if any(row['status'] in {'failed', 'rejected', 'incomplete', 'timed_out', 'instrumentation_unavailable'} for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
