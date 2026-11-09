"""Automated Hugging Face collection: discover, validate, deduplicate, download, run.

Per candidate, in order: find a model for the requested task, validate its
category from hub metadata (pipeline tag, config architectures, tags, library),
read the checkpoint's SHA-256 from the hub's LFS metadata, skip it if that hash
is already in the registry or its family quota is full, and only then download
and verify the bytes. The existing Collector then quarantines, runs the
checkpoint in a disposable runsc sandbox, stores raw events and features, and
records the outcome in the same SHA-256 registry.

The host only speaks HTTPS and hashes bytes; it never imports or deserializes a
downloaded artifact. The sandbox itself stays offline.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .categories import ARCHITECTURES, identify_category
from .collection import Collector
from .datasets import CATEGORIES, DatasetBuilder
from .policy import MAX_MODEL_BYTES
from .quarantine import normalize_sha256

HUB_URL = 'https://huggingface.co'
# Single-file pickle-based PyTorch checkpoints, in preference order. Sharded
# checkpoints and safetensors-only repositories are skipped: safetensors cannot
# execute code on load, so it carries no pickle-loading behaviour to profile.
CHECKPOINT_FILES = ('pytorch_model.bin',)
PIPELINE_TAGS = {category: category.replace('_', '-') for category in CATEGORIES}
SELECTION_POLICY = ('automated: public, ungated Hugging Face transformers model ranked by downloads; '
                    'not manually reviewed for benignness')


class HubError(RuntimeError):
    """Hugging Face API or download failure."""


class HubClient:
    """Minimal read-only Hugging Face Hub API client over HTTPS."""

    def __init__(self, base_url=HUB_URL, token=None, timeout=60):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout

    def _open(self, url):
        headers = {'User-Agent': 'modelcage-collector/1'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        try:
            return urlopen(Request(url, headers=headers), timeout=self.timeout)
        except (HTTPError, URLError, TimeoutError) as error:
            raise HubError(f'{url}: {error}') from error

    def _json(self, url):
        with self._open(url) as response:
            return json.load(response)

    def list_models(self, category, limit):
        """Most-downloaded transformers models whose pipeline tag is the category's task."""
        query = urlencode({'pipeline_tag': PIPELINE_TAGS[category], 'filter': 'transformers',
                           'sort': 'downloads', 'direction': -1, 'limit': min(limit, 1000)})
        return self._json(f'{self.base_url}/api/models?{query}')

    def model_info(self, repo_id):
        """Repository metadata including config summary and per-file LFS SHA-256/size."""
        return self._json(f'{self.base_url}/api/models/{quote(repo_id, safe="/")}?blobs=true')

    def download(self, repo_id, revision, filename, destination, expected_sha256, expected_size):
        """Stream one file at a pinned revision, enforcing size and SHA-256."""
        url = f'{self.base_url}/{quote(repo_id, safe="/")}/resolve/{revision}/{quote(filename)}'
        digest, size = hashlib.sha256(), 0
        with self._open(url) as response, Path(destination).open('xb') as target:
            Path(destination).chmod(0o600)
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > min(expected_size, MAX_MODEL_BYTES):
                    raise ValueError('Download exceeds the expected or permitted size')
                digest.update(chunk)
                target.write(chunk)
        value = digest.hexdigest()
        if size != expected_size or value != expected_sha256:
            raise ValueError('Downloaded bytes do not match the hub SHA-256/size')
        return value


def select_checkpoint(info):
    """Pick the single pickle-based checkpoint with LFS identity, or explain why not."""
    siblings = {s.get('rfilename'): s for s in info.get('siblings') or []}
    for name in CHECKPOINT_FILES:
        entry = siblings.get(name)
        if entry is None:
            continue
        lfs = entry.get('lfs') or {}
        try:
            sha256 = normalize_sha256(lfs.get('sha256'))
        except (TypeError, ValueError):
            return None, f'{name} has no LFS SHA-256 in hub metadata'
        size = lfs.get('size')
        if not isinstance(size, int) or size <= 0:
            return None, f'{name} has no LFS size in hub metadata'
        if size > MAX_MODEL_BYTES:
            return None, f'{name} is {size} bytes, above the {MAX_MODEL_BYTES} byte limit'
        return dict(filename=name, sha256=sha256, size=size), None
    if any(name.startswith('pytorch_model-') for name in siblings):
        return None, 'only a sharded PyTorch checkpoint is available'
    return None, 'no single-file pickle-based PyTorch checkpoint (pytorch_model.bin)'


def hub_metadata(info):
    """Task-relevant hub fields in the shape identify_category understands."""
    config = info.get('config') or {}
    architectures = config.get('architectures') or []
    return {
        'pipeline_tag': info.get('pipeline_tag'),
        'tags': [t for t in info.get('tags') or [] if t in {'text-generation', 'text-classification',
                                                           'image-classification', 'text2text-generation'}],
        'architectures': [a for a in architectures if isinstance(a, str)][:64],
        'model_type': config.get('model_type'),
        'library': info.get('library_name'),
    }


def validate_hub_model(info, category):
    """Accept only models whose hub task and architecture evidence agree with ``category``.

    Returns ``(details, None)`` for an accepted model or ``(None, reason)``.
    """
    repo_id, revision = info.get('id'), info.get('sha')
    if not repo_id or not revision:
        return None, 'hub metadata lacks repository id or revision'
    if info.get('private') or info.get('gated') or info.get('disabled'):
        return None, 'repository is private, gated or disabled'
    meta = hub_metadata(info)
    if meta['library'] != 'transformers':
        return None, f"library is {meta['library']!r}, not transformers"
    if meta['pipeline_tag'] != PIPELINE_TAGS[category]:
        return None, f"pipeline tag {meta['pipeline_tag']!r} does not match {PIPELINE_TAGS[category]!r}"
    known = [a for a in meta['architectures'] if a in ARCHITECTURES]
    if not known:
        return None, f"config architectures {meta['architectures']} contain no recognised {category} head"
    assignment = identify_category(b'', None, {key: value for key, value in meta.items() if value is not None})
    if assignment.task_category != category:
        return None, assignment.metadata.get('category_reason', 'hub metadata does not identify the category')
    family = meta['model_type'] if isinstance(meta['model_type'], str) and meta['model_type'] else 'unknown'
    reference = f'{HUB_URL}/{repo_id}/blob/{revision}/config.json'
    accepted = (f"pipeline_tag={meta['pipeline_tag']}, architectures={known}, "
                f"library={meta['library']}, model_type={family}")
    return {
        'model_id': repo_id, 'family': family, 'architecture': known[0],
        'provider': info.get('author') or repo_id.split('/')[0],
        'source': f'huggingface:{repo_id}@{revision}', 'library': meta['library'], 'revision': revision,
        'benign_review': SELECTION_POLICY, 'category_validated': False,
        'category_acceptance': accepted,
        'category_evidence': {'task': category, 'reference': reference,
                              'reviewed_by': 'automatic:huggingface-metadata', 'hub_metadata': meta},
    }, None


class HubCollector(Collector):
    """Collector whose candidates come from Hugging Face instead of a local manifest."""

    def __init__(self, workspace, *, client=None, **options):
        super().__init__(workspace, **options)
        self.client = client or HubClient()
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS hub_candidates (repo_id TEXT, revision TEXT, category TEXT, '
                       'filename TEXT, sha256 TEXT, status TEXT, reason TEXT, PRIMARY KEY (repo_id, revision))')

    def _seen(self, repo_id, revision):
        with self.db() as db:
            return db.execute('SELECT status FROM hub_candidates WHERE repo_id=? AND revision=?',
                              (repo_id, revision)).fetchone()

    def _note(self, info, category, checkpoint, status, reason):
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO hub_candidates VALUES (?,?,?,?,?,?,?)',
                       (info.get('id'), info.get('sha'), category, (checkpoint or {}).get('filename'),
                        (checkpoint or {}).get('sha256'), status, reason))

    def candidate(self, info, category, retry_failed=False):
        """Run one listed model through validation, pre-download dedup and collection."""
        repo_id = info.get('id', '')
        stub = {'model_id': repo_id}
        try:
            info = self.client.model_info(repo_id)
        except HubError as error:
            return self._record(stub, 'rejected', str(error)[:500], None, None)
        details, reason = validate_hub_model(info, category)
        checkpoint = None
        if details is not None:
            checkpoint, reason = select_checkpoint(info)
        if checkpoint is None:
            self._note(info, category, None, 'rejected', reason)
            return self._record(stub, 'rejected', reason, None, None)
        digest = checkpoint['sha256']
        known = self.known(digest)
        if known and known[0] != category:
            reason = f'SHA-256 already registered under {known[0]}'
            self._note(info, category, checkpoint, 'rejected', reason)
            return self._record(stub, 'rejected', reason, digest, known[2])
        if known and (known[1] == 'succeeded' or not retry_failed):
            # Exact duplicate by hub-reported SHA-256: skipped before any download.
            self._note(info, category, checkpoint, 'duplicate', 'SHA-256 already in registry')
            return self._record(stub, 'duplicate', 'SHA-256 already in registry; not downloaded', digest, known[2])
        if self.family_full(category, details['family']):
            reason = f"family {details['family']!r} reached its quota of {self.family_limit}"
            self._note(info, category, checkpoint, 'quota_skipped', reason)
            return self._record(stub, 'quota_skipped', reason + '; not downloaded', digest, None)
        details['checkpoint'] = checkpoint['filename']
        candidate = {'model_id': repo_id, 'category': category, 'sha256': digest,
                     'metadata': {'task_category': category, **hub_metadata(info)}}
        result = self.collect(candidate, details,
            lambda artifact: self.client.download(repo_id, info['sha'], checkpoint['filename'], artifact,
                                                  digest, checkpoint['size']),
            retry_failed=retry_failed, original_filename=f"{repo_id.replace('/', '__')}__{checkpoint['filename']}")
        self._note(info, category, checkpoint, result['status'], result['reason'] or details['category_acceptance'])
        return result

    def run_hub(self, categories, *, per_category=5, scan_limit=100, retry_failed=False):
        """Collect up to ``per_category`` successful models per category."""
        results = []
        with (self.workspace / '.collection.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.db() as db:
                db.execute("UPDATE models SET status='interrupted' WHERE status='running'")
            for category in categories:
                succeeded = 0
                for listed in self.client.list_models(category, scan_limit):
                    if succeeded >= per_category:
                        break
                    seen = self._seen(listed.get('id'), listed.get('sha')) if listed.get('sha') else None
                    if seen and seen[0] in {'succeeded', 'rejected', 'duplicate'} and not retry_failed:
                        continue
                    result = self.candidate(listed, category, retry_failed)
                    result['category'] = category
                    results.append(result)
                    succeeded += result['status'] == 'succeeded'
            output = DatasetBuilder(self.workspace / 'data').build()
            (self.workspace / 'collection-report.json').write_text(json.dumps(results, indent=2) + '\n')
        return results, output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--category', choices=sorted(CATEGORIES), action='append',
                        help='Repeat for several categories; default is all three')
    parser.add_argument('--per-category', type=int, default=5, help='Successful models to collect per category')
    parser.add_argument('--scan-limit', type=int, default=100, help='Hub listing size searched per category')
    parser.add_argument('--family-limit', type=int, default=20)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    collector = HubCollector(args.workspace, family_limit=args.family_limit,
                             client=HubClient(token=os.environ.get('HF_TOKEN')))
    results, path = collector.run_hub(args.category or sorted(CATEGORIES), per_category=args.per_category,
                                      scan_limit=args.scan_limit, retry_failed=args.retry_failed)
    print(json.dumps({'results': results, 'datasets': str(path)}, indent=2))


if __name__ == '__main__':
    main()
