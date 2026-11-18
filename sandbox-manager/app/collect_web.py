"""Localhost dashboard for the automated Hugging Face collection.

Start, watch and stop ``modelcage.hub`` runs from a browser. Every line shown
comes from the same Report hooks the terminal uses: real Hugging Face requests,
category and checkpoint decisions, downloads, SHA-256 checks, sandbox lifecycle
events and SecCheck syscall counts. Nothing on the page is simulated.

Run it with the same privileges as the CLI collector (sudo, for SecCheck):

    sudo .venv/bin/python -m modelcage.collect_web --workspace hf-data
"""

import argparse
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3
from threading import RLock, Thread
import time
from urllib.parse import parse_qs, urlparse

from .datasets import CATEGORIES
from .hub import (HUB_URL, HubClient, HubCollector, LifecycleToReport, PIPELINE_TAGS, Report, TerminalReport,
                  preview)


class CollectionState:
    """One collection run at a time; its event journal is what the page renders."""

    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.lock = RLock()
        self.events, self.live = [], {}
        self.status, self.options, self.error = 'idle', None, None
        self.started = self.finished = None
        self.collector = self.report = None

    def add(self, kind, label, text, mark=None):
        with self.lock:
            self.events.append(dict(seq=len(self.events) + 1, time=time.strftime('%H:%M:%S'), kind=kind,
                                    label=label, text=text, mark=mark))
            self.live.clear()

    def snapshot(self, after=0):
        with self.lock:
            return dict(status=self.status, options=self.options, error=self.error, started=self.started,
                        finished=self.finished, events=self.events[after:], total=len(self.events),
                        live=dict(self.live), root=os.geteuid() == 0, workspace=str(self.workspace))

    def start(self, options):
        with self.lock:
            if self.status in {'running', 'stopping'}:
                raise RuntimeError('A collection is already running')
            self.events, self.live, self.error = [], {}, None
            self.status, self.options = 'running', options
            self.started, self.finished = time.strftime('%Y-%m-%d %H:%M:%S'), None
            self.report = WebReport(self)
        Thread(target=self._run, args=(options,), daemon=True).start()

    def stop(self):
        with self.lock:
            if self.status != 'running':
                return
            self.status = 'stopping'
            self.report.stop_requested = True
            if self.collector is not None:
                self.collector.stop_requested = True
        self.add('heading', '', 'Stop requested: the current model finishes, then the run ends')

    def _run(self, options):
        report = self.report
        try:
            token = os.environ.get('HF_TOKEN')
            client = HubClient(token=token, report=report)
            report.heading(f"Source: Hugging Face public REST API at {HUB_URL} "
                           f"({'HF_TOKEN set' if token else 'anonymous, no token'}); read-only, nothing is uploaded")
            if options['dry_run']:
                rows = preview(client, options['categories'], options['scan_limit'], report)
                accepted = sum(row['decision'] == 'would_download' for row in rows)
                report.heading(f'Dry run done: {accepted} would download, {len(rows) - accepted} rejected')
            else:
                if os.geteuid() != 0:
                    report.step('sandbox', 'Not running as root: SecCheck cannot attach, so runs will finish as '
                                           'instrumentation_unavailable. Restart this server with sudo.', 'fail')
                report.heading(f'Workspace: {self.workspace}')
                collector = HubCollector(self.workspace, family_limit=options['family_limit'], report=report,
                                         client=client)
                with self.lock:
                    self.collector = collector
                    collector.stop_requested = self.status == 'stopping'
                results, path = collector.run_hub(options['categories'], per_category=options['per_category'],
                                                  scan_limit=options['scan_limit'],
                                                  retry_failed=options['retry_failed'])
                counts = {}
                for result in results:
                    counts[result['status']] = counts.get(result['status'], 0) + 1
                report.heading('Done: ' + (', '.join(f'{n} {s}' for s, n in sorted(counts.items())) or 'no models'))
                report.heading(f'Datasets: {path}')
            final = 'finished'
        except BlockingIOError:
            final = 'failed'
            self.error = 'Another collector is using this workspace (is the terminal command still running?)'
        except Exception as error:
            logging.getLogger(__name__).exception('Collection run failed')
            final = 'failed'
            self.error = f'{type(error).__name__}: {str(error)[:300]}'
        with self.lock:
            if self.error:
                self.add('heading', '', self.error, 'fail')
            self.status, self.finished, self.collector = final, time.strftime('%Y-%m-%d %H:%M:%S'), None

    def summary(self):
        """Registry and dataset totals for the workspace, read-only."""
        registry = self.workspace / 'registry.sqlite'
        result = dict(models=[], candidates={}, datasets={})
        if registry.is_file():
            with sqlite3.connect(f'file:{registry}?mode=ro', uri=True) as db:
                for sha256, category, family, status, details in db.execute(
                        'SELECT sha256, category, family, status, details FROM models ORDER BY rowid DESC'):
                    model_id = json.loads(details or '{}').get('model_id')
                    result['models'].append(dict(model_id=model_id, category=category, family=family,
                                                 status=status, sha256=sha256))
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='hub_candidates'").fetchone():
                    result['candidates'] = dict(db.execute('SELECT status, count(*) FROM hub_candidates GROUP BY 1'))
        for category in CATEGORIES:
            result['datasets'][category] = {view: _rows(self.workspace / 'data' / view / f'{category}.csv')
                                            for view in ('datasets', 'training')}
        return result


def _rows(path):
    try:
        with path.open() as stream:
            return max(sum(1 for _ in stream) - 1, 0)
    except OSError:
        return 0


class WebReport(Report):
    """Report sink that records steps for the page and mirrors them to the server terminal."""

    def __init__(self, state):
        self.state = state
        self.terminal = TerminalReport()
        self.stop_requested = False

    def heading(self, text):
        self.state.add('heading', '', text)
        self.terminal.heading(text)

    def step(self, label, text, mark=None):
        self.state.add('step', label, text, mark)
        self.terminal.step(label, text, mark)

    def live(self, label, text, fraction=None):
        with self.state.lock:
            self.state.live[label] = dict(text=text, fraction=fraction)
        self.terminal.live(label, text, fraction)


def parse_options(body):
    categories = body.get('categories') or sorted(CATEGORIES)
    if not isinstance(categories, list) or not categories or set(categories) - set(CATEGORIES):
        raise ValueError('Choose one or more of: ' + ', '.join(sorted(CATEGORIES)))

    def number(key, default, low, high):
        value = body.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f'{key} must be a whole number from {low} to {high}')
        return value

    return dict(categories=[c for c in sorted(CATEGORIES) if c in categories],
                per_category=number('per_category', 5, 1, 200), scan_limit=number('scan_limit', 100, 1, 1000),
                family_limit=number('family_limit', 20, 1, 1000),
                retry_failed=body.get('retry_failed') is True, dry_run=body.get('dry_run') is True)


class CollectionServer(ThreadingHTTPServer):
    def __init__(self, address, state):
        super().__init__(address, CollectionHandler)
        self.state = state


class CollectionHandler(BaseHTTPRequestHandler):
    server: CollectionServer

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def _send(self, status, body, content_type='application/json'):
        encoded = (json.dumps(body) if content_type == 'application/json' else body).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', f'{content_type}; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path == '/':
            self._send(HTTPStatus.OK, PAGE, 'text/html')
        elif url.path == '/api/state':
            after = parse_qs(url.query).get('after', ['0'])[0]
            self._send(HTTPStatus.OK, self.server.state.snapshot(int(after) if after.isdigit() else 0))
        elif url.path == '/api/summary':
            self._send(HTTPStatus.OK, self.server.state.summary())
        elif url.path.startswith('/api/download/'):
            self._download(url.path.removeprefix('/api/download/'))
        else:
            self._send(HTTPStatus.NOT_FOUND, {'error': 'not found'})

    def _download(self, relative):
        """Serve one dataset file the root-owned workspace keeps private (mode 600).

        Only names in DOWNLOADS are served, so raw traces, quarantine and models stay private.
        """
        view, _, name = relative.partition('/')
        if name not in DOWNLOADS.get(view, ()):
            self._send(HTTPStatus.NOT_FOUND, {'error': 'not a downloadable dataset file'})
            return
        try:
            data = (self.server.state.workspace / 'data' / view / name).read_bytes()
        except OSError:
            self._send(HTTPStatus.NOT_FOUND, {'error': 'file not built yet; run a collection first'})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/csv; charset=utf-8' if name.endswith('.csv') else 'application/json')
        # Prefix the view so training and measurement copies of the same name don't overwrite each other.
        self.send_header('Content-Disposition', f'attachment; filename="{view}-{name}"')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        # JSON-only POSTs from this page's own origin: a cross-site form cannot start a run.
        origin = self.headers.get('Origin')
        host = self.headers.get('Host', '')
        if not self.headers.get('Content-Type', '').startswith('application/json') or (
                origin is not None and origin != f'http://{host}'):
            self._send(HTTPStatus.FORBIDDEN, {'error': 'JSON requests from this page only'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            body = json.loads(self.rfile.read(length) or b'{}') if 0 <= length <= 4096 else None
            if not isinstance(body, dict):
                raise ValueError('Request body must be a JSON object of at most 4 KiB')
            if path == '/api/start':
                self.server.state.start(parse_options(body))
            elif path == '/api/stop':
                self.server.state.stop()
            else:
                self._send(HTTPStatus.NOT_FOUND, {'error': 'not found'})
                return
        except (ValueError, json.JSONDecodeError) as error:
            self._send(HTTPStatus.BAD_REQUEST, {'error': str(error)})
            return
        except RuntimeError as error:
            self._send(HTTPStatus.CONFLICT, {'error': str(error)})
            return
        self._send(HTTPStatus.OK, {'status': self.server.state.status})

    def log_message(self, format, *args):
        """Polling would flood the terminal; collection progress is printed instead."""


# data/training is the strict OCSVM training view; data/datasets holds every complete measurement.
DOWNLOADS = {view: {*(f'{c}{suffix}' for c in CATEGORIES for suffix in ('.csv', '.normalized.csv')),
                    'metadata.csv', 'quality.json', 'schema.json'}
             for view in ('training', 'datasets')}

STEPS = [('category', 'Category check'), ('checkpoint', 'Checkpoint'), ('dedup', 'Deduplication'),
         ('download', 'Download'), ('verify', 'SHA-256 verify'), ('sandbox', 'gVisor sandbox'),
         ('capture', 'Syscall capture'), ('result', 'Result')]

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ModelCage | Dataset collection</title>
<style>
:root{color-scheme:dark;--bg:#09111f;--panel:#111d31;--soft:#17253c;--line:#2b3c59;--text:#edf4ff;--muted:#9fb0c9;--accent:#6ee7c7;--accent2:#37cfa9;--warn:#ffd27a;--danger:#ff8c9e;--shadow:0 24px 70px rgba(0,0,0,.28)}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 15% 0,#17345a 0,transparent 38%),var(--bg);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}
.shell{max-width:1180px;margin:0 auto;padding:36px 20px 64px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:28px}
.brand{display:flex;align-items:center;gap:12px;font-weight:750}.mark{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--accent),#6ca8ff);color:#07121e;font-weight:900}
.pill{border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:12px;padding:5px 11px;white-space:nowrap}.pill.ok{color:var(--accent);border-color:rgba(110,231,199,.4)}.pill.bad{color:var(--danger);border-color:rgba(255,140,158,.4)}.pill.run{color:var(--warn);border-color:rgba(255,210,122,.4)}
h1{font-size:clamp(1.8rem,4vw,2.8rem);line-height:1.08;letter-spacing:-.04em;margin:0 0 10px}h1 span{color:var(--accent)}.lead{color:var(--muted);margin:0 0 26px;max-width:760px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.card{background:rgba(17,29,49,.9);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);padding:22px;min-width:0}
.eyebrow{color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.card h2{font-size:1.1rem;margin:2px 0 12px}.muted{color:var(--muted);font-size:13px}
.checks{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 14px}.checks label{display:flex;align-items:center;gap:7px;background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:7px 11px;cursor:pointer;font-size:14px}
.fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.fields label{font-size:12px;color:var(--muted)}input[type=number]{width:100%;margin-top:4px;border:1px solid var(--line);border-radius:9px;background:#0b1628;color:var(--text);font:inherit;padding:8px 10px}
.toggles{display:flex;flex-wrap:wrap;gap:16px;margin:14px 0 4px;font-size:14px}.toggles label{display:flex;gap:7px;align-items:center;cursor:pointer}
.actions{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}button{border:0;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#06151a;cursor:pointer;font:inherit;font-weight:800;padding:11px 18px}button.secondary{background:var(--soft);color:var(--text);border:1px solid var(--line)}button:disabled{opacity:.45;cursor:not-allowed}
.error{border:1px solid rgba(255,140,158,.4);border-radius:10px;background:rgba(255,140,158,.1);color:var(--danger);padding:10px 13px;margin-top:12px;font-size:14px}
.flow{list-style:none;margin:0;padding:0;counter-reset:s}.flow li{counter-increment:s;display:grid;grid-template-columns:26px 1fr;gap:8px;padding:7px 0;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}.flow li::before{content:counter(s);display:grid;place-items:center;width:22px;height:22px;border-radius:7px;background:var(--soft);color:var(--accent);font-weight:800;font-size:11px}.flow b{color:var(--text);font-weight:650}
code{color:var(--accent);font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}.stat{background:rgba(17,29,49,.9);border:1px solid var(--line);border-radius:14px;padding:14px 16px}.stat .n{font-size:1.7rem;font-weight:800;letter-spacing:-.03em}.stat .k{color:var(--muted);font-size:12px}
.current{margin-top:18px}.model{font:600 15px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere;margin:2px 0 12px}
.steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.st{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:10px 12px;min-width:0}.st .h{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);font-weight:700}.st .t{font-size:12px;margin-top:5px;color:#c9d8ed;overflow-wrap:anywhere;max-height:3.9em;overflow:hidden}
.st.ok{border-color:rgba(110,231,199,.5)}.st.fail{border-color:rgba(255,140,158,.6)}.st.skip{border-color:rgba(255,210,122,.5)}.st.active{border-color:#6ca8ff;box-shadow:0 0 0 2px rgba(108,168,255,.15)}
.bar{height:6px;border-radius:9px;background:#0b1628;margin-top:7px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent),#6ca8ff);transition:width .3s}
#log{background:#070f1c;border:1px solid var(--line);border-radius:12px;height:420px;overflow:auto;padding:12px 14px;font:12px/1.65 ui-monospace,SFMono-Regular,monospace}#log div{white-space:pre-wrap;overflow-wrap:anywhere}#log .hd{color:var(--text);font-weight:700;margin-top:8px}#log .tm{color:#5d7090}#log .lb{color:#6ca8ff}#log .ok{color:var(--accent)}#log .fail{color:var(--danger)}#log .skip{color:var(--warn)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px 8px;border-top:1px solid var(--line);overflow-wrap:anywhere}th{color:var(--muted);font-weight:600;font-size:12px}td.s-succeeded{color:var(--accent)}td.s-failed,td.s-incomplete,td.s-timed_out,td.s-instrumentation_unavailable{color:var(--danger)}
.dl{display:flex;flex-wrap:wrap;gap:6px}.dl a,#extras a{display:inline-block;border:1px solid var(--line);background:var(--soft);border-radius:8px;color:var(--accent);font-size:12px;font-weight:700;padding:5px 10px;text-decoration:none;white-space:nowrap}.dl a:hover{border-color:var(--accent)}#extras{margin-top:12px}
.wide{grid-column:1/-1}@media(max-width:860px){.grid{grid-template-columns:1fr}.stats,.steps{grid-template-columns:repeat(2,minmax(0,1fr))}.fields{grid-template-columns:1fr}.top .pill.hide-sm{display:none}}
</style></head><body><main class="shell">
<header class="top"><div class="brand"><span class="mark">M</span> ModelCage</div><div style="display:flex;gap:8px"><span id="priv" class="pill hide-sm">checking privileges…</span><span id="runstate" class="pill">idle</span></div></header>
<h1>Automated <span>dataset collection</span></h1>
<p class="lead">Pulls popular pickle-based PyTorch checkpoints from Hugging Face, loads each one inside a disposable gVisor sandbox, and records every system call into per-category datasets. Each line below is a real request, check or sandbox event.</p>
<div class="grid">
<section class="card"><div class="eyebrow">Run</div><h2>Collection settings</h2>
<div class="muted">Categories</div><div class="checks" id="cats"></div>
<div class="fields"><label>New models per category<input id="per" type="number" min="1" max="200" value="5"></label><label>Models scanned per category<input id="scan" type="number" min="1" max="1000" value="100"></label><label>Max per family<input id="fam" type="number" min="1" max="1000" value="20"></label></div>
<div class="toggles"><label><input id="retry" type="checkbox"> Retry failed models</label><label><input id="dry" type="checkbox"> Dry run (check only, no download)</label></div>
<div class="actions"><button id="start">Start collection</button><button id="stop" class="secondary" disabled>Stop after current model</button></div>
<div id="err" class="error" hidden></div>
<p class="muted" style="margin:14px 0 0">Workspace: <code id="ws"></code></p></section>
<section class="card"><div class="eyebrow">Process</div><h2>What happens to each model</h2>
<ol class="flow">
<li><span><b>List</b> · <code>GET huggingface.co/api/models?pipeline_tag=…&amp;sort=downloads</code></span></li>
<li><span><b>Metadata</b> · <code>GET /api/models/&lt;repo&gt;?blobs=true</code>: config, tags, file SHA-256</span></li>
<li><span><b>Category check</b> · library, pipeline tag and architecture must agree</span></li>
<li><span><b>Dedup</b> · known SHA-256 or full family quota is skipped before download</span></li>
<li><span><b>Download + verify</b> · <code>pytorch_model.bin</code> at a pinned revision; size and SHA-256 must match</span></li>
<li><span><b>Sandbox</b> · gVisor <code>runsc</code>, no network, read-only root, 4 GiB, 1 CPU, 60 s</span></li>
<li><span><b>Load</b> · <code>torch.load(weights_only=False)</code> unpickles <code>data.pkl</code> inside the sandbox</span></li>
<li><span><b>Capture</b> · gVisor SecCheck streams every syscall; any loss rejects the run</span></li>
<li><span><b>Store</b> · raw JSONL, features, category CSV row; sandbox destroyed</span></li>
</ol></section>
</div>
<div class="stats" id="stats"></div>
<section class="card current" id="current" hidden><div class="eyebrow">Now processing</div><div class="model" id="model">—</div><div class="steps" id="steps"></div></section>
<div class="grid" style="margin-top:18px">
<section class="card wide"><div class="eyebrow">Activity</div><h2>Live log</h2><div id="log"><div class="tm">No run yet. Choose settings and press Start.</div></div></section>
<section class="card wide"><div class="eyebrow">Download</div><h2>Datasets</h2><p class="muted" style="margin:0 0 10px"><b>Training</b> is the strict set for OCSVM training (successful, complete, verified runs only). <b>All measurements</b> also includes complete captures of models that failed to load. Files are rebuilt after every run.</p>
<div style="overflow:auto"><table><thead><tr><th>Category</th><th>Training rows</th><th>Measured rows</th><th>Download</th></tr></thead><tbody id="downloads"></tbody></table></div>
<div class="dl" id="extras"></div></section>
<section class="card wide"><div class="eyebrow">Registry</div><h2>Collected models</h2><div class="muted" id="cand"></div><div style="overflow:auto"><table><thead><tr><th>Model</th><th>Category</th><th>Family</th><th>Status</th><th>SHA-256</th></tr></thead><tbody id="models"><tr><td colspan="5" class="muted">Nothing collected yet.</td></tr></tbody></table></div></section>
</div>
</main>
<script>
const CATEGORIES = __CATEGORIES__, STEPS = __STEPS__;
const $ = id => document.getElementById(id);
for (const c of CATEGORIES) { const l = document.createElement('label'); l.innerHTML = '<input type="checkbox" checked>'; l.firstChild.value = c; l.append(c.replace('_', ' ')); $('cats').append(l); }
let seen = 0, events = [], lastStatus = null, lastModel = null, lastLive = {};
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
async function post(path, body) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
  const v = await r.json(); if (!r.ok) throw new Error(v.error || 'Request failed'); return v;
}
$('start').onclick = async () => {
  $('err').hidden = true;
  try {
    await post('/api/start', {categories: [...document.querySelectorAll('#cats input:checked')].map(i => i.value),
      per_category: +$('per').value, scan_limit: +$('scan').value, family_limit: +$('fam').value,
      retry_failed: $('retry').checked, dry_run: $('dry').checked});
    seen = 0; events = []; $('log').textContent = ''; poll();
  } catch (e) { $('err').textContent = e.message; $('err').hidden = false; }
};
$('stop').onclick = () => post('/api/stop').catch(e => { $('err').textContent = e.message; $('err').hidden = false; });
function logLine(e) {
  const d = el('div', e.kind === 'heading' ? 'hd' : '');
  d.append(el('span', 'tm', e.time + '  '));
  if (e.kind === 'step') { d.append(el('span', 'lb', e.label.padEnd(11))); d.append(el('span', e.mark || '', (e.mark === 'ok' ? '✓ ' : e.mark === 'fail' ? '✗ ' : e.mark === 'skip' ? '– ' : '  ')));
    d.append(el('span', '', e.text)); }
  else d.append(el('span', e.mark || '', e.text));
  return d;
}
function renderCurrent(live, running) {
  let start = -1;
  for (let i = events.length - 1; i >= 0; i--) if (events[i].kind === 'heading' && events[i].text.startsWith('[')) { start = i; break; }
  if (start < 0) { $('current').hidden = true; return; }
  $('current').hidden = false;
  $('model').textContent = events[start].text;
  const state = {};
  for (const e of events.slice(start + 1)) {
    if (e.kind !== 'step') continue;
    const key = e.label === 'http' ? (e.text.includes('/resolve/') ? 'download' : null) : e.label;
    if (!key || !STEPS.some(s => s[0] === key)) continue;
    const prev = state[key] || {};
    state[key] = {text: e.text, mark: e.mark || prev.mark || (key === 'sandbox' ? 'ok' : null)};
    if (e.mark === 'fail') state[key].mark = 'fail';
  }
  // Live values vanish once a step completes; keep the last one so the capture count stays visible.
  if (lastModel !== events[start].text) { lastModel = events[start].text; lastLive = {}; }
  for (const [key, v] of Object.entries(lastLive)) if (!state[key]) state[key] = {text: v.text, mark: 'ok'};
  for (const [key, v] of Object.entries(live || {})) { lastLive[key] = v; state[key] = {text: v.text, mark: 'active', fraction: v.fraction}; }
  const steps = $('steps'); steps.textContent = '';
  for (const [key, name] of STEPS) {
    const s = state[key], box = el('div', 'st ' + (s ? s.mark || 'ok' : ''));
    const h = el('div', 'h'); h.append(el('span', '', name));
    h.append(el('span', '', !s ? '·' : s.mark === 'fail' ? '✗' : s.mark === 'skip' ? '–' : s.mark === 'active' ? '…' : '✓'));
    box.append(h, el('div', 't', s ? s.text : (running ? 'waiting' : '—')));
    if (s && s.fraction != null) { const b = el('div', 'bar'), i = el('i'); i.style.width = Math.round(s.fraction * 100) + '%'; b.append(i); box.append(b); }
    steps.append(box);
  }
}
function stat(n, k) { const d = el('div', 'stat'); d.append(el('div', 'n', String(n)), el('div', 'k', k)); return d; }
async function summary() {
  try {
    const v = await (await fetch('/api/summary', {cache: 'no-store'})).json();
    const s = $('stats'); s.textContent = '';
    for (const c of CATEGORIES) s.append(stat(v.datasets[c].training, c.replace('_', ' ') + ' · training rows (' + v.datasets[c].datasets + ' measured)'));
    const ok = v.models.filter(m => m.status === 'succeeded').length;
    s.append(stat(ok + ' / ' + v.models.length, 'models succeeded / downloaded'));
    const link = (href, text) => { const a = el('a', '', text); a.href = '/api/download/' + href; a.download = ''; return a; };
    const dl = $('downloads'); dl.textContent = '';
    for (const c of CATEGORIES) {
      const r = el('tr'), td = el('td'), box = el('div', 'dl');
      box.append(link('training/' + c + '.csv', 'Training CSV'), link('training/' + c + '.normalized.csv', 'Training normalized'),
        link('datasets/' + c + '.csv', 'All measurements'), link('datasets/' + c + '.normalized.csv', 'All normalized'));
      td.append(box);
      r.append(el('td', '', c.replace('_', ' ')), el('td', '', String(v.datasets[c].training)), el('td', '', String(v.datasets[c].datasets)), td);
      dl.append(r);
    }
    if (!$('extras').childElementCount) $('extras').append(link('datasets/metadata.csv', 'metadata.csv (every run)'),
      link('training/quality.json', 'quality.json (balance, exclusions)'), link('datasets/schema.json', 'schema.json (column order)'));
    $('cand').textContent = 'Hub repositories examined: ' + (Object.entries(v.candidates).map(([k, n]) => n + ' ' + k).join(' · ') || 'none yet');
    const body = $('models'); body.textContent = '';
    if (!v.models.length) { const r = el('tr'); const td = el('td', 'muted', 'Nothing collected yet.'); td.colSpan = 5; r.append(td); body.append(r); }
    for (const m of v.models) { const r = el('tr'); r.append(el('td', '', m.model_id || '?'), el('td', '', m.category), el('td', '', m.family), el('td', 's-' + m.status, m.status), el('td', '', (m.sha256 || '').slice(0, 16) + '…')); body.append(r); }
  } catch (e) {}
}
async function poll() {
  let v;
  try { v = await (await fetch('/api/state?after=' + seen, {cache: 'no-store'})).json(); }
  catch (e) { $('runstate').textContent = 'server unreachable'; $('runstate').className = 'pill bad'; return setTimeout(poll, 2000); }
  const log = $('log'), atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  if (v.total < seen) { seen = 0; events = []; log.textContent = ''; }
  if (seen === 0 && v.events.length) log.textContent = '';
  for (const e of v.events) { events.push(e); log.append(logLine(e)); }
  seen = v.total;
  if (atBottom) log.scrollTop = log.scrollHeight;
  const running = v.status === 'running' || v.status === 'stopping';
  $('runstate').textContent = v.status; $('runstate').className = 'pill ' + (running ? 'run' : v.status === 'failed' ? 'bad' : v.status === 'finished' ? 'ok' : '');
  $('priv').textContent = v.root ? 'running as root · capture available' : 'not root · dry run only';
  $('priv').className = 'pill hide-sm ' + (v.root ? 'ok' : 'bad');
  $('ws').textContent = v.workspace;
  $('start').disabled = running; $('stop').disabled = v.status !== 'running';
  renderCurrent(v.live, running);
  if (lastStatus !== v.status || running) summary();
  lastStatus = v.status;
  setTimeout(poll, running ? 700 : 3000);
}
poll(); summary();
</script></body></html>""".replace('__CATEGORIES__', json.dumps(sorted(CATEGORIES))).replace('__STEPS__', json.dumps(STEPS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    state = CollectionState(args.workspace)
    lifecycle = logging.getLogger('modelcage.lifecycle')
    lifecycle.addHandler(LifecycleToReport(_Relay(state)))
    lifecycle.setLevel(logging.INFO)
    lifecycle.propagate = False
    server = CollectionServer(('127.0.0.1', args.port), state)
    print(f'ModelCage collection dashboard: http://127.0.0.1:{args.port}  (workspace {state.workspace})')
    if os.geteuid() != 0:
        print('Not running as root: dry runs work, but sandbox capture needs sudo.')
    server.serve_forever()


class _Relay(Report):
    """Route sandbox lifecycle log records to whichever run is current."""

    def __init__(self, state):
        self.state = state

    def step(self, label, text, mark=None):
        if self.state.report:
            self.state.report.step(label, text, mark)

    def live(self, label, text, fraction=None):
        if self.state.report:
            self.state.report.live(label, text, fraction)


if __name__ == '__main__':
    main()
