"""Small localhost-only UI for manually exercising the ModelCage POC."""

import argparse
import cgi
import json
import logging
from threading import Thread, BoundedSemaphore
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse

from .modelcage import ModelCage, ModelInputError
from .instrumentation import require_category


MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_EXECUTION_ID = re.compile(r"[0-9a-f-]{36}")


class ModelCageWebServer(ThreadingHTTPServer):
    """HTTP server state kept outside the untrusted sandbox boundary."""

    def __init__(self, address: tuple[str, int], cage: ModelCage, upload_dir: Path):
        super().__init__(address, ModelCageWebHandler)
        self.cage = cage
        self.upload_dir = upload_dir
        self.slots = BoundedSemaphore(2)


class ModelCageWebHandler(BaseHTTPRequestHandler):
    """Serve a single upload form and selected-execution CSV downloads."""

    server: ModelCageWebServer

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._page()
            return
        match = re.fullmatch(r"/analyses/([0-9a-f-]{36})", path)
        if match:
            try:
                snapshot = self.server.cage.events.snapshot(match.group(1))
            except (FileNotFoundError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            encoded = json.dumps(snapshot).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        match = re.fullmatch(r"/exports/([0-9a-f-]{36})\.csv", path)
        if match:
            self._csv(match.group(1))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/execute":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            self._page(error="Upload must be between 1 byte and 64 MiB.", status=HTTPStatus.BAD_REQUEST)
            return
        if not self.headers.get("Content-Type", "").startswith("multipart/form-data"):
            self._page(error="Use the model upload form.", status=HTTPStatus.BAD_REQUEST)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]},
        )
        if "model" not in form or not form["model"].filename:
            self._page(error="Choose a .pkl or .pickle model file.", status=HTTPStatus.BAD_REQUEST)
            return

        uploaded = form["model"]
        category = form.getfirst("category")
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in {".pkl", ".pickle"}:
            self._page(error="Only .pkl and .pickle files are supported.", status=HTTPStatus.BAD_REQUEST)
            return

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=suffix, prefix="upload-", dir=self.server.upload_dir, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                remaining = MAX_UPLOAD_BYTES + 1
                while chunk := uploaded.file.read(min(1024 * 1024, remaining)):
                    temporary_file.write(chunk)
                    remaining -= len(chunk)
                    if remaining <= 0:
                        raise ModelInputError("Upload exceeds the 64 MiB limit.")

            category = require_category(category).value if category else None
            manifest_text = form.getfirst("manifest", "")
            if len(manifest_text) > 16384:
                raise ValueError("Manifest must be at most 16 KiB")
            manifest = json.loads(manifest_text) if manifest_text.strip() else None
            family = form.getfirst('family', '').strip()
            if family:
                if len(family) > 200:
                    raise ValueError('Model family must be at most 200 characters')
                if manifest is not None and not isinstance(manifest, dict):
                    raise ValueError('Manifest must be a JSON object')
                manifest = {**(manifest or {}), 'family': family}
            if not self.server.slots.acquire(blocking=False):
                self._page(error="Two analyses are already running. Try again after one finishes.",
                           status=HTTPStatus.SERVICE_UNAVAILABLE)
                return
            try:
                analysis_id = self.server.cage.events.begin(category=category)
                worker_path = temporary_path

                def execute():
                    try:
                        self.server.cage.execute_model(worker_path, category=category, analysis_id=analysis_id,
                                                       manifest=manifest, original_filename=Path(uploaded.filename).name)
                    except Exception:
                        # The facade records failure without exposing model output or exception content.
                        pass
                    finally:
                        worker_path.unlink(missing_ok=True)
                        self.server.slots.release()

                Thread(target=execute, daemon=False).start()
                temporary_path = None  # worker owns the upload from this point
            except Exception:
                self.server.slots.release()
                raise
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", f"/?analysis={analysis_id}")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (ModelInputError, ValueError) as error:
            self._page(error=str(error), status=HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._page(error="Unable to start analysis.", status=HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _csv(self, execution_id: str) -> None:
        try:
            csv_data = self.server.cage.export_csv(execution_id).encode("utf-8")
        except KeyError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{execution_id}.csv"')
        self.send_header("Content-Length", str(len(csv_data)))
        self.end_headers()
        self.wfile.write(csv_data)

    def _page(
        self,
        *,
        result=None,
        trace=None,
        error: str | None = None,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        message = ""
        if error:
            message = f'<p class="error">{escape(error)}</p>'
        elif result is not None and trace is not None:
            execution_id = escape(result.execution_id)
            message = f"""
            <section class="result">
              <h2>Execution complete</h2>
              <dl>
                <dt>Execution ID</dt><dd><code>{execution_id}</code></dd>
                <dt>Exit code</dt><dd>{result.exit_code}</dd>
                <dt>Duration</dt><dd>{trace.duration_seconds}s</dd>
                <dt>Syscalls</dt><dd>{len(trace.syscalls)}</dd>
                <dt>Termination</dt><dd>{escape(trace.termination_reason or "unknown")}</dd>
              </dl>
              <a href="/exports/{execution_id}.csv">Download syscall CSV</a>
            </section>"""

        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ModelCage | Secure model analysis</title>
<style>
:root{{color-scheme:dark;--bg:#09111f;--panel:#111d31;--panel-soft:#17253c;--line:#2b3c59;--text:#edf4ff;--muted:#9fb0c9;--accent:#6ee7c7;--accent-strong:#37cfa9;--danger:#ff8c9e;--shadow:0 24px 70px rgba(0,0,0,.28)}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;background:radial-gradient(circle at 15% 0,#17345a 0,transparent 38%),var(--bg);color:var(--text);font:15px/1.6 Inter,ui-sans-serif,system-ui,sans-serif}}
.shell{{max-width:1000px;margin:0 auto;padding:48px 22px 72px}}.topbar{{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:52px}}
.brand{{display:flex;align-items:center;gap:12px;font-weight:750;letter-spacing:.02em}}.mark{{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--accent),#6ca8ff);color:#07121e;font-weight:900}}
.badge{{border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:12px;padding:5px 11px}}.hero{{max-width:720px;margin-bottom:34px}}
h1{{font-size:clamp(2.1rem,5vw,3.7rem);line-height:1.05;letter-spacing:-.055em;margin:0 0 18px}}h1 span{{color:var(--accent)}}.hero p{{color:var(--muted);font-size:1.08rem;margin:0}}
.layout{{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(260px,.7fr);gap:20px}}.card{{background:rgba(17,29,49,.88);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow);padding:26px}}
.card h2{{font-size:1.15rem;margin:0 0 5px}}.eyebrow{{color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}}.subtle{{color:var(--muted);font-size:13px;margin:0 0 22px}}
label{{display:block;color:#dbe6f8;font-weight:650;margin:18px 0 7px}}input,textarea{{display:block;width:100%;border:1px solid var(--line);border-radius:10px;background:#0b1628;color:var(--text);font:inherit;padding:11px 12px;outline:none}}input:focus,textarea:focus{{border-color:var(--accent);box-shadow:0 0 0 3px rgba(110,231,199,.12)}}textarea{{min-height:108px;resize:vertical}}input[type=file]{{padding:9px;color:var(--muted)}}::placeholder{{color:#647793}}
button{{border:0;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent-strong));color:#06151a;cursor:pointer;font:inherit;font-weight:800;margin-top:22px;padding:12px 18px;transition:transform .15s,filter .15s}}button:hover{{filter:brightness(1.08);transform:translateY(-1px)}}.error{{border:1px solid rgba(255,140,158,.4);border-radius:10px;background:rgba(255,140,158,.1);color:var(--danger);padding:12px 15px}}.result{{margin-top:20px;border:1px solid rgba(110,231,199,.35);border-radius:14px;background:rgba(110,231,199,.07);padding:20px}}.result h2{{margin-top:0}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;margin:18px 0}}dt{{color:var(--muted)}}dd{{margin:0;overflow-wrap:anywhere}}code{{color:var(--accent);font:12px ui-monospace,SFMono-Regular,monospace}}a{{color:var(--accent);font-weight:700}}
.side-list{{list-style:none;padding:0;margin:22px 0 0}}.side-list li{{border-top:1px solid var(--line);padding:13px 0;color:var(--muted)}}.side-list strong{{display:block;color:var(--text);font-size:14px}}.side-list strong::before{{content:"✓";color:var(--accent);margin-right:8px}}
#live{{margin-top:20px}}#connection{{color:var(--accent);font-size:13px}}#status{{background:#091321;border:1px solid var(--line);border-radius:12px;color:#c9d8ed;overflow:auto;padding:18px;white-space:pre-wrap;word-break:break-word}}#events{{display:grid;gap:8px;list-style:none;margin:18px 0;padding:0}}#events li{{background:var(--panel-soft);border-left:3px solid var(--accent);border-radius:7px;padding:8px 12px;color:#c7d4e8;font-size:13px}}.paths{{color:var(--muted);font-size:13px;margin-top:22px}}@media(max-width:760px){{.shell{{padding-top:25px}}.topbar{{margin-bottom:35px}}.layout{{grid-template-columns:1fr}}.badge{{display:none}}}}
</style></head><body><main class="shell">
<header class="topbar"><div class="brand"><span class="mark">M</span> ModelCage</div><span class="badge">LOCAL ANALYSIS CONSOLE</span></header>
<section class="hero"><div class="eyebrow">Policy-driven runtime instrumentation</div><h1>Inspect models in a <span>disposable sandbox.</span></h1><p>Upload a pickle model and ModelCage will quarantine, identify, and execute it behind gVisor isolation while collecting syscall evidence.</p></section>
<div class="layout"><section class="card"><div class="eyebrow">New analysis</div><h2>Submit a model</h2><p class="subtle">Supported formats: .pkl and .pickle · Maximum size: 64 MiB</p>
<form method="post" action="/execute" enctype="multipart/form-data">
<label for="model">Pickle model</label><input id="model" type="file" name="model" accept=".pkl,.pickle" required><p id="file-hint" class="subtle file-hint">Choose a .pkl or .pickle file to begin.</p>
<p class="subtle">Category identification is automatic. Models with insufficient or conflicting task information are held for review.</p>
<label for="manifest">Model config or metadata <span class="subtle">(optional JSON)</span></label><textarea id="manifest" name="manifest" maxlength="16384" placeholder='{{"architectures":["BertForSequenceClassification"]}}'></textarea>
<label for="family">Model family <span class="subtle">(optional)</span></label><input id="family" name="family" maxlength="200" placeholder="e.g. GPT, T5, ResNet">
<button type="submit">Run securely <span aria-hidden="true">→</span></button></form>{message}</section>
<aside class="card"><div class="eyebrow">What happens next</div><h2>Evidence, not assumptions</h2><p class="subtle">Every run follows the same guarded workflow.</p>
<ul class="side-list"><li><strong>Quarantine</strong>Bytes are hashed and verified before staging.</li><li><strong>Identify</strong>Metadata is inspected without loading the model on the host.</li><li><strong>Observe</strong>Syscall traces and execution events are captured.</li><li><strong>Review</strong>Evidence is routed to the matching dataset or held safely.</li></ul></aside></div>
<section id="live" class="card" hidden><div class="eyebrow">Live telemetry</div><h2>Analysis in progress</h2><p id="connection"></p><pre id="status"></pre><ol id="events"></ol><p class="paths">Raw traces and category syscall CSVs are saved automatically.<br>Counts: <code>data/features/&lt;category&gt;.csv</code> · Frequencies: <code>data/features/&lt;category&gt;.normalized.csv</code> · Metadata: <code>data/metadata.csv</code></p></section>
</main></body></html>"""
        html = html.replace("</body>", LIVE_SCRIPT + "</body>")
        encoded = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        """Keep test-model output out of the terminal except explicit errors."""


LIVE_SCRIPT = r"""<script>
const modelInput = document.getElementById('model');
const fileHint = document.getElementById('file-hint');
if (modelInput && fileHint) {
  modelInput.addEventListener('change', () => {
    const file = modelInput.files[0];
    if (!file) {
      fileHint.textContent = 'Choose a .pkl or .pickle file to begin.';
      return;
    }
    const size = file.size < 1024 * 1024
      ? Math.max(1, Math.round(file.size / 1024)) + ' KB'
      : (file.size / (1024 * 1024)).toFixed(1) + ' MB';
    fileHint.textContent = file.name + ' · ' + size + ' · ready to upload';
  });
}
const analysis = new URLSearchParams(location.search).get('analysis');
if (analysis && /^[0-9a-f-]{36}$/.test(analysis)) {
  document.getElementById('live').hidden = false;
  const state = document.getElementById('status');
  const connection = document.getElementById('connection');
  const events = document.getElementById('events');
  let seen = 0;
  async function refresh() {
    try {
      const response = await fetch('/analyses/' + analysis, {cache:'no-store'});
      if (!response.ok) throw new Error('Status unavailable');
      const value = await response.json();
      connection.textContent = 'Connected — progress comes from backend operations.';
      state.textContent = 'Analysis: ' + analysis + '\nCategory: ' + (value.task_category || value.category || 'unclassified') +
        '\nModel SHA-256: ' + (value.model_sha256 || (value.hashing_status === 'paused' ? 'paused' : 'pending')) +
        '\nIdentity verified: ' + (value.identity_verified === true ? 'yes' : 'not yet verified') +
        '\nQuarantine copy: ' + (value.quarantine_reused === true ? 'reused after verification' : 'new') +
        '\nSandbox: ' + (value.sandbox_id || 'pending') +
        '\nRuntime: ' + (value.runtime || 'not yet verified') +
        '\nStage: ' + value.stage + '\nStatus: ' + value.status +
        '\nCategory source: ' + (value.category_source || 'pending') +
        '\nCategory validation: ' + (value.validation_status || 'pending') +
        '\nCategory evidence: ' + (value.metadata?.category_reason || 'Inspecting model metadata') +
        '\nCapture: ' + value.capture_status + '\nEvents: ' + value.event_count +
        '\nRecord count: ' + (value.record_count ?? 0) +
        '\nReported losses: ' + (value.reported_losses ?? 'unavailable') +
        '\nObservations: ' + (value.observation_count ?? 0) +
        '\nError: ' + (value.failure_reason || (value.capture_errors || []).join(', ') || 'none') +
        '\nStorage: ' + value.storage_status + '\nCleanup: ' + value.cleanup_status +
        '\nFeature CSV directory: ' + (value.feature_directory || 'pending') +
        '\nPer-analysis frequency CSV directory: ' + (value.frequency_directory || 'pending') +
        '\nMetadata CSV: ' + (value.metadata_csv || 'pending');
      for (const event of value.events) {
        if (event.sequence <= seen) continue;
        const row = document.createElement('li');
        row.textContent = event.timestamp + ' [' + event.level + '] ' + event.message +
          (event.observation ? ' — ' + event.observation.reason + ' (' + event.observation.outcome +
            '; source events: ' + event.observation.source_sequences.join(', ') + ')' : '');
        events.appendChild(row);
        seen = event.sequence;
      }
      if (value.stage === 'finalized') {
        return;
      }
    } catch (error) {
      connection.textContent = 'Connection interrupted; retrying. This URL retains the analysis ID.';
    }
    setTimeout(refresh, 1000);
  }
  refresh();
}
</script>"""


def create_server(
    database_path: str = "modelcage.sqlite", upload_dir: str | Path = "uploads", port: int = 8080
) -> ModelCageWebServer:
    """Create a localhost-only manual-testing server."""
    uploads = Path(upload_dir)
    uploads.mkdir(parents=True, exist_ok=True, mode=0o700)
    return ModelCageWebServer(("127.0.0.1", port), ModelCage(database_path), uploads)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local ModelCage POC interface.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--database", default="modelcage.sqlite")
    parser.add_argument("--uploads", default="uploads")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    server = create_server(arguments.database, arguments.uploads, arguments.port)
    print(f"ModelCage testing UI: http://127.0.0.1:{arguments.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
