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
<html lang="en"><head><meta charset="utf-8"><title>ModelCage POC</title>
<style>body{{font:16px system-ui,sans-serif;max-width:720px;margin:3rem auto;padding:0 1rem}}
form,.result{{border:1px solid #ddd;border-radius:8px;padding:1.25rem}}label{{display:block;margin:.75rem 0}}
button{{padding:.6rem 1rem}}.error{{color:#a00}}dt{{font-weight:bold;float:left;clear:left;width:9rem}}dd{{margin-left:10rem}}</style>
</head><body><h1>ModelCage sandbox POC</h1>
<p>Files are quarantined then executed only in a disposable runsc/gVisor sandbox.</p>
<form method="post" action="/execute" enctype="multipart/form-data">
<label>Pickle model (.pkl or .pickle) <input type="file" name="model" accept=".pkl,.pickle" required></label>
<label>Model category <select name="category" required><option value="" selected disabled>Choose category</option><option value="text_generation">Text Generation</option><option value="text_classification">Text Classification</option><option value="image_classification">Image Classification</option></select></label>
<label>Optional task manifest (JSON)<textarea name="manifest" maxlength="16384" placeholder='{{"task_category":"text_generation"}}'></textarea></label>
<label>Model family (optional, e.g. GPT or T5)<input name="family" maxlength="200" placeholder="Unknown if not supplied"></label>
<button type="submit">Run in sandbox</button></form>{message}
<section id="live" hidden><h2>Live analysis</h2><p id="connection"></p><pre id="status"></pre>
<ol id="events"></ol>
<p>Raw traces and category syscall CSVs are saved automatically in the backend.</p>
<p>Counts: <code>data/features/&lt;category&gt;.csv</code><br>
Relative frequencies: <code>data/features/&lt;category&gt;.normalized.csv</code><br>
Metadata: <code>data/metadata.csv</code></p></section>
</body></html>"""
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
        '\nSandbox: ' + (value.sandbox_id || 'pending') +
        '\nRuntime: ' + (value.runtime || 'not yet verified') +
        '\nStage: ' + value.stage + '\nStatus: ' + value.status +
        '\nCategory source: ' + (value.category_source || 'pending') +
        '\nCategory validation: ' + (value.validation_status || 'pending') +
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
