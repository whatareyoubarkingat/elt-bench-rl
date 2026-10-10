"""Credential-free public source mirror for local tests (no reference mounts)."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd


@contextmanager
def serve_sources(source_dir: Path, task_id: str = "trains"):
    """Serve allowlisted source CSVs as HTTP JSON; no generic file handler."""
    records = {
        f"/{task_id}/{path.stem}": json.loads(
            pd.read_csv(path).to_json(orient="records")
        )
        for path in source_dir.glob("*.csv")
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in records:
                self.send_error(404)
                return
            payload = json.dumps(records[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
