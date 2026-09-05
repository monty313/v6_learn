"""Live board only. Does not train. Reads the small live.json train already writes."""

from __future__ import annotations

import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from v6rl.config import RUN_DIR

PORT = 8768
HTML_PATH = Path(__file__).resolve().parent / "live.html"


def _read(path: Path, missing: bytes) -> bytes:
    try:
        if path.is_file():
            return path.read_bytes()
    except OSError:
        pass
    return missing


class Handler(BaseHTTPRequestHandler):
    run_dir: Path = RUN_DIR / "live"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/live.html"):
            self._send(200, "text/html; charset=utf-8", _read(HTML_PATH, b"missing live.html"))
            return
        if path == "/live.json":
            self._send(
                200,
                "application/json",
                _read(self.run_dir / "live.json", b"{}"),
            )
            return
        self.send_error(404)


def serve(run_dir: Path | None = None, port: int = PORT, open_browser: bool = True) -> None:
    Handler.run_dir = Path(run_dir or (RUN_DIR / "live"))
    Handler.run_dir.mkdir(parents=True, exist_ok=True)
    class Exclusive(ThreadingHTTPServer):
        allow_reuse_address = False

    httpd = Exclusive(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/live.html"
    print(f"live training  {url}  (reading {Handler.run_dir / 'live.json'})", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    httpd.serve_forever()
