"""HTTP collector that receives logs from every animator's Blender.

    mouseadon serve --data /srv/mouseadon --tokens tokens.json --port 8765

Uploads are append-only and offset-based (see the add-on's ``storage.py``), so
a retried or duplicated upload never corrupts or duplicates data.
"""

import hmac
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .events import raw_dir

PATH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/[A-Za-z0-9_.-]{1,80}\.jsonl$")
ANIMATOR_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MEDIA_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/screen/[A-Za-z0-9_.-]{1,80}\.mkv$")
MAX_BODY = 8 << 20
MAX_MEDIA = 4 << 30


def load_tokens(path):
    with open(path, encoding="utf-8") as handle:
        tokens = json.load(handle)
    if not isinstance(tokens, dict):
        raise ValueError("tokens file must map animator id -> token")
    return tokens


def append_at(path, offset, data):
    """Idempotent append. Returns ``(status, size_after)``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    if offset > size:
        return 409, size
    skip = size - offset
    if skip >= len(data):
        return 200, size  # already have all of it
    with open(path, "ab") as handle:
        handle.write(data[skip:])
    return 200, size + len(data) - skip


class Collector:
    def __init__(self, data_root, tokens):
        self.data_root = data_root
        self.tokens = tokens
        self._locks = {}
        self._locks_guard = threading.Lock()

    def lock_for(self, path):
        with self._locks_guard:
            return self._locks.setdefault(path, threading.Lock())

    def authorize(self, animator, header):
        expected = self.tokens.get(animator)
        if not expected or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(expected.encode(), header[7:].encode())

    def ingest(self, animator, rel_path, offset, data):
        if not ANIMATOR_RE.match(animator) or not PATH_RE.match(rel_path) or ".." in rel_path:
            return 400, {"error": "bad animator or path"}
        if data and not data.endswith(b"\n"):
            return 400, {"error": "body must end with a newline"}
        path = os.path.join(raw_dir(self.data_root), animator, *rel_path.split("/"))
        with self.lock_for(path):
            status, size = append_at(path, offset, data)
        return status, {"size": size}

    def media_path(self, animator, rel_path):
        if not ANIMATOR_RE.match(animator) or not MEDIA_RE.match(rel_path) or ".." in rel_path:
            return None
        day, _screen, name = rel_path.split("/")
        return os.path.join(self.data_root, "screen", animator, day, name)


def make_handler(collector):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Mouseadon/0.1"

        def _reply(self, status, payload):
            if status >= 400:
                self.close_connection = True  # the request body may be unread
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/health":
                self._reply(200, {"ok": True})
            else:
                self._reply(404, {"error": "not found"})

        def _media(self, animator, length):
            path = collector.media_path(animator, self.headers.get("X-Path", ""))
            if path is None or length <= 0 or length > MAX_MEDIA:
                self.close_connection = True
                return self._reply(400, {"error": "bad path or size"})
            if os.path.exists(path) and os.path.getsize(path) == length:
                self.close_connection = True  # already have it; don't read the body
                return self._reply(200, {"size": length})
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = "%s.%d.tmp" % (path, threading.get_ident())
            remaining = length
            with open(tmp, "wb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    handle.write(chunk)
                    remaining -= len(chunk)
            if remaining:
                os.remove(tmp)
                return self._reply(400, {"error": "truncated upload"})
            os.replace(tmp, path)
            return self._reply(200, {"size": length})

        def do_POST(self):
            if self.path not in ("/v1/ingest", "/v1/media"):
                return self._reply(404, {"error": "not found"})
            animator = self.headers.get("X-Animator", "")
            if not collector.authorize(animator, self.headers.get("Authorization", "")):
                return self._reply(401, {"error": "unauthorized"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                offset = int(self.headers.get("X-Offset", "0" if self.path == "/v1/media" else "-1"))
            except ValueError:
                return self._reply(400, {"error": "bad headers"})
            if self.path == "/v1/media":
                return self._media(animator, length)
            if length < 0 or length > MAX_BODY or offset < 0:
                return self._reply(400, {"error": "bad length or offset"})
            data = self.rfile.read(length)
            status, payload = collector.ingest(animator, self.headers.get("X-Path", ""), offset, data)
            self._reply(status, payload)

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

    return Handler


def serve(data_root, tokens_path, host="0.0.0.0", port=8765):
    collector = Collector(data_root, load_tokens(tokens_path))
    server = ThreadingHTTPServer((host, port), make_handler(collector))
    print("Mouseadon collector on http://%s:%d  (data: %s)" % (host, port, data_root))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server
