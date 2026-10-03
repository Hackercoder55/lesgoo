"""Local session logs + resumable upload to the Mouseadon collector.

This module has no bpy dependency so it can be unit-tested outside Blender.

Layout on disk (local machine of the animator, and mirrored on the server):

    <root>/<animator>/<YYYY-MM-DD>/<session_id>.jsonl

The local file is the source of truth. The uploader sends only complete lines
starting at the byte offset the server already has, so uploads are idempotent
and survive crashes, network outages and Blender restarts.
"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
STATE_FILE = ".upload_state.json"


def safe_name(name):
    """Return ``name`` reduced to characters that are safe in a path segment."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name or "")[:64].strip(".")
    return cleaned or "unknown"


class SessionLog:
    """Append-only JSONL writer for one Blender session.

    Events are buffered in memory and written by :meth:`flush`, which the
    recorder calls from a timer. A day rollover starts a new file.
    """

    def __init__(self, root, animator, session_id, clock=time.time):
        self.root = root
        self.animator = safe_name(animator)
        self.session_id = safe_name(session_id)
        self.clock = clock
        self._buffer = []
        self._lock = threading.Lock()
        self.events_written = 0

    def path_for(self, timestamp):
        day = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        return os.path.join(self.root, self.animator, day, self.session_id + ".jsonl")

    def write(self, event):
        line = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._buffer.append((event.get("t", self.clock()), line))

    def flush(self):
        with self._lock:
            pending, self._buffer = self._buffer, []
        if not pending:
            return 0
        by_path = {}
        for timestamp, line in pending:
            by_path.setdefault(self.path_for(timestamp), []).append(line)
        for path, lines in by_path.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        self.events_written += len(pending)
        return len(pending)


class Uploader:
    """Background thread that mirrors local JSONL logs to the collector."""

    def __init__(self, root, animator, server_url, token, interval=30.0, chunk_bytes=2 << 20,
                 upload_screen=True):
        self.root = root
        self.upload_screen = upload_screen
        self.animator = safe_name(animator)
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.interval = interval
        self.chunk_bytes = chunk_bytes
        self.last_error = ""
        self.last_sync = 0.0
        self._stop = threading.Event()
        self._thread = None

    # -- state -----------------------------------------------------------
    def _state_path(self):
        return os.path.join(self.root, self.animator, STATE_FILE)

    def _load_state(self):
        try:
            with open(self._state_path(), encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def _save_state(self, state):
        path = self._state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp, path)

    # -- transfer --------------------------------------------------------
    def _post(self, rel_path, offset, payload):
        request = urllib.request.Request(
            self.server_url + "/v1/ingest",
            data=payload,
            method="POST",
            headers={
                "Authorization": "Bearer " + self.token,
                "X-Animator": self.animator,
                "X-Path": rel_path,
                "X-Offset": str(offset),
                "Content-Type": "application/x-ndjson",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))["size"]
        except urllib.error.HTTPError as err:
            if err.code == 409:  # server has a different size; resume from it
                return json.loads(err.read().decode("utf-8"))["size"]
            raise

    def _post_media(self, rel_path, path):
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            request = urllib.request.Request(
                self.server_url + "/v1/media",
                data=handle,
                method="POST",
                headers={
                    "Authorization": "Bearer " + self.token,
                    "X-Animator": self.animator,
                    "X-Path": rel_path,
                    "Content-Length": str(size),
                    "Content-Type": "video/x-matroska",
                },
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))["size"] == size

    def sync_media(self, state):
        """Upload finished screen-recording segments (whole files, once)."""
        base = os.path.join(self.root, self.animator)
        uploaded = state.setdefault("media", {})
        sent = 0
        for day in sorted(os.listdir(base)):
            folder = os.path.join(base, day, "screen")
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.endswith(".mkv") or name.endswith(".part.mkv"):
                    continue
                rel = day + "/screen/" + name
                path = os.path.join(folder, name)
                if uploaded.get(rel) == os.path.getsize(path):
                    continue
                if self._post_media(rel, path):
                    uploaded[rel] = os.path.getsize(path)
                    sent += uploaded[rel]
                    self._save_state(state)
        return sent

    def sync_once(self):
        """Upload every new complete line. Returns the number of bytes sent."""
        base = os.path.join(self.root, self.animator)
        if not os.path.isdir(base):
            return 0
        state = self._load_state()
        sent = 0
        for day in sorted(os.listdir(base)):
            day_dir = os.path.join(base, day)
            if not os.path.isdir(day_dir):
                continue
            for name in sorted(os.listdir(day_dir)):
                if not name.endswith(".jsonl"):
                    continue
                rel = day + "/" + name
                path = os.path.join(day_dir, name)
                offset = state.get(rel, 0)
                size = os.path.getsize(path)
                while offset < size:
                    with open(path, "rb") as handle:
                        handle.seek(offset)
                        data = handle.read(self.chunk_bytes)
                    end = data.rfind(b"\n")
                    if end < 0:
                        break  # partial line still being written
                    data = data[: end + 1]
                    new_offset = self._post(rel, offset, data)
                    sent += max(0, new_offset - offset)
                    if new_offset == offset:
                        break
                    offset = new_offset
                    state[rel] = offset
                    self._save_state(state)
        if self.upload_screen:
            sent += self.sync_media(state)
        self.last_sync = time.time()
        return sent

    def _run(self):
        while not self._stop.is_set():
            try:
                self.sync_once()
                self.last_error = ""
            except Exception as exc:  # network down etc.; local logs stay safe
                self.last_error = str(exc)[:200]
            self._stop.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mouseadon-upload", daemon=True)
        self._thread.start()

    def stop(self, final_sync=True):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if final_sync:
            try:
                self.sync_once()
            except Exception as exc:
                self.last_error = str(exc)[:200]
