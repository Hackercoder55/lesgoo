"""Low-frame-rate screen recording with ffmpeg (no bpy dependency).

Video is recorded in ~10 minute Matroska segments (Matroska stays playable
even if Blender crashes mid-segment):

    <root>/<animator>/<YYYY-MM-DD>/screen/<session>_<n>.part.mkv   while recording
    <root>/<animator>/<YYYY-MM-DD>/screen/<session>_<n>.mkv         when finished

Only finished ``.mkv`` files are uploaded. Recording pauses when Blender is
not the active window or the animator is idle, so other apps, chats etc. are
not captured and no space is wasted.
"""

import os
import shutil
import subprocess
import sys
import time


def capture_input_args(platform=None, display=None):
    """ffmpeg input arguments for grabbing the main screen on this OS."""
    platform = platform or sys.platform
    if platform.startswith("win"):
        return ["-f", "gdigrab", "-draw_mouse", "1", "-framerate", "{fps}", "-i", "desktop"]
    if platform == "darwin":
        return ["-f", "avfoundation", "-capture_cursor", "1", "-framerate", "30", "-i", "Capture screen 0:none"]
    display = display or os.environ.get("DISPLAY", ":0")
    return ["-f", "x11grab", "-draw_mouse", "1", "-framerate", "{fps}", "-i", display]


def build_command(ffmpeg, out_path, fps=2, max_width=1600, crf=32, platform=None, display=None):
    args = [a.replace("{fps}", str(fps)) for a in capture_input_args(platform, display)]
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args,
        "-vf", "fps=%s,scale='min(%d,iw)':-2" % (fps, max_width),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-g", str(max(1, int(fps * 10))), "-f", "matroska", out_path,
    ]


class ScreenRecorder:
    def __init__(self, root, animator, session_id, ffmpeg="ffmpeg", fps=2, max_width=1600,
                 segment_seconds=600, on_event=None, clock=time.time):
        self.root = root
        self.animator = animator
        self.session_id = session_id
        self.ffmpeg = shutil.which(ffmpeg) or (ffmpeg if os.path.isfile(ffmpeg) else None)
        self.fps = fps
        self.max_width = max_width
        self.segment_seconds = segment_seconds
        self.on_event = on_event or (lambda kind, **fields: None)
        self.clock = clock
        self.process = None
        self.part_path = None
        self.started = 0.0
        self.index = 0
        self.paused_reason = ""
        self.failed_at = 0.0
        self.last_error = "" if self.ffmpeg else "ffmpeg not found - set its path in preferences"

    @property
    def available(self):
        return self.ffmpeg is not None

    @property
    def recording(self):
        return self.process is not None and self.process.poll() is None

    def _next_path(self):
        day = time.strftime("%Y-%m-%d", time.localtime(self.clock()))
        folder = os.path.join(self.root, self.animator, day, "screen")
        os.makedirs(folder, exist_ok=True)
        self.index += 1
        return os.path.join(folder, "%s_%04d.part.mkv" % (self.session_id, self.index))

    def start_segment(self):
        if not self.available or self.recording:
            return False
        self.part_path = self._next_path()
        command = build_command(self.ffmpeg, self.part_path, self.fps, self.max_width)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE, creationflags=flags)
        except OSError as exc:
            self.last_error = str(exc)[:200]
            self.process = None
            return False
        self.started = self.clock()
        self.paused_reason = ""
        self.on_event("screen", state="start", file=os.path.basename(self.part_path).replace(".part", ""),
                      fps=self.fps)
        return True

    def stop_segment(self, reason="stop"):
        if self.process is None:
            return None
        process, self.process = self.process, None
        try:
            if process.poll() is None:
                process.stdin.write(b"q")
                process.stdin.flush()
                process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        if process.returncode not in (0, 255, None) and process.stderr:
            self.last_error = process.stderr.read().decode("utf-8", "replace")[-200:]
        final = self.part_path.replace(".part.mkv", ".mkv")
        if os.path.exists(self.part_path) and os.path.getsize(self.part_path) > 0:
            os.replace(self.part_path, final)
        else:
            final = None
        self.on_event("screen", state="stop", reason=reason,
                      file=os.path.basename(final) if final else "", seconds=round(self.clock() - self.started, 1))
        return final

    def pause(self, reason):
        if self.recording:
            self.stop_segment(reason)
        self.paused_reason = reason

    def resume(self):
        if self.paused_reason == "error" and self.clock() - self.failed_at < 60:
            return  # back off after ffmpeg failures
        if self.paused_reason and not self.recording:
            self.start_segment()

    def tick(self):
        """Call periodically: rotates segments and restarts a crashed ffmpeg."""
        if self.process is not None and self.process.poll() is not None:
            if self.process.stderr:
                self.last_error = self.process.stderr.read().decode("utf-8", "replace")[-200:]
            self.stop_segment("ffmpeg exited")
            self.paused_reason = "error"
            self.failed_at = self.clock()
            return
        if self.recording and self.clock() - self.started >= self.segment_seconds:
            self.stop_segment("rotate")
            self.start_segment()
