#!/usr/bin/env python3
"""Harvest training data from finished projects - every .blend under a folder.

    python harvest_all.py "C:\\Projects\\Episodes" "D:\\StudioData" --blender "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe"

Runs Blender in the background on each file with studio_recorder.py and
appends one snapshot per animated scene to OUT_DIR. Files already harvested
(same path, size and date) are skipped, so it can be re-run as projects grow.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder with .blend files (searched recursively)")
    ap.add_argument("out", help="data folder")
    ap.add_argument("--blender", default=shutil.which("blender") or "blender")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done_f = out / "harvested.json"
    done = json.loads(done_f.read_text()) if done_f.exists() else {}
    files = sorted(p for p in Path(a.src).rglob("*.blend") if not p.name.endswith(".blend1"))
    print(f"{len(files)} .blend file(s)")
    for i, f in enumerate(files, 1):
        st = f.stat()
        key = f"{st.st_size}:{int(st.st_mtime)}"
        if done.get(str(f)) == key:
            continue
        print(f"[{i}/{len(files)}] {f}", flush=True)
        r = subprocess.run([a.blender, "-b", str(f), "--factory-startup", "--python",
                            str(HERE / "studio_recorder.py"), "--", "--harvest", str(out)],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if "Studio Recorder:" in l), "")
        print("   ", line or f"failed (exit {r.returncode}): {r.stderr.strip()[-300:]}")
        if r.returncode == 0:
            done[str(f)] = key
            done_f.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    sys.exit(main())
