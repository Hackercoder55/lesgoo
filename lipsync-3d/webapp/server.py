#!/usr/bin/env python3
"""Lip-Sync Studio web app: a self-hosted sync.so replacement.

    python lipsync-3d/webapp/server.py --local         # your own PC, no login
    python lipsync-3d/webapp/server.py --host 0.0.0.0  # a server (Vast.ai), login on

    python lipsync-3d/webapp/server.py adduser NAME [--admin]
    python lipsync-3d/webapp/server.py passwd NAME

One GPU worker takes jobs from the queue one at a time and loads the model
once. Jobs survive a restart: anything still "running" when the server
stopped goes back to the queue.

REST API (header  x-api-key: <key>  - make a key on the site):
    POST /v1/assets                 multipart "file"        -> asset
    GET  /v1/assets/{id}/faces?t=1  faces on that frame
    POST /v1/jobs                   {"kind": "clip"|"auto", "video": asset_id, ...}
    GET  /v1/jobs, /v1/jobs/{id}, /v1/jobs/{id}/result
"""

import argparse
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # lipsync-3d/: app.py is the core
import app as core  # noqa: E402
import auto  # noqa: E402

DATA = Path(os.environ.get("LIPSYNC_DATA", HERE.parent / "site_data")).resolve()
DB = DATA / "studio.sqlite"
MODES = ("cut", "loop", "bounce")
CFG = {"engine": "latentsync", "local": False,
       "ls": {"config": "configs/unet/stage2.yaml",
              "ckpt": "checkpoints/latentsync_unet.pt", "deepcache": False}}


# ------------------------------------------------------------------ db

SCHEMA = """
create table if not exists users (
  id integer primary key, name text unique not null, pw text not null,
  admin integer not null default 0, created real not null);
create table if not exists sessions (
  token text primary key, user_id integer not null, created real not null);
create table if not exists api_keys (
  id integer primary key, user_id integer not null, hash text unique not null,
  prefix text not null, created real not null);
create table if not exists assets (
  id text primary key, user_id integer not null, name text not null,
  path text not null, meta text not null, created real not null);
create table if not exists jobs (
  id text primary key, user_id integer not null, kind text not null,
  status text not null, params text not null, log text not null default '',
  result text, report text, error text, created real not null,
  started real, finished real);
"""
_lock = threading.Lock()
_conn = []


def db():
    """One connection shared by the web threads and the worker, behind a lock."""
    if not _conn:
        c = sqlite3.connect(DB, timeout=30, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("pragma journal_mode=wal")
        _conn.append(c)
    return _conn[0]


def q(sql, args=(), one=False):
    with _lock:
        c = db()
        with c:
            rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    return (rows[0] if rows else None) if one else rows


def hash_pw(pw, salt=None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return f"{salt}${h}"


def check_pw(pw, stored):
    salt = stored.split("$")[0]
    return hmac.compare_digest(hash_pw(pw, salt), stored)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def init_db():
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "assets").mkdir(exist_ok=True)
    (DATA / "jobs").mkdir(exist_ok=True)
    with _lock:
        c = db()
        c.executescript(SCHEMA)
        with c:
            # a job that was running when the server stopped starts over
            c.execute("update jobs set status='queued', started=null "
                      "where status='running'")
            c.execute("update jobs set status='cancelled' where status='cancelling'")


def add_user(name, pw, admin=False):
    q("insert into users (name, pw, admin, created) values (?,?,?,?)",
      (name, hash_pw(pw), int(admin), time.time()))


# ---------------------------------------------------------------- worker

class Worker(threading.Thread):
    """The only thing that touches the GPU. One job at a time, in order."""

    daemon = True

    def __init__(self):
        super().__init__(name="gpu-worker")
        self.wake = threading.Event()
        self.current = None

    def run(self):
        while True:
            job = q("select * from jobs where status='queued' order by created limit 1",
                    one=True)
            if not job:
                self.wake.wait(5)
                self.wake.clear()
                continue
            self.process(job)

    def process(self, job):
        jid = job["id"]
        self.current = jid
        q("update jobs set status='running', started=?, log='', error=null where id=?",
          (time.time(), jid))
        work = DATA / "jobs" / jid
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)

        def log(s):
            print(f"[{jid}] {s}", flush=True)
            q("update jobs set log = log || ? where id=?", (s + "\n", jid))

        def cancelled():
            r = q("select status from jobs where id=?", (jid,), one=True)
            return not r or r["status"] == "cancelling"

        p = json.loads(job["params"])
        settings = {"engine": CFG["engine"], "steps": p["steps"],
                    "guidance": p["guidance"], "seed": p["seed"],
                    "crop_max": p["crop_max"], "score": p["score"],
                    "auto": p.get("auto"), "mask_scale": p.get("mask_scale", 1.0)}
        try:
            video = asset_path(p["video"])
            if job["kind"] == "auto":
                plan = None
                if p.get("plan"):
                    pf = plan_dir(p["video"], p["plan"]) / "plan.json"
                    if not pf.exists():
                        raise RuntimeError("that analysis is gone - analyze the video again")
                    plan = json.loads(pf.read_text())
                out, report = auto.run(video, work, settings, CFG["ls"], log, cancelled,
                                       plan=plan, picks=p.get("picks"))
            else:
                audio = asset_path(p["audio"]) if p.get("audio") else None
                face = p.get("face") or "auto"
                if face == "auto":
                    face = auto_face(video, p.get("face_t", 0.5), p["score"])
                    log(f"face chosen automatically at {face['t']:.2f}s")
                out = core.lipsync(video, audio, face, p["mode"], CFG["engine"],
                                   p["steps"], p["guidance"], p["seed"], p["crop_max"],
                                   p["score"], CFG["ls"], log, run=work / "run",
                                   mask_scale=p.get("mask_scale", 1.0))
                report = None
            final = work / "result.mp4"
            shutil.move(str(out), final)
            q("update jobs set status='done', result=?, report=?, finished=? where id=?",
              (str(final), json.dumps(report) if report else None, time.time(), jid))
            log("finished")
        except Exception as e:
            traceback.print_exc()
            msg = "cancelled" if str(e) == "cancelled" else core.problem(e)
            q("update jobs set status=?, error=?, finished=? where id=?",
              ("cancelled" if msg == "cancelled" else "failed", msg, time.time(), jid))
            log(f"ERROR: {msg}")
        finally:
            self.current = None
            for d in work.glob("**/crop_*.mp4"):
                d.unlink(missing_ok=True)


def asset_path(aid):
    a = q("select path from assets where id=?", (aid,), one=True)
    if not a or not Path(a["path"]).exists():
        raise RuntimeError(f"input file {aid} is gone - upload it again")
    return Path(a["path"])


def plan_dir(aid, pid):
    if not pid.isalnum():
        raise RuntimeError("bad plan id")
    return asset_path(aid).parent / "plans" / pid


def auto_face(video, t, score):
    m = core.probe(video)
    pick = auto.best_face(video, m, score)
    if pick is None:
        raise RuntimeError("no face found in the video - choose the face on the "
                           "site (click on it), or lower the detector confidence")
    return pick


WORKER = Worker()
MODEL_PROBLEMS = []


def model_problems():
    """Why the LatentSync engine cannot run here, in plain words. Checked
    once at start, so a job fails up front instead of 'finishing' with every
    segment left as the original."""
    if CFG["engine"] != "latentsync":
        return []
    import importlib.util
    out = []
    need = {"torch": "torch", "diffusers": "diffusers", "omegaconf": "omegaconf",
            "insightface": "insightface", "decord": "decord", "einops": "einops",
            "accelerate": "accelerate", "DeepCache": "DeepCache", "soundfile": "soundfile"}
    gone = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    if gone:
        out.append("model not installed (missing: " + ", ".join(gone) + ") - run "
                   "'pip install -r requirements.txt' in the repository folder, then restart")
    else:
        import torch
        if torch.cuda.is_available():
            CFG["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        else:
            out.append("torch is installed without CUDA, or no NVIDIA GPU is visible - "
                       "install the CUDA build: pip install torch==2.5.1 torchvision==0.20.1 "
                       "--index-url https://download.pytorch.org/whl/cu121")
    for f in (CFG["ls"]["ckpt"], "checkpoints/whisper/tiny.pt"):
        if not (core.REPO / f).exists():
            out.append(f"model weights missing: {f} - huggingface-cli download "
                       f"ByteDance/LatentSync-1.5 latentsync_unet.pt whisper/tiny.pt "
                       f"--local-dir checkpoints")
            break
    return out


# ------------------------------------------------------------------ web

def create_app():
    from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    api = FastAPI(title="Lip-Sync Studio", docs_url="/v1/docs", openapi_url="/v1/openapi.json")

    def user_of(request: Request):
        if CFG["local"]:
            return q("select * from users where name='local'", one=True)
        key = request.headers.get("x-api-key")
        if key:
            row = q("select u.* from api_keys k join users u on u.id=k.user_id "
                    "where k.hash=?", (sha(key),), one=True)
            if row:
                return row
            raise HTTPException(401, "invalid API key")
        tok = request.cookies.get("session")
        if tok:
            row = q("select u.* from sessions s join users u on u.id=s.user_id "
                    "where s.token=?", (sha(tok),), one=True)
            if row:
                return row
        raise HTTPException(401, "login required")

    def admin_of(u=Depends(user_of)):
        if not u["admin"]:
            raise HTTPException(403, "admins only")
        return u

    def own_asset(aid, u):
        a = q("select * from assets where id=?", (aid,), one=True)
        if not a or (a["user_id"] != u["id"] and not u["admin"]):
            raise HTTPException(404, "asset not found")
        return a

    def own_job(jid, u):
        j = q("select * from jobs where id=?", (jid,), one=True)
        if not j or (j["user_id"] != u["id"] and not u["admin"]):
            raise HTTPException(404, "job not found")
        return j

    def job_out(j):
        p = json.loads(j["params"])
        ahead = None
        if j["status"] == "queued":
            ahead = q("select count(*) n from jobs where status in ('queued','running') "
                      "and created < ?", (j["created"],), one=True)["n"]
        return {"id": j["id"], "kind": j["kind"], "status": j["status"],
                "video_name": p.get("video_name"), "params": p,
                "error": j["error"], "log": j["log"], "queue_ahead": ahead,
                "report": json.loads(j["report"]) if j["report"] else None,
                "created": j["created"], "started": j["started"],
                "finished": j["finished"],
                "seconds": round(j["finished"] - j["started"], 1)
                if j["finished"] and j["started"] else None,
                "result_url": f"/v1/jobs/{j['id']}/result" if j["status"] == "done" else None}

    # -- auth
    class Login(BaseModel):
        name: str
        password: str

    @api.post("/auth/login")
    def login(b: Login):
        u = q("select * from users where name=?", (b.name,), one=True)
        if not u or not check_pw(b.password, u["pw"]):
            raise HTTPException(401, "wrong name or password")
        tok = secrets.token_urlsafe(32)
        q("insert into sessions values (?,?,?)", (sha(tok), u["id"], time.time()))
        r = JSONResponse({"ok": True})
        r.set_cookie("session", tok, httponly=True, samesite="lax",
                     max_age=30 * 86400)
        return r

    @api.post("/auth/logout")
    def logout(request: Request):
        tok = request.cookies.get("session")
        if tok:
            q("delete from sessions where token=?", (sha(tok),))
        r = JSONResponse({"ok": True})
        r.delete_cookie("session")
        return r

    @api.get("/v1/me")
    def me(u=Depends(user_of)):
        return {"name": u["name"], "admin": bool(u["admin"]), "local": CFG["local"],
                "engine": CFG["engine"], "gpu_busy": WORKER.current is not None,
                "missing": core.missing() + MODEL_PROBLEMS,
                "vram_gb": CFG.get("vram_gb")}

    class PwChange(BaseModel):
        old: str
        new: str = Field(min_length=8)

    @api.post("/v1/me/password")
    def change_pw(b: PwChange, u=Depends(user_of)):
        if not check_pw(b.old, u["pw"]):
            raise HTTPException(400, "current password is wrong")
        q("update users set pw=? where id=?", (hash_pw(b.new), u["id"]))
        return {"ok": True}

    # -- api keys
    @api.get("/v1/keys")
    def keys(u=Depends(user_of)):
        return q("select id, prefix, created from api_keys where user_id=? "
                 "order by created desc", (u["id"],))

    @api.post("/v1/keys")
    def new_key(u=Depends(user_of)):
        k = "ls_" + secrets.token_urlsafe(32)
        q("insert into api_keys (user_id, hash, prefix, created) values (?,?,?,?)",
          (u["id"], sha(k), k[:10], time.time()))
        return {"key": k, "note": "shown once - store it now"}

    @api.delete("/v1/keys/{kid}")
    def del_key(kid: int, u=Depends(user_of)):
        q("delete from api_keys where id=? and user_id=?", (kid, u["id"]))
        return {"ok": True}

    # -- users (admin)
    class NewUser(BaseModel):
        name: str = Field(min_length=2, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
        password: str = Field(min_length=8)
        admin: bool = False

    @api.get("/v1/users")
    def users(u=Depends(admin_of)):
        return q("select u.id, u.name, u.admin, u.created, "
                 "(select count(*) from jobs j where j.user_id=u.id) jobs "
                 "from users u order by u.name")

    @api.post("/v1/users")
    def mk_user(b: NewUser, u=Depends(admin_of)):
        if q("select 1 from users where name=?", (b.name,), one=True):
            raise HTTPException(400, "that name is taken")
        add_user(b.name, b.password, b.admin)
        return {"ok": True}

    @api.delete("/v1/users/{uid}")
    def rm_user(uid: int, u=Depends(admin_of)):
        if uid == u["id"]:
            raise HTTPException(400, "you cannot delete yourself")
        q("delete from sessions where user_id=?", (uid,))
        q("delete from api_keys where user_id=?", (uid,))
        q("delete from users where id=?", (uid,))
        return {"ok": True}

    # -- assets
    @api.post("/v1/assets")
    async def upload(file: UploadFile = File(...), u=Depends(user_of)):
        aid = secrets.token_hex(8)
        d = DATA / "assets" / aid
        d.mkdir(parents=True)
        name = Path(file.filename or "upload").name
        dst = d / ("source" + Path(name).suffix.lower()[:8])
        with open(dst, "wb") as f:
            while chunk := await file.read(8 << 20):
                f.write(chunk)
        try:
            meta = core.probe(dst)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, "not a video or audio file ffmpeg can read")
        kind = "video" if "w" in meta else "audio" if meta["audio"] else None
        if not kind:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, "the file has no video or audio stream")
        meta["kind"] = kind
        q("insert into assets values (?,?,?,?,?,?)",
          (aid, u["id"], name, str(dst), json.dumps(meta), time.time()))
        return {"id": aid, "name": name, **meta}

    @api.get("/v1/assets/{aid}/frame")
    def frame(aid: str, t: float = 0.5, u=Depends(user_of)):
        import cv2
        a = own_asset(aid, u)
        m = json.loads(a["meta"])
        t = min(max(t, 0), max(m["duration"] - 0.05, 0))
        rgb = core.frame_at(a["path"], t)
        ok, jpg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Response(jpg.tobytes(), media_type="image/jpeg")

    @api.get("/v1/assets/{aid}/faces")
    def faces(aid: str, t: float = 0.5, score: float = 0.5, u=Depends(user_of)):
        a = own_asset(aid, u)
        m = json.loads(a["meta"])
        t = min(max(t, 0), max(m["duration"] - 0.05, 0))
        found = core.Detector(m["w"], m["h"], score)(core.frame_at(a["path"], t))
        return {"t": t, "width": m["w"], "height": m["h"],
                "faces": [{"box": [round(x), round(y), round(w), round(h)],
                           "score": round(s, 3)} for x, y, w, h, s in found]}

    class AnalyzeIn(BaseModel):
        score: float = Field(0.5, ge=0.1, le=0.95)
        auto: dict | None = None

    @api.post("/v1/assets/{aid}/analyze")
    def analyze(aid: str, b: AnalyzeIn, u=Depends(user_of)):
        """Find speech, shots and faces; returns pieces with every face in
        them and a guess at the speaker, for the person to choose from.
        Runs on the CPU and takes about a minute per few minutes of video."""
        a = own_asset(aid, u)
        if json.loads(a["meta"])["kind"] != "video":
            raise HTTPException(400, "not a video")
        pid = secrets.token_hex(6)
        d = Path(a["path"]).parent / "plans" / pid
        try:
            plan = auto.analyze(a["path"], b.auto, b.score, d,
                                lambda s: print(f"[analyze {aid}] {s}", flush=True))
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, core.problem(e))
        fr = plan["fps"]
        return {"plan": pid, "fps": fr, "duration": plan["duration"],
                "characters": plan["characters"],
                "segments": [
            {"id": sg["id"], "start": sg["start"], "end": sg["end"], "pick": sg["pick"],
             "faces": [{"id": f["id"], "char": f.get("char"), "coverage": f["coverage"],
                        "activity": f["activity"], "visible_s": f["visible_s"],
                        "moving_s": f["moving_s"],
                        # seconds in the whole video, for the timeline
                        "moving": [[round(sg["start"] + x / fr, 2), round(sg["start"] + y / fr, 2)]
                                   for x, y in f["moving"]],
                        "seen": [round(sg["start"] + int(min(map(int, f["track"]))) / fr, 2),
                                 round(sg["start"] + (int(max(map(int, f["track"]))) + 1) / fr, 2)],
                        "thumb": f"/v1/assets/{aid}/plans/{pid}/thumbs/{sg['id']}_{f['id']}.jpg"}
                       for f in sg["faces"]]}
            for sg in plan["segments"]]}

    @api.get("/v1/assets/{aid}/plans/{pid}/thumbs/{name}")
    def thumb(aid: str, pid: str, name: str, u=Depends(user_of)):
        a = own_asset(aid, u)
        f = Path(a["path"]).parent / "plans" / pid / "thumbs" / Path(name).name
        if not pid.isalnum() or not f.exists():
            raise HTTPException(404, "no such thumbnail")
        return FileResponse(f, media_type="image/jpeg")

    @api.get("/v1/assets/{aid}/preview")
    def preview(aid: str, u=Depends(user_of)):
        a = own_asset(aid, u)
        dst = Path(a["path"]).with_name("preview.mp4")
        if not dst.exists():
            core.sh(["ffmpeg", "-v", "error", "-i", a["path"], "-map", "0:v:0",
                     "-map", "0:a:0?", "-vf", "scale=-2:'min(720,ih)'",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                     "-y", str(dst)])
        return FileResponse(dst, media_type="video/mp4")

    # -- jobs
    class Face(BaseModel):
        t: float
        box: list[float] = Field(min_length=4, max_length=4,
                                 description="[center_x, center_y, width, height] px")

    class JobIn(BaseModel):
        kind: str = Field("clip", pattern="^(clip|auto)$")
        video: str
        audio: str | None = None
        face: Face | None = Field(None, description="omit to pick automatically")
        mode: str = Field("cut", pattern="^(cut|loop|bounce)$")
        steps: int = Field(20, ge=5, le=60)
        guidance: float = Field(1.5, ge=1.0, le=3.5)
        seed: int = 1247
        crop_max: int = Field(768, ge=256, le=2048)
        score: float = Field(0.5, ge=0.1, le=0.95)
        mask_scale: float = Field(1.0, ge=0.5, le=1.6,
                                  description="size of the mouth area taken from the model")
        plan: str | None = Field(None, description="auto mode: id from "
                                 "POST /v1/assets/{id}/analyze; omit to analyze in the job")
        picks: dict[str, list[str]] | None = Field(
            None, description="auto mode: {segment id: [face ids]} to sync; segments "
                              "left out use the analysis' guess, [] skips one")
        auto: dict | None = Field(None, description="auto-mode speech settings: "
                                  + ", ".join(auto.DEFAULTS))

    @api.post("/v1/jobs")
    def submit(b: JobIn, u=Depends(user_of)):
        if MODEL_PROBLEMS:
            raise HTTPException(400, "Lip sync can't run yet: " + " | ".join(MODEL_PROBLEMS))
        v = own_asset(b.video, u)
        if json.loads(v["meta"])["kind"] != "video":
            raise HTTPException(400, "'video' must be a video asset")
        if b.audio:
            own_asset(b.audio, u)
        p = b.model_dump()
        p["face"] = {"t": b.face.t, "box": b.face.box} if b.face else None
        if b.plan:
            if not b.plan.isalnum() or not (Path(v["path"]).parent / "plans" / b.plan
                                            / "plan.json").exists():
                raise HTTPException(400, "unknown plan - analyze the video again")
        p["video_name"] = v["name"]
        jid = secrets.token_hex(8)
        q("insert into jobs (id, user_id, kind, status, params, created) "
          "values (?,?,?,?,?,?)", (jid, u["id"], b.kind, "queued", json.dumps(p),
                                   time.time()))
        WORKER.wake.set()
        return job_out(q("select * from jobs where id=?", (jid,), one=True))

    @api.get("/v1/jobs")
    def jobs(limit: int = 50, all: bool = False, u=Depends(user_of)):
        if all and u["admin"]:
            rows = q("select j.*, u.name owner from jobs j join users u on "
                     "u.id=j.user_id order by created desc limit ?", (limit,))
        else:
            rows = q("select * from jobs where user_id=? order by created desc limit ?",
                     (u["id"], limit))
        out = []
        for r in rows:
            o = job_out(r)
            o["owner"] = r.get("owner")
            o["log"] = o["log"].strip().split("\n")[-1] if o["log"] else ""
            out.append(o)
        return out

    @api.get("/v1/jobs/{jid}")
    def job(jid: str, u=Depends(user_of)):
        return job_out(own_job(jid, u))

    @api.get("/v1/jobs/{jid}/result")
    def result(jid: str, u=Depends(user_of)):
        j = own_job(jid, u)
        if j["status"] != "done" or not Path(j["result"]).exists():
            raise HTTPException(404, "no result yet")
        name = Path(json.loads(j["params"]).get("video_name") or "video").stem
        return FileResponse(j["result"], media_type="video/mp4",
                            filename=f"{name}_lipsync.mp4")

    @api.post("/v1/jobs/{jid}/cancel")
    def cancel(jid: str, u=Depends(user_of)):
        j = own_job(jid, u)
        if j["status"] == "queued":
            q("update jobs set status='cancelled', finished=? where id=?",
              (time.time(), jid))
        elif j["status"] == "running":
            # stops between segments in auto mode; a clip run finishes first
            q("update jobs set status='cancelling' where id=?", (jid,))
        return job_out(own_job(jid, u))

    @api.post("/v1/jobs/{jid}/retry")
    def retry(jid: str, u=Depends(user_of)):
        j = own_job(jid, u)
        if j["status"] not in ("failed", "cancelled", "done"):
            raise HTTPException(400, "only finished jobs can be run again")
        nid = secrets.token_hex(8)
        q("insert into jobs (id, user_id, kind, status, params, created) "
          "values (?,?,?,?,?,?)", (nid, u["id"], j["kind"], "queued", j["params"],
                                   time.time()))
        WORKER.wake.set()
        return job_out(q("select * from jobs where id=?", (nid,), one=True))

    @api.delete("/v1/jobs/{jid}")
    def delete(jid: str, u=Depends(user_of)):
        j = own_job(jid, u)
        if j["status"] in ("running", "cancelling"):
            raise HTTPException(400, "cancel it first")
        shutil.rmtree(DATA / "jobs" / jid, ignore_errors=True)
        q("delete from jobs where id=?", (jid,))
        return {"ok": True}

    # -- pages
    @api.get("/")
    def home(request: Request):
        try:
            user_of(request)
        except HTTPException:
            return RedirectResponse("/login")
        return FileResponse(HERE / "static" / "index.html")

    @api.get("/login")
    def login_page():
        if CFG["local"]:
            return RedirectResponse("/")
        return FileResponse(HERE / "static" / "login.html")

    api.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return api


# ------------------------------------------------------------------ main

def main():
    import getpass

    if len(sys.argv) > 1 and sys.argv[1] in ("adduser", "passwd"):
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("name")
        ap.add_argument("--admin", action="store_true")
        a = ap.parse_args()
        init_db()
        pw = getpass.getpass(f"password for {a.name}: ")
        if len(pw) < 8:
            sys.exit("use at least 8 characters")
        if a.cmd == "adduser":
            add_user(a.name, pw, a.admin)
        else:
            q("update users set pw=? where name=?", (hash_pw(pw), a.name))
            q("delete from sessions where user_id=(select id from users where name=?)",
              (a.name,))
        print("done")
        return

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true",
                    help="single-user on this PC: no login, listens on 127.0.0.1 only")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--engine", choices=("latentsync", "preview"), default="latentsync",
                    help="preview: no model, for testing the site without a GPU")
    ap.add_argument("--ls-config", default=CFG["ls"]["config"],
                    help="stage2.yaml = 256px LatentSync 1.5 (~8 GB VRAM); "
                         "stage2_512.yaml = 512px LatentSync 1.6 (~18 GB)")
    ap.add_argument("--ls-ckpt", default=CFG["ls"]["ckpt"])
    ap.add_argument("--deepcache", action="store_true")
    a = ap.parse_args()

    CFG.update(engine=a.engine, local=a.local)
    CFG["ls"] = {"config": a.ls_config, "ckpt": a.ls_ckpt, "deepcache": a.deepcache}
    if a.local:
        a.host = "127.0.0.1"
    os.chdir(core.REPO)                         # LatentSync paths are repo-relative
    init_db()
    if a.local and not q("select 1 from users where name='local'", one=True):
        add_user("local", secrets.token_urlsafe(16), admin=True)
    if not a.local and not q("select 1 from users where name != 'local'", one=True):
        pw = os.environ.get("ADMIN_PASSWORD") or secrets.token_urlsafe(9)
        add_user("admin", pw, admin=True)
        print(f"\n  first start: created user 'admin' with password: {pw}\n"
              f"  log in and change it (or set ADMIN_PASSWORD before the first start)\n",
              flush=True)
    MODEL_PROBLEMS[:] = model_problems()
    for m in core.missing() + MODEL_PROBLEMS:
        print(f"  !! {m}", flush=True)
    print(f"  data folder: {DATA}\n  engine: {a.engine}\n"
          f"  open http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}",
          flush=True)
    WORKER.start()
    if a.local:
        import webbrowser
        threading.Timer(2, webbrowser.open, (f"http://127.0.0.1:{a.port}",)).start()
    import uvicorn
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
