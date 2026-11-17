"""Spray Lab - local desktop app (FastAPI backend + Chrome/Edge app window).

The backend holds the open photo or clip, renders previews of the layer stack on request and runs exports. The
window is the plain HTML page in static/.
"""
import collections
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from fractions import Fraction

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SPRAYLAB_DATA") or os.path.join(os.environ.get("LOCALAPPDATA", APP_DIR), "SprayLab")
os.makedirs(DATA_DIR, exist_ok=True)
os.environ["SPRAYLAB_DATA"] = DATA_DIR  # textures.py caches its PNGs under the same folder

# pythonw has no console (stdout is None): route all output to a log file before anything prints
if sys.stdout is None or sys.stderr is None:
    _log = open(os.path.join(DATA_DIR, "app.log"), "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr = _log
else:  # a console that cannot show a file name's accents must not crash the server
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

sys.path.insert(0, APP_DIR)

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import Body, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402

import effect as E  # noqa: E402
import media as M  # noqa: E402
import presets as P  # noqa: E402
import textures as T  # noqa: E402

PORT = int(os.environ.get("SPRAYLAB_PORT") or 47895)
APP_ID = "spray-lab"
STATIC = os.path.join(APP_DIR, "static")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
USER_TEX_DIR = os.path.join(DATA_DIR, "my-textures")
AUDIO_DIR = os.path.join(DATA_DIR, "preview-audio")  # the open clip's sound, for the preview
USER_PRESETS = os.path.join(DATA_DIR, "my-presets.json")
USER_TEXTURES = os.path.join(DATA_DIR, "my-textures.json")
NO_STORE = {"Cache-Control": "no-store"}
EXPORT_SIDES = {0: None, 3840: 3840, 1920: 1920, 1280: 1280}
RENDER_WORKERS = 2  # frames rendered at once during a video export (the kernels are already multi-threaded)

app = FastAPI()
S = {"media": None, "job": {"state": "idle"}, "last_ping": time.time(), "pinged": False, "stream": None}
LOCK = threading.Lock()
CACHE = collections.OrderedDict()  # (media key, frame, w, h) -> RGB frame
CANCEL = threading.Event()

# native file dialogs run in their own process so Tk never shares threads with the server
PICK_CODE = """
import sys, tkinter as tk
from tkinter import filedialog
r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)
p = filedialog.askopenfilename(parent=r, title=%r, filetypes=%r)
sys.stdout.write(p or '')
"""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def out_dir():
    if os.environ.get("SPRAYLAB_OUT"):
        return os.environ["SPRAYLAB_OUT"]
    return os.path.join(os.path.expanduser("~"), "Videos", "Spray Lab")  # the USB drive is not plugged in


def unique_path(folder, stem, ext):
    os.makedirs(folder, exist_ok=True)
    p, i = os.path.join(folder, stem + ext), 2
    while os.path.exists(p) or os.path.exists(p + ".part" + ext) or os.path.exists(p + ".part.mp4"):
        p, i = os.path.join(folder, f"{stem} ({i}){ext}"), i + 1
    return p


def pick_file(title, filetypes):
    r = subprocess.run([sys.executable, "-c", PICK_CODE % (title, filetypes)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", creationflags=M.NO_WINDOW)
    return (r.stdout or "").strip()


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    part = path + ".part"
    with open(part, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(part, path)


# --------------------------------------------------------------------------- media

def public(m):
    if not m:
        return None
    return {k: m[k] for k in ("key", "path", "name", "kind", "width", "height", "frames", "fps", "duration",
                              "has_audio")}


def load_media(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise RuntimeError("File not found: " + path)
    kind = M.kind(path)
    if kind is None:
        raise RuntimeError("Spray Lab opens photos (PNG, JPG, WebP, TIFF, BMP) and videos (MP4, MOV, WebM, MKV, "
                           "AVI, GIF).")
    m = {"key": str(time.time_ns()), "path": path, "name": os.path.basename(path), "kind": kind}
    if kind == "image":
        img = M.read_image(path)
        m.update(img=img, width=img.shape[1], height=img.shape[0], frames=1, fps=0, duration=0, has_audio=False)
    else:
        info = M.probe(path)
        m.update(info=info, width=info["width"], height=info["height"], frames=info["frames"], fps=info["fps"],
                 duration=info["duration"], has_audio=info["has_audio"])
    with LOCK:
        old, S["media"] = S["media"], m
        CACHE.clear()
        if S["stream"]:
            S["stream"].close()
            S["stream"] = None
    stop_preview_audio(old)
    if m.get("has_audio"):
        start_preview_audio(m)
    log("loaded", kind, path)
    return public(m)


def start_preview_audio(m):
    """Cut the clip's sound for the window in the background, once per clip. /api/audio waits for it."""
    a = m["audio"] = {"path": None, "error": None, "proc": None, "stop": False, "done": threading.Event()}

    def work():
        proc = None
        try:
            os.makedirs(AUDIO_DIR, exist_ok=True)
            for f in os.listdir(AUDIO_DIR):  # the sound of clips opened earlier
                try:
                    os.remove(os.path.join(AUDIO_DIR, f))
                except OSError:
                    pass  # still open somewhere: it goes next time
            out = os.path.join(AUDIO_DIR, m["key"] + ".m4a")
            part = out + ".part"
            if a["stop"]:
                return
            proc = a["proc"] = subprocess.Popen(M.preview_audio_cmd(m["path"], m["info"], part),
                                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                                stderr=subprocess.PIPE, creationflags=M.NO_WINDOW)
            M.no_throttle(proc.pid)
            _, err = proc.communicate(timeout=900)
            if a["stop"]:
                return
            if proc.returncode != 0 or not os.path.isfile(part):
                raise RuntimeError(err.decode(errors="replace").strip()[-300:] or "ffmpeg failed")
            os.replace(part, out)
            a["path"] = out
        except Exception as e:
            if proc and proc.poll() is None:
                proc.kill()
            a["error"] = "No sound for the preview: " + str(e)
            log("preview sound failed:", e)
        finally:
            a["done"].set()

    threading.Thread(target=work, daemon=True).start()


def stop_preview_audio(m):
    a = (m or {}).get("audio")
    if a and not a["done"].is_set():
        a["stop"] = True
        if a["proc"] and a["proc"].poll() is None:
            a["proc"].kill()


def source(frame, side):
    """The open photo, or one frame of the open clip, scaled to fit `side` (cached)."""
    m = S["media"]
    if not m:
        raise HTTPException(404, "Nothing is open")
    side = int(min(max(side, 48), 8192))
    frame = 0 if m["kind"] == "image" else int(min(max(frame, 0), m["frames"] - 1))
    W, H = M.fit(m["width"], m["height"], side)
    key = (m["key"], frame, W, H)
    with LOCK:
        hit = CACHE.get(key)
        if hit is not None:
            CACHE.move_to_end(key)
            return hit
    if m["kind"] == "image":
        img = m["img"] if (W, H) == (m["width"], m["height"]) else cv2.resize(m["img"], (W, H),
                                                                                interpolation=cv2.INTER_AREA)
    else:
        with LOCK:
            st = S["stream"]
            if st is None or (st.W, st.H) != (W, H) or st.path != m["path"]:
                if st:
                    st.close()
                st = S["stream"] = M.FrameStream(m["path"], m["info"], W, H)
        img = st.read(frame)
    with LOCK:
        if S["media"] is m:
            while len(CACHE) >= 24:
                CACHE.popitem(last=False)
            CACHE[key] = img
    return img


# --------------------------------------------------------------------------- export

def export_plan(m, fmt, fps_choice, seconds, max_side):
    """What the export will be: kind ('png'/'jpg'/'mp4'), size, frame count and frame rate."""
    W, H = M.even_size(m["width"], m["height"], max_side or 16384)
    if m["kind"] == "image":
        if fmt in ("png", "jpg"):
            w, h = M.fit(m["width"], m["height"], max_side) if max_side else (m["width"], m["height"])
            return {"kind": fmt, "w": w, "h": h, "frames": 1}
        fps = Fraction(int(fps_choice)) if str(fps_choice) in ("24", "25", "30", "60") else Fraction(24)
        n = max(2, int(round(seconds * fps)))
        return {"kind": "mp4", "w": W, "h": H, "frames": n, "fps": fps, "resample": None}
    info = m["info"]
    src_fps = Fraction(info["fps_num"], info["fps_den"])
    fps, resample = src_fps, None
    if str(fps_choice) in ("24", "25", "30", "60") and Fraction(int(fps_choice)) != src_fps:
        fps = Fraction(int(fps_choice))
        resample = f"{fps.numerator}/{fps.denominator}"
    n = info["frames"] if resample is None else max(1, int(info["duration"] * fps + 1e-6))
    return {"kind": "mp4", "w": W, "h": H, "frames": n, "fps": fps, "resample": resample}


def run_export(m, p, plan):
    job = S["job"]
    stem = os.path.splitext(m["name"])[0] + " - Spray Lab"
    writer = jobs = None
    try:
        if plan["kind"] in ("png", "jpg"):
            img = m["img"]
            if (plan["w"], plan["h"]) != (img.shape[1], img.shape[0]):
                img = cv2.resize(img, (plan["w"], plan["h"]), interpolation=cv2.INTER_AREA)
            path = unique_path(out_dir(), stem, "." + plan["kind"])
            M.write_image(path, E.render(img, p, clean=True))
            job.update(done=1, out=path, state="done")
            log("exported", path, f"{time.time() - job['started']:.1f}s")
            return
        W, H, n, fps = plan["w"], plan["h"], plan["frames"], plan["fps"]
        if m["kind"] == "image":
            img = m["img"]
            if (W, H) != (m["width"], m["height"]):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            info = {"frames": n, "fps_num": fps.numerator, "fps_den": fps.denominator, "has_audio": False,
                    "aoff": 0.0}
            jobs = ((img, i) for i in range(n))
        else:
            info = dict(m["info"], frames=n, fps_num=fps.numerator, fps_den=fps.denominator)

            def frames():
                for i, f in M.decode_frames(m["path"], m["info"], W, H, fps=plan["resample"]):
                    if i >= n:
                        return
                    yield f, i

            jobs = frames()
        path = unique_path(out_dir(), stem, ".mp4")
        writer = M.VideoWriter(path, W, H, info, m["path"])
        pending = collections.deque()  # frames being rendered, oldest first: written in order

        def drain(keep):
            while len(pending) > keep:
                writer.write(pending.popleft().result())
                job["done"] += 1

        with ThreadPoolExecutor(RENDER_WORKERS, thread_name_prefix="render") as pool:
            for f, i in jobs:
                if CANCEL.is_set():
                    break
                pending.append(pool.submit(E.render, f, p, i, True))
                drain(RENDER_WORKERS)
            if not CANCEL.is_set():
                drain(0)
        if CANCEL.is_set():
            writer.cancel()
            job.update(state="cancelled")
            log("export cancelled")
            return
        rep = writer.finish(job["done"])
        writer = None
        job.update(out=path, report=rep, state="done")
        if rep["problems"]:
            log("export check:", rep["problems"])
        log("exported", path, f"{time.time() - job['started']:.1f}s")
    except Exception as e:
        traceback.print_exc()
        if writer:
            writer.cancel()
        job.update(state="error", error=str(e))
    finally:
        if hasattr(jobs, "close"):
            jobs.close()  # stops the decoder if the export ended early


# --------------------------------------------------------------------------- routes: app shell

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers=NO_STORE)


@app.get("/static/{path:path}")
def static(path: str):
    full = os.path.normpath(os.path.join(STATIC, path))
    if not full.startswith(STATIC + os.sep) or not os.path.isfile(full):
        raise HTTPException(404)
    return FileResponse(full, headers=NO_STORE)


@app.get("/favicon.ico")
def favicon():
    return FileResponse(os.path.join(APP_DIR, "icon.ico"))


@app.get("/api/ping")
def ping_get():
    return {"app": APP_ID}


@app.post("/api/ping")
def ping():
    S["last_ping"], S["pinged"] = time.time(), True
    return {"ok": True}


def texture_list():
    out = [{"id": k, "label": v["label"], "builtin": True} for k, v in T.BUILTIN.items()]
    for t in read_json(USER_TEXTURES, []):
        if os.path.isfile(t.get("path", "")):
            out.append({"id": "file:" + t["path"], "label": t.get("label") or os.path.basename(t["path"]),
                        "builtin": False})
    return out


@app.get("/api/state")
def state():
    return {"media": public(S["media"]), "job": job_public(), "out_dir": out_dir(),
            "presets": [{"name": p["name"], "note": p["note"], "params": p["params"]} for p in P.PRESETS],
            "my_presets": read_json(USER_PRESETS, []), "textures": texture_list(), "blend_modes": E.BLEND_MODES,
            "fits": E.FITS, "channels": E.CHANNELS, "layer_types": E.LAYER_TYPES, "layer_names": E.LAYER_NAMES,
            "ranges": E.RANGES, "defaults": E.DEFAULTS}


# --------------------------------------------------------------------------- routes: media

@app.post("/api/browse")
def browse():
    imgs = " ".join("*" + e for e in sorted(M.IMAGE_EXT))
    vids = " ".join("*" + e for e in sorted(M.VIDEO_EXT))
    path = pick_file("Open a photo or video", [("Photos and videos", imgs + " " + vids), ("Photos", imgs),
                                               ("Videos", vids), ("All files", "*.*")])
    if not path:
        return {"media": public(S["media"]), "cancelled": True}
    return _load(path)


@app.post("/api/load")
def load(body: dict = Body(...)):
    return _load(body.get("path", ""))


def _load(path):
    try:
        return {"media": load_media(path)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, str(e))


def _safe_name(name, fallback):
    return "".join(c for c in os.path.basename(name or "") if c not in '<>:"/\\|?*').strip() or fallback


@app.post("/api/upload")
async def upload(req: Request, name: str):
    """Files dropped into the window arrive as bytes (the browser does not reveal their path)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, _safe_name(name, "dropped"))
    with open(path, "wb") as f:
        async for chunk in req.stream():
            f.write(chunk)
    return _load(path)


@app.post("/api/preview")
def preview(body: dict = Body(...)):
    try:
        img = source(int(body.get("frame", 0)), int(body.get("side", 1280)))
        t = time.time()
        out = img if body.get("original") else E.render(img, body.get("params") or {}, int(body.get("frame", 0)))
        ms = int((time.time() - t) * 1000)
        return Response(M.jpeg(out, 90), media_type="image/jpeg",
                        headers={**NO_STORE, "X-Render-Ms": str(ms), "X-Size": f"{out.shape[1]}x{out.shape[0]}"})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/audio")
def preview_audio(key: str = ""):
    """The open clip's sound for preview playback (AAC, lined up with frame 0 like the export)."""
    m = S["media"]
    a = m.get("audio") if m and m.get("key") == key else None
    if not a:
        raise HTTPException(404, "No sound for this clip")
    a["done"].wait(timeout=900)
    if not a["path"]:
        raise HTTPException(404, a["error"] or "The sound is not ready")
    return FileResponse(a["path"], media_type="audio/mp4", headers=NO_STORE)


@app.post("/api/thumb")
def thumb(body: dict = Body(...)):
    """A preset (or any params) on the open picture, small, for the preset strip."""
    try:
        img = source(int(body.get("frame", 0)), int(body.get("side", 320)))
        return Response(M.jpeg(E.render(img, body.get("params") or {}), 84), media_type="image/jpeg",
                        headers=NO_STORE)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# --------------------------------------------------------------------------- routes: layers, textures, presets

@app.post("/api/normalize")
def normalize(body: dict = Body(...)):
    """Settings saved by an older version (or edited by hand) come back clean."""
    return E.normalize(body.get("params"))


@app.post("/api/layer/new")
def layer_new(body: dict = Body(...)):
    kind = body.get("type")
    if kind not in E.LAYER_TYPES:
        raise HTTPException(400, "Unknown layer type")
    return E.new_layer(kind)


@app.post("/api/layer/randomize")
def layer_randomize(body: dict = Body(...)):
    lay = E.normalize_layer(body.get("layer"))
    if not lay:
        raise HTTPException(400, "Not a layer")
    return E.randomize_layer(lay, body.get("seed"))


@app.get("/api/texture-thumb")
def texture_thumb(id: str, side: int = 96):
    try:
        return Response(M.jpeg(T.thumbnail(id, int(min(max(side, 16), 512))), 88), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(404, str(e))


def _add_texture(path, label=None):
    try:
        T.load_file(path)  # readable?
    except Exception as e:
        raise HTTPException(400, str(e))
    lst = [t for t in read_json(USER_TEXTURES, []) if t.get("path") != path]
    lst.insert(0, {"path": path, "label": label or os.path.splitext(os.path.basename(path))[0]})
    write_json(USER_TEXTURES, lst[:16])
    return {"id": "file:" + path, "textures": texture_list()}


@app.post("/api/texture/browse")
def texture_browse():
    exts = " ".join("*" + e for e in sorted(T.TEXTURE_EXT))
    path = pick_file("Choose a texture image", [("Textures", exts), ("All files", "*.*")])
    if not path:
        return {"cancelled": True}
    return _add_texture(os.path.abspath(path))


@app.post("/api/texture/upload")
async def texture_upload(req: Request, name: str):
    os.makedirs(USER_TEX_DIR, exist_ok=True)
    path = os.path.join(USER_TEX_DIR, _safe_name(name, "texture.png"))
    with open(path, "wb") as f:
        async for chunk in req.stream():
            f.write(chunk)
    return _add_texture(path)


@app.post("/api/my-presets")
def save_preset(body: dict = Body(...)):
    name = str(body.get("name") or "").strip()[:60]
    if not name:
        raise HTTPException(400, "Give the preset a name")
    lst = [p for p in read_json(USER_PRESETS, []) if p.get("name") != name]
    lst.insert(0, {"name": name, "params": E.normalize(body.get("params"))})
    write_json(USER_PRESETS, lst[:60])
    return {"my_presets": lst[:60]}


@app.post("/api/my-presets/delete")
def delete_preset(body: dict = Body(...)):
    lst = [p for p in read_json(USER_PRESETS, []) if p.get("name") != body.get("name")]
    write_json(USER_PRESETS, lst)
    return {"my_presets": lst}


# --------------------------------------------------------------------------- routes: export

def job_public():
    j = dict(S["job"])
    j.pop("thread", None)
    if j.get("state") == "running" and j.get("done"):
        el = time.time() - j["started"]
        j["fps"] = j["done"] / max(el, 1e-6)
        j["eta"] = (j["total"] - j["done"]) / max(j["fps"], 1e-6)
    return j


@app.post("/api/export")
def export(body: dict = Body(...)):
    m = S["media"]
    if not m:
        raise HTTPException(400, "Open a photo or video first")
    if S["job"].get("state") == "running":
        raise HTTPException(409, "An export is already running")
    p = E.normalize(body.get("params"))
    max_side = EXPORT_SIDES.get(int(body.get("max_side") or 0))
    fmt = body.get("format") if body.get("format") in ("png", "jpg", "mp4") else (
        "png" if m["kind"] == "image" else "mp4")
    seconds = min(60.0, max(1.0, float(body.get("seconds") or 5)))
    plan = export_plan(m, fmt, body.get("fps") or "source", seconds, max_side)
    CANCEL.clear()
    S["job"] = {"state": "running", "kind": plan["kind"], "done": 0, "total": plan["frames"],
                "started": time.time(), "out": None, "error": None, "name": m["name"],
                "size": f"{plan['w']}x{plan['h']}"}
    threading.Thread(target=run_export, args=(m, p, plan), daemon=True).start()
    return job_public()


@app.get("/api/job")
def job():
    return job_public()


@app.post("/api/cancel")
def cancel():
    CANCEL.set()
    return job_public()


@app.post("/api/reveal")
def reveal(body: dict = Body(...)):
    path = body.get("path") or ""
    if path and os.path.exists(path):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        folder = out_dir()
        os.makedirs(folder, exist_ok=True)
        os.startfile(folder)
    return {"ok": True}


# --------------------------------------------------------------------------- lifecycle

def watchdog():
    """Quit once the window has been closed for a while (and no export is running)."""
    while True:
        time.sleep(2)
        idle = time.time() - S["last_ping"]
        if S["job"].get("state") == "running":
            continue
        if idle > (45 if S["pinged"] else 180):
            log("no window for %ds, exiting" % idle)
            os._exit(0)


def warm_up():
    E.warm_up()  # compiles (or loads) the kernels
    T.warm_up()  # builds any missing built-in texture


def serve():
    M.no_throttle()  # pythonw has no window: keep Windows from slowing the renders down
    threading.Thread(target=warm_up, daemon=True).start()
    if not os.environ.get("SPRAYLAB_STAY"):
        threading.Thread(target=watchdog, daemon=True).start()
    log("serving on", PORT)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        load_media(sys.argv[1])
    serve()
