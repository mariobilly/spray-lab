"""ffmpeg/ffprobe and image plumbing: probe, unicode-safe image read/write, single-frame grab, a frame stream for
playback, the preview's sound, frame pipe, MP4 writer that keeps the source audio. Frames are HxWx3 uint8 RGB
everywhere.
(Started from Pixel Drip's media.py.)"""
import json
import os
import queue
import shutil
import subprocess
import threading
from fractions import Fraction

import cv2
import numpy as np

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".wmv", ".mpg", ".mpeg", ".mts", ".m2ts", ".3gp",
             ".gif"}
_ENC = {}


def ffmpeg_bin(name="ffmpeg"):
    p = shutil.which(name)
    if p:
        return p
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(base):
        for d in os.listdir(base):
            if d.lower().startswith("gyan.ffmpeg"):
                for root, _, files in os.walk(os.path.join(base, d)):
                    if name + ".exe" in files:
                        return os.path.join(root, name + ".exe")
    return name


def no_throttle(pid=None):
    """Opt a process (default: this one) out of Windows power throttling (EcoQoS).

    Windows can quietly slow down busy processes that have no window of their own (pythonw, ffmpeg pipes); on this
    PC a frame was measured taking 8-10x longer. Returns True when the opt-out was applied. (From Blob Scope.)
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes as wt

        class _PPT(ctypes.Structure):
            _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wt.HANDLE
        k.OpenProcess.restype = wt.HANDLE
        k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        k.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        k.SetProcessInformation.restype = wt.BOOL
        k.CloseHandle.argtypes = [wt.HANDLE]
        h = k.GetCurrentProcess() if pid is None else k.OpenProcess(0x0200 | 0x1000, False, pid)
        if not h:
            return False
        st = _PPT(1, 0x1, 0x0)  # execution-speed throttling: controlled by us, and off
        ok = bool(k.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))  # 4 = ProcessPowerThrottling
        if pid is not None:
            k.CloseHandle(h)
        return ok
    except Exception:
        return False


def kind(path):
    ext = os.path.splitext(path)[1].lower()
    return "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else None


def even_size(w, h, max_side=None):
    """Never upscale; long side <= max_side; even dimensions (yuv420)."""
    s = min(1.0, max_side / max(w, h)) if max_side else 1.0
    W, H = int(round(w * s)), int(round(h * s))
    return max(2, W - W % 2), max(2, H - H % 2)


def fit(w, h, max_side):
    s = min(1.0, max_side / max(w, h))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


# --------------------------------------------------------------------------- images

def read_image(path):
    """RGB uint8. Reads through a byte buffer so accented / non-ASCII paths work; EXIF rotation applied."""
    img = None
    try:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", path, "-frames:v", "1", "-f", "image2pipe", "-c:v", "png",
                        "-"], capture_output=True, timeout=60, creationflags=NO_WINDOW)
    img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR) if r.stdout else None
    if img is None:
        raise RuntimeError("Could not read this image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path, rgb):
    ext = os.path.splitext(path)[1].lower() or ".png"
    opts = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), opts)
    if not ok:
        raise RuntimeError("Could not encode " + ext)
    part = path + ".part" + ext
    buf.tofile(part)
    os.replace(part, path)


def jpeg(rgb, q=88):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


# --------------------------------------------------------------------------- video

def _frac(s):
    try:
        f = Fraction(s)
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def probe(path):
    """Exact stream facts: fractional fps, frame count, display size after rotation, audio offset."""
    r = subprocess.run([ffmpeg_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format",
                        "-show_streams", path], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=30, creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("Could not read this file: " + (r.stderr.strip()[-300:] or "ffprobe failed"))
    j = json.loads(r.stdout or "{}")
    fmt = j.get("format", {})
    vs = [s for s in j.get("streams", []) if s.get("codec_type") == "video"
          and not (s.get("disposition") or {}).get("attached_pic")]
    aus = [s for s in j.get("streams", []) if s.get("codec_type") == "audio"]
    if not vs:
        raise RuntimeError("No video stream in this file")
    v = vs[0]
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    rot = int((v.get("tags") or {}).get("rotate", rot) or rot)
    if abs(rot) % 180 == 90:
        w, h = h, w
    rf, af = _frac(v.get("r_frame_rate")), _frac(v.get("avg_frame_rate"))
    fps = rf or af or Fraction(30)
    if fps > 240 and af:
        fps = af
    vfr = bool(rf and af and abs(float(rf - af)) / float(rf) > 0.01)
    if vfr:
        fps = af.limit_denominator(1001)
    dur = float(v.get("duration") or fmt.get("duration") or 0)
    n = int(v.get("nb_frames") or 0)
    if n <= 0 or vfr or (dur > 0 and abs(n / float(fps) - dur) > 1.0):
        n = max(1, int(round(dur * float(fps))))
    fmt_start = float(fmt.get("start_time") or 0)
    v_start = float(v.get("start_time") or fmt_start)
    a_start = float(aus[0].get("start_time") or fmt_start) if aus else 0.0
    return {"width": w, "height": h, "fps_num": fps.numerator, "fps_den": fps.denominator, "fps": float(fps),
            "frames": n, "duration": n / float(fps), "vfr": vfr, "has_audio": bool(aus),
            "aoff": max(0.0, v_start - a_start), "codec": v.get("codec_name", "")}


def read_frame(path, info, index, W, H):
    """One frame (by index) at W x H. Input seeking is frame-accurate when ffmpeg decodes."""
    size = W * H * 3
    t = index * info["fps_den"] / info["fps_num"]
    for back in (0.0, 0.25, 1.0):  # the very last frames can sit past the container duration
        r = subprocess.run([ffmpeg_bin(), "-v", "error", "-ss", f"{max(0.0, t - back):.6f}", "-i", path,
                            "-map", "0:v:0", "-frames:v", "1", "-vf", f"scale={W}:{H}:flags=area",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True, timeout=60, creationflags=NO_WINDOW)
        if len(r.stdout) >= size:
            return np.frombuffer(r.stdout[:size], np.uint8).reshape(H, W, 3).copy()
    raise RuntimeError("Could not decode frame %d: %s" % (index, r.stderr.decode(errors="replace")[-300:]))


class FrameStream:
    """Frames i, i+1, i+2... from one running ffmpeg, for smooth preview playback.

    read(i) returns frame i: from the running decoder when i is just ahead of it, otherwise it restarts there.
    """

    AHEAD = 45  # skipping up to this many frames is cheaper than a new seek

    def __init__(self, path, info, W, H):
        self.path, self.info, self.W, self.H = path, info, W, H
        self.p, self.next = None, 0
        self.lock = threading.Lock()

    def _start(self, i):
        self.close()
        t = max(0.0, (i - 0.25) * self.info["fps_den"] / self.info["fps_num"])  # frame i is the first one kept
        vf = f"scale={self.W}:{self.H}:flags=area"
        if self.info.get("vfr"):
            vf = f"fps={self.info['fps_num']}/{self.info['fps_den']}," + vf
        self.p = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-ss", f"{t:.6f}", "-i", self.path, "-map", "0:v:0",
                                   "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW,
                                  bufsize=self.W * self.H * 3 * 2)
        no_throttle(self.p.pid)
        self.next = i

    def _pull(self):
        size = self.W * self.H * 3
        buf = bytearray(size)
        view, got = memoryview(buf), 0
        while got < size:
            k = self.p.stdout.readinto(view[got:])
            if not k:
                return None
            got += k
        self.next += 1
        return np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3)

    def read(self, i):
        with self.lock:
            if self.p is None or not (self.next <= i <= self.next + self.AHEAD):
                self._start(i)
            f = None
            while self.next <= i:
                f = self._pull()
                if f is None:
                    break
            if f is None:  # past the end of the stream: fall back to a one-off grab
                self.close()
                return read_frame(self.path, self.info, i, self.W, self.H)
            return f

    def close(self):
        if self.p is not None:
            try:
                self.p.kill()
                self.p.wait(timeout=5)
                self.p.stdout.close()
            except Exception:
                pass
            self.p = None


def decode_frames(path, info, W, H, prefetch=6, fps=None):
    """Yield (index, HxWx3 uint8 RGB) at W x H, constant frame rate, rotation applied.

    `fps` ("num/den") resamples to another frame rate (frames are dropped or repeated, like ffmpeg's fps filter).
    A reader thread keeps a few frames decoded ahead so the effect never waits on the pipe.
    """
    vf = [f"scale={W}:{H}:flags=bicubic"]
    if fps:
        vf.insert(0, f"fps={fps}")
    elif info.get("vfr"):
        vf.insert(0, f"fps={info['fps_num']}/{info['fps_den']}")
    cmd = [ffmpeg_bin(), "-v", "error", "-i", path, "-map", "0:v:0", "-vf", ",".join(vf),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW,
                         bufsize=W * H * 3 * 2)
    no_throttle(p.pid)
    size = W * H * 3
    q = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                buf = bytearray(size)
                view, got = memoryview(buf), 0
                while got < size:
                    k = p.stdout.readinto(view[got:])
                    if not k:
                        break
                    got += k
                if got < size:
                    break
                q.put(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
        finally:
            q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    i = 0
    try:
        while True:
            f = q.get()
            if f is None:
                break
            yield i, f
            i += 1
    finally:
        stop.set()
        p.kill()
        while t.is_alive():  # unblock a reader waiting on a full queue
            try:
                q.get(timeout=0.05)
            except queue.Empty:
                pass
        p.wait()
        p.stdout.close()


def preview_audio_cmd(path, info, out):
    """ffmpeg command: the clip's first audio track as a small stereo AAC .m4a the app window can play.

    It starts where frame 0 is and stops at the last frame, with the same offset the export uses, so the preview
    sounds like the exported file.
    """
    cmd = [ffmpeg_bin(), "-v", "error", "-y"]
    if info["aoff"] > 0.001:
        cmd += ["-ss", f"{info['aoff']:.6f}"]
    return cmd + ["-i", path, "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "2", "-c:a", "aac", "-b:a", "160k",
                  "-t", f"{info['duration']:.6f}", "-movflags", "+faststart", "-f", "mp4", out]


def encoder_args():
    if "nvenc" not in _ENC:
        try:
            r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True,
                               timeout=20, creationflags=NO_WINDOW)
            _ENC["nvenc"] = "h264_nvenc" in r.stdout and _nvenc_works()
        except Exception:
            _ENC["nvenc"] = False
    if _ENC["nvenc"]:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "17", "-b:v", "0", "-profile:v", "high"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "16"]


def _nvenc_works():
    r = subprocess.run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                        "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30,
                       creationflags=NO_WINDOW)
    return r.returncode == 0


BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1"]


class VideoWriter:
    """RGB frames -> ffmpeg stdin -> H.264 MP4 with the source's audio, written to a .part file first."""

    def __init__(self, out_path, w, h, info, src_path):
        self.out, self.part = out_path, out_path + ".part.mp4"
        self.w, self.h, self.info = w, h, info
        n, fn, fd = info["frames"], info["fps_num"], info["fps_den"]
        dur = n * fd / fn
        cmd = [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-framerate", f"{fn}/{fd}", "-i", "-"]
        if info["has_audio"]:
            if info["aoff"] > 0.001:
                cmd += ["-ss", f"{info['aoff']:.6f}"]
            cmd += ["-i", src_path, "-map", "0:v:0", "-map", "1:a:0?", "-af", f"apad=whole_dur={dur:.6f}",
                    "-c:a", "aac", "-b:a", "256k"]
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv"] + BT709 + encoder_args() + [
            "-pix_fmt", "yuv420p", "-t", f"{dur:.6f}", "-movflags", "+faststart", self.part]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
        no_throttle(self.p.pid)
        self._err = b""
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self):
        self._err = self.p.stderr.read()

    def _reap(self):
        self._t.join(timeout=5)
        self.p.stderr.close()

    def write(self, rgb):
        try:
            self.p.stdin.write(np.ascontiguousarray(rgb).data)  # blocks while ffmpeg is busy: backpressure
        except OSError:
            self.p.wait()
            raise RuntimeError("Encoder stopped: " + self._err.decode(errors="replace")[-500:])

    def finish(self, frames_written):
        self.p.stdin.close()
        code = self.p.wait()
        self._reap()
        if code != 0:
            raise RuntimeError("Encoder failed: " + self._err.decode(errors="replace")[-500:])
        os.replace(self.part, self.out)
        return verify(self.out, self.w, self.h, frames_written, self.info["fps_num"], self.info["fps_den"])

    def cancel(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        self.p.kill()
        self.p.wait()
        self._reap()
        if os.path.exists(self.part):
            os.remove(self.part)


def verify(path, w, h, n, fps_num, fps_den):
    """ffprobe the finished file; returns a small report (problems listed, never raises)."""
    rep = {"path": path, "problems": []}
    try:
        r = subprocess.run([ffmpeg_bin("ffprobe"), "-v", "error", "-count_frames", "-print_format", "json",
                            "-show_streams", "-show_format", path], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120, creationflags=NO_WINDOW)
        j = json.loads(r.stdout or "{}")
        v = next(s for s in j["streams"] if s["codec_type"] == "video")
        a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
        frames = int(v.get("nb_read_frames") or 0)
        rep.update(codec=v.get("codec_name"), width=v.get("width"), height=v.get("height"), frames=frames,
                   audio=bool(a), size=int(j["format"].get("size", 0)),
                   duration=float(j["format"].get("duration", 0)))
        if (v.get("width"), v.get("height")) != (w, h):
            rep["problems"].append(f"size {v.get('width')}x{v.get('height')} != {w}x{h}")
        if frames != n:
            rep["problems"].append(f"{frames} frames written, expected {n}")
        if _frac(v.get("r_frame_rate")) != Fraction(fps_num, fps_den):
            rep["problems"].append(f"frame rate {v.get('r_frame_rate')} != {fps_num}/{fps_den}")
    except Exception as e:
        rep["problems"].append("verify failed: " + str(e))
    return rep
