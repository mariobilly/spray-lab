"""The spray-paint effect: a stack of layers applied bottom to top, like Effect.app's layer panel.

Layer types:
  blur      Gaussian Blur: strength, aspect, opacity, blend mode
  mix       Layer Mix: a spray texture blended over the picture (fit, texture scale, blend channel, the two
            blend-if ramps, opacity, blend mode); `boil` re-places the texture every N video frames
  gradient  Gradient Map: colour stops with midpoints, shift, repeat, opacity, blend mode

Pictures are float32 RGB 0..1 inside; render() takes and returns HxWx3 uint8 RGB. Sizes (blur radius, texture
droplets) scale with the picture, so a small preview looks like the full-size export.
"""
import colorsys
import copy
import math
import os
import random
import threading
import uuid

# numba's OpenMP threads would otherwise spin after each kernel and slow OpenCV's blur down (about 25%)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import cv2  # noqa: E402
import numpy as np
from numba import config as _numba_config
from numba import njit, prange

import textures as T

_numba_config.THREADING_LAYER = "threadsafe"  # exports render several frames at once from worker threads

BLEND_MODES = ["normal", "multiply", "screen", "overlay", "soft-light", "hard-light", "linear-light",
               "vivid-light", "color-dodge", "color-burn", "darken", "lighten", "difference", "exclusion", "add",
               "subtract"]
FITS = ["fill", "stretch", "tile"]
CHANNELS = ["gray", "red", "green", "blue"]
LAYER_TYPES = ["blur", "mix", "gradient"]
LAYER_NAMES = {"blur": "Gaussian Blur", "mix": "Layer Mix", "gradient": "Gradient Map"}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)  # Rec.709
LUT_SIZE = 1024
REF_SIDE = 1080.0  # sizes are defined for a picture whose short side is 1080 px

FIRE = [("#000000", 0.0), ("#0a0000", 0.16), ("#6a0808", 0.24), ("#dc2416", 0.31), ("#f44a18", 0.42),
        ("#ff7008", 0.5), ("#ffae00", 0.58), ("#ffd21a", 0.68), ("#ffeb70", 0.84), ("#ffffff", 1.0)]
RAMP_ALL = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)  # full strength everywhere

DEFAULTS = {
    "blur": {"strength": 14.0, "aspect": 0.0, "opacity": 0.34, "blend": "normal"},
    "mix": {"texture": T.DEFAULT, "fit": "fill", "scale": 1.5, "channel": "gray", "ramp_top": list(RAMP_ALL),
            "ramp_under": list(RAMP_ALL), "opacity": 1.0, "blend": "hard-light", "boil": 0},
    "gradient": {"stops": [{"pos": p, "color": c, "mid": 0.5} for c, p in FIRE], "shift": 0.0, "repeat": 1,
                 "opacity": 1.0, "blend": "normal"},
}
RANGES = {  # slider limits (also enforced on load)
    "strength": (0.0, 100.0), "aspect": (-1.0, 1.0), "opacity": (0.0, 1.0), "scale": (0.25, 4.0),
    "shift": (-1.0, 1.0), "repeat": (1, 8), "boil": (0, 12),
}


# --------------------------------------------------------------------------- params

def _num(v, lo, hi, default, integer=False):
    try:
        v = float(v)
        if not np.isfinite(v):
            raise ValueError
    except (TypeError, ValueError):
        v = float(default)
    v = min(max(v, lo), hi)
    return int(round(v)) if integer else v


def _hex(c, default="#000000"):
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) == 6:
            try:
                int(s, 16)
                return "#" + s.lower()
            except ValueError:
                pass
    return default


def _ramp(r):
    """[a, b, c, d, lo, hi]: the corners of the trapezoid and the heights of its two ends."""
    try:
        v = [min(max(float(x), 0.0), 1.0) for x in list(r)[:6]]
    except (TypeError, ValueError):
        v = []
    if len(v) == 4:
        v += [0.0, 0.0]
    if len(v) != 6:
        return list(RAMP_ALL)
    v[1] = max(v[1], v[0])  # handles never cross: a <= b <= c <= d
    v[2] = max(v[2], v[1])
    v[3] = max(v[3], v[2])
    return v


def new_layer(kind):
    lay = copy.deepcopy(DEFAULTS[kind])
    lay.update(type=kind, id=uuid.uuid4().hex[:8], visible=True)
    return lay


def normalize_layer(raw):
    """A clean copy of one layer: known type, every setting present and inside its range."""
    if not isinstance(raw, dict) or raw.get("type") not in LAYER_TYPES:
        return None
    kind = raw["type"]
    d = DEFAULTS[kind]
    out = {"type": kind, "id": str(raw.get("id") or uuid.uuid4().hex[:8])[:32],
           "visible": bool(raw.get("visible", True))}
    out["opacity"] = _num(raw.get("opacity"), *RANGES["opacity"], d["opacity"])
    out["blend"] = raw.get("blend") if raw.get("blend") in BLEND_MODES else d["blend"]
    if kind == "blur":
        out["strength"] = _num(raw.get("strength"), *RANGES["strength"], d["strength"])
        out["aspect"] = _num(raw.get("aspect"), *RANGES["aspect"], d["aspect"])
    elif kind == "mix":
        tex = raw.get("texture")
        out["texture"] = tex if isinstance(tex, str) and (tex in T.BUILTIN or tex.startswith("file:")) \
            else d["texture"]
        out["fit"] = raw.get("fit") if raw.get("fit") in FITS else d["fit"]
        out["scale"] = _num(raw.get("scale"), *RANGES["scale"], d["scale"])
        out["channel"] = raw.get("channel") if raw.get("channel") in CHANNELS else d["channel"]
        out["ramp_top"] = _ramp(raw.get("ramp_top", d["ramp_top"]))
        out["ramp_under"] = _ramp(raw.get("ramp_under", d["ramp_under"]))
        out["boil"] = _num(raw.get("boil"), *RANGES["boil"], d["boil"], integer=True)
    else:
        stops = []
        for s in raw.get("stops") or []:
            if isinstance(s, dict):
                stops.append({"pos": _num(s.get("pos"), 0.0, 1.0, 0.0), "color": _hex(s.get("color")),
                              "mid": _num(s.get("mid"), 0.05, 0.95, 0.5)})
        if len(stops) < 2:
            stops = copy.deepcopy(d["stops"])
        out["stops"] = sorted(stops, key=lambda s: s["pos"])[:16]
        out["shift"] = _num(raw.get("shift"), *RANGES["shift"], d["shift"])
        out["repeat"] = _num(raw.get("repeat"), *RANGES["repeat"], d["repeat"], integer=True)
    return out


def normalize(params):
    """Clean params: {"layers": [...]} bottom layer first. Bad or unknown layers are dropped."""
    layers = (params or {}).get("layers") if isinstance(params, dict) else None
    if not isinstance(layers, list):
        layers = []
    out = [lay for lay in (normalize_layer(x) for x in layers[:24]) if lay]
    return {"layers": out}


# --------------------------------------------------------------------------- blend modes

def _soft_light(b, s):
    d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(b))
    return np.where(s <= 0.5, b - (1.0 - 2.0 * s) * b * (1.0 - b), b + (2.0 * s - 1.0) * (d - b))


def _dodge(b, s):
    return np.where(b <= 0.0, 0.0, np.where(s >= 1.0, 1.0, np.minimum(1.0, b / np.maximum(1.0 - s, 1e-6))))


def _burn(b, s):
    return np.where(b >= 1.0, 1.0, np.where(s <= 0.0, 0.0, 1.0 - np.minimum(1.0, (1.0 - b) / np.maximum(s, 1e-6))))


def _hard(b, s):
    return np.where(s <= 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))


def blend(mode, b, s):
    """Blend source `s` over base `b` (both 0..1, broadcastable). Returns the fully-opaque result."""
    if mode == "normal":
        return np.broadcast_to(s, np.broadcast_shapes(b.shape, s.shape))
    if mode == "multiply":
        return b * s
    if mode == "screen":
        return 1.0 - (1.0 - b) * (1.0 - s)
    if mode == "overlay":
        return _hard(s, b)
    if mode == "hard-light":
        return _hard(b, s)
    if mode == "soft-light":
        return _soft_light(b, s)
    if mode == "linear-light":
        return np.clip(b + 2.0 * s - 1.0, 0.0, 1.0)
    if mode == "vivid-light":
        return np.where(s <= 0.5, _burn(b, 2.0 * s), _dodge(b, 2.0 * s - 1.0))
    if mode == "color-dodge":
        return _dodge(b, s)
    if mode == "color-burn":
        return _burn(b, s)
    if mode == "darken":
        return np.minimum(b, s)
    if mode == "lighten":
        return np.maximum(b, s)
    if mode == "difference":
        return np.abs(b - s)
    if mode == "exclusion":
        return b + s - 2.0 * b * s
    if mode == "add":
        return np.minimum(b + s, 1.0)
    if mode == "subtract":
        return np.maximum(b - s, 0.0)
    raise ValueError("unknown blend mode " + str(mode))


def composite(base, src, mode, weight):
    """base, blended with src through `mode`, then mixed in by weight (scalar or HxW map)."""
    out = blend(mode, base, src)
    if np.isscalar(weight):
        if weight >= 1.0:
            return np.ascontiguousarray(np.broadcast_to(out, base.shape), np.float32)
        w = np.float32(weight)
    else:
        w = weight[:, :, None] if weight.ndim == 2 else weight
    return (base + (out - base) * w).astype(np.float32, copy=False)



# --------------------------------------------------------------------------- compiled kernels
# One pass per layer over the pixels: blend mode, ramps, opacity and the gradient lookup are fused, so nothing
# big is allocated per step. blend() above is the plain-numpy reference these must match (see the tests).

MODE_ID = {m: i for i, m in enumerate(BLEND_MODES)}
CHANNEL_ID = {c: i for i, c in enumerate(CHANNELS)}


@njit(cache=True, inline="always")
def _dodge1(b, s):
    if b <= 0.0:
        return 0.0
    if s >= 1.0:
        return 1.0
    return min(1.0, b / (1.0 - s))


@njit(cache=True, inline="always")
def _burn1(b, s):
    if b >= 1.0:
        return 1.0
    if s <= 0.0:
        return 0.0
    return 1.0 - min(1.0, (1.0 - b) / s)


@njit(cache=True, inline="always")
def _blend1(m, b, s):
    if m == 0:
        return s
    if m == 1:
        return b * s
    if m == 2:
        return 1.0 - (1.0 - b) * (1.0 - s)
    if m == 3:  # overlay = hard light with the layers swapped
        return 2.0 * b * s if b <= 0.5 else 1.0 - 2.0 * (1.0 - b) * (1.0 - s)
    if m == 4:  # soft light (W3C)
        if s <= 0.5:
            return b - (1.0 - 2.0 * s) * b * (1.0 - b)
        d = ((16.0 * b - 12.0) * b + 4.0) * b if b <= 0.25 else math.sqrt(b)
        return b + (2.0 * s - 1.0) * (d - b)
    if m == 5:  # hard light
        return 2.0 * b * s if s <= 0.5 else 1.0 - 2.0 * (1.0 - b) * (1.0 - s)
    if m == 6:
        return min(max(b + 2.0 * s - 1.0, 0.0), 1.0)
    if m == 7:  # vivid light
        return _burn1(b, 2.0 * s) if s <= 0.5 else _dodge1(b, 2.0 * s - 1.0)
    if m == 8:
        return _dodge1(b, s)
    if m == 9:
        return _burn1(b, s)
    if m == 10:
        return min(b, s)
    if m == 11:
        return max(b, s)
    if m == 12:
        return abs(b - s)
    if m == 13:
        return b + s - 2.0 * b * s
    if m == 14:
        return min(b + s, 1.0)
    return max(b - s, 0.0)


@njit(cache=True, inline="always")
def _ramp1(x, r):
    a, b, c, d, lo, hi = r[0], r[1], r[2], r[3], r[4], r[5]
    if b > a:
        up = min(max((x - a) / (b - a), 0.0), 1.0)
    else:
        up = 1.0 if x >= a else 0.0
    if d > c:
        down = min(max((d - x) / (d - c), 0.0), 1.0)
    else:
        down = 1.0 if x <= d else 0.0
    return (lo + (1.0 - lo) * up) * (hi + (1.0 - hi) * down)


@njit(cache=True, inline="always")
def _chan1(ch, r, g, b):
    if ch == 1:
        return r
    if ch == 2:
        return g
    if ch == 3:
        return b
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@njit(parallel=True, cache=True, nogil=True, fastmath=True)
def _k_composite(base, src, mode, opacity, out):
    H, W = base.shape[0], base.shape[1]
    for y in prange(H):
        for x in range(W):
            for c in range(3):
                b = base[y, x, c]
                out[y, x, c] = b + (_blend1(mode, b, src[y, x, c]) - b) * opacity


@njit(parallel=True, cache=True, nogil=True, fastmath=True)
def _k_mix(img, tex, mode, opacity, ch, rt, use_t, ru, use_u, out):
    H, W = img.shape[0], img.shape[1]
    colour = tex.shape[2] == 3
    for y in prange(H):
        for x in range(W):
            b0, b1, b2 = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            t0 = tex[y, x, 0]
            t1, t2 = t0, t0
            if colour:
                t1, t2 = tex[y, x, 1], tex[y, x, 2]
            w = opacity
            if use_t:
                w *= _ramp1(_chan1(ch, t0, t1, t2), rt)
            if use_u:
                w *= _ramp1(_chan1(ch, b0, b1, b2), ru)
            out[y, x, 0] = b0 + (_blend1(mode, b0, t0) - b0) * w
            out[y, x, 1] = b1 + (_blend1(mode, b1, t1) - b1) * w
            out[y, x, 2] = b2 + (_blend1(mode, b2, t2) - b2) * w


@njit(parallel=True, cache=True, nogil=True, fastmath=True)
def _k_gradient(img, lut, rep, shift, wrap, mode, opacity, out):
    H, W = img.shape[0], img.shape[1]
    n = lut.shape[0]
    for y in prange(H):
        for x in range(W):
            b0, b1, b2 = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            v = 0.2126 * b0 + 0.7152 * b1 + 0.0722 * b2
            if wrap:
                v = v * rep + shift
                f = v - math.floor(v)
                if f == 0.0 and v > 0.0:
                    f = 1.0  # the top of each repeat is the gradient's end colour
                v = f
            i = min(max(int(v * (n - 1) + 0.5), 0), n - 1)
            out[y, x, 0] = b0 + (_blend1(mode, b0, lut[i, 0]) - b0) * opacity
            out[y, x, 1] = b1 + (_blend1(mode, b1, lut[i, 1]) - b1) * opacity
            out[y, x, 2] = b2 + (_blend1(mode, b2, lut[i, 2]) - b2) * opacity


@njit(parallel=True, cache=True, nogil=True)
def _k_to_u8(img, out):
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        for x in range(W):
            for c in range(3):
                v = img[y, x, c] * 255.0 + 0.5
                if v <= 0.0:
                    out[y, x, c] = 0
                elif v >= 255.0:
                    out[y, x, c] = 255
                else:
                    out[y, x, c] = np.uint8(v)

# --------------------------------------------------------------------------- layers

def unit(h, w):
    return min(h, w) / REF_SIDE


def gaussian(img, sx, sy):
    """Gaussian blur. Wide blurs run on a shrunken copy and are scaled back up: the same look (under 0.5/255 off
    on average) in a fraction of the time, e.g. 12 ms instead of 1 s for sigma 40 on 1440 x 2560."""
    k = int(min(sx, sy) / 2.5)
    if k < 2:
        # a sigma this small gives a 1-px kernel: no blur along that axis
        return cv2.GaussianBlur(img, (0, 0), sigmaX=max(sx, 1e-3), sigmaY=max(sy, 1e-3),
                                borderType=cv2.BORDER_REFLECT_101)
    h, w = img.shape[:2]
    sw, sh = max(1, round(w / k)), max(1, round(h / k))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    fx, fy = w / sw, h / sh
    # shrinking (box) and linear upscaling blur a little themselves: take that out of the sigma
    bx = math.sqrt(max((sx / fx) ** 2 - 0.25, 0.25))
    by = math.sqrt(max((sy / fy) ** 2 - 0.25, 0.25))
    b = cv2.GaussianBlur(small, (0, 0), sigmaX=bx, sigmaY=by, borderType=cv2.BORDER_REFLECT_101)
    return cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)


def blur_layer(img, lay):
    h, w = img.shape[:2]
    sigma = lay["strength"] * unit(h, w)  # strength = blur sigma in px when the short side is 1080
    a = lay["aspect"]
    sx, sy = sigma * (1.0 + a), sigma * (1.0 - a)  # aspect 1: horizontal only, -1: vertical only
    if sx < 0.3 and sy < 0.3:
        return img
    b = np.ascontiguousarray(gaussian(img, sx, sy))
    _k_composite(img, b, MODE_ID[lay["blend"]], np.float32(lay["opacity"]), b)
    return b


_TEX_LOCK = threading.Lock()
_TEX_SCALED = {}  # (texture id, w, h) -> pre-scaled texture


def _prescaled(tex_id, tex, sw, sh):
    key = (tex_id, sw, sh)
    with _TEX_LOCK:
        hit = _TEX_SCALED.get(key)
    if hit is not None:
        return hit
    th, tw = tex.shape[:2]
    out = cv2.resize(tex, (sw, sh), interpolation=cv2.INTER_AREA if sw < tw or sh < th else cv2.INTER_CUBIC)
    with _TEX_LOCK:
        while len(_TEX_SCALED) >= 6:
            _TEX_SCALED.pop(next(iter(_TEX_SCALED)))
        _TEX_SCALED[key] = out
    return out


def boil_offset(boil, frame):
    """Where the texture sits for this frame: still unless boil > 0, then a new spot every `boil` frames."""
    if not boil:
        return 0.0, 0.0
    r = random.Random(7919 * (frame // boil) + 13)
    return r.uniform(-0.2, 0.2), r.uniform(-0.2, 0.2)


def place_texture(tex_id, w, h, fit, scale, offset=(0.0, 0.0)):
    """The texture laid over a w x h picture: fill (cover), stretch or tile, zoomed by `scale`.

    Shrinking is done once with area averaging (no aliasing, cached); the remaining zoom and the offset are one
    affine warp. Returns float32 HxW (grey texture) or HxWx3 (colour texture)."""
    tex = T.get(tex_id)
    th, tw = tex.shape[:2]
    if fit == "stretch":
        fx, fy = w / tw * scale, h / th * scale
    elif fit == "tile":
        fx = fy = unit(h, w) * scale
    else:
        fx = fy = max(w / tw, h / th) * scale
    sw, sh = max(1, int(round(tw * min(fx, 1.0)))), max(1, int(round(th * min(fy, 1.0))))
    src = _prescaled(tex_id, tex, sw, sh) if (sw, sh) != (tw, th) else tex
    gx, gy = fx * tw / sw, fy * th / sh  # zoom still left after the pre-scale (>= 1)
    if fit == "tile":
        cx, cy = 0.0, 0.0
    else:
        cx, cy = sw / 2.0, sh / 2.0
    ox, oy = offset[0] * sw, offset[1] * sh
    # destination pixel (x, y) -> source (x - w/2) / gx + cx + ox  (centred for fill / stretch)
    x0 = -w / 2.0 / gx + cx + ox if fit != "tile" else ox
    y0 = -h / 2.0 / gy + cy + oy if fit != "tile" else oy
    m = np.array([[1.0 / gx, 0.0, x0 + 0.5 / gx - 0.5], [0.0, 1.0 / gy, y0 + 0.5 / gy - 0.5]], np.float32)
    border = cv2.BORDER_WRAP if fit == "tile" else cv2.BORDER_REFLECT_101
    if abs(gx - 1.0) < 1e-6 and abs(gy - 1.0) < 1e-6 and ox == 0 and oy == 0 and fit != "tile" \
            and sw >= w and sh >= h:
        x, y = int(round(x0)), int(round(y0))
        return np.ascontiguousarray(src[y:y + h, x:x + w])
    return cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border)


def channel_of(img, channel):
    if img.ndim == 2:
        return img
    if channel == "red":
        return img[:, :, 0]
    if channel == "green":
        return img[:, :, 1]
    if channel == "blue":
        return img[:, :, 2]
    return img @ LUMA


def ramp_active(r):
    """False when the ramp is 1 for every value (nothing to weigh)."""
    a, b, c, d, lo, hi = r
    return not ((lo >= 1.0 or (a <= 0.0 and b <= 0.0)) and (hi >= 1.0 or (c >= 1.0 and d >= 1.0)))


def ramp_weight(x, r):
    """Blend-if ramp: `lo` below a, rises to 1 at b, stays 1 to c, falls to `hi` at d and stays there.
    Plain-numpy reference for the kernel; None when the ramp covers everything."""
    a, b, c, d, lo, hi = r
    if not ramp_active(r):
        return None
    up = np.clip((x - a) / (b - a), 0.0, 1.0) if b > a else (x >= a).astype(np.float32)
    down = np.clip((d - x) / (d - c), 0.0, 1.0) if d > c else (x <= d).astype(np.float32)
    up = lo + (1.0 - lo) * up
    down = hi + (1.0 - hi) * down
    return (up * down).astype(np.float32)


def mix_layer(img, lay, frame=0):
    h, w = img.shape[:2]
    tex = place_texture(lay["texture"], w, h, lay["fit"], lay["scale"], boil_offset(lay["boil"], frame))
    tex = tex[:, :, None] if tex.ndim == 2 else tex
    out = np.empty_like(img)
    _k_mix(img, tex, MODE_ID[lay["blend"]], np.float32(lay["opacity"]), CHANNEL_ID[lay["channel"]],
           np.array(lay["ramp_top"], np.float32), ramp_active(lay["ramp_top"]),
           np.array(lay["ramp_under"], np.float32), ramp_active(lay["ramp_under"]), out)
    return out


def _rgb(hexc):
    return np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def gradient_lut(stops, n=LUT_SIZE):
    """n x 3 colours across 0..1. Between two stops, `mid` is where the colours are half-and-half."""
    stops = sorted(stops, key=lambda s: s["pos"])
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    cols = np.array([_rgb(s["color"]) for s in stops], np.float32)
    pos = np.array([s["pos"] for s in stops], np.float32)
    lut = np.empty((n, 3), np.float32)
    lut[x <= pos[0]] = cols[0]
    lut[x >= pos[-1]] = cols[-1]
    for i in range(len(stops) - 1):
        p0, p1 = pos[i], pos[i + 1]
        sel = (x > p0) & (x < p1) if p1 > p0 else np.zeros_like(x, bool)
        if not sel.any():
            continue
        u = (x[sel] - p0) / (p1 - p0)
        mid = min(max(stops[i]["mid"], 0.05), 0.95)
        u = u ** (np.log(0.5) / np.log(mid))  # u = mid -> 0.5
        lut[sel] = cols[i] + (cols[i + 1] - cols[i]) * u[:, None]
    return lut


_LUT_CACHE = {}


def lut_for(stops):
    key = tuple((s["pos"], s["color"], s["mid"]) for s in stops)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        if len(_LUT_CACHE) > 32:
            _LUT_CACHE.clear()
        lut = _LUT_CACHE[key] = gradient_lut(stops)
    return lut


def gradient_layer(img, lay):
    r, shift = lay["repeat"], lay["shift"]
    out = np.empty_like(img)
    _k_gradient(img, lut_for(lay["stops"]), np.float32(r), np.float32(shift), r != 1 or shift != 0.0,
                MODE_ID[lay["blend"]], np.float32(lay["opacity"]), out)
    return out


def apply_layer(img, lay, frame=0):
    if not lay.get("visible", True) or lay.get("opacity", 1.0) <= 0.0:
        return img
    if lay["type"] == "blur":
        return blur_layer(img, lay)
    if lay["type"] == "mix":
        return mix_layer(img, lay, frame)
    if lay["type"] == "gradient":
        return gradient_layer(img, lay)
    return img


def render(rgb, params, frame=0, clean=False):
    """Run the layer stack over an HxWx3 uint8 RGB picture. `frame` drives the texture boil on videos."""
    p = params if clean else normalize(params)
    img = rgb.astype(np.float32) * np.float32(1.0 / 255.0)
    for lay in p["layers"]:
        img = apply_layer(img, lay, frame)
    out = np.empty(img.shape, np.uint8)
    _k_to_u8(np.ascontiguousarray(img), out)
    return out


# --------------------------------------------------------------------------- randomize

def _hsv_hex(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0), 1), min(max(v, 0), 1))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def random_stops(rng):
    """A random but readable gradient: a dark end, two to four colours, a light end (or flipped)."""
    n = rng.randint(2, 4)
    h0 = rng.random()
    step = rng.choice([0.05, 0.08, 0.12, 0.2, 0.33, 0.5]) * rng.choice([-1, 1])
    cols = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        cols.append(_hsv_hex(h0 + step * i, rng.uniform(0.55, 1.0), 0.35 + 0.65 * t))
    dark = rng.choice(["#000000", _hsv_hex(h0 - step, 0.8, 0.12)])
    light = rng.choice(["#ffffff", _hsv_hex(h0 + step * n, 0.25, 1.0)])
    cols = [dark] + cols + [light]
    if rng.random() < 0.2:
        cols.reverse()
    k = len(cols)
    pos = sorted([0.0, 1.0] + [rng.uniform(0.08, 0.92) for _ in range(k - 2)])
    return [{"pos": round(p, 3), "color": c, "mid": round(rng.uniform(0.35, 0.65), 2)} for p, c in zip(pos, cols)]


def randomize_layer(lay, seed=None):
    """New random settings for one layer (keeps its type, id and visibility)."""
    rng = random.Random(seed)
    out = dict(lay)
    if lay["type"] == "blur":
        out.update(strength=round(rng.uniform(2, 40), 2), opacity=round(rng.uniform(0.15, 0.8), 2),
                   aspect=round(rng.uniform(-0.7, 0.7), 2) if rng.random() < 0.3 else 0.0,
                   blend=rng.choice(["normal", "normal", "screen", "soft-light", "overlay", "lighten"]))
    elif lay["type"] == "mix":
        out.update(texture=rng.choice(list(T.BUILTIN)), fit=rng.choice(["fill", "fill", "fill", "tile"]),
                   scale=round(rng.uniform(0.8, 2.6), 2), opacity=round(rng.uniform(0.6, 1.0), 2),
                   blend=rng.choice(["hard-light", "hard-light", "overlay", "soft-light", "linear-light",
                                     "vivid-light"]))
    elif lay["type"] == "gradient":
        out.update(stops=random_stops(rng))
    return normalize_layer(out)


def warm_up():
    """First call of each OpenCV / numpy path costs a little; do it while the window opens."""
    try:
        from presets import PRESETS
        img = np.full((96, 96, 3), 128, np.uint8)
        render(img, PRESETS[0]["params"])
    except Exception:
        pass
