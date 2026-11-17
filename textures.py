"""Spray textures: the built-in ones are generated once and cached as PNG; your own texture files load too.

A spray texture is mid-grey (0.5, neutral under Hard light / Overlay) with dark and light droplets of many sizes.
The droplets come from thresholded smooth noise: smoothing white noise by sigma gives blobs about that wide, and
the threshold sets how many of them pass. Low-frequency clouds vary the density like a real spray can.
"""
import os
import threading

import cv2
import numpy as np
from scipy.special import ndtri

SIZE = 4096  # built-in textures are square; droplets are drawn for this size and scale with the picture
VERSION = 6  # bump to regenerate the cached PNGs after changing a recipe

# mottle: (sigma px at SIZE, amplitude, sharpness, light): the dense spray mist, blobs that are either a bit darker
#         or a bit lighter than grey, with crisp edges (tanh of smooth noise); `light` scales the lighter blobs
# drops:  (sigma px at SIZE, coverage 0..1, strength 0..1, sign -1 dark / +1 light): sparse solid droplets
BUILTIN = {
    "ink-spray": {
        "label": "Ink Spray", "seed": 11, "grain": 0.03, "cloud": 0.5, "cloud_px": 900,
        "mottle": [(2.2, 0.17, 3.0, 0.6), (1.0, 0.05, 2.0, 0.6)],
        "drops": [(1.6, 0.03, 0.9, -1), (2.8, 0.01, 1.0, -1), (5.0, 0.003, 1.0, -1), (1.4, 0.02, 0.7, 1)],
    },
    "fine-mist": {
        "label": "Fine Mist", "seed": 23, "grain": 0.03, "cloud": 0.35, "cloud_px": 700,
        "mottle": [(1.0, 0.16, 3.0, 0.7), (0.6, 0.06, 2.0, 0.7)],
        "drops": [(0.9, 0.04, 0.85, -1), (1.6, 0.01, 0.95, -1), (0.9, 0.03, 0.7, 1)],
    },
    "coarse-splatter": {
        "label": "Coarse Splatter", "seed": 37, "grain": 0.03, "cloud": 0.7, "cloud_px": 1200,
        "mottle": [(2.6, 0.12, 2.5, 0.8), (1.2, 0.06, 2.0, 0.8)],
        "drops": [(2.0, 0.05, 0.9, -1), (4.0, 0.025, 1.0, -1), (8.0, 0.008, 1.0, -1), (14.0, 0.0015, 1.0, -1),
                  (2.0, 0.03, 0.85, 1), (4.0, 0.012, 1.0, 1)],
    },
    "stencil-fade": {
        "label": "Stencil Fade", "seed": 51, "grain": 0.03, "cloud": 0.95, "cloud_px": 1600,
        "mottle": [(1.5, 0.2, 3.5, 0.7)],
        "drops": [(1.2, 0.06, 0.95, -1), (2.4, 0.02, 1.0, -1), (1.2, 0.03, 0.8, 1)],
    },
    "grit": {
        "label": "Grit", "seed": 67, "grain": 0.12, "cloud": 0.25, "cloud_px": 500,
        "mottle": [(0.7, 0.14, 2.0, 1.0)],
        "drops": [(0.8, 0.08, 0.7, -1), (0.8, 0.08, 0.7, 1), (1.6, 0.02, 0.9, -1)],
    },
}
DEFAULT = "ink-spray"
TEXTURE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_LOCK = threading.Lock()
_MEM = {}  # texture id -> float32 HxW (grey) or HxWx3 (colour) in 0..1
_BUILD_LOCKS = {}  # built-in name -> lock held while that texture is generated and saved


def cache_dir():
    base = os.environ.get("SPRAYLAB_DATA") or os.path.join(os.environ.get("LOCALAPPDATA", "."), "SprayLab")
    d = os.path.join(base, "textures")
    os.makedirs(d, exist_ok=True)
    return d


def _clouds(rng, size, px):
    """Smooth 0..1 noise with features about `px` wide (two octaves)."""
    out = np.zeros((size, size), np.float32)
    for k, amp in ((1, 0.7), (3, 0.3)):
        n = max(2, int(round(size / px * k)) + 2)
        g = rng.random((n, n), dtype=np.float32)
        out += amp * cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return out / max(float(out.max()), 1e-6)


def _droplets(rng, size, sigma, coverage, density):
    """0..1 mask of droplets about 2*sigma wide covering `coverage` of the area, more where density is high.

    `density` is a small smooth map; the threshold is worked out on it and scaled up (it varies slowly)."""
    n = rng.standard_normal((size, size), dtype=np.float32)
    f = cv2.GaussianBlur(n, (0, 0), sigma)
    f *= 1.0 / max(float(f[::7, ::7].std()), 1e-6)
    p = np.clip(coverage * density, 1e-5, 0.95)
    thr = cv2.resize(ndtri(1.0 - p).astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    soft = 1.0 / 0.35  # droplet edge softness (0.35 std): soft rims, solid cores
    f -= thr
    f *= soft
    return np.clip(f, 0.0, 1.0, out=f)


def generate(name, size=SIZE):
    """Build a built-in texture: float32 size x size, 0..1, mean near 0.5."""
    r = BUILTIN[name]
    rng = np.random.default_rng(r["seed"])
    low = max(16, size // 16)
    cl = _clouds(rng, low, r["cloud_px"] * low / SIZE)
    density = (1.0 - r["cloud"]) + r["cloud"] * 2.0 * cl  # mean ~1, swings with the clouds
    tex = cv2.GaussianBlur(rng.standard_normal((size, size), dtype=np.float32), (0, 0), 0.7)
    tex *= r["grain"]
    tex += 0.5
    amp_map = cv2.resize(np.sqrt(density).astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    for sigma, amp, sharp, light in r.get("mottle", []):
        f = cv2.GaussianBlur(rng.standard_normal((size, size), dtype=np.float32), (0, 0), sigma * size / SIZE)
        f *= sharp / max(float(f[::7, ::7].std()), 1e-6)
        np.tanh(f, out=f)
        f[f > 0] *= light
        f *= amp_map
        tex += amp * f
    for sigma, cov, strength, sign in r["drops"]:
        m = _droplets(rng, size, sigma * size / SIZE, cov, density)
        m *= strength
        target = 0.03 if sign < 0 else 0.97
        tex += m * (target - tex)  # tex*(1-m) + target*m
    return np.clip(tex, 0.0, 1.0, out=tex)


def builtin_path(name):
    return os.path.join(cache_dir(), f"{name}-v{VERSION}-{SIZE}.png")


def load_builtin(name):
    # One builder per texture: on a fresh install the background warm-up and a first preview or export can ask for
    # the same texture at once, and two threads writing the same file fails on Windows ("used by another process").
    with _LOCK:
        build_lock = _BUILD_LOCKS.setdefault(name, threading.Lock())
    with build_lock:
        path = builtin_path(name)
        img = None
        if os.path.isfile(path):
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (SIZE, SIZE):
            g = generate(name)
            img = np.round(g * 65535).astype(np.uint16)
            ok, buf = cv2.imencode(".png", img)
            if ok:
                part = f"{path}.{os.getpid()}-{threading.get_ident()}.part.png"
                buf.tofile(part)
                os.replace(part, path)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return (img.astype(np.float32) / scale)


def load_file(path):
    """A texture image of yours, as float32 0..1 (grey stays one channel, colour stays RGB)."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Could not read this texture: " + os.path.basename(path))
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    img = img.astype(np.float32) / scale
    if img.ndim == 3:
        if img.shape[2] == 4:  # transparent areas count as neutral grey
            a = img[:, :, 3:4]
            img = img[:, :, :3] * a + 0.5 * (1.0 - a)
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        if float(np.abs(img - img.mean(axis=2, keepdims=True)).max()) < 2.0 / 255:
            img = np.ascontiguousarray(img[:, :, 0])
    return img


def get(tex_id):
    """Texture by id: a built-in name, or 'file:<path>' for one of yours. Cached in memory."""
    tex_id = tex_id or DEFAULT
    with _LOCK:
        hit = _MEM.get(tex_id)
    if hit is not None:
        return hit
    if tex_id.startswith("file:"):
        img = load_file(tex_id[5:])
    elif tex_id in BUILTIN:
        img = load_builtin(tex_id)
    else:
        img = load_builtin(DEFAULT)
    with _LOCK:
        while len(_MEM) >= 4:
            _MEM.pop(next(iter(_MEM)))
        _MEM[tex_id] = img
    return img


def thumbnail(tex_id, side=160):
    """A small crop at roughly the scale the texture shows on a picture, for the UI."""
    t = get(tex_id)
    h, w = t.shape[:2]
    c = min(h, w) // 4
    crop = t[(h - c) // 2:(h + c) // 2, (w - c) // 2:(w + c) // 2]
    small = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    small = (small - 0.5) * 1.8 + 0.5  # shrinking greys the droplets out; push them back so the swatch reads
    small = np.clip(small * 255 + 0.5, 0, 255).astype(np.uint8)
    return small if small.ndim == 3 else cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)


def warm_up():
    """Generate every built-in texture once in the background so the first switch is instant."""
    for name in BUILTIN:
        if not os.path.isfile(builtin_path(name)):
            try:
                load_builtin(name)
            except Exception:
                pass
