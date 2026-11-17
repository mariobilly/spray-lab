"""Draws the Spray Lab icon (icon.png + multi-size icon.ico): a spray can firing a burst of fire-coloured spray.

The burst is made by the app's own effect (Spray preset on a soft glow), so it has the real speckle.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import effect as E  # noqa: E402
import presets as P  # noqa: E402

S = 512  # final icon size
R = 112  # tile corner radius
TILT = 34  # degrees the can leans left; its side nozzle then points up and to the right
K = 4  # the can is drawn this many times bigger, then shrunk, for smooth edges


def burst(nx, ny):
    """The spray cloud from nozzle (nx, ny): a soft widening cone through the Spray look. RGB uint8 S x S."""
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ang = math.radians(-TILT)
    dx, dy = math.cos(ang), math.sin(ang)
    px, py = xx - nx, yy - ny
    along = px * dx + py * dy
    across = -px * dy + py * dx
    t = np.clip(along / (S * 0.7), 0, 1)
    width = S * (0.03 + 0.3 * t)
    cone = np.exp(-(across / width) ** 2 * 1.5) * np.clip(along / (S * 0.04), 0, 1)
    cx, cy = nx + dx * S * 0.38, ny + dy * S * 0.38
    core = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (S * 0.17) ** 2)))
    lum = np.clip(cone * (0.5 + 0.62 * core) + 0.08 * core, 0, 1) * np.clip(1.2 - t * 0.55, 0, 1)
    img = np.repeat((lum * 255)[:, :, None], 3, 2).astype(np.uint8)
    p = P.get("Spray")
    p["layers"][0].update(opacity=0.0)
    # big droplets for a small icon; no spray on the black background
    p["layers"][1].update(texture="coarse-splatter", scale=4.0, ramp_under=[0.03, 0.18, 1, 1, 0, 0])
    return E.render(img, p)


def can():
    """The can, upright, K times the icon size. Returns the image and its nozzle tip (x, y)."""
    W = S * K
    im = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = W * 0.5
    bw, top, bottom = W * 0.2, W * 0.42, W * 0.86
    dome = bw * 0.2
    body = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    x0, x1 = cx - bw / 2, cx + bw / 2
    for i in range(int(bw)):  # cylinder shading: dark edges, bright stripe left of centre
        u = i / bw
        v = int(120 + 128 * math.sin(math.pi * (0.08 + 0.84 * u)) ** 2.2 * (1.0 - 0.25 * u))
        bd.line([(x0 + i, top - dome), (x0 + i, bottom)], fill=(v, v, min(255, v + 6), 255))
    mask = Image.new("L", (W, W), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((x0, top, x1, bottom), radius=bw * 0.12, fill=255)
    md.ellipse((x0, top - dome, x1, top + dome), fill=255)
    im.paste(body, (0, 0), mask)
    h = bottom - top
    d.rectangle((x0, top + h * 0.34, x1, top + h * 0.6), fill=(255, 90, 31, 255))
    d.rectangle((x0, top + h * 0.6, x1, top + h * 0.66), fill=(255, 176, 0, 255))
    cw = bw * 0.38  # valve stem and actuator cap with a nozzle on its right side
    d.rounded_rectangle((cx - cw / 2, top - dome - bw * 0.2, cx + cw / 2, top - dome * 0.4), radius=cw * 0.16,
                        fill=(28, 28, 28, 255))
    cap = (cx - cw * 0.62, top - dome - bw * 0.42, cx + cw * 0.62, top - dome - bw * 0.16)
    d.rounded_rectangle(cap, radius=cw * 0.18, fill=(255, 90, 31, 255))
    tip = (cap[2] + bw * 0.06, (cap[1] + cap[3]) / 2)
    d.rectangle((cap[2] - 2, tip[1] - bw * 0.035, tip[0], tip[1] + bw * 0.035), fill=(28, 28, 28, 255))
    return im, tip


def draw():
    c, tip = can()
    W = c.width
    centre = (W * 0.5, W * 0.62)
    c = c.rotate(TILT, resample=Image.BICUBIC, center=centre)  # counter-clockwise: leans left, nozzle up-right
    a = math.radians(TILT)
    dx, dy = tip[0] - centre[0], tip[1] - centre[1]
    tx = centre[0] + dx * math.cos(a) + dy * math.sin(a)
    ty = centre[1] - dx * math.sin(a) + dy * math.cos(a)
    c = c.resize((S, S), Image.LANCZOS)
    off = (-int(S * 0.13), int(S * 0.1))  # the can sits low and left
    nx, ny = tx / K + off[0], ty / K + off[1]

    tile = Image.new("RGBA", (S, S), (12, 10, 9, 255))
    tile = Image.alpha_composite(tile, Image.fromarray(burst(nx + S * 0.01, ny)).convert("RGBA"))
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow.putalpha(c.getchannel("A").filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(v * 0.75)))
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    layer.alpha_composite(shadow, (off[0] + 5, off[1] + 9))
    layer.alpha_composite(c, off)
    tile = Image.alpha_composite(tile, layer)

    mask = Image.new("L", (S * 4, S * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S * 4 - 1, S * 4 - 1), radius=R * 4, fill=255)
    mask = mask.resize((S, S), Image.LANCZOS)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(tile, (0, 0), mask)
    return out


if __name__ == "__main__":
    im = draw()
    im.save(os.path.join(HERE, "icon.png"))
    im.save(os.path.join(HERE, "icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128),
                                                     (256, 256)])
    print("icon.png and icon.ico written")
