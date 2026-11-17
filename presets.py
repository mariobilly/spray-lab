"""Ready-made looks. "Spray" is the one built in the Effect.app reel; the others follow the colour variations the
reel flips through while setting the gradient stops, plus a few of our own."""
import copy

import effect as E


def _stops(*pairs, mid=0.5):
    return [{"pos": p, "color": c, "mid": mid} for c, p in pairs]


# the reel pulls the underlying ramp down at the bright end: whites and highlights take less texture
SOFT_WHITES = [0.0, 0.0, 0.55, 1.0, 0.0, 0.12]


def _look(stops, blur=(14.0, 0.0, 0.34), texture="ink-spray", scale=1.51, mode="hard-light", tex_opacity=1.0,
          gradient=True, ramp_under=SOFT_WHITES):
    layers = [
        {"type": "blur", "strength": blur[0], "aspect": blur[1], "opacity": blur[2], "blend": "normal"},
        {"type": "mix", "texture": texture, "fit": "fill", "scale": scale, "channel": "gray",
         "opacity": tex_opacity, "blend": mode, "ramp_under": list(ramp_under)},
    ]
    if gradient:
        layers.append({"type": "gradient", "stops": stops, "shift": 0.0, "repeat": 1, "opacity": 1.0,
                       "blend": "normal"})
    return E.normalize({"layers": layers})


PRESETS = [
    {"name": "Spray", "note": "The signature look: black, deep red, orange, yellow, white",
     "params": _look(_stops(*E.FIRE))},
    {"name": "Bubblegum", "note": "Magenta into pink and pale rose",
     "params": _look(_stops(("#8a00b8", 0.0), ("#d61f8c", 0.38), ("#f27aa8", 0.62), ("#fde8f0", 1.0)))},
    {"name": "Toxic Teal", "note": "Black, teal green, cream",
     "params": _look(_stops(("#000000", 0.0), ("#0b1f1a", 0.18), ("#2e8f78", 0.46), ("#a9d3bd", 0.72),
                            ("#f6f2e2", 1.0)))},
    {"name": "Neon Vapor", "note": "Black, teal, electric blue, magenta",
     "params": _look(_stops(("#000000", 0.0), ("#071512", 0.16), ("#2f9a86", 0.42), ("#5b74ff", 0.68),
                            ("#ff1cf0", 1.0)))},
    {"name": "Thermal", "note": "Heat-camera blues into violet and hot pink",
     "params": _look(_stops(("#12005e", 0.0), ("#2b3bff", 0.36), ("#a63dff", 0.64), ("#ff2d6f", 0.86),
                            ("#ffd1dc", 1.0)))},
    {"name": "Acid", "note": "Olive shadows, red, orange, acid yellow",
     "params": _look(_stops(("#1f2a00", 0.0), ("#b5121b", 0.34), ("#ff7a00", 0.58), ("#ffe500", 0.82),
                            ("#f6ffb0", 1.0)), texture="fine-mist")},
    {"name": "Cobalt", "note": "Navy, cobalt, sky blue, white",
     "params": _look(_stops(("#00040f", 0.0), ("#00227a", 0.3), ("#1f5cff", 0.58), ("#8cc4ff", 0.82),
                            ("#ffffff", 1.0)))},
    {"name": "Mono Stencil", "note": "Hard black-and-white stencil with patchy spray",
     "params": _look(_stops(("#000000", 0.0), ("#0d0d0d", 0.4), ("#f2f2f2", 0.6), ("#ffffff", 1.0)),
                     texture="stencil-fade", blur=(10.0, 0.0, 0.5))},
    {"name": "Ink Only", "note": "Blur and spray texture, original colours (no gradient map)",
     "params": _look([], gradient=False)},
]


def get(name):
    for p in PRESETS:
        if p["name"].lower() == str(name).lower():
            return copy.deepcopy(p["params"])
    return copy.deepcopy(PRESETS[0]["params"])
