"""Engine tests: params cleaning, the compiled kernels against the plain-numpy maths, gradient stops, textures,
size independence, texture boil, presets and randomize."""
import os
import sys
import tempfile
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("SPRAYLAB_DATA", os.path.join(tempfile.gettempdir(), "spraylab-test-data"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import effect as E  # noqa: E402
import presets as P  # noqa: E402
import textures as T  # noqa: E402


def face(h=360, w=240):
    """A synthetic portrait: light background, a shaded oval 'face' with darker 'eyes'."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w, 3), 240, np.float32)
    d = ((xx - w / 2) / (w * 0.32)) ** 2 + ((yy - h / 2) / (h * 0.36)) ** 2
    shade = 0.45 + 0.4 * (1 - yy / h) * (0.6 + 0.4 * np.cos((xx - w / 2) / w * 3))
    skin = np.stack([210 * shade, 150 * shade, 120 * shade], -1)
    img = np.where((d < 1)[:, :, None], skin, img)
    for ex in (0.38, 0.62):
        e = ((xx - w * ex) / (w * 0.07)) ** 2 + ((yy - h * 0.45) / (h * 0.025)) ** 2
        img = np.where((e < 1)[:, :, None], 30.0, img)
    return np.clip(img, 0, 255).astype(np.uint8)


class Params(unittest.TestCase):
    def test_defaults_filled_and_clamped(self):
        p = E.normalize({"layers": [{"type": "blur", "strength": 900, "opacity": -2, "blend": "nope"},
                                    {"type": "mix", "scale": "abc", "fit": "zoom", "ramp_top": [0.9, 0.1, 2, -1]},
                                    {"type": "gradient", "stops": [{"pos": 2, "color": "#FFF"}], "repeat": 99},
                                    {"type": "mystery"}, "junk"]})
        blur, mix, grad = p["layers"]
        self.assertEqual(len(p["layers"]), 3)
        self.assertEqual(blur["strength"], 100.0)
        self.assertEqual(blur["opacity"], 0.0)
        self.assertEqual(blur["blend"], "normal")
        self.assertEqual(mix["scale"], E.DEFAULTS["mix"]["scale"])
        self.assertEqual(mix["fit"], "fill")
        a, b, c, d, lo, hi = mix["ramp_top"]
        self.assertTrue(a <= b <= c <= d)
        self.assertEqual((lo, hi), (0.0, 0.0))  # a 4-value ramp gets flat ends
        self.assertEqual(len(grad["stops"]), len(E.DEFAULTS["gradient"]["stops"]))  # one stop is not a gradient
        self.assertEqual(grad["repeat"], 8)
        for lay in p["layers"]:
            self.assertTrue(lay["id"])
            self.assertTrue(lay["visible"])

    def test_colours_cleaned(self):
        g = E.normalize_layer({"type": "gradient", "stops": [{"pos": 0, "color": "#ABC"},
                                                             {"pos": 1, "color": "zzz"}]})
        self.assertEqual([s["color"] for s in g["stops"]], ["#aabbcc", "#000000"])

    def test_stops_sorted(self):
        g = E.normalize_layer({"type": "gradient", "stops": [{"pos": 0.9, "color": "#ffffff"},
                                                             {"pos": 0.1, "color": "#000000"}]})
        self.assertEqual([s["pos"] for s in g["stops"]], [0.1, 0.9])


class Kernels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.b = rng.random((40, 50, 3), dtype=np.float32)
        self.s = rng.random((40, 50, 3), dtype=np.float32)
        self.b[0, :3] = [0, 0.25, 1]  # edge values for dodge / burn / soft light
        self.s[0, :3] = [1, 0.5, 0]

    def test_every_blend_mode_matches_reference(self):
        for mode in E.BLEND_MODES:
            for op in (1.0, 0.4):
                ref = self.b + (np.broadcast_to(E.blend(mode, self.b, self.s), self.b.shape) - self.b) * op
                out = np.empty_like(self.b)
                E._k_composite(self.b, self.s, E.MODE_ID[mode], np.float32(op), out)
                np.testing.assert_allclose(out, ref, atol=2e-4, err_msg=mode)

    def test_mix_ramps_match_reference(self):
        tex = self.s[:, :, 0]
        rt = [0.1, 0.3, 0.6, 0.9, 0.2, 0.5]
        ru = [0.0, 0.0, 0.5, 1.0, 0.0, 0.3]
        w = 0.8 * E.ramp_weight(tex, rt) * E.ramp_weight(self.b @ E.LUMA, ru)  # grey texture: gray = itself
        ref = self.b + (E.blend("hard-light", self.b, tex[:, :, None]) - self.b) * w[:, :, None]
        out = np.empty_like(self.b)
        E._k_mix(self.b, np.ascontiguousarray(tex[:, :, None]), E.MODE_ID["hard-light"], np.float32(0.8),
                 E.CHANNEL_ID["gray"], np.array(rt, np.float32), True, np.array(ru, np.float32), True, out)
        np.testing.assert_allclose(out, ref, atol=2e-4)

    def test_fast_blur_matches_exact(self):
        img = cv2.resize(face(), (480, 720), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        for sx, sy in ((6, 6), (12, 12), (30, 30), (20, 8), (25, 0.0)):
            exact = cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=max(sy, 1e-3), borderType=cv2.BORDER_REFLECT_101)
            fast = E.gaussian(img, sx, sy)
            self.assertEqual(fast.shape, img.shape)
            self.assertLess(float(np.abs(fast - exact).mean()) * 255, 0.6, (sx, sy))

    def test_full_ramp_is_inactive(self):
        self.assertFalse(E.ramp_active(list(E.RAMP_ALL)))
        self.assertTrue(E.ramp_active([0, 0, 0.5, 1, 0, 0.3]))
        self.assertIsNone(E.ramp_weight(self.b[:, :, 0], list(E.RAMP_ALL)))

    def test_ramp_shape(self):
        x = np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0], np.float32)
        w = E.ramp_weight(x, [0.1, 0.2, 0.8, 0.9, 0.0, 0.5])
        np.testing.assert_allclose(w, [0, 0, 1, 1, 1, 0.5, 0.5], atol=1e-6)


class Gradient(unittest.TestCase):
    def test_lut_ends_and_midpoint(self):
        stops = [{"pos": 0.0, "color": "#000000", "mid": 0.5}, {"pos": 1.0, "color": "#ffffff", "mid": 0.5}]
        lut = E.gradient_lut(stops, 1001)
        np.testing.assert_allclose(lut[0], 0, atol=1e-6)
        np.testing.assert_allclose(lut[-1], 1, atol=1e-6)
        np.testing.assert_allclose(lut[500], 0.5, atol=1e-3)
        stops[0]["mid"] = 0.25  # half-and-half moves to a quarter of the way
        lut = E.gradient_lut(stops, 1001)
        np.testing.assert_allclose(lut[250], 0.5, atol=2e-3)

    def test_outside_stops_hold_end_colours(self):
        stops = [{"pos": 0.3, "color": "#ff0000", "mid": 0.5}, {"pos": 0.7, "color": "#0000ff", "mid": 0.5}]
        lut = E.gradient_lut(stops, 101)
        np.testing.assert_allclose(lut[10], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(lut[90], [0, 0, 1], atol=1e-6)

    def test_map_grey_ramp(self):
        grey = np.repeat(np.linspace(0, 255, 256).astype(np.uint8)[None, :], 4, 0)
        img = np.stack([grey] * 3, -1)
        bw = [{"pos": 0, "color": "#000000"}, {"pos": 1, "color": "#ffffff"}]
        out = E.render(img, {"layers": [{"type": "gradient", "stops": bw}]})
        self.assertLessEqual(int(np.abs(out.astype(int) - img.astype(int)).max()), 1)
        red = E.render(img, {"layers": [{"type": "gradient", "stops": [{"pos": 0, "color": "#000000"},
                                                                        {"pos": 1, "color": "#ff0000"}]}]})
        self.assertEqual(int(red[..., 1].max()), 0)
        self.assertGreater(int(red[0, -1, 0]), 250)

    def test_repeat_and_shift(self):
        bw = [{"pos": 0, "color": "#000000"}, {"pos": 1, "color": "#ffffff"}]
        img = np.full((2, 2, 3), 128, np.uint8)  # luma ~0.502
        twice = E.render(img, {"layers": [{"type": "gradient", "stops": bw, "repeat": 2}]})
        self.assertLess(int(twice[0, 0, 0]), 10)  # 0.502 * 2 = 1.004 -> start of the second repeat: dark
        white = np.full((2, 2, 3), 255, np.uint8)
        top = E.render(white, {"layers": [{"type": "gradient", "stops": bw, "repeat": 3}]})
        self.assertEqual(int(top[0, 0, 0]), 255)  # the very top stays the end colour, never wraps to black
        shifted = E.render(img, {"layers": [{"type": "gradient", "stops": bw, "shift": 0.25}]})
        self.assertAlmostEqual(int(shifted[0, 0, 0]), 192, delta=2)


class Textures(unittest.TestCase):
    def test_builtins_are_neutral_and_deterministic(self):
        for name in T.BUILTIN:
            a = T.generate(name, size=384)
            self.assertEqual(a.dtype, np.float32)
            self.assertTrue(0.4 < float(a.mean()) < 0.6, name)
            self.assertGreater(float(a.std()), 0.05, name)
            np.testing.assert_array_equal(a, T.generate(name, size=384))

    def test_user_texture_file(self):
        path = os.path.join(tempfile.mkdtemp(), "tex é.png")
        g = np.tile(np.array([[0, 255], [255, 0]], np.uint8), (32, 32))
        cv2.imencode(".png", g)[1].tofile(path)
        t = T.load_file(path)
        self.assertEqual(t.ndim, 2)  # grey stays grey
        col = np.zeros((16, 16, 3), np.uint8)
        col[..., 2] = 255  # BGR red
        cv2.imencode(".png", col)[1].tofile(path)
        T._MEM.clear()
        t = T.load_file(path)
        self.assertEqual(t.shape, (16, 16, 3))
        np.testing.assert_allclose(t[0, 0], [1, 0, 0])

    def test_place_texture_fits(self):
        for fit in E.FITS:
            for scale in (0.5, 1.0, 2.0):
                t = E.place_texture(T.DEFAULT, 300, 200, fit, scale)
                self.assertEqual(t.shape, (200, 300), (fit, scale))
                self.assertGreater(float(t.min()), -1e-6)
                self.assertTrue(0.35 < float(t.mean()) < 0.65, (fit, scale))  # no black borders

    def test_thumbnail(self):
        th = T.thumbnail(T.DEFAULT, 64)
        self.assertEqual(th.shape, (64, 64, 3))
        self.assertEqual(th.dtype, np.uint8)


class Render(unittest.TestCase):
    def test_empty_stack_is_identity(self):
        img = face()
        np.testing.assert_array_equal(E.render(img, {"layers": []}), img)

    def test_hidden_layers_do_nothing(self):
        img = face()
        p = P.get("Spray")
        for lay in p["layers"]:
            lay["visible"] = False
        np.testing.assert_array_equal(E.render(img, p), img)

    def test_deterministic(self):
        img = face()
        p = P.get("Spray")
        np.testing.assert_array_equal(E.render(img, p), E.render(img, p))

    def test_spray_is_warm_and_speckled(self):
        img = face()
        out = E.render(img, P.get("Spray")).astype(np.float32)
        m = out.reshape(-1, 3).mean(0)
        self.assertTrue(m[0] >= m[1] >= m[2], m)  # fire palette: red >= green >= blue
        skin = out[150:250, 90:150]
        self.assertGreater(float(skin.std()), 25.0)  # the spray grain is there

    def test_size_independent_look(self):
        """A half-size render looks like the full-size one shrunk: same colours, grain at the same scale."""
        big = cv2.resize(face(), (720, 1080), interpolation=cv2.INTER_CUBIC)
        small = cv2.resize(big, (360, 540), interpolation=cv2.INTER_AREA)
        p = P.get("Spray")
        a = cv2.resize(E.render(big, p), (90, 135), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(E.render(small, p), (90, 135), interpolation=cv2.INTER_AREA).astype(np.float32)
        self.assertLess(float(np.abs(a - b).mean()), 12.0)

    def test_boil_moves_texture(self):
        img = face()
        p = P.get("Spray")
        f0 = E.render(img, p, frame=0)
        self.assertTrue(np.array_equal(f0, E.render(img, p, frame=7)))  # boil off: still texture
        p["layers"][1]["boil"] = 2
        a, b, c = (E.render(img, p, frame=i) for i in (0, 1, 2))
        np.testing.assert_array_equal(a, b)  # same spot for 2 frames
        self.assertFalse(np.array_equal(b, c))  # then it moves

    def test_every_preset_renders(self):
        img = face(120, 90)
        for pr in P.PRESETS:
            out = E.render(img, pr["params"])
            self.assertEqual(out.shape, img.shape, pr["name"])
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(E.normalize(pr["params"]), pr["params"], pr["name"])

    def test_every_blend_mode_renders(self):
        img = face(80, 60)
        for mode in E.BLEND_MODES:
            p = P.get("Spray")
            for lay in p["layers"]:
                lay["blend"] = mode
            out = E.render(img, p)
            self.assertEqual(out.shape, img.shape, mode)

    def test_randomize(self):
        for kind in E.LAYER_TYPES:
            lay = E.new_layer(kind)
            r1, r2 = E.randomize_layer(lay, seed=5), E.randomize_layer(lay, seed=5)
            self.assertEqual(r1, r2)
            self.assertEqual(r1, E.normalize_layer(r1))
            self.assertEqual((r1["id"], r1["type"]), (lay["id"], kind))
        g = E.randomize_layer(E.new_layer("gradient"), seed=9)
        self.assertGreaterEqual(len(g["stops"]), 4)


if __name__ == "__main__":
    unittest.main()
