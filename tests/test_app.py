"""Server + ffmpeg pipeline tests on a synthetic clip with audio and accented file names.

Run from the app folder: python -m unittest discover -s tests
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="spraylab-test-")
os.environ["SPRAYLAB_DATA"] = os.path.join(TMP, "data")
os.environ["SPRAYLAB_OUT"] = os.path.join(TMP, "out")
os.environ["SPRAYLAB_STAY"] = "1"
sys.path.insert(0, HERE)

from fastapi.testclient import TestClient  # noqa: E402

import app as A  # noqa: E402
import effect as E  # noqa: E402
import media as M  # noqa: E402
import presets as P  # noqa: E402

CLIP = os.path.join(TMP, "Clipé ü test.mp4")
PHOTO = os.path.join(TMP, "Fotó 写真.png")


def setUpModule():
    subprocess.run([M.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "1.5", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", CLIP], check=True)
    yy, xx = np.mgrid[0:300, 0:200]
    face = np.dstack([128 + 100 * np.sin(xx / 17.0), 128 + 90 * np.cos(yy / 23.0), 100 + 0 * xx])
    M.write_image(PHOTO, np.clip(face, 0, 255).astype(np.uint8))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def decode_jpeg(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def wait_job(c, timeout=180):
    t = time.time()
    while time.time() - t < timeout:
        j = c.get("/api/job").json()
        if j["state"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("export did not finish")


def probe_streams(path):
    r = subprocess.run([M.ffmpeg_bin("ffprobe"), "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,nb_read_frames,r_frame_rate,width,height", "-of", "json", path],
                       capture_output=True, text=True, encoding="utf-8")
    import json
    return json.loads(r.stdout)["streams"]


def frame_of(path, i, w, h, fps):
    r = subprocess.run([M.ffmpeg_bin(), "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{i})", "-vsync", "0",
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    return np.frombuffer(r.stdout[:w * h * 3], np.uint8).reshape(h, w, 3)


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TestClient(A.app)

    def load(self, path):
        r = self.c.post("/api/load", json={"path": path})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["media"]

    def export(self, **body):
        body.setdefault("params", P.get("Spray"))
        r = self.c.post("/api/export", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        j = wait_job(self.c)
        self.assertEqual(j["state"], "done", j)
        return j

    def test_ping_and_state(self):
        self.assertEqual(self.c.get("/api/ping").json()["app"], "spray-lab")
        st = self.c.get("/api/state").json()
        self.assertEqual([p["name"] for p in st["presets"]][0], "Spray")
        self.assertIn("hard-light", st["blend_modes"])
        self.assertEqual(st["out_dir"], os.environ["SPRAYLAB_OUT"])
        self.assertTrue(any(t["id"] == "ink-spray" for t in st["textures"]))
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)
        self.assertEqual(self.c.get("/static/../app.py").status_code, 404)

    def test_unsupported_file(self):
        bad = os.path.join(TMP, "notes.txt")
        with open(bad, "w") as f:
            f.write("hi")
        r = self.c.post("/api/load", json={"path": bad})
        self.assertEqual(r.status_code, 400)
        self.assertIn("opens photos", r.json()["detail"])

    def test_photo_preview(self):
        m = self.load(PHOTO)
        self.assertEqual((m["kind"], m["width"], m["height"]), ("image", 200, 300))
        r = self.c.post("/api/preview", json={"side": 150, "params": P.get("Spray")})
        self.assertEqual(r.status_code, 200)
        img = decode_jpeg(r.content)
        self.assertEqual(img.shape, (150, 100, 3))
        orig = decode_jpeg(self.c.post("/api/preview", json={"side": 150, "original": True}).content)
        self.assertGreater(float(np.abs(img.astype(int) - orig.astype(int)).mean()), 10)
        self.assertEqual(self.c.post("/api/thumb", json={"params": P.get("Cobalt"), "side": 90}).status_code, 200)

    def test_layer_endpoints(self):
        lay = self.c.post("/api/layer/new", json={"type": "gradient"}).json()
        self.assertEqual(lay["type"], "gradient")
        self.assertEqual(self.c.post("/api/layer/new", json={"type": "nope"}).status_code, 400)
        r = self.c.post("/api/layer/randomize", json={"layer": lay, "seed": 3}).json()
        self.assertEqual(r["id"], lay["id"])
        self.assertNotEqual(r["stops"], lay["stops"])
        clean = self.c.post("/api/normalize", json={"params": {"layers": [{"type": "mix", "ramp_top": [0, 0, 1, 1]},
                                                                           {"type": "bogus"}]}}).json()
        self.assertEqual(len(clean["layers"]), 1)
        self.assertEqual(len(clean["layers"][0]["ramp_top"]), 6)

    def test_textures(self):
        r = self.c.get("/api/texture-thumb", params={"id": "grit", "side": 64})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(decode_jpeg(r.content).shape, (64, 64, 3))
        rng = np.random.default_rng(4)
        tex = (rng.random((256, 256)) * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", tex)
        r = self.c.post("/api/texture/upload", params={"name": "my grain é.png"}, content=buf.tobytes())
        self.assertEqual(r.status_code, 200, r.text)
        tid = r.json()["id"]
        self.assertTrue(tid.startswith("file:"))
        self.assertTrue(any(t["id"] == tid for t in r.json()["textures"]))
        self.load(PHOTO)
        p = P.get("Spray")
        p["layers"][1]["texture"] = tid
        self.assertEqual(self.c.post("/api/preview", json={"side": 120, "params": p}).status_code, 200)
        bad = self.c.post("/api/texture/upload", params={"name": "broken.png"}, content=b"not an image")
        self.assertEqual(bad.status_code, 400)

    def test_my_presets(self):
        r = self.c.post("/api/my-presets", json={"name": "Mine", "params": P.get("Acid")}).json()
        self.assertEqual(r["my_presets"][0]["name"], "Mine")
        self.assertEqual(self.c.get("/api/state").json()["my_presets"][0]["name"], "Mine")
        self.assertEqual(self.c.post("/api/my-presets", json={"name": " "}).status_code, 400)
        r = self.c.post("/api/my-presets/delete", json={"name": "Mine"}).json()
        self.assertFalse(any(p["name"] == "Mine" for p in r["my_presets"]))

    def test_export_photo_png_and_jpg(self):
        self.load(PHOTO)
        p = P.get("Toxic Teal")
        j = self.export(params=p, format="png")
        self.assertTrue(j["out"].endswith(".png"))
        out = M.read_image(j["out"])
        np.testing.assert_array_equal(out, E.render(M.read_image(PHOTO), p))  # the file is exactly the render
        j = self.export(params=p, format="jpg")
        self.assertTrue(j["out"].endswith(".jpg") and os.path.getsize(j["out"]) > 1000)
        j2 = self.export(params=p, format="png")
        self.assertNotEqual(j2["out"], j["out"])  # never overwrites

    def test_export_photo_clip_boils(self):
        self.load(PHOTO)
        p = P.get("Spray")
        p["layers"][1]["boil"] = 2
        j = self.export(params=p, format="mp4", fps=24, seconds=1)
        self.assertEqual(j["report"]["problems"], [], j["report"])
        self.assertEqual(j["report"]["frames"], 24)
        a, b, c = (frame_of(j["out"], i, 200, 300, 24).astype(int) for i in (0, 1, 2))
        self.assertLess(float(np.abs(a - b).mean()), 3.0)  # same texture spot for 2 frames
        self.assertGreater(float(np.abs(b - c).mean()), 8.0)  # then it moves

    def test_export_video_with_audio(self):
        m = self.load(CLIP)
        self.assertEqual((m["kind"], m["frames"], m["has_audio"]), ("video", 45, True))
        p = P.get("Spray")
        j = self.export(params=p)
        rep = j["report"]
        self.assertEqual(rep["problems"], [], rep)
        self.assertEqual((rep["width"], rep["height"], rep["frames"]), (180, 320, 45))
        self.assertTrue(rep["audio"])
        self.assertIn("Clipé ü test - Spray Lab", os.path.basename(j["out"]))
        # an exported frame matches the preview render of that same frame, not its neighbours
        info = M.probe(CLIP)
        out = frame_of(j["out"], 20, 180, 320, 30).astype(np.float32)
        diffs = {k: float(np.abs(out - E.render(M.read_frame(CLIP, info, k, 180, 320), p, k)).mean())
                 for k in (14, 20, 26)}
        self.assertLess(diffs[20], 14.0, diffs)
        self.assertLess(diffs[20], min(diffs[14], diffs[26]), diffs)

    def test_export_video_resampled_to_24fps(self):
        self.load(CLIP)
        j = self.export(fps=24, max_side=1280)
        rep = j["report"]
        self.assertEqual(rep["problems"], [], rep)
        self.assertTrue(rep["audio"])
        streams = probe_streams(j["out"])
        v = next(s for s in streams if s["codec_type"] == "video")
        self.assertEqual(v["r_frame_rate"], "24/1")
        self.assertEqual(int(v["nb_read_frames"]), 36)  # 1.5 s at 24 fps

    def test_preview_frames_follow_the_clip(self):
        self.load(CLIP)
        a = decode_jpeg(self.c.post("/api/preview", json={"frame": 3, "side": 320, "original": True}).content)
        b = decode_jpeg(self.c.post("/api/preview", json={"frame": 30, "side": 320, "original": True}).content)
        c = decode_jpeg(self.c.post("/api/preview", json={"frame": 4, "side": 320, "original": True}).content)
        self.assertGreater(float(np.abs(a.astype(int) - b.astype(int)).mean()), 5)  # the clip moves
        info = M.probe(CLIP)
        # the streamed frame is frame 4 itself (closest to it, not to its neighbours; JPEG adds a little noise)
        d = {k: float(np.abs(c.astype(int) - cv2.cvtColor(M.read_frame(CLIP, info, k, 180, 320),
                                                           cv2.COLOR_RGB2BGR).astype(int)).mean()) for k in (3, 4, 5)}
        self.assertLess(d[4], 6, d)
        self.assertLess(d[4], min(d[3], d[5]), d)

    def test_preview_audio(self):
        m = self.load(CLIP)
        r = self.c.get("/api/audio", params={"key": m["key"]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "audio/mp4")
        path = os.path.join(TMP, "preview sound.m4a")
        with open(path, "wb") as f:
            f.write(r.content)
        self.assertEqual([s["codec_type"] for s in probe_streams(path)], ["audio"])
        dur = subprocess.run([M.ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of",
                              "csv=p=0", path], capture_output=True, text=True).stdout
        self.assertAlmostEqual(float(dur), 1.5, delta=0.1)  # as long as the clip
        pcm = subprocess.run([M.ffmpeg_bin(), "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-"],
                             capture_output=True).stdout
        self.assertGreater(float(np.sqrt((np.frombuffer(pcm, np.int16).astype(float) ** 2).mean())), 1000)  # the tone
        part = self.c.get("/api/audio", params={"key": m["key"]}, headers={"Range": "bytes=0-99"})
        self.assertEqual((part.status_code, len(part.content)), (206, 100))  # the window can seek in it
        self.assertEqual(self.c.get("/api/audio", params={"key": "an-old-clip"}).status_code, 404)
        photo = self.load(PHOTO)
        self.assertEqual(self.c.get("/api/audio", params={"key": photo["key"]}).status_code, 404)

    def test_cancel(self):
        self.load(CLIP)
        r = self.c.post("/api/export", json={"params": P.get("Spray")})
        self.assertEqual(r.status_code, 200)
        self.c.post("/api/cancel")
        j = wait_job(self.c)
        self.assertIn(j["state"], ("cancelled", "done"))
        if j["state"] == "cancelled":
            leftovers = [f for f in os.listdir(os.environ["SPRAYLAB_OUT"]) if ".part" in f]
            self.assertEqual(leftovers, [])

    def test_upload(self):
        with open(PHOTO, "rb") as f:
            r = self.c.post("/api/upload", params={"name": "dropped é.png"}, content=f.read())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["media"]["name"], "dropped é.png")


if __name__ == "__main__":
    unittest.main()
