# Spray Lab

A spray-paint effect for your photos and videos: a stack of **Gaussian Blur → Layer Mix (spray texture) → Gradient
Map**, with presets, editable layers and MP4 export that keeps the sound.

**Everything runs on your computer. No account, no API key, nothing is uploaded.**

Inspired by the spray-paint effect shown in a reel by [Effect.app](https://www.instagram.com/reel/Db-umgRuf2n/).
Spray Lab is an independent project, not affiliated with Effect.app, and shares no code with it.

## Install

You need:
- Windows 10 or 11 (it is built and tested on Windows 11)
- [Python](https://www.python.org/downloads/) 3.11 or newer
- [ffmpeg](https://ffmpeg.org/download.html) on your PATH (ffmpeg and ffprobe), for videos and MP4 export
- Google Chrome or Microsoft Edge: the app opens in its own window

```
python -m pip install -r requirements.txt
python launch.pyw
```

The first start takes a little longer: numba compiles the effect, and the built-in spray textures are generated once.

Optional: `powershell -ExecutionPolicy Bypass -File make_shortcut.ps1` puts a **Spray Lab** shortcut on your Desktop.
You can drop a photo or video onto it.

## How it works
1. **Open** a photo (PNG, JPG, WebP, TIFF, BMP) or a video (MP4, MOV, WebM, MKV, AVI, GIF).
2. Pick a **preset** on the right. Each thumbnail shows that preset on your own picture.
   - **Spray**: black, deep red, orange, yellow, white.
   - **Bubblegum**, **Toxic Teal**, **Neon Vapor**, **Thermal**, **Acid**, **Cobalt**, **Mono Stencil**: other colour
     looks.
   - **Ink Only**: blur and spray with the original colours.
3. Fine-tune the **layers** on the left. The top layer in the list is applied last.
   - Click a layer to edit it. The 🗑 and 👁 icons delete or hide it.
   - Drag a layer to reorder it. **+** adds a layer.
   - 🔀 (or **R**) randomizes the selected layer, and ↺ resets it.
   - **Gaussian Blur**: **strength** (the blur radius, in px when the picture's short side is 1080), **aspect** (−1
     blurs vertically only, +1 horizontally only), **opacity** and **blend mode**.
   - **Layer Mix** puts a spray texture over the picture:
     - **Texture**: five built-in spray textures (Ink Spray, Fine Mist, Coarse Splatter, Stencil Fade, Grit). You
       can also **load your own**, or drop an image on the texture picker. Grey textures work best.
     - **Fit**: Fill, Stretch or Tile. **Texture scale** sets the grain size.
     - **Blend channel**: Gray, Red, Green or Blue.
     - **Current layer ramp** and **Underlying layer ramp** decide how much texture shows, by brightness: left is
       dark, right is light. Drag the dots to move the four corners; the squares set the two ends. Double-click a
       ramp to reset it. The Spray preset pulls the underlying ramp down at the bright end, so whites stay clean.
     - **opacity** and **blend mode** (Spray uses Hard light).
     - **boil**: moves the texture every N frames, so videos shimmer like fresh spray. Off keeps the texture still.
   - **Gradient Map**:
     - Click the bar to add a colour stop. Drag a stop to move it, click it to change its colour, or drag it down
       to remove it.
     - The ◆ diamonds are midpoints: where two neighbouring colours are half-and-half.
     - **Reverse** flips the gradient. **Gradient shift** and **Gradient repeat** cycle it. Also opacity and blend mode.
   - 🔖 (in Layers) saves the whole stack as one of **My presets**.
4. **Viewing**:
   - Hold **Space** (or press on the source thumbnail) to see the original.
   - **Ctrl+Z** / **Ctrl+Y** undo and redo.
   - Zoom: Fit, 25 to 200 %. Drag to pan.
   - Videos: **▶** (or **P**) plays the preview with the clip's sound. The sound keeps the time and the picture follows
     it, skipping frames when the render can't keep up; the export has every frame. The speaker button (or **M**)
     mutes it. **←/→** step one frame, **Shift** steps 10.
5. **Export**:
   - Photos: **PNG**, **JPG**, or **MP4** at 24 or 30 fps (a 3 to 15 s photo clip; turn on boil so it moves).
   - Videos: **MP4** at the source frame rate, 24 fps or 30 fps. The original audio is kept, and the file is checked
     with ffprobe afterwards.
   - Size: full, or a long side of at most 3840, 1920 or 1280 px. It never upscales.
   - Speed: about 4 to 5 frames/s at 1440×2560 and about 18 frames/s at 720×1280 (it uses NVIDIA's NVENC encoder
     when your GPU has one).
   - Files go to `Videos\Spray Lab` in your user folder. Set the `SPRAYLAB_OUT` environment variable to use another
     folder. Existing files are never overwritten.

## What's where
- `effect.py`: the engine. It holds the layer stack, the blend modes, the blend-if ramps, the gradient map and the
  presets' maths. Each layer is one fused numba kernel, and the plain-numpy versions are the reference the tests
  check against. Sizes scale with the picture, so the preview matches the export.
- `textures.py`: the built-in spray textures and your own texture files. The built-ins are 4096² and generated once,
  then cached as 16-bit PNGs in `%LOCALAPPDATA%\SprayLab\textures`.
- `presets.py`: the ready-made looks.
- `media.py`: ffmpeg/ffprobe plumbing: probe, image read and write that works with accented file names, the frame
  stream for playback, the preview's sound, the decoder with optional frame-rate change, and the MP4 writer that keeps
  the audio.
- `app.py`: a FastAPI server on `127.0.0.1:47895` for previews, the preview's sound, presets, textures and exports.
  It quits about 45 s after the window closes, unless an export is still running.
- `launch.pyw`: the launcher. It opens a Chrome or Edge app window.
- `static/`: the window (HTML, CSS, JS).
- `make_icon.py`, `make_shortcut.ps1`: rebuild the icon (drawn with the app's own effect) and the Desktop shortcut.
- Your data: `%LOCALAPPDATA%\SprayLab\`: `my-presets.json`, `my-textures.json`, uploads, `app.log` and
  `preview-audio\` (only the open clip's sound; it is replaced when you open the next one).

## Tests
```
python -m unittest discover -s tests
```
The tests cover the engine (every blend mode and the ramps in the compiled kernels checked against the numpy maths,
gradient stops, textures, fit modes, size independence, boil, every preset, the fast blur against OpenCV's) and the
server (previews, every endpoint, PNG/JPG/MP4 exports, audio, frame-rate changes, streaming, cancel and upload).

## Credits
- Inspiration: the spray-paint effect shown in a reel by [Effect.app](https://www.instagram.com/reel/Db-umgRuf2n/).

## Licence

MIT, see [LICENSE](LICENSE).
