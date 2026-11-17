"""Desktop launcher: starts the Spray Lab server (if needed) and opens the app window.

Drop a photo or video onto the desktop shortcut to open it directly.
"""
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("SPRAYLAB_PORT") or 47895)
URL = f"http://127.0.0.1:{PORT}/"
LOCAL = os.environ.get("LOCALAPPDATA", APP_DIR)


def probe():
    """'ours' if Spray Lab answers on the port, 'other' if something else does, None if free."""
    try:
        with urllib.request.urlopen(URL + "api/ping", timeout=0.8) as r:
            return "ours" if b"spray-lab" in r.read() else "other"
    except urllib.error.HTTPError:
        return "other"
    except Exception:
        return None


def post_load(path):
    req = urllib.request.Request(URL + "api/load", data=json.dumps({"path": path}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass


def find_browser():
    pf, pf86 = os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for p in (os.path.join(pf, r"Google\Chrome\Application\chrome.exe"),
              os.path.join(pf86, r"Google\Chrome\Application\chrome.exe"),
              os.path.join(LOCAL, r"Google\Chrome\Application\chrome.exe"),
              os.path.join(pf86, r"Microsoft\Edge\Application\msedge.exe"),
              os.path.join(pf, r"Microsoft\Edge\Application\msedge.exe")):
        if os.path.exists(p):
            return p
    return None


def open_window():
    exe = find_browser()
    if not exe:
        webbrowser.open(URL)
        return
    profile = os.path.join(LOCAL, "SprayLab", "window")
    # the window keeps pinging the server while minimised (the server quits once the window is gone)
    subprocess.Popen([exe, f"--app={URL}", f"--user-data-dir={profile}", "--no-first-run",
                      "--no-default-browser-check", "--disable-features=Translate",
                      "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                      "--disable-backgrounding-occluded-windows", "--window-size=1560,980"],
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None
    who = probe()
    if who == "other":
        ctypes.windll.user32.MessageBoxW(None, f"Port {PORT} is used by another program, so Spray Lab cannot "
                                         "start. Close that program and try again.", "Spray Lab", 0x10)
        return
    if who == "ours":
        if path:
            post_load(path)
        open_window()
        return

    sys.path.insert(0, APP_DIR)
    import app  # redirects output to the log file

    if path:
        try:
            app.load_media(path)
        except Exception:
            pass

    def opener():
        for _ in range(300):
            if probe() == "ours":
                break
            time.sleep(0.1)
        open_window()

    threading.Thread(target=opener, daemon=True).start()
    app.serve()


if __name__ == "__main__":
    main()
