"""
Client for the GenD face-forensics worker.

The worker needs a different `transformers` version than the main app, so it
runs in its own virtualenv (../venv_gend). This module starts it as a child
process and talks to it over localhost. If the worker is unavailable, face
scores come back as "not checked" (with an error) instead of a fake 0.0.
"""
import atexit
import os
import subprocess
import time

import requests
from .calibration import calibrate_score as _calibrate_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "venv_gend", "Scripts", "python.exe")
PORT = 8001
URL = f"http://127.0.0.1:{PORT}"

_proc = None


def start(timeout: float = 240.0) -> bool:
    """Launch the worker and wait until it is healthy. Returns availability."""
    global _proc
    if not os.path.exists(PY):
        print("GenD: venv_gend not found -> face detector disabled.")
        return False
    _proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "gend_worker:app", "--host", "127.0.0.1", "--port", str(PORT),
         "--log-level", "warning"],
        cwd=APP_DIR,
    )
    atexit.register(stop)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _proc.poll() is not None:
            print("GenD: worker exited early -> face detector disabled.")
            return False
        try:
            if requests.get(f"{URL}/health", timeout=2).ok:
                return True
        except requests.RequestException:
            time.sleep(1.0)
    print("GenD: worker did not start in time -> face detector disabled.")
    return False


def stop():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
    _proc = None


def _health() -> dict:
    try:
        r = requests.get(f"{URL}/health", timeout=1.5)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def is_healthy() -> bool:
    """Return True if the worker process is running and has a face model loaded."""
    if _proc is None or _proc.poll() is not None:
        return False
    return bool(_health().get("ok"))


def get_status() -> dict:
    """Return status of the GenD worker process."""
    running = _proc is not None and _proc.poll() is None
    h = _health() if running else {}
    return {
        "running": running,
        "healthy": bool(h.get("ok")),
        "model": h.get("model"),
        "pid": _proc.pid if running else None,
        "port": PORT,
        "url": URL,
    }


def score(filename: str, data: bytes, content_type: str = "") -> dict:
    """
    Score faces with the worker. Always returns a dict:
      score        calibrated P(fake) of the face(s), or None
      raw          uncalibrated worker score, or None
      faces        number of faces detected (max over sampled frames)
      box          normalized [x, y, w, h] of the most suspicious face, or None
      model        name of the face model the worker used
      frame_scores per-frame face scores (videos)
      error        None when the worker answered; otherwise why no face score exists.
                   A missing score with error=None means "no face found", which is a
                   valid answer. A missing score with an error means "not checked".
    """
    out = {"score": None, "raw": None, "faces": 0, "box": None, "model": None, "frame_scores": [], "error": None}
    if _proc is None or _proc.poll() is not None:
        out["error"] = "Face-forensics worker is not running."
        return out
    name = (filename or "").lower()
    is_video = name.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")) or (content_type or "").startswith("video/")
    timeout_tuple = (3.0, 120.0) if is_video else (3.0, 30.0)
    try:
        r = requests.post(f"{URL}/score", files={"file": (filename, data, content_type)}, timeout=timeout_tuple)
        if not r.ok:
            out["error"] = f"Face-forensics worker returned HTTP {r.status_code}."
            return out
        j = r.json()
    except requests.exceptions.Timeout:
        out["error"] = "Face-forensics worker timed out."
        print("Notice: GenD worker timed out.")
        return out
    except Exception as e:
        out["error"] = f"Face-forensics worker error: {e}"
        print(f"Notice: GenD worker communication error ({e}).")
        return out

    raw = j.get("gend")
    out.update({
        "raw": raw,
        "faces": int(j.get("faces") or 0),
        "box": j.get("box"),
        "model": j.get("model"),
        "frame_scores": j.get("frame_scores") or [],
    })
    if j.get("model") is None:
        out["error"] = "No face-forensics model is loaded in the worker."
    elif raw is not None:
        out["score"] = round(float(_calibrate_score("gend", raw)), 6)
    return out
