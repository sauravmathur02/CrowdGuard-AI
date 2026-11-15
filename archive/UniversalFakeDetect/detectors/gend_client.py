"""
Client for the GenD face-forensics worker.

The worker needs a different `transformers` version than the main app, so it
runs in its own virtualenv (../venv_gend). This module starts it as a child
process and talks to it over localhost. If the venv is missing, GenD is simply
disabled and the app keeps working with the other detectors.
"""
import atexit
import os
import subprocess
import sys
import time

import requests

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


def is_healthy() -> bool:
    """Return True if worker process is running and answering health checks."""
    global _proc
    if _proc is None or _proc.poll() is not None:
        return False
    try:
        r = requests.get(f"{URL}/health", timeout=1.5)
        return r.ok
    except Exception:
        return False


def get_status() -> dict:
    """Return status of the GenD worker process."""
    global _proc
    running = _proc is not None and _proc.poll() is None
    healthy = is_healthy() if running else False
    return {
        "running": running,
        "healthy": healthy,
        "pid": _proc.pid if running else None,
        "port": PORT,
        "url": URL
    }


def score(filename: str, data: bytes, content_type: str = ""):
    """
    Return (gend_prob_fake or None, faces_found, face_box or None).
    Uses non-blocking fault-tolerant timeout with graceful fallback to None.
    """
    global _proc
    if _proc is None or _proc.poll() is not None:
        return None, 0, None

    name = (filename or "").lower()
    is_video = name.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")) or content_type.startswith("video/")
    timeout_tuple = (2.5, 18.0) if is_video else (2.0, 6.0)

    try:
        r = requests.post(
            f"{URL}/score",
            files={"file": (filename, data, content_type)},
            timeout=timeout_tuple
        )
        if not r.ok:
            return None, 0, None
        j = r.json()
        raw_gend = j.get("gend")
        cal_gend = calibration.calibrate_score("gend", raw_gend) if raw_gend is not None else None
        final_score = cal_gend if cal_gend is not None else raw_gend
        return (round(float(final_score), 6) if final_score is not None else None), int(j.get("faces", 0)), j.get("box")
    except requests.exceptions.Timeout:
        print("Notice: GenD worker timed out; continuing ensemble without face-forensics.")
        return None, 0, None
    except Exception as e:
        print(f"Notice: GenD worker communication error ({e}); graceful fallback active.")
        return None, 0, None


