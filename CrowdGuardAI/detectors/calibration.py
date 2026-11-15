"""
Score calibration for the deepfake detectors.

Each detector's raw P(fake) is mapped with Platt scaling,
    p_cal = sigmoid(a * logit(p_raw) + b),
using coefficients fitted by tools/calibrate.py on a labelled, held-out evaluation set
(Defactify, DeepFakeFace, GenD sample crops). The same file holds the fusion model used by
detectors/ensemble.py and the held-out accuracy ("eval").

Without data/calibration.json every detector runs uncalibrated (identity mapping).
"""
import json
import os
from typing import List, Optional, Union

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")
CALIB_PATH = os.path.join(DATA_DIR, "calibration.json")

_NAMES = ("commfor", "bfree", "clip", "gend", "pixel", "npr", "npr_resnet", "temporal")

# Identity calibration: raw network posteriors are passed through unchanged.
DEFAULT_CALIB = {
    "temperature": {n: 1.0 for n in _NAMES},
    "platt": {n: [1.0, 0.0] for n in _NAMES},
}


def _logit(p: Union[float, np.ndarray], eps: float = 1e-5) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def _sigmoid(z: Union[float, np.ndarray]) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -50.0, 50.0)))


_LOADED_CALIB: Optional[dict] = None


def load() -> dict:
    """Load data/calibration.json (written by tools/calibrate.py); identity if missing."""
    global _LOADED_CALIB
    if _LOADED_CALIB is not None:
        return _LOADED_CALIB
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                _LOADED_CALIB = json.load(f)
                return _LOADED_CALIB
        except Exception as e:
            print(f"Warning: could not read {CALIB_PATH} ({e}); detectors run uncalibrated.")
    _LOADED_CALIB = DEFAULT_CALIB
    return _LOADED_CALIB


def reload() -> dict:
    """Drop the cached calibration and read calibration.json again."""
    global _LOADED_CALIB
    _LOADED_CALIB = None
    return load()


def is_fitted() -> bool:
    return bool((load() or {}).get("fusion"))


def calibrate_score(detector_name: str, raw_score: Optional[float], method: str = "platt") -> Optional[float]:
    """Map a raw detector score to a calibrated probability in [0, 1].

    detector_name: 'commfor', 'bfree' (NPR), 'clip', 'gend', 'temporal', ...
    method: 'platt' (default) or 'temperature'.
    """
    if raw_score is None:
        return None
    s = float(np.clip(raw_score, 1e-6, 1.0 - 1e-6))
    name = detector_name.lower().strip()
    cal = load() or DEFAULT_CALIB

    if method == "temperature":
        t_val = float(cal.get("temperature", {}).get(name, 1.0))
        return float(_sigmoid(_logit(s) / max(t_val, 0.1)))

    a, b = cal.get("platt", {}).get(name, [1.0, 0.0])
    return float(_sigmoid(float(a) * _logit(s) + float(b)))


def add_samples(samples: List[dict]):
    """Store user-labelled samples (review queue) for a future recalibration."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({
                "commfor": float(s.get("commfor", 0.5)),
                "bfree": float(s.get("bfree", s.get("ufd", 0.5))),
                "is_fake": bool(s["is_fake"]),
            }) + "\n")


try:
    load()
except Exception:
    pass
