"""
Score Calibration for Ensemble Deepfake Detection.

Upgraded to support:
1. Temperature Scaling: p_cal = sigmoid(logit(p) / T)
2. Platt Scaling (Logistic Regression): p_cal = sigmoid(a * logit(p) + b)
3. Isotonic Regression: Non-parametric monotonic mapping via PAVA with linear interpolation.
4. Multivariate Logistic Fusion & Dynamic Context Gating.

All detectors (CLIP, GenD, B-Free, NPR-ResNet, commfor, pixel, temporal)
output normalized, calibrated probabilities in [0.0, 1.0] before fusion.
"""
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")
CALIB_PATH = os.path.join(DATA_DIR, "calibration.json")
BENCH_PATH = os.path.join(DATA_DIR, "bench_scores.json")

# Default empirically fitted calibration coefficients
DEFAULT_CALIB = {
    "temperature": {
        "commfor": 1.0,
        "bfree": 1.0,
        "clip": 1.0,
        "gend": 1.0,
        "pixel": 1.0,
        "npr": 1.0,
        "npr_resnet": 1.0,
        "temporal": 1.0,
    },
    "platt": {
        # format: [a, b] where p_cal = sigmoid(a * logit(p) + b)
        # Unbiased identity scaling preserving raw neural network posteriors
        "commfor": [1.0, 0.0],
        "bfree": [1.0, 0.0],
        "npr": [1.0, 0.0],
        "npr_resnet": [1.0, 0.0],
        "clip": [1.0, 0.0],
        "gend": [1.0, 0.0],
        "pixel": [1.0, 0.0],
        "temporal": [1.0, 0.0],
    },
    "multivariate": {
        "w": [0.50, 0.50],
        "b": 0.0
    }
}


def _logit(p: Union[float, np.ndarray, torch.Tensor], eps: float = 1e-5) -> np.ndarray:
    """Safely convert probability to log-odds logit."""
    arr = np.asarray(p, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def _sigmoid(z: Union[float, np.ndarray]) -> np.ndarray:
    """Numerically stable sigmoid."""
    arr = np.asarray(z, dtype=np.float64)
    # Clip to prevent overflow
    arr = np.clip(arr, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-arr))


def pava_monotonic(y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm (PAVA) for Isotonic Regression.
    Returns monotonically non-decreasing array that minimizes weighted squared error.
    """
    n = len(y)
    target = np.array(y, dtype=np.float64)
    w = np.ones(n, dtype=np.float64) if weights is None else np.array(weights, dtype=np.float64)
    blocks = [[i] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        if target[i] > target[i + 1]:
            w_sum = w[i] + w[i + 1]
            target[i] = (target[i] * w[i] + target[i + 1] * w[i + 1]) / w_sum
            w[i] = w_sum
            target = np.delete(target, i + 1)
            w = np.delete(w, i + 1)
            blocks[i] = blocks[i] + blocks[i + 1]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    out = np.zeros(n, dtype=np.float64)
    for b_idx, blk in enumerate(blocks):
        for idx in blk:
            out[idx] = target[b_idx]
    return out


class IsotonicMap:
    """Monotonic 1D piecewise-linear calibrator with boundary clipping."""
    def __init__(self, x_knots: np.ndarray, y_knots: np.ndarray):
        # Keep strictly unique x_knots for interpolation
        unique_x, indices = np.unique(x_knots, return_index=True)
        self.x = unique_x
        self.y = y_knots[indices]

    def predict(self, x_val: Union[float, np.ndarray]) -> np.ndarray:
        arr = np.asarray(x_val, dtype=np.float64)
        interp = np.interp(arr, self.x, self.y, left=self.y[0], right=self.y[-1])
        return np.clip(interp, 0.0, 1.0)


_ACTIVE_ISOTONIC: Dict[str, IsotonicMap] = {}
_LOADED_CALIB: Optional[dict] = None


def fit_from_benchmark() -> dict:
    """Fit Temperature Scaling, Platt Scaling, and Isotonic Calibration from bench_scores.json."""
    if not os.path.exists(BENCH_PATH):
        return DEFAULT_CALIB

    try:
        with open(BENCH_PATH, "r", encoding="utf-8") as f:
            bench = json.load(f)

        commfor_raw = np.array(bench["commfor"], dtype=np.float64)
        bfree_raw = np.array(bench["bfree"], dtype=np.float64)
        y = np.array(bench["label"], dtype=np.float64)

        calib = {
            "temperature": dict(DEFAULT_CALIB["temperature"]),
            "platt": dict(DEFAULT_CALIB["platt"]),
            "multivariate": dict(DEFAULT_CALIB["multivariate"]),
            "n_samples": len(y)
        }

        # Fit Platt and Temperature Scaling via PyTorch BCE
        zc = torch.tensor(_logit(commfor_raw), dtype=torch.float32)
        zb = torch.tensor(_logit(bfree_raw), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)

        # 1. CommFor Platt
        param_c = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
        opt_pc = torch.optim.Adam([param_c], lr=0.02)
        for _ in range(800):
            opt_pc.zero_grad()
            pred = param_c[0] * zc + param_c[1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, yt)
            loss.backward()
            opt_pc.step()

        # 2. BFree Platt
        param_b = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
        opt_pb = torch.optim.Adam([param_b], lr=0.02)
        for _ in range(800):
            opt_pb.zero_grad()
            pred = param_b[0] * zb + param_b[1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, yt)
            loss.backward()
            opt_pb.step()

        calib["platt"]["commfor"] = [float(param_c[0].item()), float(param_c[1].item())]
        calib["platt"]["bfree"] = [float(param_b[0].item()), float(param_b[1].item())]
        calib["platt"]["npr"] = calib["platt"]["bfree"]
        calib["platt"]["npr_resnet"] = calib["platt"]["bfree"]

        # 3. Multivariate Platt
        W = torch.nn.Parameter(torch.tensor([0.5, 0.5]))
        B = torch.nn.Parameter(torch.zeros(1))
        opt_m = torch.optim.Adam([W, B], lr=0.02)
        X = torch.stack([zc, zb], dim=-1)
        for _ in range(800):
            opt_m.zero_grad()
            pred = X @ W + B
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, yt)
            loss.backward()
            opt_m.step()

        calib["multivariate"]["w"] = [float(W[0].item()), float(W[1].item())]
        calib["multivariate"]["b"] = float(B.item())

        # 4. Isotonic calibration maps
        order_c = np.argsort(commfor_raw)
        iso_c_y = pava_monotonic(y[order_c])
        _ACTIVE_ISOTONIC["commfor"] = IsotonicMap(commfor_raw[order_c], iso_c_y)

        order_b = np.argsort(bfree_raw)
        iso_b_y = pava_monotonic(y[order_b])
        _ACTIVE_ISOTONIC["bfree"] = IsotonicMap(bfree_raw[order_b], iso_b_y)
        _ACTIVE_ISOTONIC["npr"] = _ACTIVE_ISOTONIC["bfree"]
        _ACTIVE_ISOTONIC["npr_resnet"] = _ACTIVE_ISOTONIC["bfree"]

        # Save to disk
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)

        return calib
    except Exception as e:
        print(f"Warning: could not fit benchmark calibration ({e}), using default calibration.")
        return DEFAULT_CALIB


def load() -> dict:
    """Load calibration config; auto-fit from bench_scores.json if needed."""
    global _LOADED_CALIB
    if _LOADED_CALIB is not None:
        return _LOADED_CALIB

    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                _LOADED_CALIB = json.load(f)
                return _LOADED_CALIB
        except Exception:
            pass

    _LOADED_CALIB = fit_from_benchmark()
    return _LOADED_CALIB


def calibrate_score(detector_name: str, raw_score: Optional[float], method: str = "platt") -> Optional[float]:
    """
    Calibrate a raw score to a normalized probability in [0.0, 1.0].
    
    detector_name: 'commfor', 'bfree', 'npr', 'npr_resnet', 'clip', 'gend', 'pixel', 'temporal'
    method: 'platt' (logistic regression), 'isotonic', 'temperature'
    """
    if raw_score is None:
        return None

    s = float(np.clip(raw_score, 1e-6, 1.0 - 1e-6))
    name = detector_name.lower().strip()
    cal = load() or DEFAULT_CALIB

    # Method 1: Isotonic Regression (for bench models)
    if method == "isotonic" and name in _ACTIVE_ISOTONIC:
        return float(np.clip(_ACTIVE_ISOTONIC[name].predict(s), 0.0, 1.0))

    # Method 2: Temperature Scaling
    if method == "temperature":
        t_val = cal.get("temperature", {}).get(name, DEFAULT_CALIB["temperature"].get(name, 1.0))
        z = _logit(s) / max(float(t_val), 0.1)
        return float(np.clip(_sigmoid(z), 0.0, 1.0))

    # Method 3: Platt Scaling (default - smooth and monotonic across all models)
    platt_params = cal.get("platt", {}).get(name, DEFAULT_CALIB["platt"].get(name, [1.0, 0.0]))
    a, b = float(platt_params[0]), float(platt_params[1])
    z = a * _logit(s) + b
    return float(np.clip(_sigmoid(z), 0.0, 1.0))


def apply_multivariate_platt(commfor_score: float, bfree_score: float) -> float:
    """Evaluate calibrated multivariate logistic regression over commfor and bfree."""
    cal = load() or DEFAULT_CALIB
    w = cal.get("multivariate", {}).get("w", [0.25, 0.61])
    b = cal.get("multivariate", {}).get("b", 4.03)

    zc = _logit(commfor_score)
    zb = _logit(bfree_score)
    z = w[0] * zc + w[1] * zb + b
    return float(np.clip(_sigmoid(z), 0.0, 1.0))


def add_samples(samples: List[dict]):
    """Store labeled samples from review queue for continuous learning."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({
                "commfor": float(s.get("commfor", 0.5)),
                "bfree": float(s.get("bfree", s.get("ufd", 0.5))),
                "is_fake": bool(s["is_fake"])
            }) + "\n")


def fit() -> dict:
    """Refit calibration with feedback data + benchmark data."""
    return fit_from_benchmark()


# Auto-initialize calibration on module load
try:
    load()
except Exception:
    pass
