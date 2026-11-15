"""
Shared detector scoring and score fusion.

Used by both the live API (app.py) and the calibration tool (tools/calibrate.py), so the
verdict logic that is measured during calibration is exactly the logic that runs in production.

Pipeline:
  1. Each detector returns a calibrated P(fake) (Platt scaling fitted by tools/calibrate.py).
  2. fuse() combines the available detectors with a logistic-regression model fitted on
     labelled data (calibration.json -> "fusion"). One model is used when a face score
     exists ("face"), another when it does not ("no_face").
  3. The fused probability is compared with the fitted threshold (default 0.5).
"""
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from detectors import calibration

DETECTORS = ("commfor", "bfree", "clip", "gend")


def _logit(p: float, eps: float = 1e-4) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def face_crop_for(img: Image.Image, face_box: Optional[List[float]], min_area: float = 0.02) -> Optional[Image.Image]:
    """1.3x face crop passed to the NPR detector as an extra view (only for reasonably large faces)."""
    if not face_box or len(face_box) != 4 or face_box[2] * face_box[3] <= min_area:
        return None
    W, H = img.size
    x, y, w, h = face_box
    cx, cy = (x + w / 2.0) * W, (y + h / 2.0) * H
    cw, ch = 1.3 * w * W, 1.3 * h * H
    x0, y0 = max(0, int(cx - cw / 2.0)), max(0, int(cy - ch / 2.0))
    x1, y1 = min(W, int(cx + cw / 2.0)), min(H, int(cy + ch / 2.0))
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return None
    return img.crop((x0, y0, x1, y1))


def score_image(img: Image.Image, commfor, bfree, clip, face_box=None) -> Dict[str, Optional[float]]:
    """Calibrated whole-image scores from the pixel/semantic detectors."""
    crop = face_crop_for(img, face_box)
    c = commfor.predict([img])[0]
    b = bfree.predict([img], face_crops=[crop] if crop is not None else None)[0] if bfree.is_available else None
    k = clip.predict([img])[0] if clip.is_available else None
    return {"commfor": float(c), "bfree": None if b is None else float(b), "clip": None if k is None else float(k)}


def aggregate_frames(scores: List[Optional[float]]) -> Optional[float]:
    """Video aggregation: mean of the most suspicious half of the sampled frames.
    Partially manipulated clips still register; one noisy frame cannot decide on its own."""
    s = sorted((float(x) for x in scores if x is not None), reverse=True)
    if not s:
        return None
    k = max(1, (len(s) + 1) // 2)
    return float(np.mean(s[:k]))


def score_frames(frames: List[Image.Image], commfor, bfree, clip) -> Dict[str, object]:
    """Per-frame and aggregated calibrated scores for a video."""
    c = [float(x) for x in commfor.predict(frames)]
    b = bfree.predict(frames) if bfree.is_available else [None] * len(frames)
    k = clip.predict(frames) if clip.is_available else [None] * len(frames)
    per_frame = []
    for i in range(len(frames)):
        per_frame.append(fuse({"commfor": c[i], "bfree": b[i], "clip": k[i]})["p_fake"])
    return {
        "commfor": aggregate_frames(c),
        "bfree": aggregate_frames(b),
        "clip": aggregate_frames(k),
        "frame_fused": per_frame,
    }


def _fusion_model(name: str) -> Optional[dict]:
    return (calibration.load() or {}).get("fusion", {}).get(name)


def fuse(scores: Dict[str, Optional[float]], media: str = "image") -> Dict[str, object]:
    """
    Combine calibrated detector scores into one P(fake).

    media="video" uses the "video_face" model (fitted on face-swap videos by
    tools/eval_video.py) when a face was scored; otherwise the image models are used.
    Returns {"p_fake", "threshold", "model", "contributions", "calibrated"}.
    contributions[name] = weight * logit(score): positive pushes towards fake.
    """
    use_face = scores.get("gend") is not None
    model_name = "face" if use_face else "no_face"
    if media == "video" and use_face and _fusion_model("video_face"):
        model_name = "video_face"
    m = _fusion_model(model_name)
    threshold = float((calibration.load() or {}).get("threshold", 0.5))

    if m:
        names, w, b = list(m["features"]), list(m["w"]), float(m["b"])
    else:
        # Not calibrated yet: equal-weight average of the available calibrated logits.
        names = [n for n in DETECTORS if scores.get(n) is not None and (use_face or n != "gend")]
        w, b = [1.0 / max(1, len(names))] * len(names), 0.0

    z, contrib, used_w = b, {}, 0.0
    for n, wi in zip(names, w):
        s = scores.get(n)
        if s is None:
            continue
        contrib[n] = round(wi * _logit(s), 4)
        z += contrib[n]
        used_w += abs(wi)
    if not contrib:
        return {"p_fake": None, "threshold": threshold, "model": model_name, "contributions": {}, "calibrated": bool(m)}
    p = float(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))
    return {"p_fake": p, "threshold": threshold, "model": model_name, "contributions": contrib, "calibrated": bool(m)}
