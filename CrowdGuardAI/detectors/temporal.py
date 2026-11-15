"""
Temporal Consistency & Video Anomaly Detection.

Computes:
1. Frame-to-frame deep embedding cosine similarity & variance to detect temporal flickering.
2. Optical flow motion compensation error (warp error & warp jitter).
3. High-frequency sensor noise / residual energy.
4. Per-timestamp timeline array of anomaly scores for interactive UI scrubber.
"""
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from detectors import calibration


def _extract_frame_embedding(gray_frame: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Extract normalized structural and spatial gradient feature embedding (768D) for a video frame.
    Sensitive to facial morphing, texture jitter, and neural diffusion flicker.
    """
    resized = cv2.resize(gray_frame, target_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # Horizontal and vertical spatial gradients
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)

    # 16x16 pooled spatial feature maps
    f_lum = cv2.resize(resized, (16, 16), interpolation=cv2.INTER_AREA).flatten()
    f_gx = cv2.resize(gx, (16, 16), interpolation=cv2.INTER_AREA).flatten()
    f_gy = cv2.resize(gy, (16, 16), interpolation=cv2.INTER_AREA).flatten()

    emb = np.concatenate([f_lum, f_gx, f_gy])
    norm = np.linalg.norm(emb) + 1e-7
    return emb / norm


def _read_consecutive_with_timestamps(
    path: str, n: int = 20, stride: int = 2, size: int = 320
) -> Tuple[List[np.ndarray], List[float]]:
    """Sample consecutive video frames with exact timestamps."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0

    need = n * stride
    start = max((total - need) // 2, 0) if total > 0 else 0
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    i = 0
    curr_frame_idx = start

    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = f.shape[:2]
            scale = size / max(h, w)
            f_resized = cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f_resized, cv2.COLOR_BGR2GRAY))
            timestamps.append(round(float(curr_frame_idx) / fps, 3))
        i += 1
        curr_frame_idx += 1

    cap.release()
    return frames, timestamps


def temporal_features(path: str) -> Optional[Dict]:
    """
    Compute comprehensive temporal anomaly metrics and per-timestamp anomaly timeline.
    """
    frames, timestamps = _read_consecutive_with_timestamps(path)
    if len(frames) < 4:
        return None

    # 1. Optical flow warping & luminance flicker
    warp_errs: List[float] = []
    flickers: List[float] = []
    h, w = frames[0].shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))

    for a, b in zip(frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mapx = (gx + flow[..., 0]).astype(np.float32)
        mapy = (gy + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = float(np.mean(np.abs(warped.astype(np.float32) - b.astype(np.float32))))
        warp_errs.append(err)
        flickers.append(float(np.abs(b.mean() - a.mean())))

    # 2. High-frequency sensor noise energy
    hf_energies = []
    for f in frames:
        ff = f.astype(np.float32)
        hf_energies.append(float(np.mean(np.abs(ff - cv2.GaussianBlur(ff, (0, 0), 1.5)))))

    # 3. Embedding cosine similarities & temporal variance
    embeddings = [_extract_frame_embedding(f) for f in frames]
    cosine_sims = []
    for e_curr, e_next in zip(embeddings[:-1], embeddings[1:]):
        sim = float(np.dot(e_curr, e_next))
        cosine_sims.append(sim)

    sim_arr = np.array(cosine_sims, dtype=np.float64)
    cosine_sim_variance = float(np.var(sim_arr))
    cosine_sim_mean = float(np.mean(sim_arr))

    # 4. Construct per-timestamp anomaly timeline for the interactive UI
    timeline = []
    for idx in range(len(warp_errs)):
        t_sec = timestamps[idx + 1] if (idx + 1) < len(timestamps) else timestamps[idx]
        w_e = warp_errs[idx]
        flk = flickers[idx]
        cos_dist = max(0.0, 1.0 - cosine_sims[idx])

        # Composite anomaly score at timestamp t:
        # Optical warping + luminance jump + cosine feature drop
        norm_w = min(1.0, w_e / 2.5)
        norm_flk = min(1.0, flk / 1.5)
        norm_cos = min(1.0, cos_dist * 8.0)
        anomaly = float(0.40 * norm_cos + 0.35 * norm_w + 0.25 * norm_flk)

        timeline.append({
            "timestamp": t_sec,
            "score": round(anomaly, 4),
            "flicker": round(flk, 4),
            "warp_error": round(w_e, 3)
        })

    # 5. Composite temporal fake signal [0.0, 1.0]
    mean_warp = float(np.mean(warp_errs))
    warp_jitter = float(np.std(warp_errs))
    mean_flicker = float(np.std(flickers))
    mean_hf = float(np.mean(hf_energies))

    # Calibrated temporal fake signal:
    # High cosine variance + high optical warp jitter indicates AI-generated/swapped video
    raw_signal = (
        0.40 * min(1.0, cosine_sim_variance * 400.0) +
        0.35 * min(1.0, max(0.0, mean_warp - 1.1) / 1.2) +
        0.25 * min(1.0, max(0.0, warp_jitter - 0.25) / 0.8)
    )
    temporal_fake_signal = calibration.calibrate_score("temporal", raw_signal, method="platt")

    return {
        "warp_error": round(mean_warp, 4),
        "warp_jitter": round(warp_jitter, 4),
        "flicker": round(mean_flicker, 4),
        "hf_energy": round(mean_hf, 4),
        "cosine_sim_mean": round(cosine_sim_mean, 4),
        "cosine_sim_variance": round(cosine_sim_variance, 6),
        "temporal_fake_signal": round(float(temporal_fake_signal or raw_signal), 4),
        "frames": len(frames),
        "timeline": timeline,
    }


def temporal_features_bytes(data: bytes, suffix: str = ".mp4") -> Optional[Dict]:
    """Compute temporal features from in-memory video bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        p = tmp.name
    try:
        return temporal_features(p)
    finally:
        try:
            os.remove(p)
        except OSError:
            pass
