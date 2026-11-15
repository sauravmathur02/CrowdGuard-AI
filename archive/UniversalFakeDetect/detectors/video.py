"""
Video extraction, batching, and GPU tensor preparation.

Provides:
- Temporal frame sampling with exact timestamp tracking.
- Single-tensor GPU batching (B, C, H, W) for efficient multi-frame inference.
"""
import os
import tempfile
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def sample_video_frames_with_timestamps(
    path: str, n_frames: int = 16
) -> Tuple[List[Image.Image], List[float]]:
    """
    Sample up to n_frames PIL images spread evenly over the video with their timestamps in seconds.
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0

    frames: List[Image.Image] = []
    timestamps: List[float] = []

    if total > 0:
        indices = np.linspace(0, max(total - 1, 0), num=min(n_frames, total)).astype(int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                timestamps.append(round(float(idx) / fps, 3))

    if not frames:
        cap.release()
        cap = cv2.VideoCapture(path)
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(frame)

        if all_frames:
            indices = np.linspace(0, len(all_frames) - 1, num=min(n_frames, len(all_frames))).astype(int)
            for idx in indices:
                frames.append(Image.fromarray(cv2.cvtColor(all_frames[idx], cv2.COLOR_BGR2RGB)))
                timestamps.append(round(float(idx) / fps, 3))

    cap.release()
    return frames, timestamps


def sample_video_frames(path: str, n_frames: int = 16) -> List[Image.Image]:
    """Backward-compatible helper returning only the sampled PIL frames."""
    frames, _ = sample_video_frames_with_timestamps(path, n_frames=n_frames)
    return frames


def sample_video_bytes_with_timestamps(
    data: bytes, n_frames: int = 16, suffix: str = ".mp4"
) -> Tuple[List[Image.Image], List[float]]:
    """Extract frames and timestamps from in-memory video bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return sample_video_frames_with_timestamps(tmp_path, n_frames)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def sample_video_bytes(data: bytes, n_frames: int = 16, suffix: str = ".mp4") -> List[Image.Image]:
    """Backward-compatible helper returning only the sampled PIL frames from bytes."""
    frames, _ = sample_video_bytes_with_timestamps(data, n_frames, suffix)
    return frames


def batch_video_frames(
    frames: List[Image.Image],
    target_size: Tuple[int, int] = (384, 384),
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Batch sampled video frames into a single PyTorch tensor (B, C, H, W).
    Preserves aspect ratio along shortest edge then center-crops to target_size.
    """
    if not frames:
        return torch.empty(0)

    transform = T.Compose([
        T.Resize(max(target_size), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    tensors = [transform(im.convert("RGB")) for im in frames]
    batch = torch.stack(tensors)
    if device is not None:
        batch = batch.to(device)
    return batch
