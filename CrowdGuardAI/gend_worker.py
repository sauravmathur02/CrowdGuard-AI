"""
GenD face-forensics worker (runs inside venv_gend, started automatically by app.py).

POST /score  (multipart file: image or video)
  -> {"gend": float|None, "faces": int, "box": list|None, "model": str|None, "frame_scores": list}

gend = raw P(fake) of the most suspicious detected face (images) or the mean of the
most suspicious half of sampled frames (videos). Calibration is applied by the backend.

Face model priority:
  1. GenD CLIP ViT-L/14 (yermandy/GenD_CLIP_L_14, WACV 2026) in pretrained_weights/GenD
  2. Generic ViT classifier in pretrained_weights/deepfake_face (fallback only)
"""
import os
import sys
import tempfile
import urllib.request
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "detectors", "GenD"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = os.path.join(HERE, "pretrained_weights")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
VIDEO_FRAMES = 16

# GenD's own preprocessing (detector.py -s 1.3) crops a square 1.3x the face box.
FACE_SCALE = 1.3
DNN_CONF = 0.5
MIN_FACE_PX = 40

# ── Face detection: OpenCV DNN ResNet-SSD, Haar frontal as fallback ───────────
cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_DNN_PROTOTXT = os.path.join(WEIGHTS, "deploy.prototxt")
_DNN_CAFFEMODEL = os.path.join(WEIGHTS, "res10_300x300_ssd_iter_140000.caffemodel")
_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def _load_dnn_net():
    os.makedirs(WEIGHTS, exist_ok=True)
    for path, url in [(_DNN_PROTOTXT, _PROTOTXT_URL), (_DNN_CAFFEMODEL, _CAFFEMODEL_URL)]:
        if not os.path.isfile(path):
            try:
                print(f"Downloading DNN face detector: {os.path.basename(path)}...")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Warning: could not download {os.path.basename(path)}: {e}")
                return None
    try:
        net = cv2.dnn.readNetFromCaffe(_DNN_PROTOTXT, _DNN_CAFFEMODEL)
        print("DNN face detector (ResNet SSD) loaded.")
        return net
    except Exception as e:
        print(f"DNN face detector unavailable: {e}")
        return None


_DNN_NET = _load_dnn_net()


def detect_faces(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return face boxes (x, y, w, h) in pixels, largest first."""
    H, W = bgr.shape[:2]
    faces = []
    if _DNN_NET is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        _DNN_NET.setInput(blob)
        det = _DNN_NET.forward()
        for i in range(det.shape[2]):
            if float(det[0, 0, i, 2]) < DNN_CONF:
                continue
            x0, y0 = max(0, int(det[0, 0, i, 3] * W)), max(0, int(det[0, 0, i, 4] * H))
            x1, y1 = min(W, int(det[0, 0, i, 5] * W)), min(H, int(det[0, 0, i, 6] * H))
            if (x1 - x0) >= MIN_FACE_PX and (y1 - y0) >= MIN_FACE_PX:
                faces.append((x0, y0, x1 - x0, y1 - y0))
    if not faces:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
        faces = [tuple(int(v) for v in f) for f in found]
    return sorted(faces, key=lambda r: r[2] * r[3], reverse=True)


def face_crops(bgr: np.ndarray, max_faces: int = 3) -> List[Tuple[Image.Image, List[float]]]:
    """Square crops of FACE_SCALE x the face box (GenD preprocessing) with normalized boxes."""
    if bgr is None or bgr.size == 0:
        return []
    H, W = bgr.shape[:2]
    out = []
    for (x, y, w, h) in detect_faces(bgr)[:max_faces]:
        cx, cy, side = x + w / 2.0, y + h / 2.0, FACE_SCALE * max(w, h)
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        crop = Image.fromarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        out.append((crop, [x / W, y / H, w / W, h / H]))
    return out


# ── Face-forensics model ──────────────────────────────────────────────────────
gend_model = None
fallback_model = None
fallback_proc = None
MODEL_NAME: Optional[str] = None

try:
    from src.hf.modeling_gend import GenD

    _gend_dir = os.path.join(WEIGHTS, "GenD")
    if os.path.isfile(os.path.join(_gend_dir, "model.safetensors")):
        gend_model = GenD.from_pretrained(_gend_dir).to(DEVICE).eval()
        MODEL_NAME = "GenD_CLIP_L_14"
        print(f"GenD face model loaded from {_gend_dir} onto {DEVICE}.")
    else:
        print(f"GenD weights not found in {_gend_dir}.")
except Exception as e:
    print(f"Warning: GenD failed to load: {e}")
    gend_model = None

if gend_model is None:
    _fb_dir = os.path.join(WEIGHTS, "deepfake_face")
    if os.path.isdir(_fb_dir):
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            fallback_proc = AutoImageProcessor.from_pretrained(_fb_dir)
            fallback_model = AutoModelForImageClassification.from_pretrained(_fb_dir).to(DEVICE).eval()
            MODEL_NAME = "deepfake_face_vit (fallback)"
            print(f"Fallback face ViT loaded from {_fb_dir}. Install GenD weights for reliable face-swap detection.")
        except Exception as e:
            print(f"Warning: fallback face ViT failed to load: {e}")

if MODEL_NAME is None:
    print("No face-forensics model available; /score will return gend=None.")


@torch.inference_mode()
def score_crops(crops: List[Image.Image]) -> List[float]:
    """Raw P(fake) per face crop."""
    if not crops:
        return []
    if gend_model is not None:
        x = torch.stack([gend_model.feature_extractor.preprocess(c.convert("RGB")) for c in crops]).to(DEVICE)
        ctx = torch.amp.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.autocast("cpu", enabled=False)
        with ctx:
            probs = gend_model(x).softmax(dim=-1)[:, 1]
        return [float(p) for p in probs.float().cpu().numpy()]
    if fallback_model is not None:
        inputs = fallback_proc(images=[c.convert("RGB") for c in crops], return_tensors="pt").to(DEVICE)
        probs = torch.softmax(fallback_model(**inputs).logits, dim=-1)[:, 1]
        return [float(p) for p in probs.float().cpu().numpy()]
    return []


def video_frames(data: bytes, suffix: str, n: int = VIDEO_FRAMES) -> List[np.ndarray]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(data)
        path = t.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = []
        if total > 0:
            for i in np.linspace(0, max(total - 1, 0), min(n, total)).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if ok:
                    out.append(f)
        cap.release()
        return out
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def aggregate_video(frame_scores: List[float]) -> float:
    """Mean of the most suspicious half of frames: a swap that is present in part of the clip
    still registers, while a single noisy frame cannot decide the verdict on its own."""
    s = sorted(frame_scores, reverse=True)
    k = max(1, (len(s) + 1) // 2)
    return float(np.mean(s[:k]))


def score_media(data: bytes, name: str, content_type: str = "") -> dict:
    name = (name or "").lower()
    is_video = name.endswith(VIDEO_EXT) or (content_type or "").startswith("video/")
    if is_video:
        frames = video_frames(data, os.path.splitext(name)[1] or ".mp4")
    else:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        frames = [arr] if arr is not None else []

    per_frame, best_box, best_p, max_faces = [], None, -1.0, 0
    for f in frames:
        fc = face_crops(f)
        if not fc:
            continue
        max_faces = max(max_faces, len(fc))
        probs = score_crops([c for c, _ in fc])
        if not probs:
            continue
        i = int(np.argmax(probs))
        per_frame.append(float(probs[i]))  # the most suspicious face in this frame
        if probs[i] > best_p:
            best_p, best_box = float(probs[i]), fc[i][1]

    if not per_frame:
        return {"gend": None, "faces": max_faces, "box": best_box, "model": MODEL_NAME, "frame_scores": []}
    gend = aggregate_video(per_frame) if is_video else per_frame[0]
    return {
        "gend": round(gend, 6),
        "faces": max_faces,
        "box": best_box,
        "model": MODEL_NAME,
        "frame_scores": [round(p, 4) for p in per_frame],
        "frames_with_faces": len(per_frame),
        "frames_sampled": len(frames),
    }


app = FastAPI()


@app.get("/health")
def health():
    return {"ok": MODEL_NAME is not None, "device": DEVICE, "model": MODEL_NAME, "gend": gend_model is not None}


@app.post("/score")
async def score(file: UploadFile = File(...)):
    data = await file.read()
    return score_media(data, file.filename or "", file.content_type or "")
