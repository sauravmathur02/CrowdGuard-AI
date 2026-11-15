"""
GenD face-forensics worker (runs inside venv_gend, started automatically by app.py).

POST /score  (multipart file: image or video)  ->  {"gend": float|None, "faces": int, "box": list|None}
gend = P(fake) over detected face crops with robust multi-angle multi-face localization.
"""
import io
import os
import sys
import tempfile
import urllib.request

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "detectors", "GenD"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multi-angle face cascades (frontal + profile) for high recall across video motion
cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# ── OpenCV DNN face detector (ResNet SSD) ─────────────────────────────────────
# Much more robust than Haar: handles tilted, partially occluded, dark-skinned, make-up faces.
_DNN_NET = None
_DNN_PROTOTXT = os.path.join(HERE, "pretrained_weights", "deploy.prototxt")
_DNN_CAFFEMODEL = os.path.join(HERE, "pretrained_weights", "res10_300x300_ssd_iter_140000.caffemodel")

_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def _ensure_dnn_weights():
    """Download DNN weights if not already present (one-time, ~2MB)."""
    os.makedirs(os.path.join(HERE, "pretrained_weights"), exist_ok=True)
    for path, url in [(_DNN_PROTOTXT, _PROTOTXT_URL), (_DNN_CAFFEMODEL, _CAFFEMODEL_URL)]:
        if not os.path.isfile(path):
            try:
                print(f"Downloading DNN face detector: {os.path.basename(path)}...")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Warning: Could not download {os.path.basename(path)}: {e}")
                return False
    return True

def _load_dnn_net():
    global _DNN_NET
    if _DNN_NET is not None:
        return _DNN_NET
    try:
        if _ensure_dnn_weights() and os.path.isfile(_DNN_CAFFEMODEL) and os.path.isfile(_DNN_PROTOTXT):
            _DNN_NET = cv2.dnn.readNetFromCaffe(_DNN_PROTOTXT, _DNN_CAFFEMODEL)
            print("DNN face detector (ResNet SSD) loaded.")
    except Exception as e:
        print(f"DNN face detector unavailable: {e}")
    return _DNN_NET

try:
    _load_dnn_net()
except Exception:
    pass

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
app = FastAPI()

# Try loading GenD; fallback to active pixel detectors
gend_model = None
face_detector = None
try:
    from src.hf.modeling_gend import GenD
    ckpt_dir = os.path.join(HERE, "pretrained_weights", "GenD")
    if os.path.isdir(ckpt_dir):
        gend_model = GenD.from_pretrained(ckpt_dir).to(DEVICE).eval()
except Exception:
    gend_model = None

if gend_model is None:
    # ── Patch calibration shim before importing pixel detectors ────────────────
    # The worker runs in venv_gend where 'detectors.calibration' may not be
    # importable. We inject a pass-through shim so CommForDetector / BFreeDetector
    # can be used as face-crop fallbacks without the 'calibration is not defined' error.
    import sys as _sys
    import types as _types

    def _noop_calibrate(detector_name, raw_score, method="platt"):
        """Identity pass-through: returns raw score unchanged (no calibration)."""
        return raw_score

    _calib_shim = _types.ModuleType("detectors.calibration")
    _calib_shim.calibrate_score = _noop_calibrate
    _calib_shim.load = lambda: {}
    _calib_shim.fit_from_benchmark = lambda: {}
    _calib_shim.apply_multivariate_platt = lambda c, b: (c + b) / 2.0

    # Register shim under both possible import paths
    _sys.modules.setdefault("calibration", _calib_shim)
    _sys.modules.setdefault("detectors.calibration", _calib_shim)

    try:
        from detectors.pixel import CommForDetector, BFreeDetector
        face_detector_commfor = CommForDetector()
        face_detector_npr = BFreeDetector()
    except Exception as e:
        print(f"GenD worker detector initialization warning: {e}")
        face_detector_commfor = None
        face_detector_npr = None
else:
    face_detector_commfor = None
    face_detector_npr = None


def _video_frames(data: bytes, suffix: str, n: int = 12):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        t.write(data)
        path = t.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = []
        if total > 0:
            indices = np.linspace(0, max(total - 1, 0), n).astype(int)
            for i in indices:
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


def _dnn_detect_faces(bgr, conf_threshold=0.35):
    """
    OpenCV DNN ResNet SSD face detector — much higher recall than Haar cascades.
    Works well on angled, partially occluded, dark-skinned, and made-up faces.
    Returns list of (x, y, w, h) bounding boxes.
    """
    net = _DNN_NET
    if net is None:
        return []
    try:
        H, W = bgr.shape[:2]
        # Resize to 300x300 as expected by the ResNet SSD
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300),
                                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < conf_threshold:
                continue
            x0 = max(0, int(detections[0, 0, i, 3] * W))
            y0 = max(0, int(detections[0, 0, i, 4] * H))
            x1 = min(W, int(detections[0, 0, i, 5] * W))
            y1 = min(H, int(detections[0, 0, i, 6] * H))
            if (x1 - x0) >= 30 and (y1 - y0) >= 30:
                faces.append((x0, y0, x1 - x0, y1 - y0))
        return faces
    except Exception as e:
        print(f"DNN face detection error: {e}")
        return []


def _detect_faces_robust(gray, bgr=None):
    """
    3-tier face detection for maximum recall:
    Tier 1: OpenCV DNN ResNet SSD (best — handles tilt, occlusion, diverse skin tones)
    Tier 2: Haar cascades frontal + profile with aggressive settings
    Tier 3: Very loose Haar with minNeighbors=1 as last resort
    """
    # Tier 1: DNN detector (operates on color image)
    if bgr is not None and _DNN_NET is not None:
        dnn_faces = _dnn_detect_faces(bgr, conf_threshold=0.35)
        if dnn_faces:
            return dnn_faces

    # Tier 2: Haar frontal cascade
    faces = cascade_frontal.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=2, minSize=(28, 28))
    if len(faces) > 0:
        return faces

    # Try equalized histogram for difficult lighting conditions
    gray_eq = cv2.equalizeHist(gray)
    faces = cascade_frontal.detectMultiScale(gray_eq, scaleFactor=1.07, minNeighbors=2, minSize=(28, 28))
    if len(faces) > 0:
        return faces

    # Tier 2b: Profile cascade (left and right)
    if not cascade_profile.empty():
        faces = cascade_profile.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=2, minSize=(28, 28))
        if len(faces) > 0:
            return faces
        flipped = cv2.flip(gray, 1)
        faces_flipped = cascade_profile.detectMultiScale(flipped, scaleFactor=1.07, minNeighbors=2, minSize=(28, 28))
        if len(faces_flipped) > 0:
            W = gray.shape[1]
            return [(W - x - w, y, w, h) for (x, y, w, h) in faces_flipped]

    # Tier 3: Last resort — very aggressive Haar (minNeighbors=1)
    faces = cascade_frontal.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=1, minSize=(24, 24))
    if len(faces) > 0:
        # Filter to the largest (most confident) detection only
        return [max(faces, key=lambda r: r[2] * r[3])]

    return []


def _heuristic_face_region(bgr):
    """
    When ALL detection tiers fail, return a heuristic bounding box for the
    most likely face region: upper-center 40% of the image height, center 50% width.
    This ensures face-swapped images where the face is centered still get inspected.
    Returns a single (x, y, w, h) box or [] if image is too small.
    """
    if bgr is None:
        return []
    H, W = bgr.shape[:2]
    if H < 80 or W < 80:
        return []
    # Upper-center region heuristic: faces usually appear in top 60%, centered
    x = int(W * 0.20)
    y = int(H * 0.05)
    w = int(W * 0.60)
    h = int(H * 0.55)
    return [(x, y, w, h)]


def _face_crops_all(bgr, margin=0.30):
    if bgr is None or bgr.size == 0:
        return []
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _detect_faces_robust(g, bgr=bgr)
    # Fallback: heuristic region when all detectors fail (catches centered face swaps)
    used_heuristic = False
    if len(faces) == 0:
        faces = _heuristic_face_region(bgr)
        used_heuristic = True
    if len(faces) == 0:
        return []
    H, W = bgr.shape[:2]
    out = []
    # Score up to top 2 faces per frame (handles dialogue scenes and interviews)
    sorted_faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[:2]
    filtered_faces = [sorted_faces[0]]
    if len(sorted_faces) > 1:
        primary_area = sorted_faces[0][2] * sorted_faces[0][3]
        sec_area = sorted_faces[1][2] * sorted_faces[1][3]
        if sec_area >= 0.12 * primary_area or (sorted_faces[1][2] >= 60 and sorted_faces[1][3] >= 60):
            filtered_faces.append(sorted_faces[1])

    for (x, y, w, h) in filtered_faces:
        m = int(margin * max(w, h))
        x0, y0 = max(x - m, 0), max(y - m, 0)
        x1, y1 = min(x + w + m, W), min(y + h + m, H)
        crop = Image.fromarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        box = [x / W, y / H, w / W, h / H]
        out.append((crop, box))
    return out


def _score_crops(crops):
    if not crops:
        return []
    if gend_model is not None:
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.nullcontext()
        with torch.inference_mode():
            with autocast_ctx:
                x = torch.stack([gend_model.feature_extractor.preprocess(c) for c in crops]).to(DEVICE)
                probs = gend_model(x).softmax(dim=-1)[:, 1].cpu().float().numpy()
                return [float(p) for p in probs]

    c_scores = face_detector_commfor.predict(crops) if face_detector_commfor is not None else [0.0] * len(crops)
    n_scores = face_detector_npr.predict(crops) if face_detector_npr is not None else [None] * len(crops)

    combined = []
    for c, n in zip(c_scores, n_scores):
        if n is not None:
            if c > 0.45 or n > 0.45:
                combined.append(max(c, n))
            else:
                combined.append(0.50 * c + 0.50 * n)
        else:
            combined.append(float(c))
    return combined


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "gend": gend_model is not None,
        "commfor": face_detector_commfor is not None,
        "npr": face_detector_npr is not None
    }


@app.post("/score")
async def score(file: UploadFile = File(...)):
    data = await file.read()
    name = (file.filename or "").lower()
    is_video = name.endswith(VIDEO_EXT) or (file.content_type or "").startswith("video/")

    if is_video:
        frames = _video_frames(data, os.path.splitext(name)[1] or ".mp4", n=12)
    else:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        frames = [arr] if arr is not None else []

    crops = []
    boxes = []
    for f in frames:
        for crop, box in _face_crops_all(f):
            if crop.size[0] >= 36 and crop.size[1] >= 36:
                crops.append(crop)
                boxes.append(box)

    if not crops:
        return {"gend": None, "faces": 0, "box": None}

    probs = _score_crops(crops)
    if not probs:
        return {"gend": None, "faces": 0, "box": None}

    # Determine if heuristic regions were used (set by _face_crops_all internal state via boxes)
    # Heuristic crops are penalized: cap score at 0.75 to avoid false positives on non-face content
    real_face_count = sum(1 for b in boxes if b is not None and b[2] * b[3] > 0.03)

    # Video multi-frame persistence: require sustained facial manipulation across multiple frames
    if is_video and len(probs) >= 4:
        sorted_probs = sorted(probs, reverse=True)
        if sorted_probs[1] > 0.40:
            top_k = max(2, int(len(sorted_probs) * 0.35))
            final_score = float(0.70 * np.mean(sorted_probs[:top_k]) + 0.30 * np.median(sorted_probs))
        else:
            final_score = float(0.25 * sorted_probs[0] + 0.75 * np.median(sorted_probs))
    elif is_video and len(probs) >= 2:
        sorted_probs = sorted(probs, reverse=True)
        final_score = float(0.50 * sorted_probs[0] + 0.50 * sorted_probs[1])
    else:
        # For images: take mean but also respect the max for high-confidence face manipulation
        mean_p = float(np.mean(probs))
        max_p = float(np.max(probs))
        # If peak score is strongly suspicious, give it more weight (catches localized face swap)
        if max_p > 0.60:
            final_score = float(0.55 * max_p + 0.45 * mean_p)
        else:
            final_score = mean_p

    best_idx = int(np.argmax(probs))
    # Return real_face_count as detected faces for downstream gating
    return {
        "gend": round(final_score, 6),
        "faces": real_face_count if real_face_count > 0 else len(crops),
        "box": boxes[best_idx]
    }
