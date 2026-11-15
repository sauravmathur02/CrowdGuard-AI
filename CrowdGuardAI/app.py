import asyncio
import base64
import datetime
import hashlib
import io
import json
import os
import shutil
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import ExifTags, Image, ImageChops
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from detectors import calibration, ensemble, gend_client
from detectors.pixel import BFreeDetector, CLIPDetector, CommForDetector, estimate_image_quality
from detectors.provenance import check_provenance
from detectors.temporal import temporal_features_bytes
from detectors.video import sample_video_bytes_with_timestamps
from detectors.xai import generate_ela, generate_gradcam_heatmap, generate_residual_map

from crowd import BehaviorAnalyzer, DetectionSmoother, FeedProcessor, RenderOptions, WeaponClassifier, YOLODetector, compute_risk

warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")
STATIC_DIR = os.path.join(APP_DIR, "static")
WEIGHTS_DIR = os.path.join(APP_DIR, "pretrained_weights")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
VIDEO_FRAMES = 16  # frames sampled per video for the deepfake detectors

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "uploads"), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ── INITIALIZE CORE VISION FORENSIC MODELS ─────────────────────────────────
print("==================================================")
print("  CrowdGuard AI — Unified Intelligence Engine     ")
print("==================================================")
print("Loading Deepfake Ensemble Detectors...")
commfor_model = CommForDetector()
bfree_model = BFreeDetector()
clip_model = CLIPDetector()
print("Deepfake Detectors initialized.")

# ── INITIALIZE CROWD & WEAPON SURVEILLANCE MODELS ──────────────────────────
print("Loading Crowd & Weapon Surveillance Engine...")
custom_yolo = os.path.join(WEIGHTS_DIR, "best.pt")
fallback_yolo = os.path.join(WEIGHTS_DIR, "yolo_crowd.pt")
yolo_path = custom_yolo if os.path.isfile(custom_yolo) else fallback_yolo

classifier_path = os.path.join(WEIGHTS_DIR, "classifier.pth")

yolo_detector = YOLODetector(weights_path=yolo_path)
weapon_classifier = WeaponClassifier(checkpoint_path=classifier_path)
feed_processor = FeedProcessor(
    source_id="LIVE-CAM-01",
    detector=yolo_detector,
    classifier=weapon_classifier,
)
print("Crowd & Weapon Surveillance Engine initialized.")


def warmup_models():
    """Startup warmup routine to pre-compile PyTorch CUDA kernels & caches."""
    t0 = time.time()
    try:
        dummy_img = Image.new("RGB", (384, 384), (128, 128, 128))
        if hasattr(commfor_model, "predict"):
            commfor_model.predict([dummy_img])
            if hasattr(commfor_model, "get_attention_map"):
                commfor_model.get_attention_map(dummy_img)

        if hasattr(bfree_model, "predict") and bfree_model.is_available:
            bfree_model.predict([dummy_img])

        if hasattr(clip_model, "predict") and clip_model.is_available:
            clip_model.predict([dummy_img])

        # Warm up YOLO
        dummy_cv = np.zeros((384, 384, 3), dtype=np.uint8)
        yolo_detector.detect(dummy_cv)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print(f"CrowdGuard AI warmup completed in {time.time() - t0:.2f}s. All models primed.")
    except Exception as e:
        print(f"Notice: Model warmup encountered non-fatal notice: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting GenD face-forensics worker...")
    # GenD CLIP ViT-L/14 (~1.2 GB) takes longer to load than the old fallback ViT.
    gend_client.start(timeout=240.0)
    print("Executing GPU model warmup...")
    warmup_models()
    yield
    print("Stopping GenD face-forensics worker...")
    gend_client.stop()


app = FastAPI(title="CrowdGuard AI — Unified Threat Intelligence Platform", lifespan=lifespan)

# Allow Next.js frontend on port 3000 to reach the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── HEALTH & HARDWARE STATUS ENDPOINT ──────────────────────────────────────
@app.get("/api/health")
def api_health():
    """Health check returning active models, worker status, and GPU hardware metrics."""
    cuda_avail = torch.cuda.is_available()
    device_name = "cuda" if cuda_avail else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    vram_info = {}
    if cuda_avail:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved_vram = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
        vram_info = {
            "total_gb": round(total_vram, 2),
            "allocated_gb": round(allocated_vram, 2),
            "reserved_gb": round(reserved_vram, 2),
            "free_gb": round(total_vram - reserved_vram, 2),
        }

    return {
        "status": "healthy",
        "system": "CrowdGuard AI",
        "hardware": {
            "device": device_name,
            "gpu_name": gpu_name,
            "cuda_available": cuda_avail,
            "vram": vram_info,
        },
        "deepfake_models": {
            "commfor": {
                "name": getattr(commfor_model, "name", "Community Forensics"),
                "active": getattr(commfor_model, "model", None) is not None,
            },
            "bfree_npr": {
                "name": getattr(bfree_model, "name", "B-Free / NPR"),
                "active": getattr(bfree_model, "is_available", False),
            },
            "clip": {
                "name": getattr(clip_model, "name", "UniversalFakeDetect CLIP"),
                "active": getattr(clip_model, "is_available", False),
            },
            "gend_worker": gend_client.get_status(),
            "calibrated": calibration.is_fitted(),
        },
        "crowd_models": {
            "yolo_detector": {
                "active": True,
                "weights": os.path.basename(yolo_detector.weights_path),
                "is_custom_weapon_model": yolo_detector.is_custom_weapon_model,
                "classes": yolo_detector.track_classes,
            },
            "weapon_classifier": {
                "active": weapon_classifier.available,
                "labels": weapon_classifier.labels,
            },
        },
        # Ready means: every deepfake detector loaded, the face worker has a model, and the
        # fusion has been fitted (data/calibration.json).
        "ensemble_ready": bool(
            getattr(commfor_model, "model", None) is not None
            and getattr(bfree_model, "is_available", False)
            and getattr(clip_model, "is_available", False)
            and gend_client.is_healthy()
            and calibration.is_fitted()
        ),
    }


# ── FORENSIC HELPER UTILITIES ──────────────────────────────────────────────
def extract_exif(img: Image.Image) -> Dict[str, str]:
    """Extract metadata, EXIF tags, and AI generation parameters."""
    out = {"Format": str(img.format), "Dimensions": f"{img.width} x {img.height}", "Mode": img.mode}
    try:
        for k, v in (img.getexif() or {}).items():
            name = ExifTags.TAGS.get(k, str(k))
            if isinstance(v, bytes):
                continue
            out[name] = str(v)[:80]
    except Exception:
        pass
    for k, v in (getattr(img, "info", {}) or {}).items():
        if isinstance(v, str) and k.lower() in ("software", "parameters", "prompt", "comment", "description"):
            out[f"PNG:{k}"] = v[:120]
    return out


def face_region_ela(img: Image.Image, box: List[float], quality: int = 90) -> Optional[float]:
    """Ratio of mean ELA error inside face box vs background."""
    try:
        im = img.copy()
        im.thumbnail((512, 512))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        diff = np.asarray(ImageChops.difference(im, Image.open(buf).convert("RGB")).convert("L"), dtype=np.float32)
        H, W = diff.shape
        x, y, w, h = box
        x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
        mask = np.zeros_like(diff, dtype=bool)
        mask[y0:y1, x0:x1] = True
        if mask.sum() < 25 or (~mask).sum() < 25:
            return None
        return round(float(diff[mask].mean() / (diff[~mask].mean() + 1e-6)), 3)
    except Exception:
        return None


def build_explanation(
    is_fake: bool,
    is_video: bool,
    c: float,
    b: Optional[float],
    g: Optional[float],
    is_prov_fake: bool,
    prov_result: dict,
    faces_found: int,
    region: Optional[dict] = None,
    clip: Optional[float] = None,
    quality: Optional[dict] = None,
    confidence_variance: Optional[float] = None,
    final_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a human-readable forensic report from calibrated per-model signals."""
    T = 0.5  # detector scores are calibrated probabilities, so 0.5 is each model's own decision point
    media = "video" if is_video else "image"
    findings = []
    category = "Authentic"
    b_score = b if b is not None else 0.0

    if is_fake:
        if is_prov_fake:
            findings.append("Embedded metadata/C2PA credentials explicitly declare this file as AI-generated.")
        if g is not None and g > T:
            findings.append("The face-forensics model detected facial inconsistencies (blending boundaries, texture or identity artifacts) typical of a deepfake face-swap.")
        if c > T:
            findings.append("The Community Forensics Vision Transformer detected generative pixel fingerprints in frequency statistics (GAN/diffusion models such as Midjourney, DALL-E, Flux or Stable Diffusion).")
        if b_score > T:
            findings.append("The B-Free/NPR model found synthetic neighboring pixel relationships and frequency artifacts, indicating AI inpainting or generation.")
        if clip is not None and clip > T:
            findings.append("The UniversalFakeDetect (CLIP ViT-L/14) detector identified latent visual-semantic signatures of AI synthesis.")

        face_fake = g is not None and g > T
        pix_fake = c > T or b_score > T or (clip is not None and clip > T)
        if is_prov_fake:
            category = "AI-Generated (declared)"
        elif face_fake and not pix_fake:
            category = "Deepfake (face manipulation)"
        elif face_fake and pix_fake:
            category = "Deepfake / AI-Generated"
        elif c > T and b_score <= T:
            category = "AI-Generated"
        elif b_score > T and c <= T:
            category = "Edited / Manipulated"
        else:
            category = "Synthetic or Manipulated"
        summary = f"This {media} is likely {category.lower()}."
    else:
        summary = f"This {media} appears authentic. The fused detector score stayed below the decision threshold."
        findings.append("No generator fingerprints, diffusion noise patterns, or synthetic spectral signatures were found.")
        if faces_found and g is not None:
            findings.append("Detected facial anatomy, skin texture, and boundary blending appeared natural and consistent.")
        if not faces_found:
            findings.append("Pixel statistics conform to natural optical camera capture physics.")

    if is_video:
        findings.append(f"Video verdict uses the most suspicious half of {VIDEO_FRAMES} frames sampled across the clip.")

    if quality and quality.get("is_heavily_compressed"):
        findings.append(f"The {media} is heavily compressed (blocking ratio {quality.get('block_ratio', 1.0):.2f}). Compression removes the fine pixel traces that the detectors rely on, so treat this result with extra caution.")

    if confidence_variance is not None:
        if confidence_variance < 0.02:
            findings.append("Detector agreement: the models largely agree.")
        elif confidence_variance < 0.06:
            findings.append("Detector agreement: moderate.")
        else:
            findings.append("Detector agreement: the models disagree; the fused score weighs each model by its measured reliability.")

    strong = final_score if final_score is not None else max([c, b_score] + ([g] if g is not None else []) + ([clip] if clip is not None else []))
    if is_fake and strong < 0.70 and not is_prov_fake:
        findings.append("Confidence is moderate; compression or stylized filters can produce similar artifacts.")

    face_only = False
    if is_fake and g is not None and g <= T and max(c, b_score) > T and not is_prov_fake:
        findings.append("The face-swap model found no blending seams on the face. Whole-image generative patterns take priority.")
    if is_fake and g is not None and g > T and max(c, b_score) <= T and not is_prov_fake:
        face_only = True
        findings.insert(0, "Region analysis: the face region looks manipulated while the surrounding scene looks authentic — consistent with a replaced or swapped face on an otherwise real photo/video.")
    if is_fake and region and region.get("ela_mismatch") and g is not None and g > T:
        findings.append(f"Compression (ELA) in the face area differs from the background (ratio {region['ela_ratio']:.2f}), confirming a pasted or regenerated face.")
        face_only = True

    p = 1.0 if is_prov_fake else strong
    if not is_fake:
        likelihood = "Unlikely to be manipulated"
    elif p >= 0.90:
        likelihood = "Very likely"
    elif p >= 0.70:
        likelihood = "Likely"
    else:
        likelihood = "Possible"

    if is_fake and face_only and not is_prov_fake:
        category = "Face Replaced / Swapped"
        summary = f"{likelihood}: a face in this {media} has most likely been replaced or manipulated by AI."
    elif is_fake:
        summary = f"{likelihood}: this {media} is {category.lower()}."

    return {"category": category, "summary": summary, "findings": findings, "likelihood": likelihood}


# ── NOVEL FEED INTEGRITY VERIFICATION ENGINE ──────────────────────────────
def check_frame_integrity(frame_bgr: np.ndarray) -> Tuple[bool, float, dict]:
    """
    Feed Integrity Monitor: Samples a live camera frame through the deepfake
    vision ensemble to detect video feed tampering, CGI injection, or synthetic spoofing.
    Returns: (is_tampered: bool, integrity_score: float, details: dict)
    """
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        c_scores = commfor_model.predict([pil_img])
        c_score = float(c_scores[0]) if c_scores else 0.0

        b_score = 0.0
        if bfree_model.is_available:
            b_scores = bfree_model.predict([pil_img])
            b_score = float(b_scores[0]) if b_scores and b_scores[0] is not None else 0.0

        synth_risk = max(c_score, b_score)
        integrity_score = max(0.0, min(1.0, 1.0 - synth_risk))
        is_tampered = synth_risk > 0.46

        return (
            is_tampered,
            round(integrity_score, 4),
            {"commfor": round(c_score, 4), "bfree": round(b_score, 4)},
        )
    except Exception as e:
        return (False, 1.0, {"error": str(e)})


# ── CORE DEEPFAKE DETECTION PIPELINE ──────────────────────────────────────
async def execute_pipeline(
    contents: bytes,
    filename: str,
    content_type: str = "",
    mode: str = "auto",
    stage_callback=None,
) -> Dict[str, Any]:
    """Execute 4-stage deepfake forensic analysis pipeline."""
    mode = (mode or "auto").lower().strip()
    if mode not in ("auto", "deepfake", "ai_generated"):
        mode = "auto"

    is_video = (
        filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        or (content_type or "").startswith("video/")
    )

    if stage_callback:
        await stage_callback(1, "Metadata & C2PA Provenance", "running", {
            "message": "Inspecting C2PA cryptographic manifests & embedded EXIF headers..."
        })

    prov_result = check_provenance(contents)
    is_prov_fake = prov_result.get("verdict") == "ai_declared"

    exif = {}
    if not is_video:
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            exif = extract_exif(img)
        except Exception as e:
            return {"success": False, "error": f"Cannot open image: {e}"}
    else:
        img = None
        exif = {"Type": "video", "Format": content_type or "video/mp4"}

    if stage_callback:
        await stage_callback(1, "Metadata & C2PA Provenance", "completed", {
            "verdict": prov_result.get("verdict", "unknown"),
            "ai_declared": is_prov_fake,
            "tags_found": len(exif),
        })

    if stage_callback:
        await stage_callback(2, "Frequency, ELA & Pixel Forensics", "running", {
            "message": "Analyzing compression blocking, Laplacian sharpness, and ELA gradients..."
        })

    quality = None
    ela_url = None
    residual_url = None
    first_frame_img = None

    if is_video:
        frames, timestamps = sample_video_bytes_with_timestamps(contents, n_frames=VIDEO_FRAMES)
        if not frames:
            return {"success": False, "error": "Could not extract frames. Please upload a valid video."}
        quality = estimate_image_quality(frames[0]) if frames else None
        first_frame_img = frames[0] if frames else None
        ela_url, ela_diff = generate_ela(first_frame_img) if first_frame_img else (None, None)
        residual_url = generate_residual_map(first_frame_img, ela_diff=ela_diff) if first_frame_img else None
    else:
        quality = estimate_image_quality(img)
        ela_url, ela_diff = generate_ela(img)
        residual_url = generate_residual_map(img, ela_diff=ela_diff)

    if stage_callback:
        await stage_callback(2, "Frequency, ELA & Pixel Forensics", "completed", {
            "lap_var": quality.get("lap_var") if quality else None,
            "block_ratio": quality.get("block_ratio") if quality else None,
            "is_heavily_compressed": quality.get("is_heavily_compressed") if quality else False,
        })

    if stage_callback:
        await stage_callback(3, "Deep Semantic Vision Backbones", "running", {
            "message": "Evaluating CommFor ViT-S, B-Free NPR ResNet, CLIP ViT-L/14, and GenD face worker..."
        })

    face = gend_client.score(filename, contents, content_type)
    gend_score = face["score"]
    faces_found = int(face["faces"] or 0)
    face_box = face["box"]
    face_checked = face["error"] is None

    warnings_list: List[str] = []
    if not face_checked:
        warnings_list.append(f"Face-swap check did not run ({face['error']}). This verdict uses only the whole-image detectors.")
    elif face.get("model") and "fallback" in str(face.get("model")):
        warnings_list.append("GenD face weights are missing; a generic fallback face model was used, so face-swap results are less reliable.")
    if faces_found > 0 and gend_score is None and face_checked:
        warnings_list.append("A face was detected but could not be scored by the face-forensics model.")
    if not calibration.is_fitted():
        warnings_list.append("Detectors are not calibrated (data/calibration.json has no fitted fusion). Run tools/calibrate.py.")

    temporal = None
    if is_video:
        try:
            temporal = temporal_features_bytes(contents)
        except Exception:
            temporal = None

    if is_video:
        v = ensemble.score_frames(frames, commfor_model, bfree_model, clip_model)
        avg_c, avg_b, avg_clip = v["commfor"], v["bfree"], v["clip"]
        frames_analyzed = len(frames)
        frame_scores = [round(float(x), 4) for x in v["frame_fused"] if x is not None]
        gradcam_url = generate_gradcam_heatmap(first_frame_img, commfor_detector=commfor_model, face_box=None) if first_frame_img else None
    else:
        s = ensemble.score_image(img, commfor_model, bfree_model, clip_model, face_box=face_box)
        avg_c, avg_b, avg_clip = s["commfor"], s["bfree"], s["clip"]
        frames_analyzed = 1
        frame_scores = []
        gradcam_url = generate_gradcam_heatmap(img, commfor_detector=commfor_model, face_box=face_box)

    # Face-vs-background ELA ratio: shown as evidence only, it no longer overrides the verdict.
    region = None
    if not is_video and img is not None and face_box:
        ratio = face_region_ela(img, face_box)
        if ratio is not None:
            region = {"ela_ratio": ratio, "ela_mismatch": bool(ratio > 2.40 or ratio < 0.40)}

    display_b = avg_b if avg_b is not None else avg_c

    if stage_callback:
        await stage_callback(3, "Deep Semantic Vision Backbones", "completed", {
            "commfor": round(avg_c, 4),
            "bfree": round(display_b, 4),
            "clip": round(avg_clip, 4) if avg_clip is not None else None,
            "gend": round(gend_score, 4) if gend_score is not None else None,
            "faces_found": faces_found,
        })

    if stage_callback:
        await stage_callback(4, "Calibrated Ensemble Verdict & Confidence Variance", "running", {
            "message": "Fusing calibrated detector scores with the fitted logistic model..."
        })

    valid_scores = [x for x in (avg_c, avg_b, avg_clip, gend_score) if x is not None]
    confidence_variance = float(np.var(valid_scores)) if len(valid_scores) > 1 else 0.0

    scene_scores = {"commfor": avg_c, "bfree": avg_b, "clip": avg_clip}
    media_kind = "video" if is_video else "image"
    fused_scene = ensemble.fuse(scene_scores, media=media_kind)
    fused_all = ensemble.fuse({**scene_scores, "gend": gend_score}, media=media_kind)
    threshold = float(fused_all["threshold"])
    prov_score = 1.0 if is_prov_fake else 0.0

    def _likelihood(p: float, fake: bool) -> str:
        if not fake:
            return "Unlikely to be manipulated"
        return "Very likely" if p >= 0.90 else ("Likely" if p >= 0.70 else "Possible")

    def _neutral_result(prediction: str, category: str, summary: str, findings: List[str]) -> Dict[str, Any]:
        return {
            "success": True,
            "mode": mode,
            "prediction": prediction,
            "score": 0.0,
            "confidence": "N/A",
            "confidence_variance": round(confidence_variance, 6),
            "commfor_score": round(avg_c, 6),
            "bfree_score": round(display_b, 6),
            "clip_score": round(avg_clip, 6) if avg_clip is not None else None,
            "prov_score": prov_score,
            "frames_analyzed": int(frames_analyzed),
            "temporal": temporal,
            "gend_score": None,
            "faces_found": faces_found,
            "face_model": face.get("model"),
            "face_checked": face_checked,
            "warnings": warnings_list,
            "fusion": None,
            "explanation": {"category": category, "summary": summary, "findings": findings, "likelihood": "Inconclusive"},
            "media_type": "video" if is_video else "image",
            "frame_scores": frame_scores,
            "exif": exif,
            "ela": ela_url,
            "residual_map": residual_url,
            "gradcam": gradcam_url,
            "attention_heatmap": gradcam_url,
            "face_box": None,
            "region": None,
            "quality": quality,
        }

    async def _finish_neutral(prediction: str, category: str, summary: str, findings: List[str]):
        if stage_callback:
            await stage_callback(4, "Calibrated Ensemble Verdict & Confidence Variance", "completed", {
                "final_score": 0.0, "prediction": prediction, "confidence_variance": round(confidence_variance, 5),
            })
        return _neutral_result(prediction, category, summary, findings)

    if mode == "deepfake":
        media = "video" if is_video else "image"
        if not face_checked:
            return await _finish_neutral(
                "Inconclusive", "Face Check Unavailable",
                f"The face-swap check could not run, so this {media} was not evaluated for deepfakes.",
                [face["error"] or "Face-forensics worker unavailable.",
                 "Retry, or use 'AI-Generated' / 'Auto' mode for whole-image analysis."],
            )
        if faces_found == 0:
            return await _finish_neutral(
                "No Face Detected", "No Face Detected",
                "Deepfake analysis requires a visible human face, but no face was found.",
                ["The face detector found no face of at least 40 px in this media.",
                 "Deepfake mode evaluates face swaps and facial manipulation only.",
                 "Switch to 'AI-Generated' or 'Auto' mode to test full-scene synthesis."],
            )
        if gend_score is None:
            return await _finish_neutral(
                "Inconclusive", "Face Not Scored",
                "A face was found but the face-forensics model could not score it.",
                ["Retry with a larger or clearer face."],
            )

        final_score = float(fused_all["p_fake"])
        fusion_used = fused_all
        is_fake = final_score > threshold
        findings = []
        if is_fake:
            category = "Deepfake (Face Swap)"
            summary = f"A face in this {media} shows signs of being replaced or manipulated by AI."
            if gend_score > 0.5:
                findings.append(f"GenD face model: {gend_score * 100:.0f}% probability that the face is manipulated.")
            if max(avg_c, display_b) > 0.5:
                findings.append("Whole-image detectors also found synthetic pixel patterns.")
        else:
            category = "Authentic Face"
            summary = f"No sign of face replacement or facial manipulation was found in this {media}."
            findings.append(f"GenD face model: {gend_score * 100:.0f}% probability of manipulation (below the decision threshold).")
        if region and region.get("ela_mismatch"):
            findings.append(f"Compression (ELA) in the face region differs from the background (ratio {region['ela_ratio']:.2f}).")
        explanation = {"category": category, "summary": summary, "findings": findings, "likelihood": _likelihood(final_score, is_fake)}

    elif mode == "ai_generated":
        fusion_used = fused_scene
        final_score = 1.0 if is_prov_fake else float(fused_scene["p_fake"])
        is_fake = final_score > threshold
        media = "video" if is_video else "image"
        findings = []
        if is_prov_fake:
            findings.append("C2PA content credentials or embedded metadata explicitly declare this file as AI-generated.")
        if is_fake:
            category = "AI-Generated (Synthetic)"
            summary = f"This {media} shows signs of being generated by an AI model."
            for name, val in (("Community Forensics ViT", avg_c), ("NPR ResNet-50", avg_b), ("UniversalFakeDetect CLIP", avg_clip)):
                if val is not None and val > 0.5:
                    findings.append(f"{name}: {val * 100:.0f}% probability of synthetic content.")
        else:
            category = "Authentic Capture"
            summary = f"No generative-AI fingerprints were found in this {media}."
            findings.append("The whole-image detectors stayed below the calibrated decision threshold.")
        explanation = {"category": category, "summary": summary, "findings": findings, "likelihood": _likelihood(final_score, is_fake)}

    else:
        fusion_used = fused_all
        final_score = max(float(fused_all["p_fake"]), 0.99) if is_prov_fake else float(fused_all["p_fake"])
        is_fake = final_score > threshold
        explanation = build_explanation(
            is_fake, is_video, avg_c, display_b, gend_score, is_prov_fake,
            prov_result, faces_found, region, clip=avg_clip, quality=quality,
            confidence_variance=confidence_variance, final_score=final_score,
        )

    confidence = final_score if is_fake else (1.0 - final_score)

    if stage_callback:
        await stage_callback(4, "Calibrated Ensemble Verdict & Confidence Variance", "completed", {
            "final_score": round(final_score, 4),
            "prediction": "Fake" if is_fake else "Real",
            "confidence_variance": round(confidence_variance, 5),
        })

    return {
        "success": True,
        "mode": mode,
        "prediction": "Fake" if is_fake else "Real",
        "score": round(final_score, 6),
        "confidence": f"{confidence * 100:.2f}%",
        "confidence_variance": round(confidence_variance, 6),
        "commfor_score": round(avg_c, 6),
        "bfree_score": round(display_b, 6),
        "clip_score": round(avg_clip, 6) if avg_clip is not None else None,
        "prov_score": round(prov_score, 6),
        "frames_analyzed": int(frames_analyzed),
        "temporal": temporal,
        "gend_score": round(gend_score, 6) if gend_score is not None else None,
        "gend_frame_scores": face.get("frame_scores") or [],
        "faces_found": faces_found,
        "face_model": face.get("model"),
        "face_checked": face_checked,
        "warnings": warnings_list,
        "fusion": {
            "model": fusion_used["model"],
            "threshold": threshold,
            "calibrated": fusion_used["calibrated"],
            "contributions": fusion_used["contributions"],
        },
        "explanation": explanation,
        "media_type": "video" if is_video else "image",
        "frame_scores": frame_scores,
        "exif": exif,
        "ela": ela_url,
        "residual_map": residual_url,
        "gradcam": gradcam_url,
        "attention_heatmap": gradcam_url,
        "face_box": face_box if not is_video else None,
        "region": region,
        "quality": quality,
    }

# ── FASTAPI ROUTES ─────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    # The web console is the Next.js app; send browsers that open the API root there.
    return RedirectResponse(url=FRONTEND_URL)


class UrlFetchRequest(BaseModel):
    url: str


def _sync_fetch(url: str, upload_dir: str):
    import mimetypes
    import requests
    import shutil
    import uuid

    for prefix in ("http://127.0.0.1:8000/", "http://localhost:8000/", "http://0.0.0.0:8000/", "/"):
        if url.startswith(prefix):
            local_rel = url[len(prefix):].split("?")[0]
            if os.path.exists(local_rel):
                fname = f"copy_{uuid.uuid4().hex[:8]}_{os.path.basename(local_rel)}"
                dest = os.path.join(upload_dir, fname)
                shutil.copyfile(local_rel, dest)
                ext = os.path.splitext(fname)[1].lower()
                return {
                    "success": True,
                    "filename": fname,
                    "file_url": f"/static/uploads/{fname}",
                    "content_type": "video/mp4" if ext in (".mp4", ".mov", ".avi") else "image/jpeg",
                    "size": os.path.getsize(dest),
                }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(url, headers=headers, stream=True, timeout=15)
        ct = r.headers.get("Content-Type", "").lower()
        is_media_type = any(ct.startswith(p) for p in ("image/", "video/"))
        has_media_ext = any(url.lower().split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".webm"))

        if r.status_code == 200 and not ct.startswith("text/") and (is_media_type or has_media_ext):
            clean_url = url.lower().split("?")[0]
            if clean_url.endswith(".png"): ext = ".png"
            elif clean_url.endswith(".mp4"): ext = ".mp4"
            elif clean_url.endswith(".mov"): ext = ".mov"
            elif clean_url.endswith(".webm"): ext = ".webm"
            elif clean_url.endswith(".webp"): ext = ".webp"
            else: ext = mimetypes.guess_extension(ct.split(";")[0]) or ".jpg"

            fname = f"fetched_{uuid.uuid4().hex[:10]}{ext}"
            fpath = os.path.join(upload_dir, fname)

            size = 0
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        size += len(chunk)
                        if size > 60 * 1024 * 1024:
                            f.close()
                            os.remove(fpath)
                            return {"success": False, "error": "Media exceeds 60MB size limit."}
                        f.write(chunk)

            return {
                "success": True,
                "filename": fname,
                "file_url": f"/static/uploads/{fname}",
                "content_type": ct or ("video/mp4" if ext == ".mp4" else "image/jpeg"),
                "size": size,
            }
    except Exception:
        pass

    try:
        import yt_dlp
        ffmpeg_bin = None
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None

        uid = uuid.uuid4().hex[:10]
        out_tmpl = os.path.join(upload_dir, f"social_{uid}.%(ext)s")
        ydl_opts = {
            "outtmpl": out_tmpl,
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
            "merge_output_format": "mp4",
            "max_filesize": 60 * 1024 * 1024,
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            "quiet": True,
            "no_warnings": True,
        }
        if ffmpeg_bin and os.path.exists(ffmpeg_bin):
            ydl_opts["ffmpeg_location"] = ffmpeg_bin
        else:
            ydl_opts["format"] = "best[height<=720][ext=mp4]/best[height<=720]"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            saved_path = ydl.prepare_filename(info)

        if not os.path.exists(saved_path):
            base = os.path.splitext(saved_path)[0]
            for candidate_ext in (".mp4", ".mkv", ".webm", ".mov", ".avi"):
                if os.path.exists(base + candidate_ext):
                    saved_path = base + candidate_ext
                    break

        if not os.path.exists(saved_path):
            return {"success": False, "error": "Video downloaded but output file could not be located."}

        fname = os.path.basename(saved_path)
        ext = os.path.splitext(fname)[1].lower()
        return {
            "success": True,
            "filename": fname,
            "file_url": f"/static/uploads/{fname}",
            "content_type": "video/mp4" if ext in (".mp4", ".mov", ".avi") else f"video/{ext.lstrip('.')}",
            "size": os.path.getsize(saved_path),
        }
    except Exception as e:
        return {"success": False, "error": f"Could not fetch media: {str(e)}"}


@app.post("/fetch-url")
async def fetch_url(payload: UrlFetchRequest):
    url = (payload.url or "").strip()
    if not url:
        return {"success": False, "error": "URL cannot be empty."}
    upload_dir = os.path.join(STATIC_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    return await asyncio.to_thread(_sync_fetch, url, upload_dir)


@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = Form("auto")):
    """Synchronous deepfake forensics prediction endpoint."""
    contents = await file.read()
    filename = file.filename or "media"
    content_type = file.content_type or ""
    try:
        return await execute_pipeline(contents, filename, content_type, mode)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e) or type(e).__name__}


@app.post("/api/predict/stream")
async def predict_stream(file: UploadFile = File(...), mode: str = Form("auto")):
    """Streaming deepfake prediction endpoint emitting real-time Server-Sent Events (SSE)."""
    contents = await file.read()
    filename = file.filename or "media"
    content_type = file.content_type or ""

    async def sse_event_stream():
        event_queue = asyncio.Queue()

        async def stage_callback(stage_num: int, stage_name: str, status: str, details: dict):
            await event_queue.put({
                "event": "stage",
                "stage": stage_num,
                "name": stage_name,
                "status": status,
                "progress": stage_num * 25,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "details": details,
            })

        async def worker_task():
            try:
                final_data = await execute_pipeline(
                    contents, filename, content_type, mode, stage_callback=stage_callback
                )
                await event_queue.put({"event": "complete", "result": final_data})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                await event_queue.put({"event": "error", "error": str(exc)})
            finally:
                await event_queue.put(None)

        task = asyncio.create_task(worker_task())
        yield f"data: {json.dumps({'event': 'started', 'filename': filename, 'mode': mode})}\n\n"

        while True:
            msg = await event_queue.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg)}\n\n"

        await task

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


# ── CROWD SURVEILLANCE & THREAT INTELLIGENCE ENDPOINTS ────────────────────
@app.post("/api/crowd/detect")
async def crowd_detect_image(file: UploadFile = File(...)):
    """
    Live frame / static surveillance image analysis:
    Runs YOLO person tracking, weapon classification, behavior analysis,
    and instantaneous feed integrity verification.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return {"success": False, "error": "Invalid image format"}

    # Verify frame integrity (check for deepfake/synthetic injection)
    is_tampered, integrity_score, integ_details = check_frame_integrity(frame_bgr)

    # Process frame through surveillance engine
    processed = feed_processor.process_frame(
        frame_bgr,
        feed_tampered=is_tampered,
        feed_integrity_score=integrity_score,
    )

    res = processed.to_dict()
    res["annotated_image"] = processed.to_base64_jpg(quality=85)
    res["success"] = True
    return res


@app.post("/api/crowd/video")
async def crowd_analyze_video(file: UploadFile = File(...)):
    """
    Full Surveillance Video Analysis:
    Tracks crowd dynamics, person paths, weapon appearances, and feed integrity
    across the temporal timeline of the uploaded video.
    """
    contents = await file.read()
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t:
        t.write(contents)
        temp_path = t.name

    try:
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        duration_sec = total_frames / max(fps, 1.0)

        # Sample up to 16 frames uniformly across the video
        num_samples = min(16, max(6, int(duration_sec * 2)))
        indices = np.linspace(0, max(total_frames - 1, 0), num_samples).astype(int)

        feed_processor.reset()
        timeline = []
        sampled_annotated_frames = []
        peak_crowd = 0
        all_weapon_labels = set()
        feed_tampered_count = 0
        integrity_scores = []

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if idx in indices:
                # Run feed integrity check on sampled frames
                is_tampered, integrity_score, _ = check_frame_integrity(frame)
                if is_tampered:
                    feed_tampered_count += 1
                integrity_scores.append(integrity_score)

                sec = round(idx / max(fps, 1.0), 2)
                processed = feed_processor.process_frame(
                    frame,
                    timestamp=f"{sec:.1f}s",
                    feed_tampered=is_tampered,
                    feed_integrity_score=integrity_score,
                )

                peak_crowd = max(peak_crowd, processed.crowd_count)
                for w in processed.weapon_labels:
                    all_weapon_labels.add(w)

                timeline.append({
                    "timestamp": f"{sec:.1f}s",
                    "time_seconds": sec,
                    "crowd_count": processed.crowd_count,
                    "risk_score": processed.risk["score"],
                    "risk_level": processed.risk["level"],
                    "weapon_count": len(processed.weapon_labels),
                    "feed_integrity": round(integrity_score * 100, 1),
                    "is_tampered": is_tampered,
                })

                if len(sampled_annotated_frames) < 6:
                    sampled_annotated_frames.append({
                        "timestamp": f"{sec:.1f}s",
                        "image": processed.to_base64_jpg(quality=75),
                        "crowd_count": processed.crowd_count,
                        "risk_level": processed.risk["level"],
                    })

        cap.release()
        avg_integrity = float(np.mean(integrity_scores)) if integrity_scores else 1.0
        feed_compromised = feed_tampered_count >= max(1, len(indices) // 4)

        overall_risk = compute_risk(
            crowd_count=peak_crowd,
            weapon_labels=list(all_weapon_labels),
            suspicious=False,
            feed_tampered=feed_compromised,
            feed_integrity_score=avg_integrity,
        )

        return {
            "success": True,
            "filename": file.filename,
            "duration_seconds": round(duration_sec, 2),
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "peak_crowd": peak_crowd,
            "weapons_detected": list(all_weapon_labels),
            "has_weapons": len(all_weapon_labels) > 0,
            "overall_risk": overall_risk,
            "feed_integrity": {
                "score": round(avg_integrity, 3),
                "is_compromised": feed_compromised,
                "tampered_samples": feed_tampered_count,
                "total_samples": len(indices),
            },
            "timeline": timeline,
            "sample_frames": sampled_annotated_frames,
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ── NOVEL DUAL-PIPELINE INCIDENT INVESTIGATION ENDPOINT ────────────────────
@app.post("/api/investigate")
async def incident_investigation(file: UploadFile = File(...)):
    """
    Unified Incident Investigation Mode:
    Executes BOTH the Crowd Surveillance Pipeline AND the Media Forensics Pipeline
    simultaneously on the same incident footage.
    Correlates physical security anomalies with digital tampering evidence.
    """
    contents = await file.read()
    filename = file.filename or "incident.mp4"
    content_type = file.content_type or "video/mp4"

    # 1. Run Deepfake Media Forensics
    deepfake_result = await execute_pipeline(contents, filename, content_type, mode="auto")
    if not deepfake_result.get("success"):
        # Without a forensics result the media must not be reported as authentic.
        return {"success": False, "error": deepfake_result.get("error", "Media forensics analysis failed.")}

    # 2. Run Crowd Surveillance on the media
    is_video = (
        filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        or content_type.startswith("video/")
    )

    if is_video:
        frames, timestamps = sample_video_bytes_with_timestamps(contents, n_frames=8)
        feed_processor.reset()
        peak_crowd = 0
        weapon_labels = set()
        annotated_samples = []

        for idx, frame_img in enumerate(frames):
            frame_cv = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
            processed = feed_processor.process_frame(
                frame_cv,
                timestamp=f"{timestamps[idx]:.1f}s" if idx < len(timestamps) else f"Frame {idx}",
                feed_tampered=bool(deepfake_result.get("prediction") == "Fake"),
                feed_integrity_score=float(1.0 - deepfake_result.get("score", 0.0)),
            )
            peak_crowd = max(peak_crowd, processed.crowd_count)
            for w in processed.weapon_labels:
                weapon_labels.add(w)

            if len(annotated_samples) < 3:
                annotated_samples.append({
                    "timestamp": f"{timestamps[idx]:.1f}s" if idx < len(timestamps) else f"Frame {idx}",
                    "image": processed.to_base64_jpg(quality=75),
                    "crowd_count": processed.crowd_count,
                    "risk_level": processed.risk["level"],
                })

        crowd_summary = {
            "peak_crowd": peak_crowd,
            "weapons_found": list(weapon_labels),
            "annotated_samples": annotated_samples,
        }
    else:
        nparr = np.frombuffer(contents, np.uint8)
        frame_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_cv is not None:
            processed = feed_processor.process_frame(
                frame_cv,
                feed_tampered=bool(deepfake_result.get("prediction") == "Fake"),
                feed_integrity_score=float(1.0 - deepfake_result.get("score", 0.0)),
            )
            crowd_summary = {
                "peak_crowd": processed.crowd_count,
                "weapons_found": processed.weapon_labels,
                "annotated_samples": [{
                    "timestamp": "0.0s",
                    "image": processed.to_base64_jpg(quality=80),
                    "crowd_count": processed.crowd_count,
                    "risk_level": processed.risk["level"],
                }],
            }
        else:
            crowd_summary = {"peak_crowd": 0, "weapons_found": [], "annotated_samples": []}

    # 3. Synthesize Unified Threat & Forensic Incident Verdict
    is_fake = deepfake_result.get("prediction") == "Fake"
    fake_score = float(deepfake_result.get("score", 0.0))
    has_weapons = len(crowd_summary["weapons_found"]) > 0

    if is_fake and has_weapons:
        incident_verdict = "CRITICAL / FABRICATED THREAT (DEEPFAKE DISINFORMATION)"
        verdict_summary = "ALERT: Footage shows physical weapons AND has been verified as synthetic/manipulated. High probability of AI-generated security disinformation or staged hoax."
        threat_level = "CRITICAL"
    elif is_fake and not has_weapons:
        incident_verdict = "MEDIA MANIPULATION DETECTED (DIGITAL SPOOFING)"
        verdict_summary = "This footage contains AI-generated or face-swapped deepfake content. The captured scene has been digitally altered."
        threat_level = "HIGH"
    elif not is_fake and has_weapons:
        incident_verdict = "VERIFIED PHYSICAL SECURITY INCIDENT"
        verdict_summary = "AUTHENTIC EVIDENCE: Camera physics verified as natural optical capture. Real-world weapon threat detected in the crowd."
        threat_level = "CRITICAL"
    elif crowd_summary["peak_crowd"] > 15:
        incident_verdict = "VERIFIED CROWD SURGE (AUTHENTIC)"
        verdict_summary = "AUTHENTIC EVIDENCE: Camera physics verified. High crowd density event detected without digital tampering."
        threat_level = "ELEVATED"
    else:
        incident_verdict = "ROUTINE CAPTURE (AUTHENTIC & SAFE)"
        verdict_summary = "Footage is authentic optical capture. No physical weapons or digital tampering detected."
        threat_level = "SAFE"

    return {
        "success": True,
        "filename": filename,
        "incident_verdict": incident_verdict,
        "verdict_summary": verdict_summary,
        "threat_level": threat_level,
        "media_authenticity": {
            "prediction": deepfake_result.get("prediction"),
            "manipulation_score": fake_score,
            "confidence": deepfake_result.get("confidence"),
            "category": deepfake_result.get("explanation", {}).get("category"),
            "findings": deepfake_result.get("explanation", {}).get("findings", []),
            "warnings": deepfake_result.get("warnings", []),
            "ela_heatmap": deepfake_result.get("ela"),
            "gradcam_heatmap": deepfake_result.get("gradcam"),
            "residual_map": deepfake_result.get("residual_map"),
        },
        "crowd_intelligence": crowd_summary,
        "deepfake_raw": deepfake_result,
    }


# ── ACTIVE LEARNING FEEDBACK & BENCHMARK ACCURACY ENDPOINT ─────────────────
def compute_benchmark_and_feedback_stats() -> Dict[str, Any]:
    stats = {"benchmarks": {}, "feedback": {"total_samples": 0, "fake_samples": 0, "real_samples": 0}}
    # Held-out accuracy measured by tools/calibrate.py (stored in calibration.json -> "eval").
    ev = (calibration.load() or {}).get("eval") or {}
    if ev:
        allm = ev.get("ALL", {}).get("new_fusion", {})
        det = ev.get("single_detectors", {})
        stats["benchmarks"] = {
            "total_samples": allm.get("n"),
            "ensemble_balanced_accuracy": allm.get("balanced_acc"),
            "ensemble_fake_caught": allm.get("fake_caught"),
            "ensemble_real_correct": allm.get("real_correct"),
            "detector_auc": {k: v.get("auc") for k, v in det.items()},
        }

    if os.path.isfile(FEEDBACK_PATH):
        try:
            total, fake_n, real_n = 0, 0, 0
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    total += 1
                    if rec.get("is_fake", True):
                        fake_n += 1
                    else:
                        real_n += 1
            stats["feedback"] = {"total_samples": total, "fake_samples": fake_n, "real_samples": real_n}
        except Exception as e:
            stats["feedback"]["error"] = str(e)

    return stats


@app.post("/feedback")
async def feedback_endpoint(request: Request):
    try:
        raw_body = await request.json()
    except Exception:
        raw_body = []

    items = raw_body if isinstance(raw_body, list) else [raw_body]
    stored_count = 0

    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        for item in items:
            if not isinstance(item, dict):
                continue

            scores_dict = item.get("scores") or {}
            commfor_s = scores_dict.get("commfor") if "commfor" in scores_dict else item.get("commfor")
            bfree_s = scores_dict.get("bfree") if "bfree" in scores_dict else item.get("bfree", item.get("ufd"))
            clip_s = scores_dict.get("clip") if "clip" in scores_dict else item.get("clip")
            gend_s = scores_dict.get("gend") if "gend" in scores_dict else item.get("gend")

            user_corr = item.get("user_correction") or item.get("user_label") or item.get("is_fake")
            if isinstance(user_corr, str):
                is_fake = user_corr.lower() in ("fake", "ai_generated", "deepfake", "true", "1")
            elif isinstance(user_corr, bool):
                is_fake = user_corr
            else:
                is_fake = bool(item.get("is_fake", True))

            file_hash = item.get("file_hash") or hashlib.sha256(f"{commfor_s}_{bfree_s}_{clip_s}_{gend_s}_{time.time()}".encode("utf-8")).hexdigest()[:16]

            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "file_hash": file_hash,
                "scores": {
                    "commfor": float(commfor_s) if commfor_s is not None else None,
                    "bfree": float(bfree_s) if bfree_s is not None else None,
                    "clip": float(clip_s) if clip_s is not None else None,
                    "gend": float(gend_s) if gend_s is not None else None,
                },
                "user_label": "fake" if is_fake else "real",
                "user_correction": str(user_corr),
                "is_fake": is_fake,
                "comments": str(item.get("comments", "")),
            }
            f.write(json.dumps(entry) + "\n")
            stored_count += 1

    stats = compute_benchmark_and_feedback_stats()
    return {"ok": True, "stored": stored_count, "stats": stats}
