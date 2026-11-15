"""
Layer 2: Pixel-based AI-image detectors & Image Quality Forensics.

Detectors:
- CommForDetector: Community Forensics (CVPR 2025), ViT-S/16 @384.
  Aspect-ratio preserving shortest-edge resize (440px) + center / multi-crop (384px).
- BFreeDetector: B-Free (CVPR 2025), DINOv2-reg4 ViT-B on 5 native-resolution 504px crops.
- CLIPDetector: UniversalFakeDetect CLIP ViT-L/14 with trained linear classifier.
- Image Quality Analyzer: Laplacian variance (blur) & 8x8 JPEG quantization blockiness.

Every detector outputs a normalized, calibrated probability P(fake) in [0.0, 1.0].
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from typing import List, Optional, Tuple, Union, Dict

from detectors import calibration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]


def estimate_image_quality(img: Union[Image.Image, np.ndarray]) -> Dict[str, Union[float, bool]]:
    """
    Estimate image blur and JPEG re-compression level.
    
    1. Laplacian Variance: measures high-frequency edge sharpness.
       Values < 80 signify heavy blur, downscaling, or smoothing.
    2. JPEG 8x8 Blocking: measures discontinuity at 8x8 DCT block boundaries.
       Ratio > 1.15 signifies heavy JPEG re-compression (e.g. WhatsApp, social media).
    3. Compression Score in [0.0, 1.0]: composite metric for dynamic gating.
    """
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("L"))
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            arr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if img.shape[2] == 3 else cv2.COLOR_BGR2GRAY)
        else:
            arr = img
    else:
        return {"lap_var": 150.0, "block_ratio": 1.0, "compression_score": 0.0, "is_heavily_compressed": False}

    # 1. Laplacian variance
    lap_var = float(cv2.Laplacian(arr, cv2.CV_64F).var())

    # 2. JPEG 8x8 grid blocking artifact ratio
    H, W = arr.shape
    if H >= 64 and W >= 64:
        # Differences across 8x8 block boundaries
        h_boundary = np.abs(arr[8:H:8, :].astype(np.float64) - arr[7:H-1:8, :].astype(np.float64)).mean()
        v_boundary = np.abs(arr[:, 8:W:8].astype(np.float64) - arr[:, 7:W-1:8].astype(np.float64)).mean()

        # Differences inside 8x8 blocks (4 pixels offset from boundary)
        h_internal = np.abs(arr[4:H:8, :].astype(np.float64) - arr[3:H-1:8, :].astype(np.float64)).mean()
        v_internal = np.abs(arr[:, 4:W:8].astype(np.float64) - arr[:, 3:W-1:8].astype(np.float64)).mean()

        block_ratio = float((h_boundary + v_boundary) / (h_internal + v_internal + 1e-6))
    else:
        block_ratio = 1.0

    # Composite compression severity score in [0.0, 1.0]
    # High block_ratio (>1.12) or very low lap_var (<70) indicate compression/blur
    blur_penalty = max(0.0, min(1.0, (120.0 - lap_var) / 100.0))
    block_penalty = max(0.0, min(1.0, (block_ratio - 1.02) / 0.25))
    compression_score = float(np.clip(0.55 * block_penalty + 0.45 * blur_penalty, 0.0, 1.0))
    is_heavily_compressed = bool(compression_score > 0.50 or block_ratio > 1.18 or lap_var < 55.0)

    return {
        "lap_var": round(lap_var, 2),
        "block_ratio": round(block_ratio, 3),
        "compression_score": round(compression_score, 3),
        "is_heavily_compressed": is_heavily_compressed
    }


class CommForDetector:
    """
    Community Forensics (CVPR 2025) ViT-S/16 @384.
    Trained on 4,803 generative models for cross-generator synthetic media detection.
    Evaluates shortest-edge resized multi-crops preserving aspect ratio.
    """
    name = "Community Forensics (2025)"
    repo_id = "OwensLab/commfor-model-384"

    def __init__(self):
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
        vit.head = nn.Linear(384, 1, bias=True)

        weights = hf_hub_download(self.repo_id, "model.safetensors")
        state = load_file(weights)
        state = {k[4:] if k.startswith("vit.") else k: v for k, v in state.items()}
        vit.load_state_dict(state, strict=True)
        self.model = vit.to(DEVICE).eval()

        # Aspect-ratio preserving shortest edge resize (440px)
        self.resize_short = T.Resize(440, interpolation=T.InterpolationMode.BICUBIC)
        self.center_crop = T.CenterCrop(384)
        self.five_crop = T.FiveCrop(384)
        self.norm = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.inference_mode()
    def get_attention_map(self, img: Image.Image) -> np.ndarray:
        """
        Extract 24x24 patch token classification saliency map from ViT-S/16.
        Returns normalized 2D numpy array in [0.0, 1.0].
        """
        im = img.convert("RGB")
        resized = self.resize_short(im)
        cropped = self.center_crop(resized)
        tensor = self.norm(cropped).unsqueeze(0).to(DEVICE)

        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else torch.nullcontext()
        with autocast_ctx:
            feat = self.model.forward_features(tensor)
            patch_tokens = feat[:, 1:, :]  # (1, 576, 384)
            patch_logits = self.model.head(patch_tokens)  # (1, 576, 1)
            grid = torch.sigmoid(patch_logits).view(24, 24).cpu().float().numpy()
        return grid

    @torch.inference_mode()
    def predict(self, images: Union[Image.Image, List[Image.Image]], use_multi_crop: bool = False) -> List[float]:
        """
        Evaluate images with shortest-edge 440px resize + 384px center crop or 5-crop.
        Returns calibrated P(fake) in [0.0, 1.0] for each image.
        Uses torch.inference_mode and AMP FP16 to minimize latency & VRAM.
        """
        if isinstance(images, Image.Image):
            images = [images]

        if not images:
            return []

        views_per_image = []
        all_tensors = []

        for im in images:
            im = im.convert("RGB")
            # Resize shortest edge to 440px, maintaining original aspect ratio
            resized = self.resize_short(im)

            if use_multi_crop:
                # 5 crops: top-left, top-right, bottom-left, bottom-right, center + h-flip center
                crops = list(self.five_crop(resized))
                c_center = self.norm(crops[4])
                v_tensors = [self.norm(c) for c in crops] + [torch.flip(c_center, dims=[2])]
            else:
                # Center crop + its horizontal flip (TTA)
                c = self.norm(self.center_crop(resized))
                v_tensors = [c, torch.flip(c, dims=[2])]

            views_per_image.append(len(v_tensors))
            all_tensors.extend(v_tensors)

        # Batch GPU inference in a single forward pass with AMP autocast
        batch = torch.stack(all_tensors).to(DEVICE)
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else torch.nullcontext()
        with autocast_ctx:
            logits = self.model(batch).flatten()
            probs = torch.sigmoid(logits).cpu().float().numpy()

        results = []
        cursor = 0
        for count in views_per_image:
            sub_probs = probs[cursor : cursor + count]
            raw_p = float(np.mean(sub_probs))
            # Apply score calibration
            cal_p = calibration.calibrate_score("commfor", raw_p, method="platt")
            results.append(float(cal_p if cal_p is not None else raw_p))
            cursor += count

        return results


class BFreeDetector:
    """
    B-Free (CVPR 2025, grip-unina/B-Free): DINOv2-reg4 ViT-B trained with a bias-free
    paradigm on Stable Diffusion self-conditioned reconstructions and inpaintings.

    Official inference: the full-resolution image is normalized with ImageNet statistics and
    scored on 5 crops of 504x504 (centre + 4 corners, Wrapper5crops); no resizing.
    Weights: detectors/B-Free/code/weights/BFREE_dino2reg4/ (see detectors/B-Free/code/README.md).

    The previous NPR ResNet-50 backend was removed: on the labelled evaluation set its
    output was below -10 logits for almost every real and fake image (AUC ~0.5).
    """
    name = "B-Free (CVPR 2025)"
    MAX_SIDE = 1024  # centre-crop (never resize) larger inputs to bound GPU memory and time

    def __init__(self):
        self.model = None
        self.is_available = False
        here = os.path.dirname(os.path.abspath(__file__))
        code = os.path.join(here, "B-Free", "code")
        wdir = os.path.join(code, "weights", "BFREE_dino2reg4")
        cfg_path = os.path.join(wdir, "config.yaml")
        if not os.path.isfile(cfg_path):
            print(f"B-Free weights not found in {wdir}; detector disabled.")
            return
        try:
            import importlib.util
            import yaml

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            # Load B-Free's own 'networks' package by path so it cannot clash with ours.
            spec = importlib.util.spec_from_file_location(
                "bfree_networks", os.path.join(code, "networks", "__init__.py"),
                submodule_search_locations=[os.path.join(code, "networks")])
            mod = importlib.util.module_from_spec(spec)
            import sys
            sys.modules["bfree_networks"] = mod
            spec.loader.exec_module(mod)
            self.model = mod.load_weights(mod.get_network(cfg["arch"]), os.path.join(wdir, cfg["weights_file"]))
            self.model = self.model.to(DEVICE).eval()
            self.norm = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
            self.is_available = True
            print("B-Free detector (DINOv2-reg4 ViT-B, 5x504 crops) loaded successfully.")
        except Exception as e:
            print(f"Warning: B-Free failed to load ({e}); detector disabled.")
            self.model = None

    def _prep(self, im: Image.Image) -> torch.Tensor:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > self.MAX_SIDE:
            cw, ch = min(w, self.MAX_SIDE), min(h, self.MAX_SIDE)
            l, t = (w - cw) // 2, (h - ch) // 2
            im = im.crop((l, t, l + cw, t + ch))
        return self.norm(im).unsqueeze(0)

    @torch.inference_mode()
    def predict_logits(self, images: Union[Image.Image, List[Image.Image]]) -> List[float]:
        if isinstance(images, Image.Image):
            images = [images]
        out = []
        ctx = torch.amp.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else torch.autocast("cpu", enabled=False)
        for im in images:  # images differ in size; the official code also runs one at a time
            with ctx:
                y = self.model(self._prep(im).to(DEVICE)).float().cpu().numpy()
            out.append(float(y[0, 0] if y.shape[1] == 1 else y[0, 1] - y[0, 0]))
        return out

    def predict(
        self,
        images: Union[Image.Image, List[Image.Image]],
        face_crops: Optional[Union[Image.Image, List[Optional[Image.Image]]]] = None,
    ) -> List[Optional[float]]:
        """Calibrated P(fake) per image. face_crops is accepted for API compatibility and
        ignored: B-Free is designed for whole images at native resolution."""
        if isinstance(images, Image.Image):
            images = [images]
        if not self.is_available or self.model is None:
            return [None for _ in images]
        res = []
        for z in self.predict_logits(images):
            raw_p = float(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))
            cal_p = calibration.calibrate_score("bfree", raw_p, method="platt")
            res.append(float(cal_p if cal_p is not None else raw_p))
        return res


class CLIPDetector:
    """
    UniversalFakeDetect CLIP ViT-L/14 with trained linear classifier.
    Evaluates global semantic representations and latent vision-language alignment.
    """
    name = "UniversalFakeDetect (CLIP ViT-L/14)"

    def __init__(self):
        self.model = None
        self.is_available = False
        here = os.path.dirname(os.path.abspath(__file__))
        app_root = os.path.dirname(here)
        fc_path = os.path.join(app_root, "pretrained_weights", "fc_weights.pth")

        if os.path.isfile(fc_path):
            try:
                import sys
                if app_root not in sys.path:
                    sys.path.insert(0, app_root)
                from models.clip_models import CLIPModel

                net = CLIPModel("ViT-L/14", num_classes=1)
                st = torch.load(fc_path, map_location="cpu", weights_only=False)
                net.fc.load_state_dict(st)
                self.model = net.to(DEVICE).eval()
                self.transform = T.Compose([
                    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(CLIP_MEAN, CLIP_STD),
                ])
                self.is_available = True
                print("CLIP Detector (ViT-L/14) loaded successfully.")
                return
            except Exception as e:
                print(f"Notice: CLIP ViT-L/14 not loaded ({e}).")

    @torch.inference_mode()
    def predict(self, images: Union[Image.Image, List[Image.Image]]) -> List[Optional[float]]:
        """
        Evaluate images with CLIP ViT-L/14.
        Uses torch.inference_mode and AMP FP16 to minimize latency & VRAM.
        """
        if not self.is_available or self.model is None:
            if isinstance(images, Image.Image):
                return [None]
            return [None for _ in images]

        if isinstance(images, Image.Image):
            images = [images]

        tensors = [self.transform(im.convert("RGB")) for im in images]
        batch = torch.stack(tensors).to(DEVICE)
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else torch.nullcontext()
        with autocast_ctx:
            logits = self.model(batch).flatten()
            probs = torch.sigmoid(logits).cpu().float().numpy()

        results = []
        for p in probs:
            cal_p = calibration.calibrate_score("clip", float(p), method="platt")
            results.append(float(cal_p if cal_p is not None else p))
        return results
