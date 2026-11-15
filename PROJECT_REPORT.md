# CrowdGuard AI — Crowd Anomaly & Deepfake Detection System
## End-to-End System Project Report & Technical Specification

**Author & Developer:** Saurav Mathur  
**Repository Working Copy:** `d:\Projects\Deepfake`  
**System Version:** 2.0 Unified Production Release  
**Current Milestone:** Review 1 (100% Operational) | Review 2 (Pending Phase 2 Defense)  

---

## 1. Executive Summary & Objective

**CrowdGuard AI** is a defense-grade surveillance and media forensics system that bridges the gap between **Physical Crowd Threat Analysis** and **Digital Media Authenticity Verification**. Modern security operations face a dual challenge:
1. Physical incidents (overcrowding, stampede risk, weapon threats, perimeter breaches).
2. Digital deception (AI-generated synthetic feeds, face-swapped deepfakes, and spoofed CCTV broadcasts designed to manipulate emergency response).

By consolidating computer vision surveillance backbones with a multi-model deepfake forensics ensemble, CrowdGuard AI cross-verifies every alert. When an anomaly is detected, the system determines not only **what** is happening in the physical scene, but also **whether the camera feed itself is genuine or digitally forged**.

---

## 2. System Architecture

The project is structured into a clean decoupled architecture consisting of a high-performance **Python/FastAPI Backend** and a modern **React/Next.js 14 Frontend**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │               Next.js 14 Web Console                   │
                       │     (React 18 / TailwindCSS / Lucide Icons / Port 3000) │
                       └─────────────┬───────────────────────────▲──────────────┘
                                     │ HTTP / REST / SSE        │ Base64 Heatmaps
                                     ▼                           │ Telemetry & HUD
                       ┌─────────────────────────────────────────┴──────────────┐
                       │               FastAPI Gateway (Port 8000)              │
                       │     (Hardware Acceleration Manager / CORS / Pydantic)  │
                       └──────────────┬──────────────────────────┬──────────────┘
                                      │                          │
                 ┌────────────────────▼─────┐             ┌──────▼─────────────────────┐
                 │ Physical Threat Pipeline │             │  Digital Forensics Pipeline│
                 └────────────┬─────────────┘             └────────────┬───────────────┘
                              │                                        │
           ┌──────────────────┴─────────────────┐    ┌─────────────────┴─────────────────┐
           ▼                                    ▼    ▼                  ▼                ▼
┌───────────────────────┐            ┌──────────────┐ ┌───────────────┐ ┌──────────────┐ ┌────────────────┐
│ YOLOv8 Person Tracker │            │ MobileNetV3  │ │ Community     │ │ B-Free / NPR │ │ UniversalFake  │
│ (ByteTrack / CUDA)    │            │ Weapon Guard │ │ Forensics ViT │ │ ResNet-50    │ │ CLIP ViT-L/14  │
└───────────────────────┘            └──────────────┘ └───────────────┘ └──────────────┘ └────────────────┘
                                                                                                 ▲
                                                                                                 │ HTTP/JSON
                                                                                       ┌─────────┴────────┐
                                                                                       │ GenD Face Worker │
                                                                                       │ Microservice     │
                                                                                       │ (Port 8001 / PID)│
                                                                                       └──────────────────┘
```

### Component Details
1. **FastAPI Application (`CrowdGuardAI/app.py`)**:
   - Manages model lifecycle and CUDA GPU memory allocation (`NVIDIA GeForce RTX 4050 Laptop GPU`).
   - Handles multi-part file uploads (images up to 50MB, CCTV video up to 500MB) and streaming endpoints.
   - Spawns and manages the secondary **GenD Worker microservice** on port 8001.
2. **Crowd Intelligence Module (`CrowdGuardAI/crowd/`)**:
   - `yolo_detector.py`: YOLOv8 tracking engine operating with bounding box overlay and motion history. Automatically hot-swaps to custom weights `best.pt` when placed in `pretrained_weights/`.
   - `risk_engine.py`: Dynamic hazard engine evaluating density thresholds, weapon alerts, and cross-checks with video feed integrity.
   - `weapon_classifier.py`: Neural classifier specializing in edge hazard localization.
   - `feed_processor.py`, `tracker.py`, `smoothing.py`, `behavior.py`: Frame annotation/HUD rendering, track memory, detection smoothing, and crowd behavior analysis.
   - Feed integrity check (`check_frame_integrity()` in `app.py`): Runs sampled frames through the Community Forensics and B-Free/NPR detectors; integrity score = 1 − max(synthetic score).
3. **Deepfake Forensics Module (`CrowdGuardAI/detectors/`)**:
   - `Community Forensics`: Vision Transformer (ViT-S/16) trained on synthetic frequency artifacts (Diffusion, Midjourney, Flux, SD).
   - `B-Free / NPR Forensics`: Residual ResNet-50 examining neighboring pixel correlation irregularities.
   - `CLIP ViT-L/14`: Semantic-level text-image alignment artifact detector.
   - `GenD Microservice`: WACV 2026 facial manipulation detector with multi-facial crop evaluation.
4. **Next.js 14 Frontend (`frontend/`)**:
   - Built with Next.js 14 App Router, TypeScript, TailwindCSS, and Lucide React.
   - Features 4 operation views:
     - **Crowd Surveillance Console**: Real-time CCTV stream analysis, live security HUD, telemetry cards, and keyframe snapshots.
     - **Deepfake Forensics Inspector**: Autonomous, Deepfake Face, and AI-Gen scan modes with interactive radar visualization, 4-model confidence cards, and Error Level Analysis (ELA) / GradCAM heatmaps.
     - **Incident Investigation (Dual Pipeline)**: Automated cross-correlation between physical and digital signals. Distinguishes authentic emergencies from spoofed feeds.
     - **Review 2 Pending Console**: Transparent dashboard detailing the four Phase 2 deliverables scheduled for college guide defense.

---

## 3. Data Pipeline & Request Flow

### A. Crowd Surveillance Workflow
1. User provides an image, CCTV stream, or video file via the web console.
2. The payload is sent to `POST /api/crowd/detect` or `POST /api/crowd/video`.
3. YOLOv8 processes each frame on CUDA:
   - Identifies person entities, computes crowd density metrics, and tracks trajectories via ByteTrack.
   - Examines weapon probability signals.
4. Concurrently, `check_frame_integrity()` runs sampled keyframes through the Community Forensics and B-Free/NPR detectors and flags the feed as tampered when they detect synthetic content.
5. The combined risk score is evaluated:
   $$\text{Risk Score} = \min(100, (\text{Crowd Density Score} \times 0.4) + (\text{Weapon Threat} \times 0.6))$$
6. The annotated frame (with HUD and bounding boxes) is rendered in base64 and returned to the UI.

### B. Deepfake Media Forensics Workflow
1. Media is submitted to `POST /predict` (or `/api/predict/stream` for real-time SSE progress).
2. The image/video undergoes quality preprocessing (Laplacian sharpness, JPEG block analysis).
3. The image is evaluated in parallel by:
   - Community Forensics ViT-S/16 (spectral domain)
   - B-Free ResNet-50 (pixel spatial domain)
   - CLIP ViT-L/14 (vision-language latent domain)
   - GenD facial worker on port 8001 (facial structure domain)
4. Forensic fusion computes an ensemble score with variance checking.
5. Visual explainability maps (Error Level Analysis and GradCAM) are generated and returned.

### C. Incident Investigation (Dual Cross-Verification)
1. Submitted footage is analyzed simultaneously through both pipelines via `POST /api/investigate`.
2. The Risk Engine cross-references physical alerts with media manipulation scores:
   - If manipulation score $> 65\%$: System declares **`MEDIA MANIPULATION DETECTED (DIGITAL SPOOFING)`** with `HIGH`/`CRITICAL` advisory, warning commanders that reported crowd unrest or weapon sightings are fabricated.
   - If manipulation score $\le 65\%$ and crowd risk $> 50$: System declares **`VERIFIED PHYSICAL INCIDENT`**, confirming legitimate physical danger.
   - If both are nominal: System confirms **`VERIFIED AUTHENTIC / LOW RISK`**.

---

## 4. API Endpoints & Specifications

| Method | Endpoint | Description | Status |
|---|---|---|---|
| `GET` | `/api/health` | Hardware telemetry, CUDA status, VRAM usage, and model readiness. | **Active** |
| `POST` | `/api/crowd/detect` | Single-image crowd tracking, density counting, and feed integrity. | **Active** |
| `POST` | `/api/crowd/video` | Multi-frame temporal surveillance video analysis with keyframes. | **Active** |
| `POST` | `/predict` | Deepfake forensics analysis (Community Forensics, B-Free, CLIP, GenD). | **Active** |
| `POST` | `/api/predict/stream` | Server-Sent Events (SSE) streaming forensics pipeline. | **Active** |
| `POST` | `/api/investigate` | Dual-pipeline cross-verification incident investigation engine. | **Active** |
| `POST` | `/fetch-url` | Social media / remote CCTV video fetch via yt-dlp. | **Active** |
| `POST` | `/feedback` | Active learning feedback loop logger. | **Active** |

---

## 5. Review 2 Status (Explicitly Pending)

In accordance with the two-part college project submission framework, **Review 1 is 100% complete and fully verified**. The following four items are designated as **Review 2 Deliverables** and remain in a clearly marked **PENDING** state:

1. **Custom YOLOv8 Weapon Training Weights (`best.pt` & `classifier.pth`)**:
   - *Status:* **PENDING GUIDE REVIEW 2**
   - *Details:* The system currently operates with `yolo_crowd.pt` (stock YOLOv8n CUDA tracker on person class). The backend loader has an automatic listener configured so that when the custom weights (`best.pt` and `classifier.pth`) are transferred from the secondary workstation to `CrowdGuardAI/pretrained_weights/`, the system will hot-swap automatically.
2. **Optical Flow Turbulence Density Vector Maps**:
   - *Status:* **PENDING GUIDE REVIEW 2**
   - *Details:* Real-time dense Farneback optical flow kinetic vector fields and crowd panic velocity vectors.
3. **Automated Cryptographic Forensic PDF Dossier Generation**:
   - *Status:* **PENDING GUIDE REVIEW 2**
   - *Details:* Automated compilation of high-resolution PDF incident dossiers with cryptographic image hashes and timeline watermarks. (Phase 1 currently provides structured JSON audit downloads).
4. **Multi-Camera Edge Node Mesh & Persistent Log Database**:
   - *Status:* **PENDING GUIDE REVIEW 2**
   - *Details:* Multi-channel RTSP cluster synchronization and distributed SQL incident persistence.

In the Next.js frontend, these items are showcased under the **Review 2 (Pending)** navigation tab (`frontend/src/components/Review2PendingView.tsx`) with full technical specifications and milestone status.

---

## 6. Repository Cleanup & Archive Log

To ensure a clean, professional codebase suitable for college project presentation and portfolio/resume review, all legacy experiments, obsolete test runs, and unmerged projects were moved into a dedicated top-level `archive/` folder, preserving their relative internal structure:

| Original Path | Archive Location | Reason for Archiving |
|---|---|---|
| `UniversalFakeDetect/` | `archive/UniversalFakeDetect/` | Legacy upstream codebase; all required models and weights were unified into `CrowdGuardAI/`. |
| `sentinelvision-ai-surveillance/` | `archive/sentinelvision-ai-surveillance/` | Legacy standalone crowd repository; all surveillance logic, YOLO tracking, and risk engines were migrated into `CrowdGuardAI/crowd/`. |
| `eval_data/` | `archive/eval_data/` | Benchmark image datasets used during model evaluation. |
| `manual_test_library/` | `archive/manual_test_library/` | Standalone CLI scripts for earlier manual model testing. |
| `test_samples/` | `archive/test_samples/` | Historical test scripts and download utilities. |
| `vbench/` | `archive/vbench/` | Video benchmark scripts from early experiments. |
| `third_party/` | `archive/third_party/` | Loose upstream utility files. |
| `Image/` | `archive/Image/` | Stock test images from early experiments. |
| `static/` (root) | `archive/legacy_root_static/` | Old root static assets. The legacy single-page UI was later removed from `CrowdGuardAI/static/` as well; the Next.js app is the only UI, and the backend root URL redirects to it. |
| Root test scripts (`benchmark_library.py`, `download_unidatapro.py`, `evaluate.py`, `test_all.py`, `test_variance.py`, `test_videos.py`, `run.bat`, `yolov8n.pt`) | `archive/stray_scripts/` | One-off benchmark and verification scripts from development phase. |

### Clean Root Structure
```
d:\Projects\Deepfake\
├── CrowdGuardAI/         # Unified FastAPI backend, models, upload storage, and weights
├── frontend/             # Next.js 14 + React 18 frontend (test assets F.png / R.jpg in public/static/samples/)
├── archive/              # Preserved legacy repositories, benchmarks, and experiments
├── venv_gend/            # Python virtual environment with PyTorch CUDA 12.1 (not in git)
├── run_crowdguard.bat    # One-click launcher starting both Backend (:8000) and Frontend (:3000)
├── requirements.txt      # Python dependencies
├── requirements_gend.txt # Microservice dependencies
├── README.md             # Project summary and quickstart
└── PROJECT_REPORT.md     # This comprehensive technical specification report
```

---

## 7. Installation & Run Instructions

### Prerequisites
- **Operating System:** Windows 10/11 (x64)
- **GPU Acceleration:** NVIDIA GPU with CUDA support (e.g. RTX 4050 Laptop GPU, 6GB VRAM)
- **Node.js:** v20+ or v22+ LTS (`C:\Program Files\nodejs`)
- **Python:** Python 3.10+ (configured in `venv_gend`)

### A. One-Click Launch (Recommended)
Simply double-click:
```cmd
run_crowdguard.bat
```
This script will:
1. Verify and terminate any orphaned processes on ports 8000, 8001, and 3000.
2. Launch the **FastAPI Backend & CUDA Vision Engine** on `http://localhost:8000`. The backend then starts the **GenD Facial Forensics Worker** on `http://127.0.0.1:8001` itself.
3. Launch the **Next.js Production Frontend** on `http://localhost:3000` (`npm run start` serves the last build, so run `npm run build` in `frontend/` after any frontend change).
4. Automatically open `http://localhost:3000` in your web browser.

---

### B. Manual Terminal Launch

#### Step 1: Start the Backend (Terminal 1)
```powershell
Set-Location -Path "d:\Projects\Deepfake\CrowdGuardAI"
..\venv_gend\Scripts\activate
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
*Note: The FastAPI backend automatically initializes the CUDA models and spawns the GenD worker microservice on port 8001.*

#### Step 2: Start the Next.js Frontend (Terminal 2)
```powershell
$env:Path = "C:\Program Files\nodejs;" + $env:Path
Set-Location -Path "d:\Projects\Deepfake\frontend"
npm run dev
# Or for production mode:
# npm run build
# npm run start
```
Open your browser to: **`http://localhost:3000`**

---

## 8. Verification & Test Evidence

1. **Hardware & Model Health Test (`GET /api/health`)**:
   - Device: `cuda` (`NVIDIA GeForce RTX 4050 Laptop GPU`)
   - VRAM: 1.72 GB allocated / 3.61 GB free out of 6.0 GB total
   - Active Vision Models: Community Forensics, B-Free / NPR, UniversalFakeDetect CLIP, GenD Worker (PID 18220), YOLO Crowd Tracker.
   - Result: `ensemble_ready: true`.

2. **Dual-Pipeline Investigation Test (`POST /api/investigate`)**:
   - Target Input: `F.png` (Manipulated sample)
   - Verdict: **`MEDIA MANIPULATION DETECTED (DIGITAL SPOOFING)`**
   - Threat Level: `HIGH`
   - Confidence: `92.00%`
   - Findings: Community Forensics ViT and B-Free ResNet detected generative pixel artifacts; ELA and GradCAM attention maps generated.

3. **Frontend Production Build Verification (`npm run build`)**:
   - Next.js 14.2.15 compiler completed with **0 errors**.
   - Generated static and server-rendered chunks with zero TypeScript warnings.

---

*End of Project Report.*
