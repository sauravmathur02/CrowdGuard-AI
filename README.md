# CrowdGuard AI — Crowd Anomaly & Deepfake Detection System
> **Unified Physical Crowd Threat Intelligence & Digital Media Forensics Platform**  
> *Developed for College Project Review 1 & 2 Defense*

---

## Overview

**CrowdGuard AI** is an advanced security intelligence platform that unites **Real-Time Physical Crowd Surveillance** with **Deepfake & Synthetic Media Forensics**. It addresses a critical modern security vector: verifying whether security footage of an escalating crowd anomaly or weapon threat is physical reality or a synthetic / spoofed digital forgery.

```
                      CrowdGuard AI Architecture
                ┌────────────────────────────────────┐
                │        Video / Image Input         │
                └─────────────────┬──────────────────┘
                                  │
                ┌─────────────────▼──────────────────┐
                │        Video Preprocessing         │
                └─────────────────┬──────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
   ┌─────────────────────┐                 ┌─────────────────────┐
   │ Crowd Intelligence  │                 │  Deepfake Forensics │
   │      Pipeline       │                 │      Pipeline       │
   └──────────┬──────────┘                 └──────────┬──────────┘
              │                                       │
              ▼                                       ▼
   • YOLOv8 Person Tracker                 • Community Forensics ViT
   • ByteTrack Trajectories                • B-Free / NPR ResNet-50
   • Crowd Density Score                   • UniversalFake CLIP
   • Weapon Localization                   • GenD Face Microservice
   • Feed Tampering Check                  • ELA & GradCAM Explainability
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │   Risk / Event Engine   │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │   Next.js 14 Dashboard  │
                     ├─────────────────────────┤
                     │ • Crowd Density Tracker │
                     │ • Weapon Anomaly Type   │
                     │ • Deepfake Probability  │
                     │ • Integrated Risk Level │
                     │ • Dual-Forensic Dossier │
                     └─────────────────────────┘
```

---

## Key Features

1. **Physical Crowd Threat Pipeline**:
   - **YOLOv8 Real-Time Person Detection & ByteTrack**: Tracks human flow and crowd density on CUDA GPU.
   - **Weapon Hazard Localization**: MobileNetV3 / YOLO custom heads alerting on edge weapons.
   - **Video Feed Tamper Detection**: Compares frame-by-frame structural similarity and noise continuity to catch loop playback and feed cutting.
2. **Digital Media Forensics Pipeline**:
   - **Four-Backbone Ensemble**: Community Forensics (ViT-S/16), B-Free/NPR (ResNet-50), UniversalFakeDetect (CLIP ViT-L/14), and GenD (WACV 2026 Face model).
   - **Explainability Maps**: Error Level Analysis (ELA) compression heatmaps and GradCAM attention visualizations.
   - **Natural Language Verdicts**: Translates complex spectral Fourier and spatial artifact metrics into clear security intelligence.
3. **Dual Investigation Engine**:
   - Simultaneously runs physical tracking and deepfake forensics on suspicious incidents to prevent emergency responders from being baited by spoofed or AI-generated feeds.
4. **Modern Web Console**:
   - Built on **React 18 / Next.js 14 / TailwindCSS** with dark-mode security HUD, real-time VRAM telemetry, and four operational views.

---

## Project Structure

```
d:\Projects\Deepfake\
├── CrowdGuardAI/               # FastAPI Backend & Vision Pipeline
│   ├── app.py                  # Main server, routing, CUDA device manager
│   ├── crowd/                  # Crowd intelligence (YOLO, tracker, behavior, risk engine, weapon classifier)
│   ├── detectors/              # Forensics backbones (CommFor, B-Free, CLIP, GenD client)
│   ├── models/                 # Neural architectures (CLIP, ViT, ResNet)
│   ├── pretrained_weights/     # Model weights (not in git; see Setup below)
│   └── static/uploads/         # Runtime storage for media fetched via /fetch-url
├── frontend/                   # Next.js 14 React Application (the only web UI)
│   ├── src/app/                # Next.js App Router (page.tsx, layout.tsx, globals.css)
│   ├── src/components/         # SurveillanceView, ForensicsView, InvestigationView, Review2PendingView
│   ├── src/lib/api.ts          # Typed API gateway connector
│   └── public/static/samples/  # Demo images/videos used by the "Test Sample" buttons (incl. F.png, R.jpg)
├── archive/                    # Preserved legacy repositories, benchmarks & experiments
│   ├── UniversalFakeDetect/    # Archived upstream source
│   ├── sentinelvision.../      # Archived standalone crowd repository
│   └── stray_scripts/          # Archived evaluation scripts
├── venv_gend/                  # Python virtual environment (PyTorch CUDA 12.1)
├── run_crowdguard.bat          # One-click startup script (Backend + Frontend)
└── PROJECT_REPORT.md           # Comprehensive technical report & architecture spec
```

`archive/sentinelvision-ai-surveillance` is a git submodule. Clone with `git clone --recurse-submodules`, or run `git submodule update --init` after cloning.

### Setup

- **Model weights** are not stored in git. Place them in `CrowdGuardAI/pretrained_weights/`: `yolo_crowd.pt` (YOLOv8n person tracker), `NPR.pth` (B-Free/NPR), and `fc_weights.pth` (UniversalFakeDetect CLIP head). The GenD worker downloads its OpenCV face detector files (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`) automatically if they are missing, and Community Forensics weights (`OwensLab/commfor-model-384`) are downloaded from Hugging Face on first run.
- **Python environment**: `python -m venv venv_gend`, then `venv_gend\Scripts\pip install -r requirements.txt -r requirements_gend.txt`.
- **Frontend**: `cd frontend && npm install && npm run build`.

---

## College Guide Review Schedule

| Milestone | Scope & Deliverables | Status |
|---|---|---|
| **Review 1** | Dual-pipeline architecture, live real-time webcam/CCTV inference with YOLOv8 person tracking, multi-model deepfake forensics (ViT, ResNet, CLIP, GenD), cross-correlated incident investigation engine, and full responsive Next.js operations console. | **100% COMPLETE & VERIFIED** |
| **Review 2** | Custom YOLOv8 weapon weights (`best.pt` / `classifier.pth`), optical flow density vector maps, automated cryptographic PDF incident export, and multi-node cluster database synchronization. | **PENDING (Phase 2 Review)** |

---

## Quickstart & How to Run

### Option 1: One-Click Startup (Recommended)
Double-click:
```cmd
run_crowdguard.bat
```
This script launches the FastAPI backend on port 8000, starts the Next.js frontend on port 3000, and opens your browser automatically.

### Option 2: Manual Startup

**Terminal 1 — FastAPI Backend**:
```powershell
cd CrowdGuardAI
..\venv_gend\Scripts\activate
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Next.js 14 Frontend**:
```powershell
$env:Path = "C:\Program Files\nodejs;" + $env:Path
cd frontend
npm run build   # required after any frontend change; `npm run start` serves the last build
npm run start
# Or for live development:
# npm run dev
```

Visit the application at: **`http://localhost:3000`**. The backend API runs at `http://localhost:8000` (interactive docs at `/docs`); opening its root URL redirects to the web console.

---

*See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for full technical documentation, mathematical models, and benchmark results.*
