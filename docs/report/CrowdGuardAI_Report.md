<!--
RRU dissertation format: Times New Roman 12 pt, 1.5 line spacing, left margin 38 mm, other margins 25.4 mm.
The print-ready version with this exact formatting is overleaf/main.tex (upload CrowdGuardAI_Report_Overleaf.zip to Overleaf).
Fill in every [bracketed] field before submission.
-->

<div align="center">

**CROWDGUARD AI: AN INTEGRATED SYSTEM FOR CROWD THREAT MONITORING AND DEEPFAKE MEDIA VERIFICATION**

Dissertation submitted

**to**

**RASHTRIYA RAKSHA UNIVERSITY**

(An Institution of National Importance)

For the partial fulfilment for the award of the degree of

## [Degree Name]

Submitted by

**SAURAV MATHUR**

**([Enrolment No.])**

**Under the Guidance of**

**[NAME OF SUPERVISOR]**

[Designation],
[School Name],
[Place]

<img src="images/image1.png" height="120">

**[NAME OF THE SCHOOL]**

**RASHTRIYA RAKSHA UNIVERSITY,**
**Lavad, Dehgam, Gandhinagar-382305, Gujarat, India**

**[Month & Year]**

</div>

---

<div align="center">

## DECLARATION

</div>

The work embodied in this Dissertation titled "CrowdGuard AI: An Integrated System for Crowd Threat Monitoring and Deepfake Media Verification" submitted for the partial fulfillment of the degree of [Degree Name] is the original research work carried out by me. The research work does not form the basis for the award of any degree, diploma, associateship, fellowship or other titles in the Rashtriya Raksha University or similar institutions of higher learning. All the ideas and references have been duly acknowledged.

<div align="right">(Name & Signature of the Candidate)</div>

**Date:**

---

<div align="center">

## CERTIFICATE

</div>

This is to certify that the Dissertation titled "CrowdGuard AI: An Integrated System for Crowd Threat Monitoring and Deepfake Media Verification" was carried out by Mr. Saurav Mathur (Enrollment no. [Enrolment No.]) studying at [School Name] for partial fulfillment of [Degree Name] degree to be awarded by Rashtriya Raksha University. This research work has been carried out under my guidance and supervision and it is up to my satisfaction. The Dissertation is fit to be considered for evaluation for the degree of [Degree Name].

Date:
Place:

| Signature and Name of Supervisor | Signature and Name of School Director |
|---|---|

<div align="center">Round Seal of RRU</div>

---

<div align="center">

## Acknowledgements

</div>

I would like to thank my supervisor, [Name of Supervisor], for the guidance given throughout this project. The suggestion to test the system on footage of real incidents, and not only on clean sample images, changed the direction of the work in a useful way.

I am thankful to the faculty of [School Name] for the review sessions, where several weak points of the first version were pointed out. Those comments are the reason the deepfake module was measured again and rebuilt in the later part of the project.

This work depends on models and datasets that other researchers have released openly. I acknowledge the authors of Community Forensics, B-Free, UniversalFakeDetect, GenD, YOLOv8, Defactify, DeepFakeFace, FaceForensics++ and Celeb-DF. Without their public code and data, a project of this size would not have been possible in one semester.

Finally, I thank my family and friends for their patience during the long evenings spent waiting for models to finish running.

<div align="right">Saurav Mathur</div>

---

<div align="center">

## TABLE OF CONTENTS

</div>

| | Page |
|---|---|
| Declaration | i |
| Certificate | ii |
| Acknowledgements | iii |
| List of Figures | |
| List of Tables | |
| Abstract | |
| **Chapter 1 Introduction** | 1 |
| 1.1 Background | |
| 1.2 Problem Statement | |
| 1.3 Objectives | |
| 1.4 Scope of the Work | |
| 1.5 Organisation of the Report | |
| **Chapter 2 Literature Review** | |
| 2.1 Generative Media and Deepfakes | |
| 2.2 Detection of AI-Generated Images | |
| 2.3 Detection of Face Manipulation | |
| 2.4 Calibration and Combination of Detectors | |
| 2.5 Crowd Monitoring | |
| 2.6 Explainability and Provenance | |
| 2.7 Research Gap | |
| **Chapter 3 System Requirements and Design** | |
| 3.1 Requirements | |
| 3.2 Hardware and Software | |
| 3.3 Overall Architecture | |
| 3.4 Crowd Analysis Design | |
| 3.5 Media Forensics Design | |
| 3.6 Investigation Logic | |
| 3.7 API Design | |
| 3.8 User Interface Design | |
| **Chapter 4 Implementation** | |
| 4.1 Project Structure | |
| 4.2 Crowd Module | |
| 4.3 Whole-Image Detectors | |
| 4.4 Face Worker | |
| 4.5 Calibration and Fusion | |
| 4.6 Video Handling | |
| 4.7 Explainability and Provenance | |
| 4.8 Error Handling | |
| 4.9 Web Console | |
| **Chapter 5 Evaluation and Results** | |
| 5.1 Problems Found in Version 1 | |
| 5.2 Evaluation Data | |
| 5.3 Protocol | |
| 5.4 Fitted Calibration | |
| 5.5 Overall Results | |
| 5.6 Results by Generator | |
| 5.7 Individual Detectors | |
| 5.8 Running Time | |
| 5.9 Manual Testing | |
| 5.10 Discussion and Limitations | |
| **Chapter 6 Conclusions** | |
| 6.1 Conclusions | |
| 6.2 Future Work | |
| **Bibliography and References** | |
| **Appendix A** List of Abbreviations | |
| **Appendix B** Setup and Reproduction | |

---

<div align="center">

## List of Figures

</div>

| Figure | Title |
|---|---|
| 3.1 | Overall architecture of CrowdGuard AI |
| 3.2 | Deepfake forensics pipeline |
| 4.1 | Surveillance view with person tracking and risk level |
| 4.2 | Forensics view showing the verdict, model scores and heatmaps |
| 4.3 | Investigation view with the combined incident verdict |
| 5.1 | Fake detection rate by generator, version 1 and version 2 |

<div align="center">

## List of Tables

</div>

| Table | Title |
|---|---|
| 3.1 | Hardware and software used |
| 3.2 | Risk levels used by the crowd risk engine |
| 3.3 | Incident verdict rules |
| 3.4 | Main API endpoints |
| 5.1 | Labelled image data used for calibration and testing |
| 5.2 | Fitted Platt coefficients and fusion weights |
| 5.3 | Held-out results, version 1 and version 2 |
| 5.4 | Correct decisions by subset |
| 5.5 | Single detectors on the test split |
| 5.6 | Manual test cases |

---

<div align="center">

**Abstract: CrowdGuard AI: An Integrated System for Crowd Threat Monitoring and Deepfake Media Verification**

Submitted By
Saurav Mathur

Supervised By [Supervisor's name and designation]

**Abstract**

</div>

**Objective.** Security control rooms now receive crowd footage from CCTV, phones and social media, and some of it may be synthetic or edited. This work builds one system that analyses the physical scene and also checks whether the media itself is authentic.

**Work done.** CrowdGuard AI has a FastAPI backend and a Next.js web console. The crowd module uses YOLOv8 with ByteTrack to count and track people, flags fast-moving tracks and computes a risk level. The forensics module combines Community Forensics, B-Free and UniversalFakeDetect for whole-image analysis with the GenD model for faces. Testing showed that the first version performed poorly. The causes were a pixel detector whose weights gave the same output for every image, a generic face classifier, no score calibration and hand-tuned decision rules. These were replaced, and Platt calibration with logistic-regression fusion was fitted on 1,031 labelled public images, each scored as the original file and as a JPEG copy.

**Results.** On 1,010 held-out image variants, detection of fakes rose from 62% to 79% and correct acceptance of real images from 76% to 96%. Balanced accuracy rose from 69% to 88% (AUC 0.943). False alarms on real faces fell from 43% to 3%. DALL-E 3 images (40% detected) and InsightFace swaps (46%) remain weak.

**Conclusions.** Fusion weights measured on labelled data were more reliable than hand-set rules. Quantitative video evaluation and a trained weapon detector are left for the next phase.

---

# Chapter 1 Introduction

## 1.1 Background

Video is now the main evidence used during a public-order incident. Control rooms watch CCTV, police units forward phone recordings, and clips spread on social media within minutes of an event. Two separate questions have to be answered about such footage. The first is what is happening in the scene: how many people are present, whether the crowd is growing, whether someone is moving abnormally fast and whether a weapon is visible. The second is whether the footage can be trusted at all.

The second question has become harder in the last few years. Generative adversarial networks<sup>1</sup> and later diffusion models<sup>2,3</sup> can produce photographs that most people cannot tell apart from camera images. Face-swap tools can replace the identity of a person in an existing video<sup>4</sup>. A fabricated clip of a crowd attack or a fake image of an armed person can push responders to the wrong place, or be used to discredit real evidence later.

Most existing tools deal with only one of these two questions. Crowd analytics software assumes the input is genuine, and deepfake detectors are usually separate websites or research scripts that know nothing about the scene. This project brings the two checks together in one workflow.

## 1.2 Problem Statement

The aim is to design and build a system that takes an image or video of a crowd incident and reports (a) the physical threat picture and (b) whether the media appears to be AI-generated or manipulated, and then combines both into one incident verdict. The deepfake verdict must be measured on labelled data. A detector that looks convincing on a few hand-picked samples is not enough for security use.

## 1.3 Objectives

1. Detect and track people in images and video, estimate crowd density and flag abnormal movement.
2. Detect whole-image AI generation (diffusion and GAN images) and face manipulation (face swaps).
3. Combine several published detectors so that one weak model does not decide the verdict.
4. Calibrate the detector scores and fit the fusion weights on public labelled data, then report accuracy on a held-out split.
5. Report clearly when a check could not run, without silently returning "authentic".
6. Give explanations an operator can read: per-model scores, Error Level Analysis (ELA) and attention heatmaps.
7. Provide a web console with separate views for surveillance, forensics and combined investigation.

## 1.4 Scope of the Work

The project covers still images and recorded video files uploaded through the web console, or fetched from a URL. Live RTSP camera streams, a trained weapon detector, optical-flow crowd analysis and PDF incident reports are planned for the second review and are not part of the results in this report. The deepfake evaluation covers images. A labelled video set was prepared, but its full evaluation is listed as pending work in Chapter 6.

## 1.5 Organisation of the Report

Chapter 2 reviews previous work on synthetic image detection, face-manipulation detection, calibration and crowd monitoring. Chapter 3 gives the requirements and the system design. Chapter 4 describes the implementation. Chapter 5 explains how the first version was tested, what went wrong, and the results after the pipeline was rebuilt. Chapter 6 gives conclusions and future work. Appendix A lists abbreviations and Appendix B gives setup commands.

---

# Chapter 2 Literature Review

## 2.1 Generative Media and Deepfakes

Goodfellow et al.<sup>1</sup> introduced GANs, in which a generator and a discriminator are trained against each other. Denoising diffusion models<sup>2</sup> later gave higher quality and more stable training, and latent diffusion<sup>3</sup> (the basis of Stable Diffusion) made high-resolution generation possible on ordinary GPUs. Commercial systems such as DALL-E 3 and Midjourney are trained in a similar way, but their details are not public. Tolosana et al.<sup>4</sup> group face manipulation into entire face synthesis, identity swap, attribute editing and expression swap. Each type leaves different traces.

## 2.2 Detection of AI-Generated Images

Wang et al.<sup>5</sup> showed that a ResNet-50 trained on images from a single GAN can detect images from several other GANs, provided strong blur and JPEG augmentation is used during training. Generalisation to diffusion models was weaker. Ojha et al.<sup>6</sup> (UniversalFakeDetect) kept a CLIP ViT-L/14 encoder<sup>7</sup> frozen and trained only a linear layer on top. Their argument was that a general feature space avoids overfitting to one generator. Tan et al.<sup>8</sup> proposed Neighboring Pixel Relationships (NPR), which looks at the local up-sampling pattern that most generators leave behind.

Two recent works are central to this project. Park and Owens<sup>9</sup> built Community Forensics, a training set drawn from thousands of different generators, and showed that diversity of generators matters more than model size. Their released detector is a ViT-S/16<sup>10</sup> working at 384 pixels. Guillaro et al.<sup>11</sup> argued that many detectors learn dataset bias (image size, JPEG format, content) and not generation artefacts. Their B-Free method trains on fake images made by regenerating real ones through Stable Diffusion, so real and fake versions share the same content. It uses a DINOv2<sup>12</sup> ViT-B with registers<sup>13</sup> and is applied at native resolution on 504-pixel crops.

## 2.3 Detection of Face Manipulation

FaceForensics++<sup>14</sup> and Celeb-DF<sup>15</sup> are the standard face-swap benchmarks. Detectors trained on one benchmark often fail on another. Yermakov et al.<sup>16</sup> (GenD) fine-tune only the layer-normalisation parameters of a CLIP ViT-L/14 and add metric learning on a hyperspherical feature space. They report state-of-the-art cross-dataset AUROC over 14 benchmarks. GenD expects a face crop that is 1.3 times the detected face box, which matters for the implementation in Chapter 4. The DeepFakeFace dataset<sup>17</sup> contains real IMDB-WIKI faces together with InsightFace swaps, Stable Diffusion inpainting and text-to-image faces, so it tests both swap and generation cases.

## 2.4 Calibration and Combination of Detectors

Raw outputs of neural networks are often badly calibrated. A score of 0.9 does not mean a 90% chance of being fake<sup>18</sup>. Platt scaling<sup>19</sup> fits a logistic function on held-out data to turn a raw score into a probability. When several detectors are combined, calibrated log-odds can be added with weights learned by logistic regression. This gives a measured weight to each detector, where hand-set rules rely on guesswork. One aim of this project was to replace manual thresholds with this kind of fitted fusion.

## 2.5 Crowd Monitoring

Single-stage detectors from the YOLO family are widely used for real-time person detection. YOLOv8<sup>20</sup> provides pretrained COCO models and built-in tracking. ByteTrack<sup>21</sup> improves multi-object tracking by also associating low-confidence detections, which helps in dense crowds where people are partly hidden. Light classifiers such as MobileNetV3<sup>22</sup> are suitable for a second-stage weapon check on cropped regions.

## 2.6 Explainability and Provenance

Error Level Analysis<sup>23</sup> re-saves an image as JPEG and shows the difference. Regions edited after the last compression often stand out. The C2PA standard<sup>24</sup> defines signed provenance manifests, and some generators also write prompts or software tags into metadata. Neither method is reliable alone. ELA is easily confused by re-compression, and metadata can be removed. Both are still useful as supporting evidence shown next to the model scores.

## 2.7 Research Gap

The detectors above are evaluated mostly in isolation, on their own test sets. Systems for operators usually stop at a single model and report its raw score. Crowd analytics and media forensics are rarely combined. This project combines published detectors under one calibration fitted on mixed public data, links the result to a crowd threat engine, and measures the final verdict rather than only the separate models.

---

# Chapter 3 System Requirements and Design

## 3.1 Requirements

**Functional requirements**

- Accept images (JPEG, PNG, WebP) and videos (MP4, MOV, AVI, MKV, WebM) by upload or URL.
- Count and track people, detect fast movement and compute a crowd risk level.
- Run deepfake analysis in three modes: *Auto*, *Deepfake (face)* and *AI-Generated (whole image)*.
- Combine both analyses into an incident verdict.
- Show per-model scores, warnings and heatmaps, and allow the operator to submit feedback.

**Non-functional requirements**

- Run on a single laptop GPU with 6 GB of memory.
- Return an image result within a few seconds.
- Never report "authentic" when a required check did not run.
- Keep model weights out of version control and document how to obtain them.

## 3.2 Hardware and Software

**Table 3.1: Hardware and software used**

| Item | Version / Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6 GB |
| Operating system | Windows 10/11 (x64) [confirm] |
| Python | 3.12.10 |
| PyTorch | 2.5.1 with CUDA 12.1 |
| Transformers / timm | 4.56.2 / 1.0.30 |
| Backend framework | FastAPI with Uvicorn<sup>25</sup> |
| Frontend | Next.js 14.2.15, React 18, Tailwind CSS<sup>26</sup> |
| Detection and tracking | Ultralytics YOLOv8<sup>20</sup> with ByteTrack |
| Image processing | OpenCV<sup>27</sup>, Pillow |

## 3.3 Overall Architecture

Figure 3.1 shows the main parts. The web console sends requests to the FastAPI backend on port 8000. The backend holds the crowd models and the whole-image deepfake detectors on the GPU. The face model runs as a separate worker process on port 8001, started by the backend. It has its own Python environment because its library versions differ from the main application. If the worker is not available, the backend still answers, but it marks the face check as not done.

```
                    Web Console (Next.js, port 3000)
                                 |
                            HTTP / SSE
                                 |
                    FastAPI Backend (port 8000)
                  /                              \
     Crowd Module                         Forensics Module  ----->  Face Worker
     YOLOv8 + ByteTrack                   CommFor, B-Free, CLIP     GenD (port 8001)
     behaviour, risk                      calibration, fusion
                  \                              /
                     Investigation Engine (combined verdict)
```

**Figure 3.1: Overall architecture of CrowdGuard AI**

## 3.4 Crowd Analysis Design

Each frame is passed to YOLOv8, restricted to the person class, with ByteTrack keeping identities across frames. A short smoothing step removes boxes that flicker for a single frame. The behaviour analyser measures how far each track centre moves between processed frames. A displacement above 35 pixels is marked as suspicious. The risk engine then adds points: 5 per person, 70 to 100 per detected weapon depending on the class, and 25 for suspicious movement. The total is capped at 100 and mapped to a level, as shown in Table 3.2. If the forensics module finds that the feed itself is manipulated, the score is raised to at least 95 and the level becomes CRITICAL.

**Table 3.2: Risk levels used by the crowd risk engine**

| Score | Level |
|---|---|
| below 20 | SAFE |
| 20 to 49 | ELEVATED |
| 50 to 99 | ALERT |
| 100 or more | DANGER (CRITICAL if a weapon is present) |
| feed manipulated | CRITICAL |

## 3.5 Media Forensics Design

Figure 3.2 shows the forensics pipeline. Metadata and C2PA provenance are checked first. Image quality (sharpness and JPEG blocking) and the ELA map are computed next. The whole image then goes to three detectors, and in parallel the face worker finds faces and scores them with GenD. Each raw score is mapped to a probability by its own Platt function. The calibrated scores are combined by a logistic model. A separate model is used depending on whether a face was scored and whether the input is a video. The final probability is compared with a threshold of 0.5.

```
Input image or video (16 frames)
        |
Metadata and C2PA check
        |
Quality estimate, ELA map
        |---------------------------------------|
Whole-image detectors                   Face detection (SSD)
CommFor, B-Free, CLIP                   GenD on 1.3x face crops
        |---------------------------------------|
Platt calibration per detector
        |
Logistic fusion (no-face / face / video model)
        |
Verdict, warnings, heatmaps
```

**Figure 3.2: Deepfake forensics pipeline**

## 3.6 Investigation Logic

The investigation endpoint runs both pipelines on the same file and applies the rules in Table 3.3, in order.

**Table 3.3: Incident verdict rules**

| Condition | Verdict | Threat level |
|---|---|---|
| Fake and weapon seen | Fabricated threat (deepfake disinformation) | CRITICAL |
| Fake, no weapon | Media manipulation detected | HIGH |
| Authentic and weapon seen | Verified physical security incident | CRITICAL |
| Authentic, more than 15 people | Verified crowd surge | ELEVATED |
| Otherwise | Routine capture | SAFE |

If the forensics analysis fails, for example on an unreadable file, the endpoint returns an error and does not produce any verdict. An earlier version treated a failure as a score of zero, which would have marked the footage as authentic.

## 3.7 API Design

**Table 3.4: Main API endpoints**

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /api/health | GPU memory, loaded models, calibration status |
| POST | /api/crowd/detect | Crowd analysis of one image |
| POST | /api/crowd/video | Crowd analysis of a video with key frames |
| POST | /predict | Deepfake analysis (mode: auto, deepfake, ai_generated) |
| POST | /api/predict/stream | Same analysis with stage-by-stage progress (SSE) |
| POST | /api/investigate | Combined crowd and forensics verdict |
| POST | /fetch-url | Download media from a URL for analysis |
| POST | /feedback | Store an operator correction |

## 3.8 User Interface Design

The console has four views. *Surveillance* shows annotated frames with counts and risk. *Forensics* shows the verdict, the four model cards, warnings and the evidence heatmaps. *Investigation* shows the combined incident verdict. *Review 2* lists the planned features. A neutral amber style is used when there is no verdict (no face found, or a check could not run), so that such a result is not mistaken for "authentic".

---

# Chapter 4 Implementation

## 4.1 Project Structure

```
CrowdGuardAI/
  app.py               FastAPI server, routes, pipeline
  gend_worker.py       face-forensics worker (port 8001)
  crowd/               YOLO detector, tracker, behaviour, risk engine
  detectors/           CommFor, B-Free, CLIP, calibration, fusion,
                       video sampling, ELA/heatmaps, provenance
  data/                calibration.json, eval_scores.json
  tools/               dataset download, calibration, video evaluation
  pretrained_weights/  model weights (not in git)
frontend/              Next.js console (src/components, src/lib/api.ts)
```

## 4.2 Crowd Module

`YOLODetector` wraps the Ultralytics tracker with a generated ByteTrack configuration (high threshold 0.30, low threshold 0.05, new-track threshold 0.30, match threshold 0.80). The person confidence threshold is 0.20 with IoU 0.70, and very small boxes need a slightly higher score. The loader looks for custom weights (`best.pt`) first and falls back to `yolo_crowd.pt`, a YOLOv8n model for the person class. A MobileNetV3-Small weapon classifier (gun and knife) is wired in, but its trained weights are part of the second review, so it is currently inactive. For video, between 6 and 16 frames are sampled depending on the length of the clip.

## 4.3 Whole-Image Detectors

**Community Forensics.** The released ViT-S/16 weights are downloaded from Hugging Face (`OwensLab/commfor-model-384`). The image is resized so its shorter side is 440 pixels, centre-cropped to 384, and scored together with its horizontal flip. The two outputs are averaged.

**B-Free.** The official code and weights (DINOv2-reg4 ViT-B/14) were taken from the authors' repository. The model scores five 504-pixel crops at native resolution, without resizing, and averages them. To keep memory within 6 GB, images with a side longer than 1024 pixels are centre-cropped (never resized, since resizing removes the traces B-Free looks for), and half precision is used. The wrapper was checked against the four demo images published by the authors, and the logits matched to four decimal places.

**UniversalFakeDetect.** A frozen CLIP ViT-L/14 with the authors' trained linear layer (`fc_weights.pth`) scores a 224-pixel centre crop.

## 4.4 Face Worker

Faces are found with the OpenCV ResNet-10 SSD detector<sup>27</sup> at confidence 0.5, with a Haar cascade as fallback. Faces smaller than 40 pixels are ignored. Each face is cropped as a square 1.3 times the size of the detected box, which follows GenD's own preprocessing, and up to three faces per frame are scored. For an image, the score of the most suspicious face is returned. For a video, 16 frames are sampled. The most suspicious face is taken per frame, and the mean of the top half of these frame scores is returned. The worker also reports which model it used and how many frames had a face. If the GenD weights are missing, a generic fallback classifier is loaded, and the result carries a warning.

## 4.5 Calibration and Fusion

`tools/calibrate.py` scores every image in the evaluation set with all detectors, using raw (uncalibrated) outputs, and then fits two things on the *fit* half:

1. a Platt function p = σ(a · logit(s) + b) for each detector, with class-balanced weights;
2. a logistic regression on the calibrated logits, z = b + Σ wᵢ · logit(pᵢ), once for images without a scored face (CommFor, B-Free, CLIP) and once for images with one (adding GenD).

The fitted values are written to `data/calibration.json` and loaded by the live server, so the server applies exactly the logic that was measured. The same file stores the held-out metrics, and `/api/health` reports whether a fitted calibration is present.

## 4.6 Video Handling

Sixteen frames are sampled evenly across the clip. For each detector, the per-frame calibrated scores are reduced to one number by taking the mean of the most suspicious half. Version 1 used eight frames and a trimmed mean, which averaged away manipulations that appear in only part of a clip. The per-frame fused scores are also returned, so that the console can show where in the clip the suspicious frames are. `tools/eval_video.py` fits a separate video fusion model on Celeb-DF when it is run, and the server picks it up automatically.

## 4.7 Explainability and Provenance

The response contains an ELA map, a noise residual map and a 24×24 patch map taken from the Community Forensics ViT head. When a face is found, the ratio of ELA error inside the face box to the background is also reported. In version 1 this ratio could override the verdict. It is now shown only as supporting evidence, because JPEG re-compression changes it a lot. If the metadata or a C2PA manifest declares that the file was AI-generated, the final score is raised to at least 0.99.

## 4.8 Error Handling

The face-worker client returns a structured result with an `error` field. A missing face score now has two possible meanings: "no face found" (a valid answer) or "check did not run" (worker down, timeout, or model missing). In Deepfake mode the second case returns *Inconclusive* rather than "Authentic Face". In Auto mode the verdict is computed from the whole-image detectors and a warning is shown. The console displays every warning above the score bar.

## 4.9 Web Console

The frontend is written in TypeScript with Next.js 14. `src/lib/api.ts` holds typed request functions. These also treat a backend response of `success:false` as an error, so the views never try to draw a missing result. Sample images and videos are served from `public/static/samples` so that the test buttons work on a fresh clone.

> **[Insert screenshot]** Figure 4.1: Surveillance view with person tracking and risk level

> **[Insert screenshot]** Figure 4.2: Forensics view showing the verdict, model scores and heatmaps

> **[Insert screenshot]** Figure 4.3: Investigation view with the combined incident verdict

---

# Chapter 5 Evaluation and Results

## 5.1 Problems Found in Version 1

The first complete version worked end to end, but its deepfake results on new images and videos were poor. Before changing anything, each part was measured separately. The main findings were:

- **The pixel detector produced nearly constant output.** The NPR ResNet-50 file in the project gave logits around −100 for almost every image, real or fake. Its AUC on labelled data was about 0.50, which is no better than chance. The file size matched a training checkpoint in the NPR repository, not the released model. The released model also stayed near chance on this data.
- **The face model was not GenD.** The documentation said GenD, but the worker was loading a generic ViT classifier. On the face images in the evaluation set its AUC was 0.58, and it flagged 43% of real faces as fake.
- **No calibration.** The calibration file contained identity values, so raw scores from different models were compared directly.
- **Hand-set rules.** Weights and thresholds between 0.38 and 0.48 had been chosen by hand. One rule capped the score at 0.36 when two detectors disagreed, which forced some fakes to be labelled real.
- **Silent failure.** If the face worker timed out, Deepfake mode used a face score of 0.0 and reported "Authentic Face".
- **Unreliable old benchmark.** The 400-image benchmark file in the project was 82% fake and its source and labels could not be confirmed, so it was discarded.

## 5.2 Evaluation Data

All evaluation data was downloaded from public sources with their original labels (Table 5.1).

**Table 5.1: Labelled image data used for calibration and testing**

| Source and subset | Label | Images |
|---|---|---|
| Defactify<sup>28</sup> validation, MS-COCO photographs | Real | 250 |
| Defactify: SD 2.1, SDXL, SD 3, DALL-E 3, Midjourney | Fake | 5 × 50 |
| DeepFakeFace<sup>17</sup>: IMDB-WIKI faces | Real | 250 |
| DeepFakeFace: InsightFace swap, SD inpainting, SD 1.5 | Fake | 3 × 83 |
| GenD sample crops: FF++<sup>14</sup> and Celeb-DF<sup>15</sup> | Both | 32 |
| **Total** | 512 real, 519 fake | **1,031** |

For video, 189 test videos (100 real, 89 fake) were downloaded from the official Celeb-DF v2 test list.

## 5.3 Protocol

Every image was scored twice: as downloaded and after re-saving as JPEG at quality 85. The second copy is closer to images shared on messaging apps. This gives 2,062 scored variants. Images were split into a fit half and a test half using the parity of the SHA-1 hash of the file path, and both copies of an image always fall in the same half. All numbers below are on the 1,010 test variants, which were not used for fitting.

Version 1 was replayed on the same test split with its original NPR weights, its original face classifier and its original decision rules. Its compression re-weighting and ELA override were left out of the replay, so the version 1 figures are a close approximation, not an exact copy.

The metrics used are: *fakes detected* (true positive rate), *real correct* (true negative rate), *balanced accuracy* (their mean) and the area under the ROC curve (AUC).

## 5.4 Fitted Calibration

**Table 5.2: Fitted Platt coefficients and fusion weights**

| Detector | Platt a | Platt b | No-face weight | Face weight |
|---|---|---|---|---|
| Community Forensics | 0.550 | 3.471 | 0.660 | 0.616 |
| B-Free | 0.454 | 0.819 | 0.484 | 0.392 |
| UniversalFakeDetect CLIP | 0.290 | 1.927 | 0.479 | 0.224 |
| GenD (face) | 0.713 | −0.031 | – | 1.112 |
| Intercept | | | 0.085 | −0.427 |

The large positive b values for Community Forensics and CLIP mean that their raw scores were much too low on this data. Many fakes had raw scores close to zero. This explains why the fixed thresholds of version 1 missed so many fakes. When a face is present, GenD gets the largest weight, which matches its purpose.

## 5.5 Overall Results

**Table 5.3: Held-out results, version 1 and version 2 (percentages)**

| Group | n | V1 Fake | V1 Real | V1 Bal. | V2 Fake | V2 Real | V2 Bal. |
|---|---|---|---|---|---|---|---|
| All test variants | 1010 | 62 | 76 | 69 | 79 | 96 | 88 |
| Original files | 505 | 62 | 75 | 69 | 79 | 96 | 87 |
| JPEG quality 85 | 505 | 63 | 77 | 70 | 80 | 96 | 88 |
| Face scored | 615 | 78 | 55 | 66 | 83 | 94 | 88 |
| No face | 395 | 31 | 100 | 65 | 73 | 99 | 86 |

The AUC of version 2 is 0.943 overall (0.941 on original files, 0.944 on JPEG copies, 0.954 when a face is scored and 0.924 without a face). Version 1 was a rule system without a single continuous score, so no AUC is given for it. Detection of fakes rose by 17 points while false alarms on real images fell from 24% to 4%. The results on original files and JPEG copies are almost the same, so moderate compression does not hurt the new pipeline.

## 5.6 Results by Generator

**Table 5.4: Correct decisions by subset (percentages)**

| Subset | Label | n | Version 1 | Version 2 |
|---|---|---|---|---|
| Defactify, MS-COCO | Real | 226 | 95 | 99 |
| Stable Diffusion 2.1 | Fake | 50 | 96 | 100 |
| SDXL | Fake | 52 | 52 | 96 |
| Stable Diffusion 3 | Fake | 42 | 36 | 95 |
| DALL-E 3 | Fake | 62 | 13 | 40 |
| Midjourney | Fake | 48 | 4 | 65 |
| DeepFakeFace, IMDB-WIKI | Real | 222 | 57 | 97 |
| InsightFace swap | Fake | 98 | 55 | 46 |
| SD inpainting | Fake | 96 | 99 | 100 |
| SD 1.5 text-to-image | Fake | 86 | 100 | 100 |

**Figure 5.1: Fake detection rate by generator (%)** (bar chart in the Overleaf version)

| Generator | Version 1 | Version 2 |
|---|---|---|
| SD 2.1 | 96 | 100 |
| SDXL | 52 | 96 |
| SD 3 | 36 | 95 |
| DALL-E 3 | 13 | 40 |
| Midjourney | 4 | 65 |
| InsightFace | 55 | 46 |
| Inpainting | 99 | 100 |
| SD 1.5 | 100 | 100 |

The largest gains are on SDXL, Stable Diffusion 3 and Midjourney. Midjourney went from 4% to 65%. On real faces the error rate fell from 43% to 3%, which was the most visible problem for users of version 1.

InsightFace swaps are the one subset that got worse (55% to 46%). The version 1 face classifier flagged a large share of all faces, real or fake, so its 55% on swaps came together with 43% false alarms on real faces. Version 2 makes far fewer mistakes overall, but face swaps on small, low-resolution IMDB-WIKI photographs are still hard for GenD. DALL-E 3 remains the weakest generator at 40%.

On the 28 GenD sample crops (FF++ and Celeb-DF), version 2 detected 86% of fakes but accepted only 36% of the real crops. This subset is too small to draw firm conclusions. Its pre-cropped faces also differ from full photographs, and Chapter 6 returns to this point.

## 5.7 Individual Detectors

**Table 5.5: Single detectors on the test split (calibrated, threshold 0.5)**

| Detector | n | AUC | Fake (%) | Real (%) |
|---|---|---|---|---|
| Community Forensics | 1010 | 0.887 | 76 | 92 |
| B-Free | 1010 | 0.866 | 73 | 90 |
| UniversalFakeDetect CLIP | 1010 | 0.692 | 59 | 68 |
| GenD (faces only) | 615 | 0.828 | 67 | 87 |
| NPR, version 1 | 1010 | ≈0.50 | – | – |
| Generic face ViT, version 1 | 615 | 0.582 | 66 | 51 |

No single detector reaches the fused result (AUC 0.943). Community Forensics is the strongest model alone. B-Free is close behind and has replaced NPR, which carried no information. CLIP is weak alone, but it still gets a useful fusion weight because its errors differ from those of the other two.

## 5.8 Running Time

After warm-up, typical GPU times per image were about 0.03 s for Community Forensics, 0.25 to 0.35 s for B-Free, 0.06 s for CLIP, and 0.02 to 0.6 s for face detection with GenD, depending on the number and size of faces. A full image analysis with ELA and heatmaps takes well under two seconds. With all models loaded in one process, GPU memory use was between 4.2 and 4.6 GB of the 6 GB available.

## 5.9 Manual Testing

After the update, the web console was tested by hand with the sample files supplied with the project and with additional images and videos. The verdicts matched expectations on these files. Table 5.6 records the individual cases.

**Table 5.6: Manual test cases (to be completed with observed outputs)**

| File | Mode | Expected | Observed verdict and score |
|---|---|---|---|
| F.png | Investigation | Manipulated | |
| R.jpg | Investigation | Authentic | |
| deepfake_face.jpg | Deepfake | Fake | |
| authentic.jpg | Auto | Real | |
| sample_diffusion.jpg | AI-Generated | Fake | |
| sample_deepfake_video.mp4 | Auto | Fake | |
| surveillance_sample.mp4 | Surveillance | Crowd counted | |

## 5.10 Discussion and Limitations

The main lesson is that the poor results of version 1 came from the integration, not from the published detectors themselves. One detector carried no information, one model was not the model described in the documentation, and hand-set thresholds sat in the wrong place for raw scores. Measuring each part on labelled data found all of these problems within a day. Guessing would have taken much longer.

The limitations are:

- Video results are not yet measured. The Celeb-DF test videos are prepared, but the evaluation run was not completed in time.
- The calibration data comes from five commercial and open generators and two face datasets. Newer generators such as Flux, or heavy social-media compression, may behave differently.
- DALL-E 3 images and InsightFace swaps are still detected less than half the time.
- The weapon classifier is not trained yet, so weapon-based verdicts depend only on what the YOLO model detects.
- B-Free's licence allows only non-profit use, which matters if the system is ever deployed commercially.

---

# Chapter 6 Conclusions

## 6.1 Conclusions

CrowdGuard AI joins crowd monitoring and media forensics in one tool, and the combined verdict separates real incidents from fabricated footage. The main technical result of this report is the rebuilt deepfake module. Published detectors were wired in correctly, and their scores were calibrated and fused with weights fitted on public labelled data. Measured on images the module had never seen, balanced accuracy rose from 69% to 88% (AUC 0.943), and false alarms on real faces fell from 43% to 3%. The system also states clearly when a check did not run. In version 1, a failed check could be reported as authentic.

## 6.2 Future Work

1. Run `tools/eval_video.py` on the 189 Celeb-DF test videos, fit the video fusion model and report video accuracy.
2. Add recent generators (Flux, SD 3.5) and images re-shared through messaging apps to the calibration set, and look at DALL-E 3 and face-swap failures in particular.
3. Train the weapon detector and classifier (second review).
4. Add optical-flow crowd motion analysis and live RTSP streams from several cameras.
5. Generate signed PDF incident reports and store incidents in a database.
6. Use operator feedback, collected through `/feedback`, for periodic recalibration.

---

# Bibliography and References

1. Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D, Ozair S, Courville A and Bengio Y, "Generative adversarial nets." Advances in Neural Information Processing Systems. 2014, 27, 2672-2680.
2. Ho J, Jain A and Abbeel P, "Denoising diffusion probabilistic models." Advances in Neural Information Processing Systems. 2020, 33, 6840-6851.
3. Rombach R, Blattmann A, Lorenz D, Esser P and Ommer B, "High-resolution image synthesis with latent diffusion models." IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022, 10684-10695.
4. Tolosana R, Vera-Rodriguez R, Fierrez J, Morales A and Ortega-Garcia J, "Deepfakes and beyond: A survey of face manipulation and fake detection." Information Fusion. 2020, 64, 131-148.
5. Wang SY, Wang O, Zhang R, Owens A and Efros AA, "CNN-generated images are surprisingly easy to spot... for now." IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020, 8695-8704.
6. Ojha U, Li Y and Lee YJ, "Towards universal fake image detectors that generalize across generative models." IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, 24480-24489.
7. Radford A, Kim JW, Hallacy C, et al., "Learning transferable visual models from natural language supervision." International Conference on Machine Learning. 2021, PMLR 139, 8748-8763.
8. Tan C, Zhao Y, Wei S, Gu G, Liu P and Wei Y, "Rethinking the up-sampling operations in CNN-based generative network for generalizable deepfake detection." IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024, 28130-28139.
9. Park J and Owens A, "Community Forensics: Using thousands of generators to train fake image detectors." IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2025.
10. Dosovitskiy A, Beyer L, Kolesnikov A, et al., "An image is worth 16x16 words: Transformers for image recognition at scale." International Conference on Learning Representations. 2021.
11. Guillaro F, Zingarini G, Usman B, Sud A, Cozzolino D and Verdoliva L, "A bias-free training paradigm for more general AI-generated image detection." IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2025, 18685-18694.
12. Oquab M, Darcet T, Moutakanni T, et al., "DINOv2: Learning robust visual features without supervision." Transactions on Machine Learning Research. 2024.
13. Darcet T, Oquab M, Mairal J and Bojanowski P, "Vision transformers need registers." International Conference on Learning Representations. 2024.
14. Rössler A, Cozzolino D, Verdoliva L, Riess C, Thies J and Nießner M, "FaceForensics++: Learning to detect manipulated facial images." IEEE/CVF International Conference on Computer Vision. 2019, 1-11.
15. Li Y, Yang X, Sun P, Qi H and Lyu S, "Celeb-DF: A large-scale challenging dataset for DeepFake forensics." IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020, 3207-3216.
16. Yermakov A, Cech J, Matas J and Fritz M, "Deepfake detection that generalizes across benchmarks." IEEE/CVF Winter Conference on Applications of Computer Vision. 2026, 773-783.
17. Song H, Huang S, Dong Y and Tu WW, "Robustness and generalizability of deepfake detection: A study with diffusion models." arXiv preprint arXiv:2309.02218. 2023.
18. Guo C, Pleiss G, Sun Y and Weinberger KQ, "On calibration of modern neural networks." International Conference on Machine Learning. 2017, PMLR 70, 1321-1330.
19. Platt J. In Advances in Large Margin Classifiers, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods"; MIT Press, Cambridge, 1999, pp 61-74.
20. Jocher G, Chaurasia A and Qiu J, "Ultralytics YOLOv8", 2023, https://github.com/ultralytics/ultralytics
21. Zhang Y, Sun P, Jiang Y, et al., "ByteTrack: Multi-object tracking by associating every detection box." European Conference on Computer Vision. 2022, 1-21.
22. Howard A, Sandler M, Chu G, et al., "Searching for MobileNetV3." IEEE/CVF International Conference on Computer Vision. 2019, 1314-1324.
23. Krawetz N, "A picture's worth: Digital image analysis and forensics." Black Hat Briefings, USA, 2007.
24. Coalition for Content Provenance and Authenticity, "C2PA Technical Specification", https://c2pa.org/specifications/
25. Ramírez S, "FastAPI", https://fastapi.tiangolo.com
26. Vercel, "Next.js Documentation", https://nextjs.org/docs
27. Bradski G, "The OpenCV Library." Dr. Dobb's Journal of Software Tools. 2000.
28. Roy R, et al., "Defactify Image Dataset", Hugging Face, https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset

---

# Appendix A List of Abbreviations

| Abbreviation | Meaning |
|---|---|
| AUC | Area Under the ROC Curve |
| API | Application Programming Interface |
| C2PA | Coalition for Content Provenance and Authenticity |
| CCTV | Closed-Circuit Television |
| CLIP | Contrastive Language-Image Pre-training |
| CUDA | Compute Unified Device Architecture |
| DFF | DeepFakeFace dataset |
| ELA | Error Level Analysis |
| FF++ | FaceForensics++ |
| GAN | Generative Adversarial Network |
| GPU | Graphics Processing Unit |
| IoU | Intersection over Union |
| NPR | Neighboring Pixel Relationships |
| ROC | Receiver Operating Characteristic |
| SD | Stable Diffusion |
| SSE | Server-Sent Events |
| SSD | Single Shot Detector |
| ViT | Vision Transformer |
| YOLO | You Only Look Once |

# Appendix B Setup and Reproduction

**Model weights** (not stored in git), placed in `CrowdGuardAI/pretrained_weights/`: `yolo_crowd.pt`, `fc_weights.pth`, and the folder `GenD/` (from `yermandy/GenD_CLIP_L_14`). B-Free weights go in `CrowdGuardAI/detectors/B-Free/code/weights/BFREE_dino2reg4/`. Community Forensics and the face detector files are downloaded automatically on first run.

**Running the system**

```
cd frontend
npm install
npm run build
cd ..
run_crowdguard.bat
```

**Reproducing the evaluation**

```
cd CrowdGuardAI
..\venv_gend\Scripts\python.exe tools\build_eval_set.py
..\venv_gend\Scripts\python.exe tools\calibrate.py
..\venv_gend\Scripts\python.exe tools\build_video_eval.py
..\venv_gend\Scripts\python.exe tools\eval_video.py
```

The first two commands download the labelled images and write `data/calibration.json` together with the held-out metrics reported in Chapter 5. The last two prepare and evaluate the Celeb-DF test videos.

---

<div align="center">

**PLAGIARISM VERIFICATION**

(This certificate has to be attached at the backside of the dissertation)

</div>

Title of the dissertation: CrowdGuard AI: An Integrated System for Crowd Threat Monitoring and Deepfake Media Verification

This is to report that the above thesis was scanned for similarity detection. Process and outcome is given below:

**Software used** .................................... **Date** ....................

**Similarity Index** ............................. **Total word count** ....................

The complete report is submitted for review by the Supervisor/Co-supervisor.

Checked by Name & Signature &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Library Head Name & Sign**

The complete report of the above thesis has been reviewed by the undersigned. (Check Box)

- [ ] The similarity index is below accepted norms.
- [ ] The similarity index is above accepted norms, because of the following reasons:
  ..........................................................................
- [ ] The dissertation may be considered for the award of degree. (Relevant documents attached).

| Name & Sign of Student | Name & Sign of Supervisor | Name & Sign of Co-Supervisor |
|---|---|---|
