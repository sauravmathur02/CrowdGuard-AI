const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export interface HealthResponse {
  status: string;
  system: string;
  hardware: {
    device: string;
    gpu_name: string;
    cuda_available: boolean;
    vram: {
      total_gb: number;
      allocated_gb: number;
      reserved_gb: number;
      free_gb: number;
    };
  };
  deepfake_models: {
    commfor: { name: string; active: boolean };
    bfree_npr: { name: string; active: boolean };
    clip: { name: string; active: boolean };
    gend_worker: { running: boolean; healthy: boolean; model?: string | null; pid: number; port: number; url: string };
    calibrated?: boolean;
  };
  crowd_models: {
    yolo_detector: { active: boolean; weights: string; is_custom_weapon_model: boolean; classes: number[] };
    weapon_classifier: { active: boolean; labels: string[] };
  };
  ensemble_ready: boolean;
}

export interface CrowdDetectionResponse {
  success: boolean;
  source_id: string;
  timestamp: string;
  crowd_count: number;
  active_tracks: number;
  dominant_weapon: string;
  alert_active: boolean;
  weapon_labels: string[];
  feed_tampered: boolean;
  feed_integrity_score: number;
  annotated_image: string;
  risk: {
    score: number;
    level: "SAFE" | "ELEVATED" | "ALERT" | "DANGER" | "CRITICAL";
    weapon_count: number;
    suspicious: boolean;
    reasons: string[];
  };
  detections: Array<{
    label: string;
    display_label: string;
    confidence: number;
    bbox: [number, number, number, number];
    track_id: number | null;
  }>;
}

export interface CrowdVideoResponse {
  success: boolean;
  filename: string;
  duration_seconds: number;
  total_frames: number;
  fps: number;
  peak_crowd: number;
  weapons_detected: string[];
  has_weapons: boolean;
  overall_risk: {
    score: number;
    level: "SAFE" | "ELEVATED" | "ALERT" | "DANGER" | "CRITICAL";
    reasons: string[];
  };
  feed_integrity: {
    score: number;
    is_compromised: boolean;
    tampered_samples: number;
    total_samples: number;
  };
  timeline: Array<{
    timestamp: string;
    time_seconds: number;
    crowd_count: number;
    risk_score: number;
    risk_level: string;
    weapon_count: number;
    feed_integrity: number;
    is_tampered: boolean;
  }>;
  sample_frames: Array<{
    timestamp: string;
    image: string;
    crowd_count: number;
    risk_level: string;
  }>;
}

export interface ForensicsResponse {
  success: boolean;
  mode: string;
  prediction: "Real" | "Fake" | "No Face Detected" | "Inconclusive";
  score: number;
  confidence: string;
  confidence_variance: number;
  commfor_score: number;
  bfree_score: number;
  clip_score: number | null;
  prov_score: number;
  gend_score: number | null;
  faces_found: number;
  face_model?: string | null;
  face_checked?: boolean;
  /** Checks that did not run or ran degraded (e.g. face worker down). */
  warnings?: string[];
  fusion?: {
    model: "face" | "no_face";
    threshold: number;
    calibrated: boolean;
    contributions: Record<string, number>;
  } | null;
  explanation: {
    category: string;
    summary: string;
    findings: string[];
    likelihood: string;
  };
  media_type: "image" | "video";
  frame_scores: number[];
  exif: Record<string, string>;
  ela: string | null;
  residual_map: string | null;
  gradcam: string | null;
  attention_heatmap: string | null;
  quality: {
    lap_var?: number;
    block_ratio?: number;
    is_heavily_compressed?: boolean;
  } | null;
}

export interface InvestigationResponse {
  success: boolean;
  filename: string;
  incident_verdict: string;
  verdict_summary: string;
  threat_level: "SAFE" | "ELEVATED" | "ALERT" | "HIGH" | "CRITICAL";
  media_authenticity: {
    prediction: string;
    manipulation_score: number;
    confidence: string;
    category: string;
    findings: string[];
    warnings?: string[];
    ela_heatmap: string | null;
    gradcam_heatmap: string | null;
    residual_map: string | null;
  };
  crowd_intelligence: {
    peak_crowd: number;
    weapons_found: string[];
    annotated_samples: Array<{
      timestamp: string;
      image: string;
      crowd_count: number;
      risk_level: string;
    }>;
    frames_analyzed?: number;
    timeline?: Array<{ timestamp: string; crowd_count: number; weapon_count: number }>;
  };
  media_type?: "image" | "video";
}

/** One progress event from a streaming (SSE) endpoint. */
export interface StageEvent {
  event: "started" | "stage" | "complete" | "error";
  stage?: number;
  name?: string;
  status?: "running" | "completed" | "failed";
  details?: Record<string, any>;
  result?: any;
  error?: string;
}

/** POST a form to an SSE endpoint, call onEvent for each event, resolve with the final result. */
async function postStream<T>(path: string, form: FormData, onEvent: (e: StageEvent) => void): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  if (!res.ok || !res.body) throw new Error(`Request failed: ${res.status} ${res.statusText}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final: T | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 2);
      if (!chunk.startsWith("data:")) continue;
      const evt: StageEvent = JSON.parse(chunk.slice(5).trim());
      onEvent(evt);
      if (evt.event === "error") throw new Error(evt.error || "Analysis failed.");
      if (evt.event === "complete") {
        if (evt.result && evt.result.success === false) throw new Error(evt.result.error || "Analysis failed.");
        final = evt.result as T;
      }
    }
  }
  if (!final) throw new Error("The analysis stream ended without a result.");
  return final;
}

export function runInvestigationStream(file: File, onEvent: (e: StageEvent) => void) {
  const form = new FormData();
  form.append("file", file);
  return postStream<InvestigationResponse>("/api/investigate/stream", form, onEvent);
}

export function detectForensicsStream(file: File, mode: string, onEvent: (e: StageEvent) => void) {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  return postStream<ForensicsResponse>("/api/predict/stream", form, onEvent);
}

/** Backend images are full data URLs; older responses were bare base64. Accept both. */
export function asImageSrc(img: string | null | undefined): string | undefined {
  if (!img) return undefined;
  return img.startsWith("data:") ? img : `data:image/jpeg;base64,${img}`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function detectCrowdImage(file: File): Promise<CrowdDetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/crowd/detect`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Crowd detection failed: ${res.statusText}`);
  return res.json();
}

export async function detectCrowdVideo(file: File): Promise<CrowdVideoResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/crowd/video`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Video analysis failed: ${res.statusText}`);
  return res.json();
}

export async function detectForensics(file: File, mode: string = "auto"): Promise<ForensicsResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", mode);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Forensics analysis failed: ${res.statusText}`);
  const data = await res.json();
  // The backend reports analysis errors as { success: false, error } with HTTP 200.
  if (data && data.success === false) throw new Error(data.error || "Forensics analysis failed.");
  return data;
}

export async function runInvestigation(file: File): Promise<InvestigationResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/investigate`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Investigation failed: ${res.statusText}`);
  const data = await res.json();
  if (data && data.success === false) throw new Error(data.error || "Investigation failed.");
  return data;
}

export async function fetchRemoteUrl(url: string) {
  const res = await fetch(`${API_BASE}/fetch-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  return res.json();
}
