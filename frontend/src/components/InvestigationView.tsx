"use client";

import React, { useState } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  UploadCloud,
  FileSearch,
  Scan,
  Users,
  Crosshair,
  Cpu,
  Layers,
  CheckCircle,
  Eye,
  FileCode,
  Gavel,
} from "lucide-react";
import { runInvestigationStream, InvestigationResponse, StageEvent, asImageSrc } from "@/lib/api";
import MediaPreview from "./MediaPreview";
import PipelineTracker, { PipelineStep } from "./PipelineTracker";

const INITIAL_STEPS: PipelineStep[] = [
  { title: "Crowd Surveillance", subtitle: "YOLOv8 people & weapons", status: "pending", icon: Users },
  { title: "Deepfake Forensics", subtitle: "CommFor · B-Free · CLIP · GenD", status: "pending", icon: Cpu },
  { title: "Cross-Verification", subtitle: "Combined incident verdict", status: "pending", icon: Gavel },
];

const isVideoFile = (f: File | null) => !!f && (f.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm)$/i.test(f.name));

function describe(evt: StageEvent): string | undefined {
  const d = evt.details || {};
  if (evt.status === "failed") return d.message || "Stage failed";
  if (evt.stage === 1 && evt.status === "completed")
    return `${d.frames_analyzed} frame(s) · peak ${d.peak_crowd} people · ${d.weapons?.length ? `weapons: ${d.weapons.join(", ")}` : "no weapons"}`;
  if (evt.stage === 2 && evt.status === "completed")
    return `${d.prediction} · ${(Number(d.score) * 100).toFixed(1)}% fake probability · ${d.faces_found ?? 0} face(s)`;
  if (evt.stage === 2 && d.substage_name) return `${d.substage}/4 ${d.substage_name}${d.substage_status === "completed" ? " ✓" : "…"}`;
  if (evt.stage === 3 && evt.status === "completed") return `${d.threat_level} · ${d.verdict}`;
  return d.message;
}

export default function InvestigationView() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InvestigationResponse | null>(null);
  const [activeArtifactTab, setActiveArtifactTab] = useState<"ela" | "gradcam" | "residual">("ela");
  const [steps, setSteps] = useState<PipelineStep[]>(INITIAL_STEPS);
  const [elapsed, setElapsed] = useState(0);
  const [keyframe, setKeyframe] = useState(0);

  const selectFile = (selected: File) => {
    setFile(selected);
    setError(null);
    setResult(null);
    setSteps(INITIAL_STEPS);
    setPreviewUrl(URL.createObjectURL(selected));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) selectFile(e.target.files[0]);
  };

  const investigate = async (target: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setKeyframe(0);
    setSteps(INITIAL_STEPS);
    setElapsed(0);
    const t0 = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - t0) / 1000), 500);
    try {
      const res = await runInvestigationStream(target, (evt) => {
        if (evt.event !== "stage" || !evt.stage) return;
        setSteps((prev) =>
          prev.map((s, i) => {
            if (i + 1 !== evt.stage) return s;
            return { ...s, status: evt.status === "failed" ? "failed" : evt.status === "completed" ? "completed" : "running", detail: describe(evt) ?? s.detail };
          })
        );
      });
      setResult(res);
    } catch (err: any) {
      setError(err?.message || "Investigation failed to execute.");
      setSteps((prev) => prev.map((s) => (s.status === "running" ? { ...s, status: "failed" } : s)));
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  };

  const handleRunInvestigation = () => {
    if (file) investigate(file);
  };

  const loadSample = async (filename: string) => {
    try {
      setError(null);
      const res = await fetch(`/static/samples/${filename}`);
      if (!res.ok) {
        throw new Error(`Sample file ${filename} not found in public/static/samples.`);
      }
      const blob = await res.blob();
      const sampleFile = new File([blob], filename, { type: blob.type || "image/png" });
      selectFile(sampleFile);
      await investigate(sampleFile);
    } catch (err: any) {
      setError(err?.message || `Failed to run sample ${filename}`);
    }
  };

  const getVerdictTheme = (threat: string, verdict: string) => {
    if (verdict.includes("SPOOF") || threat === "CRITICAL") {
      return {
        bg: "bg-red-500/10 border-red-500/30 text-red-400",
        pill: "bg-red-500/20 text-red-300 border-red-500/40",
        icon: ShieldAlert,
        glow: "shadow-[0_0_30px_rgba(239,68,68,0.2)]",
      };
    }
    if (threat === "HIGH" || threat === "ALERT") {
      return {
        bg: "bg-amber-500/10 border-amber-500/30 text-amber-400",
        pill: "bg-amber-500/20 text-amber-300 border-amber-500/40",
        icon: AlertTriangle,
        glow: "shadow-[0_0_30px_rgba(245,158,11,0.2)]",
      };
    }
    return {
      bg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
      pill: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
      icon: ShieldCheck,
      glow: "shadow-[0_0_30px_rgba(16,185,129,0.2)]",
    };
  };

  return (
    <div className="space-y-6">
      {/* Overview Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-md">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-2">
              <Scan className="w-3.5 h-3.5" />
              Dual-Pipeline Incident Engine
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Incident Investigation & Cross-Verification
            </h2>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl">
              Cross-correlates physical crowd anomalies (person counts, weapon localization, crowd panic) with forensic deepfake analysis to definitively distinguish real-world threats from synthetic misinformation and spoofed camera feeds.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => loadSample("F.png")}
              className="px-3 py-2 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              disabled={loading}
            >
              Test Spoofed Feed (F.png)
            </button>
            <button
              onClick={() => loadSample("R.jpg")}
              className="px-3 py-2 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              disabled={loading}
            >
              Test Authentic Feed (R.jpg)
            </button>
            <button
              onClick={() => loadSample("surveillance_sample.mp4")}
              className="px-3 py-2 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
              disabled={loading}
            >
              Test CCTV Video
            </button>
          </div>
        </div>

        {/* Upload Zone */}
        <div className="mt-6">
          <label className="border-2 border-dashed border-slate-700 hover:border-cyan-500/50 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition bg-slate-950/40 group">
            <input
              type="file"
              accept="image/*,video/*"
              className="hidden"
              onChange={handleFileChange}
              disabled={loading}
            />
            <div className="p-4 rounded-full bg-cyan-500/10 text-cyan-400 group-hover:scale-110 transition">
              <UploadCloud className="w-8 h-8" />
            </div>
            <p className="mt-3 text-sm font-semibold text-slate-200">
              {file ? file.name : "Select or drag incident footage / image"}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports MP4, AVI, MOV, JPG, PNG (Max 50MB)
            </p>
          </label>

          {file && (
            <div className="mt-4 flex items-center justify-between">
              <span className="text-xs text-slate-400">
                Selected: <strong className="text-white">{file.name}</strong> ({(file.size / 1024 / 1024).toFixed(2)} MB)
              </span>
              <button
                onClick={handleRunInvestigation}
                disabled={loading}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-medium text-sm flex items-center gap-2 shadow-lg shadow-cyan-500/20 disabled:opacity-50 transition"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Executing Dual-Pipeline Analysis...
                  </>
                ) : (
                  <>
                    <FileSearch className="w-4 h-4" />
                    Run Investigation
                  </>
                )}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      {/* Live pipeline + uploaded media */}
      {(loading || result || steps.some((s) => s.status !== "pending")) && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8">
            <PipelineTracker steps={steps} elapsed={loading || result ? elapsed : undefined} />
          </div>
          <div className="lg:col-span-4">
            <MediaPreview url={previewUrl} isVideo={isVideoFile(file)} name={file?.name} title="Incident Footage" maxHeight="max-h-56" />
          </div>
        </div>
      )}
      {!loading && !result && file && steps.every((s) => s.status === "pending") && (
        <MediaPreview url={previewUrl} isVideo={isVideoFile(file)} name={file.name} title="Selected Footage" maxHeight="max-h-72" />
      )}

      {/* Investigation Results */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          {/* Main Verdict Banner */}
          {(() => {
            const theme = getVerdictTheme(result.threat_level, result.incident_verdict);
            const VerdictIcon = theme.icon;
            return (
              <div
                className={`p-6 rounded-2xl border ${theme.bg} ${theme.glow} flex flex-col md:flex-row md:items-center justify-between gap-4`}
              >
                <div className="flex items-start gap-4">
                  <div className={`p-3 rounded-xl ${theme.pill} shrink-0`}>
                    <VerdictIcon className="w-8 h-8" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${theme.pill}`}>
                        THREAT: {result.threat_level}
                      </span>
                      <span className="text-xs text-slate-400">
                        Target File: {result.filename}
                      </span>
                    </div>
                    <h3 className="text-xl font-extrabold text-white mt-1">
                      {result.incident_verdict}
                    </h3>
                    <p className="text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
                      {result.verdict_summary}
                    </p>
                  </div>
                </div>

                <div className="text-right shrink-0">
                  <div className="text-xs uppercase tracking-wider text-slate-400 font-mono">
                    {result.threat_level === "SAFE" ? "Authenticity Confidence" : "Manipulation Score"}
                  </div>
                  <div className="text-3xl font-extrabold text-white mt-0.5">
                    {result.threat_level === "SAFE"
                      ? `${Math.max(1, (100 - result.media_authenticity.manipulation_score * 100)).toFixed(1)}%`
                      : `${(result.media_authenticity.manipulation_score * 100).toFixed(1)}%`}
                  </div>
                  <div className="text-xs text-slate-400">
                    Confidence: {result.media_authenticity.confidence}
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Two-Column Cross-Correlated Evidence */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column: Digital Forensics Pipeline */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-cyan-400" />
                    <h4 className="font-bold text-white text-base">
                      Digital Media Forensics
                    </h4>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-semibold ${
                      result.media_authenticity.prediction.toLowerCase().includes("fake")
                        ? "bg-red-500/20 text-red-400 border border-red-500/30"
                        : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                    }`}
                  >
                    {result.media_authenticity.prediction}
                  </span>
                </div>

                {/* Score Progress Bar */}
                <div className="space-y-1.5 mb-5">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">
                      {result.media_authenticity.prediction.toLowerCase().includes("fake")
                        ? "Deepfake / Manipulation Risk"
                        : "Authenticity Confidence"}
                    </span>
                    <span className="text-white font-mono font-semibold">
                      {result.media_authenticity.prediction.toLowerCase().includes("fake")
                        ? `${(result.media_authenticity.manipulation_score * 100).toFixed(1)}% Fake Risk`
                        : `${Math.max(1, (100 - result.media_authenticity.manipulation_score * 100)).toFixed(1)}% Authentic`}
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        result.media_authenticity.manipulation_score > 0.60
                          ? "bg-gradient-to-r from-orange-500 to-red-500"
                          : "bg-gradient-to-r from-blue-500 to-emerald-500"
                      }`}
                      style={{
                        width: `${
                          result.media_authenticity.prediction.toLowerCase().includes("fake")
                            ? Math.min(100, Math.max(5, result.media_authenticity.manipulation_score * 100))
                            : Math.min(100, Math.max(5, (1 - result.media_authenticity.manipulation_score) * 100))
                        }%`,
                      }}
                    />
                  </div>
                </div>

                {/* Findings List */}
                <div className="space-y-2 mb-6">
                  {result.media_authenticity.warnings && result.media_authenticity.warnings.length > 0 && (
                    <div className="space-y-1">
                      {result.media_authenticity.warnings.map((w, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5">
                          <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                          <span>{w}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Forensic Indicators
                  </span>
                  <div className="bg-slate-950/50 rounded-xl p-3 border border-slate-800/80 space-y-2">
                    {result.media_authenticity.findings && result.media_authenticity.findings.length > 0 ? (
                      result.media_authenticity.findings.map((f, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-xs text-slate-300">
                          <CheckCircle className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
                          <span>{f}</span>
                        </div>
                      ))
                    ) : (
                      <p className="text-xs text-slate-500">No anomalies detected in frequency transform.</p>
                    )}
                  </div>
                </div>

                {/* Artifacts Tabs (ELA / GradCAM / Residual) */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Visual Forensics Artifacts
                    </span>
                    <div className="flex gap-1">
                      <button
                        onClick={() => setActiveArtifactTab("ela")}
                        className={`px-2.5 py-1 text-xs rounded-md transition ${
                          activeArtifactTab === "ela"
                            ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                            : "bg-slate-800 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        ELA Map
                      </button>
                      <button
                        onClick={() => setActiveArtifactTab("gradcam")}
                        className={`px-2.5 py-1 text-xs rounded-md transition ${
                          activeArtifactTab === "gradcam"
                            ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                            : "bg-slate-800 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        GradCAM
                      </button>
                      <button
                        onClick={() => setActiveArtifactTab("residual")}
                        className={`px-2.5 py-1 text-xs rounded-md transition ${
                          activeArtifactTab === "residual"
                            ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                            : "bg-slate-800 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        Noise Residual
                      </button>
                    </div>
                  </div>

                  <div className="h-60 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center overflow-hidden relative">
                    {activeArtifactTab === "ela" && result.media_authenticity.ela_heatmap && (
                      <img
                        src={asImageSrc(result.media_authenticity.ela_heatmap)}
                        alt="ELA Heatmap"
                        className="w-full h-full object-contain"
                      />
                    )}
                    {activeArtifactTab === "gradcam" && result.media_authenticity.gradcam_heatmap && (
                      <img
                        src={asImageSrc(result.media_authenticity.gradcam_heatmap)}
                        alt="GradCAM Attention"
                        className="w-full h-full object-contain"
                      />
                    )}
                    {activeArtifactTab === "residual" && result.media_authenticity.residual_map && (
                      <img
                        src={asImageSrc(result.media_authenticity.residual_map)}
                        alt="Noise Residual"
                        className="w-full h-full object-contain"
                      />
                    )}
                    {(!result.media_authenticity.ela_heatmap && activeArtifactTab === "ela") && (
                      <span className="text-xs text-slate-500">No ELA map generated</span>
                    )}
                    {(!result.media_authenticity.gradcam_heatmap && activeArtifactTab === "gradcam") && (
                      <span className="text-xs text-slate-500">No GradCAM map generated</span>
                    )}
                    {(!result.media_authenticity.residual_map && activeArtifactTab === "residual") && (
                      <span className="text-xs text-slate-500">No Residual map generated</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Physical Crowd Intelligence */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Users className="w-5 h-5 text-indigo-400" />
                    <h4 className="font-bold text-white text-base">
                      Physical Crowd Anomaly Intelligence
                    </h4>
                  </div>
                  <span className="px-2 py-0.5 rounded text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    Live Surveillance
                  </span>
                </div>

                {/* Key Metrics */}
                <div className="grid grid-cols-2 gap-3 mb-5">
                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                    <div className="flex items-center gap-2 text-slate-400 text-xs">
                      <Users className="w-3.5 h-3.5 text-blue-400" />
                      Peak Crowd Count
                    </div>
                    <div className="text-2xl font-bold text-white mt-1">
                      {result.crowd_intelligence.peak_crowd}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                    <div className="flex items-center gap-2 text-slate-400 text-xs">
                      <Crosshair className="w-3.5 h-3.5 text-red-400" />
                      Weapon Localization
                    </div>
                    <div className="text-sm font-bold text-slate-200 mt-2 truncate">
                      {result.crowd_intelligence.weapons_found.length > 0
                        ? result.crowd_intelligence.weapons_found.join(", ")
                        : "None Detected"}
                    </div>
                  </div>
                </div>

                {/* Annotated Surveillance Feed Samples */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Annotated Tracking Evidence
                    </span>
                    <span className="text-xs text-slate-500">
                      {result.crowd_intelligence.annotated_samples.length} Keyframe(s)
                    </span>
                  </div>

                  <div className="h-60 rounded-xl bg-slate-950 border border-slate-800 overflow-hidden relative">
                    {result.crowd_intelligence.annotated_samples.length > 0 ? (() => {
                      const samples = result.crowd_intelligence.annotated_samples;
                      const k = Math.min(keyframe, samples.length - 1);
                      const s = samples[k];
                      return (
                        <div className="w-full h-full flex flex-col">
                          <img src={asImageSrc(s.image)} alt="Annotated Crowd" className="w-full h-full object-contain" />
                          <div className="absolute bottom-2 left-2 right-2 px-3 py-1.5 rounded-lg bg-slate-900/80 backdrop-blur-md border border-slate-700/60 flex items-center justify-between gap-2 text-xs text-slate-300">
                            <span>Timestamp: {s.timestamp}</span>
                            {samples.length > 1 && (
                              <div className="flex gap-1">
                                {samples.map((_, i) => (
                                  <button
                                    key={i}
                                    onClick={() => setKeyframe(i)}
                                    className={`w-6 h-5 rounded text-[10px] font-mono ${i === k ? "bg-cyan-500/30 text-cyan-200" : "bg-slate-800 text-slate-400 hover:text-white"}`}
                                  >
                                    {i + 1}
                                  </button>
                                ))}
                              </div>
                            )}
                            <span className="font-semibold text-cyan-400">
                              Crowd: {s.crowd_count} | Threat: {s.risk_level}
                            </span>
                          </div>
                        </div>
                      );
                    })() : (
                      <div className="w-full h-full flex items-center justify-center text-xs text-slate-500">
                        No annotated frames captured
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Action buttons */}
              <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center justify-between">
                <span className="text-xs text-slate-400 flex items-center gap-1.5">
                  <FileCode className="w-3.5 h-3.5" />
                  Full evidence audit available
                </span>
                <button
                  onClick={() => {
                    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
                    const dl = document.createElement("a");
                    dl.setAttribute("href", dataStr);
                    dl.setAttribute("download", `investigation_${result.filename}_report.json`);
                    dl.click();
                  }}
                  className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 transition"
                >
                  Download Evidence JSON
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
