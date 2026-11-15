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
} from "lucide-react";
import { runInvestigation, InvestigationResponse } from "@/lib/api";

export default function InvestigationView() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InvestigationResponse | null>(null);
  const [activeArtifactTab, setActiveArtifactTab] = useState<"ela" | "gradcam" | "residual">("ela");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setError(null);
      if (selected.type.startsWith("image/")) {
        setPreviewUrl(URL.createObjectURL(selected));
      } else {
        setPreviewUrl(null);
      }
    }
  };

  const handleRunInvestigation = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runInvestigation(file);
      setResult(res);
    } catch (err: any) {
      setError(err?.message || "Investigation failed to execute.");
    } finally {
      setLoading(false);
    }
  };

  const loadSample = async (filename: string) => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`/static/samples/${filename}`);
      if (!res.ok) {
        throw new Error(`Sample file ${filename} not found in public/static/samples.`);
      }
      const blob = await res.blob();
      const sampleFile = new File([blob], filename, { type: blob.type || "image/png" });
      setFile(sampleFile);
      setPreviewUrl(URL.createObjectURL(blob));
      const invRes = await runInvestigation(sampleFile);
      setResult(invRes);
    } catch (err: any) {
      setError(err?.message || `Failed to run sample ${filename}`);
    } finally {
      setLoading(false);
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
                        src={`data:image/jpeg;base64,${result.media_authenticity.ela_heatmap}`}
                        alt="ELA Heatmap"
                        className="w-full h-full object-contain"
                      />
                    )}
                    {activeArtifactTab === "gradcam" && result.media_authenticity.gradcam_heatmap && (
                      <img
                        src={`data:image/jpeg;base64,${result.media_authenticity.gradcam_heatmap}`}
                        alt="GradCAM Attention"
                        className="w-full h-full object-contain"
                      />
                    )}
                    {activeArtifactTab === "residual" && result.media_authenticity.residual_map && (
                      <img
                        src={`data:image/jpeg;base64,${result.media_authenticity.residual_map}`}
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
                    {result.crowd_intelligence.annotated_samples.length > 0 ? (
                      <div className="w-full h-full flex flex-col">
                        <img
                          src={`data:image/jpeg;base64,${result.crowd_intelligence.annotated_samples[0].image}`}
                          alt="Annotated Crowd"
                          className="w-full h-full object-contain"
                        />
                        <div className="absolute bottom-2 left-2 right-2 px-3 py-1.5 rounded-lg bg-slate-900/80 backdrop-blur-md border border-slate-700/60 flex items-center justify-between text-xs text-slate-300">
                          <span>Timestamp: {result.crowd_intelligence.annotated_samples[0].timestamp}</span>
                          <span className="font-semibold text-cyan-400">
                            Crowd: {result.crowd_intelligence.annotated_samples[0].crowd_count} | Threat: {result.crowd_intelligence.annotated_samples[0].risk_level}
                          </span>
                        </div>
                      </div>
                    ) : (
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
