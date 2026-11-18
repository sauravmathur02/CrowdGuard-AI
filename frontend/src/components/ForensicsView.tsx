"use client";

import React, { useState } from "react";
import {
  Upload,
  Sparkles,
  Bot,
  Drama,
  Fingerprint,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Eye,
  Layers,
  FileSearch,
  ChevronRight,
} from "lucide-react";
import { detectForensicsStream, ForensicsResponse } from "@/lib/api";
import MediaPreview from "./MediaPreview";

const MODES = [
  { key: "auto", label: "Auto Detect", icon: Sparkles, desc: "Autonomous mode — identifies authentic, deepfake, or AI-generated." },
  { key: "deepfake", label: "Deepfake Face", icon: Drama, desc: "Facial forensics — seam blending, eye inconsistencies, identity swaps." },
  { key: "ai_generated", label: "AI Generated", icon: Bot, desc: "Full-scene diffusion — Midjourney, DALL-E, Stable Diffusion, Flux." },
] as const;

export default function ForensicsView() {
  const [mode, setMode] = useState<"auto" | "deepfake" | "ai_generated">("auto");
  const [loading, setLoading] = useState(false);
  const [scanStep, setScanStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ForensicsResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"ela" | "gradcam" | "metadata">("ela");
  const [dragOver, setDragOver] = useState(false);
  const [media, setMedia] = useState<{ url: string; isVideo: boolean; name: string } | null>(null);
  const [stepDetail, setStepDetail] = useState<string>("");

  const runAnalysis = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setScanStep(0);
    setStepDetail("Uploading…");
    setMedia({
      url: URL.createObjectURL(file),
      isVideo: file.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm)$/i.test(file.name),
      name: file.name,
    });

    try {
      // Real stage-by-stage progress from the backend (Server-Sent Events).
      const res = await detectForensicsStream(file, mode, (evt) => {
        if (evt.event !== "stage" || !evt.stage) return;
        setScanStep(evt.status === "completed" ? evt.stage : evt.stage - 0.5);
        setStepDetail(evt.details?.message || `${evt.name} ${evt.status === "completed" ? "done" : "…"}`);
      });
      setResult(res);
    } catch (err: any) {
      setError(err?.message || "Analysis failed. Is the backend running?");
    } finally {
      setLoading(false);
      setScanStep(0);
      setStepDetail("");
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files[0]) runAnalysis(e.dataTransfer.files[0]);
  };

  const loadSample = async (path: string, name: string) => {
    try {
      setLoading(true); setError(null);
      const r = await fetch(path);
      if (!r.ok) throw new Error("Sample not found");
      const blob = await r.blob();
      await runAnalysis(new File([blob], name, { type: blob.type || "image/jpeg" }));
    } catch (err: any) { setError(err.message); setLoading(false); }
  };

  const isFake = result?.prediction === "Fake";
  // "No Face Detected" / "Inconclusive" carry no verdict, so they must not render as "Authentic".
  const isNeutral = !!result && result.prediction !== "Fake" && result.prediction !== "Real";
  // For Fake: show manipulation probability. For Real: show authentic confidence.
  const displayScore = result && !isNeutral
    ? isFake
      ? Math.round(result.score * 100)
      : Math.max(1, Math.round((1 - result.score) * 100))
    : 0;
  const tone = isNeutral ? "neutral" : isFake ? "fake" : "real";
  const toneBanner = { fake: "bg-rose-950/30 border-rose-500/40", real: "bg-emerald-950/30 border-emerald-500/40", neutral: "bg-amber-950/20 border-amber-500/40" }[tone];
  const toneBadge = { fake: "bg-rose-500/20 text-rose-300 border-rose-500/30", real: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30", neutral: "bg-amber-500/15 text-amber-300 border-amber-500/30" }[tone];

  return (
    <div className="space-y-6">
      {/* Top Section: Mode + Upload */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Controls */}
        <div className="lg:col-span-4 space-y-4">

          {/* Mode Selector */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Analysis Mode</p>
            <div className="space-y-2">
              {MODES.map(({ key, label, icon: Icon, desc }) => (
                <button
                  key={key}
                  onClick={() => setMode(key)}
                  className={`w-full flex items-center gap-3 p-3 rounded-xl text-left transition-all ${mode === key
                    ? "bg-purple-600/20 border border-purple-500/50 text-white"
                    : "bg-slate-950/40 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700"
                    }`}
                >
                  <div className={`p-1.5 rounded-lg shrink-0 ${mode === key ? "bg-purple-600/30 text-purple-300" : "bg-slate-800 text-slate-500"}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold">{label}</div>
                    <div className="text-[11px] text-slate-500 mt-0.5 leading-tight">{desc}</div>
                  </div>
                  {mode === key && <ChevronRight className="w-4 h-4 text-purple-400 ml-auto shrink-0" />}
                </button>
              ))}
            </div>
          </div>

          {/* Upload Zone */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Upload Media</p>
            <label
              className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition text-center block ${dragOver ? "border-purple-500 bg-purple-500/10" : "border-slate-700 hover:border-purple-500/60 bg-slate-950/40"
                }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept="image/*,video/*"
                className="hidden"
                onChange={(e) => { if (e.target.files?.[0]) runAnalysis(e.target.files[0]); }}
              />
              <Upload className="w-8 h-8 text-slate-500 mb-2" />
              <span className="text-sm font-semibold text-slate-200">Drop or Click to Upload</span>
              <span className="text-xs text-slate-500 mt-1">JPG, PNG, MP4, MOV, WEBP</span>
            </label>

            {/* Quick Test Samples */}
            <div className="mt-4 grid grid-cols-2 gap-2">
              <button
                onClick={() => loadSample("/static/samples/deepfake_face.jpg", "deepfake_face.jpg")}
                disabled={loading}
                className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 hover:border-rose-500/40 text-xs font-medium text-slate-300 flex items-center gap-1.5 transition disabled:opacity-50"
              >
                <Drama className="w-3.5 h-3.5 text-rose-400" />
                Sample Deepfake
              </button>
              <button
                onClick={() => loadSample("/static/samples/authentic.jpg", "authentic.jpg")}
                disabled={loading}
                className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 hover:border-emerald-500/40 text-xs font-medium text-slate-300 flex items-center gap-1.5 transition disabled:opacity-50"
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                Sample Authentic
              </button>
            </div>
          </div>
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-8 space-y-4">

          {/* Loading */}
          {loading && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center gap-4 mb-5">
                <div className="relative w-10 h-10 shrink-0">
                  <div className="absolute inset-0 rounded-full border-2 border-purple-500/30" />
                  <div className="absolute inset-0 rounded-full border-t-2 border-purple-400 animate-spin" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">Running {MODES.find(m => m.key === mode)?.label} Analysis…</p>
                  <p className="text-xs text-slate-400 mt-0.5">{stepDetail || "Multi-model ensemble fusion on CUDA GPU"}</p>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-2 text-xs">
                {["Provenance & EXIF", "ELA Compression", "ViT Backbones", "Ensemble Verdict"].map((step, i) => (
                  <div
                    key={i}
                    className={`p-2 rounded-lg border text-center transition-all ${scanStep >= i + 1
                      ? "bg-purple-950/40 border-purple-500/40 text-purple-300"
                      : scanStep > i
                        ? "bg-cyan-950/40 border-cyan-500/60 text-cyan-200 animate-pulse"
                        : "bg-slate-950 border-slate-800 text-slate-600"
                      }`}
                  >
                    <div className="font-semibold text-[11px]">{i + 1}. {step.split(" ")[0]}</div>
                    <div className="text-[10px] mt-0.5 text-slate-500">{step.split(" ").slice(1).join(" ")}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-sm text-rose-300 flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          {/* Empty State */}
          {!loading && !result && !error && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-16 flex flex-col items-center text-center">
              <Fingerprint className="w-14 h-14 text-slate-700 stroke-1 mb-4" />
              <h3 className="text-base font-bold text-slate-300">No Media Analyzed Yet</h3>
              <p className="text-xs text-slate-500 mt-2 max-w-xs">
                Choose a mode on the left, then upload a suspicious image or video — or try one of the sample files.
              </p>
            </div>
          )}

          {/* Result */}
          {result && !loading && (
            <div className="space-y-4">
              {/* Verdict Banner */}
              <div
                className={`p-5 rounded-2xl border ${toneBanner}`}
              >
                <div className="flex items-start gap-4">
                  <div className={`w-20 h-20 rounded-xl flex flex-col items-center justify-center border shrink-0 ${toneBadge}`}>
                    <span className="text-2xl font-black">{isNeutral ? "—" : `${displayScore}%`}</span>
                    <span className="text-[10px] font-bold uppercase tracking-wider opacity-80">
                      {isNeutral ? "No Verdict" : isFake ? "Fake Risk" : "Authentic"}
                    </span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`text-xs font-bold uppercase px-2.5 py-0.5 rounded-full border ${toneBadge}`}>
                        {result.prediction}
                      </span>
                      <span className="text-xs text-slate-400">{result.explanation.category} · {result.explanation.likelihood}</span>
                      <span className="text-xs text-slate-400 font-mono">Confidence: {result.confidence}</span>
                    </div>
                    <p className="text-base font-bold text-white mt-1">{result.explanation.summary}</p>

                    {/* Warnings: checks that did not run or ran in a degraded state */}
                    {result.warnings && result.warnings.length > 0 && (
                      <div className="mt-3 space-y-1">
                        {result.warnings.map((w, i) => (
                          <div key={i} className="flex items-start gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5">
                            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                            <span>{w}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Score bar */}
                    {!isNeutral && (
                    <div className="mt-3">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">
                          {isFake ? "Deepfake / Manipulation Probability" : "Authenticity Confidence"}
                        </span>
                        <span className={`font-mono font-semibold ${isFake ? "text-rose-400" : "text-emerald-400"}`}>
                          {displayScore}% {isFake ? "Likelihood" : "Verified Real"}
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-700 ${isFake ? "bg-rose-500" : "bg-emerald-500"}`}
                          style={{ width: `${displayScore}%` }}
                        />
                      </div>
                    </div>
                    )}

                    {/* Findings */}
                    <div className="mt-3 space-y-1">
                      {result.explanation.findings.slice(0, 3).map((f, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs text-slate-300">
                          <span className="w-1.5 h-1.5 rounded-full bg-slate-500 mt-1.5 shrink-0" />
                          {f}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* What was analysed */}
              <MediaPreview url={media?.url ?? null} isVideo={!!media?.isVideo} name={media?.name} />

              {/* 4 Model Score Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { name: "CommFor ViT-S", value: result.commfor_score, icon: Cpu, color: "text-sky-400", desc: "4,803 Generators" },
                  { name: "B-Free ViT-B", value: result.bfree_score, icon: Layers, color: "text-indigo-400", desc: "Bias-free (CVPR 2025)" },
                  { name: "CLIP ViT-L/14", value: result.clip_score, icon: Eye, color: "text-purple-400", desc: "Semantic Head" },
                  { name: "GenD Face", value: result.gend_score, icon: Drama, color: "text-amber-400", desc: result.face_checked === false ? "Not Checked" : result.faces_found > 0 ? `${result.faces_found} Face(s)` : "No Face Found" },
                ].map(({ name, value, icon: Icon, color, desc }) => {
                  const hasVal = value !== null && value !== undefined;
                  const isModFake = hasVal && value > 0.5;
                  const modPercent = hasVal ? Math.round(value * 100) : null;
                  const modAuthPercent = hasVal ? Math.round((1 - value) * 100) : null;

                  return (
                    <div key={name} className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-[11px] font-semibold text-slate-400">{name}</span>
                        <Icon className={`w-3.5 h-3.5 ${color}`} />
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-xl font-bold font-mono text-white">
                          {hasVal ? value.toFixed(3) : "N/A"}
                        </span>
                        {hasVal && (
                          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                            isModFake ? "bg-rose-500/20 text-rose-400" : "bg-emerald-500/20 text-emerald-400"
                          }`}>
                            {isModFake ? `${modPercent}% Fake` : `${modAuthPercent}% Real`}
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1">{desc}</div>
                      {hasVal && (
                        <div className="mt-2 w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 ${isModFake ? "bg-rose-500" : "bg-emerald-500"}`}
                            style={{ width: `${isModFake ? modPercent : modAuthPercent}%` }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* XAI Inspector (ELA / GradCAM / Metadata) */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <FileSearch className="w-4 h-4 text-purple-400" />
                    <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Evidence Heatmaps</h3>
                  </div>
                  <div className="flex gap-1 p-1 bg-slate-950 rounded-lg border border-slate-800">
                    {(["ela", "gradcam", "metadata"] as const).map((tab) => (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-3 py-1 rounded text-xs font-medium transition ${activeTab === tab ? "bg-purple-600 text-white" : "text-slate-400 hover:text-white"
                          }`}
                      >
                        {tab === "ela" ? "ELA" : tab === "gradcam" ? "GradCAM" : "EXIF"}
                      </button>
                    ))}
                  </div>
                </div>

                {activeTab === "ela" && (
                  <div>
                    <p className="text-xs text-slate-500 mb-3">Compression differentials — regenerated or pasted regions appear brighter.</p>
                    <div className="flex justify-center bg-slate-950 rounded-xl border border-slate-800 p-2 min-h-32">
                      {result.ela
                        ? <img src={result.ela} alt="ELA" className="max-h-80 object-contain rounded-lg" />
                        : <span className="text-xs text-slate-600 self-center">No ELA map generated</span>}
                    </div>
                  </div>
                )}

                {activeTab === "gradcam" && (
                  <div>
                    <p className="text-xs text-slate-500 mb-3">Spatial patches in the ViT that triggered synthetic classification.</p>
                    <div className="flex justify-center bg-slate-950 rounded-xl border border-slate-800 p-2 min-h-32">
                      {result.gradcam
                        ? <img src={result.gradcam} alt="GradCAM" className="max-h-80 object-contain rounded-lg" />
                        : <span className="text-xs text-slate-600 self-center">No GradCAM map generated</span>}
                    </div>
                  </div>
                )}

                {activeTab === "metadata" && (
                  <div className="overflow-x-auto max-h-72 overflow-y-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead className="bg-slate-950 text-slate-500 sticky top-0">
                        <tr>
                          <th className="p-2 font-medium">Tag</th>
                          <th className="p-2 font-medium">Value</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-slate-300">
                        {Object.entries(result.exif || {}).length > 0
                          ? Object.entries(result.exif).map(([k, v], i) => (
                            <tr key={i} className="hover:bg-slate-900/50">
                              <td className="p-2 text-slate-400">{k}</td>
                              <td className="p-2 truncate max-w-xs">{v}</td>
                            </tr>
                          ))
                          : <tr><td colSpan={2} className="p-4 text-center text-slate-600">No EXIF metadata found</td></tr>
                        }
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
