"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Upload,
  Users,
  AlertTriangle,
  Crosshair,
  ShieldCheck,
  ShieldAlert,
  Film,
  CheckCircle2,
  FileVideo,
  Camera,
  Activity,
} from "lucide-react";
import {
  detectCrowdImage,
  detectCrowdVideo,
  CrowdDetectionResponse,
  CrowdVideoResponse,
} from "@/lib/api";

export default function SurveillanceView() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Results state
  const [imageResult, setImageResult] = useState<CrowdDetectionResponse | null>(null);
  const [videoResult, setVideoResult] = useState<CrowdVideoResponse | null>(null);
  const [currentFrameUrl, setCurrentFrameUrl] = useState<string | null>(null);
  const [animIndex, setAnimIndex] = useState<number>(0);
  const animIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setImageResult(null);
    setVideoResult(null);
    setCurrentFrameUrl(null);
    if (animIntervalRef.current) clearInterval(animIntervalRef.current);

    const isVideo = file.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);

    try {
      if (isVideo) {
        const res = await detectCrowdVideo(file);
        setVideoResult(res);
        if (res.sample_frames && res.sample_frames.length > 0) {
          setCurrentFrameUrl(res.sample_frames[0].image);
          // Start auto-playback
          let idx = 0;
          animIntervalRef.current = setInterval(() => {
            idx = (idx + 1) % res.sample_frames.length;
            setAnimIndex(idx);
            setCurrentFrameUrl(res.sample_frames[idx].image);
          }, 1000);
        }
      } else {
        const res = await detectCrowdImage(file);
        setImageResult(res);
        setCurrentFrameUrl(res.annotated_image);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to analyze media");
    } finally {
      setLoading(false);
    }
  };

  // Sample quick tests
  const loadSample = async (sampleType: "crowd" | "cctv") => {
    try {
      setLoading(true);
      setError(null);
      const url =
        sampleType === "crowd"
          ? "/static/samples/authentic.jpg"
          : "/static/samples/surveillance_sample.mp4";
      const resp = await fetch(url);
      if (!resp.ok) throw new Error("Could not load sample file");
      const blob = await resp.blob();
      const file = new File(
        [blob],
        sampleType === "crowd" ? "crowd_sample.jpg" : "surveillance_clip.mp4",
        { type: sampleType === "crowd" ? "image/jpeg" : "video/mp4" }
      );
      await handleFileUpload(file);
    } catch (err: any) {
      setError(err.message || "Sample loading error");
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (animIntervalRef.current) clearInterval(animIntervalRef.current);
    };
  }, []);

  // Compute displayed values
  const crowdCount = videoResult ? videoResult.peak_crowd : imageResult ? imageResult.crowd_count : 0;
  const riskLevel = videoResult
    ? videoResult.overall_risk.level
    : imageResult
    ? imageResult.risk.level
    : "SAFE";
  const riskScore = videoResult
    ? videoResult.overall_risk.score
    : imageResult
    ? imageResult.risk.score
    : 0;
  const weaponsDetected = videoResult
    ? videoResult.weapons_detected
    : imageResult
    ? imageResult.weapon_labels
    : [];
  const hasWeapons = weaponsDetected.length > 0;
  const feedIntegrity = videoResult
    ? Math.round(videoResult.feed_integrity.score * 100)
    : imageResult
    ? Math.round(imageResult.feed_integrity_score * 100)
    : 100;
  const isTampered = videoResult
    ? videoResult.feed_integrity.is_compromised
    : imageResult
    ? imageResult.feed_tampered
    : false;

  const getRiskBadgeColor = (level: string) => {
    switch (level) {
      case "CRITICAL":
      case "DANGER":
        return "text-rose-400 bg-rose-500/10 border-rose-500/30";
      case "ALERT":
        return "text-amber-400 bg-amber-500/10 border-amber-500/30";
      case "ELEVATED":
        return "text-sky-400 bg-sky-500/10 border-sky-500/30";
      default:
        return "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
    }
  };

  return (
    <div className="space-y-6">
      {/* 4 Primary Telemetry Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: Crowd Count */}
        <div className="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Active Crowd Count</span>
            <Users className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-3xl font-extrabold text-white tracking-tight">{crowdCount}</div>
          <div className="text-xs text-slate-500 mt-1">
            {videoResult
              ? `Peak crowd across ${videoResult.duration_seconds}s video`
              : imageResult
              ? `${imageResult.active_tracks} unique tracks identified`
              : "Awaiting video or camera feed"}
          </div>
        </div>

        {/* Metric 2: Threat Level */}
        <div className="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Security Threat Level</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`text-2xl font-black px-3 py-0.5 rounded-lg border ${getRiskBadgeColor(
                riskLevel
              )}`}
            >
              {riskLevel}
            </span>
          </div>
          <div className="text-xs text-slate-500 mt-1">Unified Threat Score: {riskScore} / 100</div>
        </div>

        {/* Metric 3: Weapon Localization */}
        <div className="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Weapon Localization</span>
            <Crosshair className="w-4 h-4 text-rose-400" />
          </div>
          <div
            className={`text-2xl font-bold tracking-tight truncate ${
              hasWeapons ? "text-rose-400" : "text-white"
            }`}
          >
            {hasWeapons ? weaponsDetected.join(", ").toUpperCase() : "NONE DETECTED"}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {hasWeapons
              ? `${weaponsDetected.length} active armed threat localized`
              : "No firearms or edged weapons identified"}
          </div>
        </div>

        {/* Metric 4: Feed Integrity */}
        <div className="glass-panel p-5 rounded-2xl relative overflow-hidden group hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Feed Integrity Monitor</span>
            {isTampered ? (
              <ShieldAlert className="w-4 h-4 text-rose-400 animate-pulse" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            )}
          </div>
          <div
            className={`text-3xl font-extrabold tracking-tight ${
              isTampered ? "text-rose-400" : "text-emerald-400"
            }`}
          >
            {feedIntegrity}%
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {isTampered
              ? "CRITICAL: Synthetic frames / deepfake injection"
              : "Continuous 1-Sec Sensor Optics Verified"}
          </div>
        </div>
      </div>

      {/* Main Workspace Layout: Controls Sidebar + Live Viewport */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Input & Controls (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="glass-panel p-5 rounded-2xl space-y-5">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Camera className="w-4 h-4 text-sky-400" /> Camera Feed Source
            </h2>

            {/* Drag & Drop File Zone */}
            <label className="border-2 border-dashed border-slate-700/80 hover:border-sky-500/60 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition bg-slate-950/40 group text-center block">
              <input
                type="file"
                accept="video/*,image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
              />
              <Upload className="w-8 h-8 text-slate-500 group-hover:text-sky-400 mb-2 transition" />
              <span className="text-sm font-semibold text-slate-200">
                Upload Surveillance Video or Frame
              </span>
              <span className="text-xs text-slate-500 mt-1">MP4, AVI, MOV, JPG, PNG (up to 60MB)</span>
            </label>

            {/* Quick Test Footage Presets */}
            <div>
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                Quick Test Footage
              </span>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => loadSample("crowd")}
                  disabled={loading}
                  className="px-3 py-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-xs font-medium text-slate-200 flex items-center justify-center gap-1.5 transition border border-slate-700/50"
                >
                  <Users className="w-3.5 h-3.5 text-sky-400" />
                  <span>Crowd Photo</span>
                </button>
                <button
                  type="button"
                  onClick={() => loadSample("cctv")}
                  disabled={loading}
                  className="px-3 py-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-xs font-medium text-slate-200 flex items-center justify-center gap-1.5 transition border border-slate-700/50"
                >
                  <FileVideo className="w-3.5 h-3.5 text-amber-400" />
                  <span>CCTV Video</span>
                </button>
              </div>
            </div>

            {/* Technical Info Badge */}
            <div className="border-t border-slate-800 pt-4">
              <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3 text-[11px] text-slate-400 space-y-1.5 font-mono">
                <div className="flex justify-between">
                  <span>Inference:</span>
                  <span className="text-sky-400">CUDA GPU (YOLOv8)</span>
                </div>
                <div className="flex justify-between">
                  <span>Tracker:</span>
                  <span className="text-slate-200">ByteTrack v2</span>
                </div>
                <div className="flex justify-between">
                  <span>Feed Check:</span>
                  <span className="text-emerald-400">Active</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Live Viewport & Timeline (8 Cols) */}
        <div className="lg:col-span-8 space-y-4">
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                  Tactical Surveillance Viewport
                </h2>
                {videoResult && (
                  <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono">
                    Frame {animIndex + 1} of {videoResult.sample_frames.length}
                  </span>
                )}
              </div>
              {loading && (
                <div className="flex items-center gap-2 text-xs text-sky-400 animate-pulse font-mono">
                  <Activity className="w-3.5 h-3.5 animate-spin" />
                  <span>Processing Video Frames...</span>
                </div>
              )}
            </div>

            {/* Viewport Canvas/Image Box */}
            <div className="relative aspect-video bg-slate-950 rounded-xl border border-slate-800/80 overflow-hidden flex items-center justify-center shadow-inner">
              {currentFrameUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={currentFrameUrl}
                  alt="Surveillance Output"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-center p-8 text-slate-500 space-y-2">
                  <Camera className="w-12 h-12 mx-auto stroke-1 text-slate-600 mb-2" />
                  <p className="text-sm font-medium text-slate-400">No Video Feed Loaded</p>
                  <p className="text-xs text-slate-600 max-w-sm">
                    Upload a surveillance clip or click &ldquo;CCTV Video&rdquo; to begin real-time multi-person tracking and feed integrity monitoring.
                  </p>
                </div>
              )}

              {/* Feed Integrity HUD Floating Badge */}
              {currentFrameUrl && (
                <div className="absolute top-3 right-3 flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-md border shadow-lg bg-slate-950/80">
                  {isTampered ? (
                    <>
                      <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                      <span className="text-rose-400">FEED COMPROMISED ({feedIntegrity}%)</span>
                    </>
                  ) : (
                    <>
                      <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400">FEED VERIFIED ({feedIntegrity}%)</span>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-300 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Sample Keyframe Timeline Gallery */}
          {videoResult && videoResult.sample_frames.length > 0 && (
            <div className="glass-panel p-5 rounded-2xl space-y-3">
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                <Film className="w-4 h-4 text-slate-400" /> Temporal Frame Sequence
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
                {videoResult.sample_frames.map((sf, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      if (animIntervalRef.current) clearInterval(animIntervalRef.current);
                      setCurrentFrameUrl(sf.image);
                      setAnimIndex(i);
                    }}
                    className={`rounded-lg overflow-hidden border transition text-left group bg-slate-950 ${
                      animIndex === i
                        ? "border-sky-500 ring-2 ring-sky-500/30"
                        : "border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={sf.image}
                      alt={`Frame ${sf.timestamp}`}
                      className="w-full h-16 object-cover"
                    />
                    <div className="p-1.5 text-[10px] text-slate-400 flex justify-between">
                      <span>{sf.timestamp}</span>
                      <span className="text-sky-400 font-mono">P: {sf.crowd_count}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
