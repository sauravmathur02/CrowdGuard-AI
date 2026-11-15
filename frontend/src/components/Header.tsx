"use client";

import React from "react";
import { Shield, Video, Fingerprint, Scale, Cpu, Activity } from "lucide-react";
import { HealthResponse } from "@/lib/api";

interface HeaderProps {
  activeTab: "surveillance" | "forensics" | "investigation";
  setActiveTab: (tab: "surveillance" | "forensics" | "investigation") => void;
  health: HealthResponse | null;
}

export default function Header({ activeTab, setActiveTab, health }: HeaderProps) {
  const isCuda = health?.hardware?.cuda_available;
  const gpuName = health?.hardware?.gpu_name || "CPU Mode";
  const vramFree = health?.hardware?.vram?.free_gb ?? 0;
  const vramTotal = health?.hardware?.vram?.total_gb ?? 0;

  return (
    <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-xl sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo & Project Identity */}
          <div className="flex items-center space-x-3">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-600 via-indigo-600 to-sky-500 p-0.5 shadow-lg shadow-blue-500/20">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Shield className="w-6 h-6 text-blue-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-1.5">
                  CrowdGuard <span className="text-blue-400">AI</span>
                </h1>
                <span className="text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  v2.0 Unified
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Physical Crowd Anomaly, Weapon Localization & Digital Deepfake Defense
              </p>
            </div>
          </div>

          {/* Hardware & API Status Bar */}
          <div className="hidden md:flex items-center space-x-3 text-xs">
            {health ? (
              <div className="flex items-center gap-3 bg-slate-950/70 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-300">
                <div className="flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-sky-400" />
                  <span className="font-mono text-slate-200">{gpuName}</span>
                </div>
                {isCuda && vramTotal > 0 && (
                  <div className="flex items-center gap-1.5 border-l border-slate-800 pl-3">
                    <span className="text-slate-500">VRAM:</span>
                    <span className="font-mono text-emerald-400">
                      {vramFree}GB / {vramTotal}GB free
                    </span>
                  </div>
                )}
                <div className="flex items-center gap-1.5 border-l border-slate-800 pl-3">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-emerald-400 font-medium">Gateway Active</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 bg-slate-950/70 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-400">
                <Activity className="w-3.5 h-3.5 text-amber-400 animate-spin" />
                <span>Connecting to Engine...</span>
              </div>
            )}
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex justify-center -mb-px pt-1 pb-3">
          <nav className="inline-flex p-1.5 rounded-full bg-slate-900/90 border border-slate-800 shadow-xl shadow-black/40 gap-1">
            <button
              onClick={() => setActiveTab("surveillance")}
              className={`flex items-center gap-2 px-5 py-2 rounded-full text-xs sm:text-sm font-semibold transition-all ${
                activeTab === "surveillance"
                  ? "bg-gradient-to-r from-sky-500 to-blue-600 text-white shadow-md shadow-sky-500/25"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50"
              }`}
            >
              <Video className="w-4 h-4" />
              <span>Crowd Surveillance</span>
            </button>

            <button
              onClick={() => setActiveTab("forensics")}
              className={`flex items-center gap-2 px-5 py-2 rounded-full text-xs sm:text-sm font-semibold transition-all ${
                activeTab === "forensics"
                  ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-md shadow-purple-500/25"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50"
              }`}
            >
              <Fingerprint className="w-4 h-4" />
              <span>Deepfake Forensics</span>
            </button>

            <button
              onClick={() => setActiveTab("investigation")}
              className={`flex items-center gap-2 px-5 py-2 rounded-full text-xs sm:text-sm font-semibold transition-all ${
                activeTab === "investigation"
                  ? "bg-gradient-to-r from-amber-500 to-orange-600 text-white shadow-md shadow-amber-500/25"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50"
              }`}
            >
              <Scale className="w-4 h-4" />
              <span>Dual Investigation</span>
            </button>


          </nav>
        </div>
      </div>
    </header>
  );
}
