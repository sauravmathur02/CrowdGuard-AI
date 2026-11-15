"use client";

import React, { useState, useEffect } from "react";
import Header from "@/components/Header";
import SurveillanceView from "@/components/SurveillanceView";
import ForensicsView from "@/components/ForensicsView";
import InvestigationView from "@/components/InvestigationView";
import { fetchHealth, HealthResponse } from "@/lib/api";
import { Shield, CheckCircle2 } from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"surveillance" | "forensics" | "investigation">("surveillance");
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    const poll = async () => {
      try {
        const data = await fetchHealth();
        if (isMounted) setHealth(data);
      } catch (e) {
        console.warn("Backend poll warning:", e);
      }
    };

    poll();
    const interval = setInterval(poll, 8000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === "surveillance" && <SurveillanceView />}
        {activeTab === "forensics" && <ForensicsView />}
        {activeTab === "investigation" && <InvestigationView />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 mt-12 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-blue-500" />
            <span className="font-semibold text-slate-400">CrowdGuard AI</span>
            <span>—</span>
            <span>Unified Crowd Surveillance &amp; Deepfake Media Forensics</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" />
              FastAPI + CUDA Engine Online
            </span>
            <span className="text-slate-700">|</span>
            <span className="text-slate-400 font-mono">Next.js 14 App Router</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
