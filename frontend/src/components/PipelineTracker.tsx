"use client";

import React from "react";
import { CheckCircle2, Loader2, Circle, XCircle } from "lucide-react";

export type StepStatus = "pending" | "running" | "completed" | "failed";

export interface PipelineStep {
  title: string;
  subtitle: string;
  status: StepStatus;
  /** Short live text: current sub-step while running, result summary when done. */
  detail?: string;
  icon?: React.ElementType;
}

/** Horizontal step tracker showing which stage of an analysis pipeline is running. */
export default function PipelineTracker({ steps, elapsed }: { steps: PipelineStep[]; elapsed?: number }) {
  const done = steps.filter((s) => s.status === "completed").length;
  const pct = Math.round((done / steps.length) * 100);
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Analysis Pipeline</h3>
        <span className="text-xs text-slate-500 font-mono">
          {done}/{steps.length} stages{elapsed !== undefined ? ` · ${elapsed.toFixed(0)}s` : ""}
        </span>
      </div>
      <div className="w-full h-1 rounded-full bg-slate-800 overflow-hidden mb-5">
        <div className="h-full bg-cyan-500 transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` }}>
        {steps.map((s, i) => {
          const Icon = s.icon;
          const tone =
            s.status === "completed" ? "border-emerald-500/40 bg-emerald-500/5"
            : s.status === "running" ? "border-cyan-500/60 bg-cyan-500/10 shadow-[0_0_20px_rgba(6,182,212,0.15)]"
            : s.status === "failed" ? "border-rose-500/50 bg-rose-500/10"
            : "border-slate-800 bg-slate-950/40 opacity-60";
          return (
            <div key={i} className={`relative rounded-xl border p-3 transition-all ${tone}`}>
              <div className="flex items-center gap-2">
                {s.status === "completed" && <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />}
                {s.status === "running" && <Loader2 className="w-4 h-4 text-cyan-300 animate-spin shrink-0" />}
                {s.status === "failed" && <XCircle className="w-4 h-4 text-rose-400 shrink-0" />}
                {s.status === "pending" && <Circle className="w-4 h-4 text-slate-600 shrink-0" />}
                <span className="text-[11px] font-mono text-slate-500">STAGE {i + 1}</span>
                {Icon && <Icon className="w-3.5 h-3.5 text-slate-500 ml-auto" />}
              </div>
              <div className="mt-1.5 text-sm font-semibold text-white">{s.title}</div>
              <div className="text-[11px] text-slate-500">{s.subtitle}</div>
              {s.detail && (
                <div className={`mt-2 text-[11px] leading-snug ${s.status === "running" ? "text-cyan-200" : "text-slate-300"}`}>
                  {s.detail}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
