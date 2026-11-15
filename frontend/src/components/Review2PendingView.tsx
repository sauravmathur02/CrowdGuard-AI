"use client";

import React from "react";
import {
  Clock,
  Lock,
  Layers,
  FileText,
  Activity,
  HardDrive,
  Cpu,
  CheckCircle2,
  AlertCircle,
  Database,
  Crosshair,
  ArrowRight,
} from "lucide-react";

export default function Review2PendingView() {
  const pendingModules = [
    {
      id: "R2-01",
      title: "Custom Weapon Weight Integration",
      scope: "Deployment of fine-tuned YOLOv8 + MobileNetV3 weights for specialized weapon localization (Knives, Pistols, Long Guns).",
      deliverable: "pretrained_weights/best.pt & classifier.pth",
      status: "PENDING GUIDE REVIEW 2",
      notes: "System currently runs stock YOLOv8n CUDA tracker on person class with automatic hot-swap listener ready for weights transfer from secondary workstation.",
      icon: Crosshair,
      tag: "Computer Vision",
    },
    {
      id: "R2-02",
      title: "Optical Flow Turbulence Vector Maps",
      scope: "Farneback dense optical flow vector field visualization and crowd stampede turbulence kinetic velocity calculation.",
      deliverable: "crowd/turbulence_engine.py & Stream Canvas HUD",
      status: "PENDING GUIDE REVIEW 2",
      notes: "Mathematical formulas and pipeline hooks defined; visualization renderer queued for Phase 2 guide presentation.",
      icon: Activity,
      tag: "Fluid Dynamics / Crowd AI",
    },
    {
      id: "R2-03",
      title: "Automated PDF Forensic Audit Export",
      scope: "Production-grade cryptographic PDF incident dossier generator with forensic hashes, timestamp watermark, and GradCAM attachments.",
      deliverable: "services/forensic_pdf_builder.py",
      status: "PENDING GUIDE REVIEW 2",
      notes: "JSON structured evidence export is fully operational in Phase 1; PDF templating engine scheduled for Review 2.",
      icon: FileText,
      tag: "Forensics & Reporting",
    },
    {
      id: "R2-04",
      title: "Multi-Camera Edge Node Mesh & Persistent Log",
      scope: "Distributed RTSP multi-channel ingestion with persistent SQLite/PostgreSQL incident database and edge mesh heartbeat.",
      deliverable: "storage/incident_mesh_db.sqlite",
      status: "PENDING GUIDE REVIEW 2",
      notes: "Single-channel live CCTV feed and batch processing completed; multi-node clustering planned for Review 2 final defense.",
      icon: Database,
      tag: "Distributed Systems",
    },
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Banner */}
      <div className="bg-gradient-to-r from-amber-500/10 via-slate-900/60 to-slate-900/60 border border-amber-500/30 rounded-2xl p-6 backdrop-blur-md shadow-[0_0_30px_rgba(245,158,11,0.1)]">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-amber-500/20 text-amber-300 border border-amber-500/30 shrink-0">
              <Clock className="w-8 h-8" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 uppercase tracking-wider">
                  Review 2 Pending State
                </span>
                <span className="text-xs text-slate-400">
                  College Project Phase 2 Defense Checklist
                </span>
              </div>
              <h2 className="text-xl font-bold text-white mt-1">
                Phase 2 Modules & Scheduled Deliverables
              </h2>
              <p className="text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
                As per the two-phase project schedule, all core foundational systems (Dual-Pipeline Incident Engine, CUDA YOLOv8 crowd tracking, multi-backbone deepfake forensics, and Next.js operations console) are 100% complete and fully operational. The items below are reserved for the subsequent college guide Review 2 evaluation.
              </p>
            </div>
          </div>

          <div className="text-right shrink-0">
            <div className="text-xs text-slate-400 uppercase font-mono">
              Review 1 Status
            </div>
            <div className="text-emerald-400 font-extrabold text-lg flex items-center gap-1.5 justify-end">
              <CheckCircle2 className="w-5 h-5" />
              100% READY
            </div>
            <div className="text-xs text-amber-400 font-mono mt-0.5">
              Review 2: 4 Items Queued
            </div>
          </div>
        </div>
      </div>

      {/* Grid of Pending Modules */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {pendingModules.map((mod) => {
          const Icon = mod.icon;
          return (
            <div
              key={mod.id}
              className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 relative overflow-hidden flex flex-col justify-between"
            >
              <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/5 rounded-full blur-2xl pointer-events-none" />

              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-slate-800 text-amber-400 border border-slate-700">
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-mono font-semibold text-slate-400">
                      {mod.id}
                    </span>
                  </div>
                  <span className="px-2.5 py-0.5 rounded text-[11px] font-bold bg-amber-500/10 text-amber-300 border border-amber-500/30">
                    {mod.status}
                  </span>
                </div>

                <h3 className="text-base font-bold text-white mb-1.5">
                  {mod.title}
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed mb-4">
                  {mod.scope}
                </p>

                <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80 mb-4">
                  <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-1">
                    Scheduled Target Deliverable
                  </div>
                  <div className="text-xs font-mono font-semibold text-cyan-300 truncate">
                    {mod.deliverable}
                  </div>
                </div>
              </div>

              <div>
                <div className="p-3 rounded-xl bg-slate-950/40 border border-slate-800 text-xs text-slate-400 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <span>{mod.notes}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Guide Review Instructions */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
        <h4 className="text-sm font-bold text-white uppercase tracking-wider mb-2 flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          Submission & Milestone Roadmap
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-emerald-500/20">
            <div className="font-semibold text-emerald-400 mb-1 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" />
              Phase 1 (Current Submission - 100% Complete)
            </div>
            <p className="text-slate-400 leading-relaxed">
              Dual-pipeline architecture, live real-time webcam/CCTV inference with YOLOv8 person tracking, multi-model deepfake forensics (ViT, ResNet, CLIP, GenD), cross-correlated incident investigation engine, and full responsive Next.js operations console.
            </p>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/60 border border-amber-500/20">
            <div className="font-semibold text-amber-400 mb-1 flex items-center gap-1.5">
              <Clock className="w-4 h-4" />
              Phase 2 (Guide Review 2 - Pending)
            </div>
            <p className="text-slate-400 leading-relaxed">
              Model weight hot-swap from secondary workstation (<code className="text-cyan-300">best.pt</code> / <code className="text-cyan-300">classifier.pth</code>), optical flow turbulence density maps, exportable forensic PDF dossiers, and multi-camera cluster database synchronization.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
