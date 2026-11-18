"use client";

import React from "react";
import { Film, ImageIcon } from "lucide-react";

/** Shows the uploaded image or video so the operator can see what was analysed. */
export default function MediaPreview({
  url,
  isVideo,
  name,
  title = "Analysed Media",
  maxHeight = "max-h-80",
}: {
  url: string | null;
  isVideo: boolean;
  name?: string;
  title?: string;
  maxHeight?: string;
}) {
  if (!url) return null;
  const Icon = isVideo ? Film : ImageIcon;
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-slate-400" />
          <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">{title}</h3>
        </div>
        {name && <span className="text-xs text-slate-500 truncate max-w-[60%]">{name}</span>}
      </div>
      <div className="flex justify-center bg-slate-950 rounded-xl border border-slate-800 overflow-hidden">
        {isVideo ? (
          <video src={url} controls playsInline className={`w-full ${maxHeight} object-contain`} />
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url} alt={name || "uploaded media"} className={`${maxHeight} object-contain`} />
        )}
      </div>
    </div>
  );
}
