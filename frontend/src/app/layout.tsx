import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CrowdGuard AI — Physical & Digital Threat Intelligence Platform",
  description: "Next-Generation Crowd Anomaly, Weapon Detection, and Deepfake Media Forensics Suite",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 antialiased selection:bg-blue-600 selection:text-white">
        <div className="fixed inset-0 pointer-events-none bg-glow-blue z-0" />
        <div className="fixed inset-0 pointer-events-none bg-glow-purple z-0" />
        <div className="relative z-10 min-h-screen flex flex-col">
          {children}
        </div>
      </body>
    </html>
  );
}
