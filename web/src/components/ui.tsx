import type { ReactNode } from "react";

export const SEV: Record<string, { bg: string; fg: string; dot: string; label: string }> = {
  CRITICAL: { bg: "rgba(239,68,68,.15)", fg: "#fca5a5", dot: "#ef4444", label: "CRITICAL" },
  HIGH: { bg: "rgba(249,115,22,.15)", fg: "#fdba74", dot: "#f97316", label: "HIGH" },
  MEDIUM: { bg: "rgba(245,197,66,.15)", fg: "#fde68a", dot: "#f5c542", label: "MEDIUM" },
  LOW: { bg: "rgba(96,165,250,.15)", fg: "#93c5fd", dot: "#60a5fa", label: "LOW" },
  NORMAL: { bg: "rgba(52,211,153,.15)", fg: "#6ee7b7", dot: "#34d399", label: "NORMAL" },
};

export function SeverityBadge({ sev }: { sev: string }) {
  const s = SEV[sev] ?? SEV.LOW;
  return (
    <span className="pill inline-flex items-center gap-1.5 px-2.5 py-1"
      style={{ background: s.bg, color: s.fg }}>
      <span style={{ width: 7, height: 7, borderRadius: 999, background: s.dot }} />
      {s.label}
    </span>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`card p-4 ${className}`}>{children}</div>;
}

export function StatCard({ label, value, sub, accent }:
  { label: string; value: ReactNode; sub?: string; accent?: string }) {
  return (
    <Card className="flex flex-col gap-1">
      <div className="text-[11px] uppercase tracking-wider text-slate-400">{label}</div>
      <div className="text-2xl font-extrabold" style={{ color: accent ?? "#e6ecf7" }}>{value}</div>
      {sub && <div className="text-xs text-slate-400">{sub}</div>}
    </Card>
  );
}

export function Meter({ value, color = "#3b82f6" }: { value: number; color?: string }) {
  return (
    <div style={{ height: 8, background: "#1a2338", borderRadius: 999, overflow: "hidden" }}>
      <div style={{ width: `${Math.min(100, Math.max(0, value * 100))}%`, height: "100%",
        background: color, transition: "width .4s ease" }} />
    </div>
  );
}

export function Button({ children, onClick, variant = "ghost", disabled }:
  { children: ReactNode; onClick?: () => void; variant?: "primary" | "ghost"; disabled?: boolean }) {
  return (
    <button className={`btn ${variant === "primary" ? "btn-primary" : "btn-ghost"}`}
      onClick={onClick} disabled={disabled} style={{ opacity: disabled ? 0.5 : 1 }}>
      {children}
    </button>
  );
}

export function Tag({ children, color = "#93c5fd" }: { children: ReactNode; color?: string }) {
  return <span className="pill px-2 py-0.5" style={{ background: "rgba(255,255,255,.05)", color }}>{children}</span>;
}
