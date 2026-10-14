import { useEffect, useState } from "react";
import {
  api, money, type IncidentCard, type Investigation, type Recommendation, type RootCause,
} from "../lib/api";
import { Button, Card, Meter, SeverityBadge, SEV, Tag } from "./ui";
import { DependencyGraph } from "./DependencyGraph";
import { MetricChart } from "./Charts";

type Detail = IncidentCard & { ranked_root_causes: RootCause[] };

export function IncidentDetail({ id, edges, onClose }:
  { id: number; edges: { source: string; target: string }[]; onClose: () => void }) {
  const [d, setD] = useState<Detail | null>(null);
  const [timeline, setTimeline] = useState<{ minute: number; kind: string; detail: string }[]>([]);
  const [dbSeries, setDbSeries] = useState<{ minute: number; value: number }[]>([]);
  const [inv, setInv] = useState<Investigation | null>(null);
  const [recs, setRecs] = useState<Recommendation[] | null>(null);
  const [action, setAction] = useState<any>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    setD(null); setInv(null); setRecs(null); setAction(null);
    api.incident(id).then(setD);
    api.timeline(id).then((t) => setTimeline(t.events));
    api.timeseries("database", "db_latency_ms").then((t) => setDbSeries(t.points));
  }, [id]);

  const runInvestigate = async () => { setBusy("inv"); setInv(await api.investigate(id)); setBusy(null); };
  const runSimulate = async () => { setBusy("sim"); setRecs((await api.recommendations(id)).recommendations); setBusy(null); };
  const runApprove = async () => {
    if (!d) return;
    setBusy("act");
    const sub = await api.submitAction({
      action_type: d.recommended_action, params: { service: d.root_cause_service ?? "order_service" },
      incident_id: d.id, reason: "operator approved from console",
    });
    let result = sub;
    if (sub.decision !== "blocked") {
      await api.approve(sub.action_id, { name: "operator", role: "approver" });
      result = await api.execute(sub.action_id, { name: "operator", role: "operator" });
    }
    setAction({ ...sub, ...result });
    setBusy(null);
  };

  if (!d) return null;
  const focusMin = dbSeries.filter((p) => p.minute >= d.start_min - 40 && p.minute <= d.end_min + 15);

  return (
    <div className="fixed inset-0 z-40 flex justify-end" style={{ background: "rgba(3,6,15,.6)" }} onClick={onClose}>
      <div className="w-full max-w-3xl h-full overflow-y-auto scroll-thin p-5"
        style={{ background: "#0b1020", borderLeft: "1px solid #26324d" }} onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="flex items-center gap-3">
              <SeverityBadge sev={d.severity} />
              <Tag>{d.fault_type.replace(/_/g, " ")}</Tag>
            </div>
            <h2 className="text-xl font-bold mt-2">{d.title}</h2>
            <div className="text-xs text-slate-400 mt-1">Incident #{d.id} · affected: {d.affected_services.join(", ")}</div>
          </div>
          <Button onClick={onClose}>✕</Button>
        </div>

        <div className="grid grid-cols-3 gap-3 mb-4">
          <Stat label="Predicted probability"
            value={d.predicted_probability != null ? `${(d.predicted_probability * 100).toFixed(1)}%` : "NA"} />
          <Stat label="Warning time" value={d.eta_minutes != null ? `${d.eta_minutes} min` : "NA"} />
          <Stat label="Est. revenue loss" value={money(d.business_impact?.estimated_revenue_loss)} accent="#fca5a5" />
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <Card>
            <SectionTitle>Root-cause chain</SectionTitle>
            <DependencyGraph edges={edges} affected={d.affected_services} rootCause={d.root_cause_service} />
            <div className="mt-2 text-sm">
              Likely cause: <b>{d.root_cause_service}</b>{" "}
              <Tag>{d.root_cause_type?.replace(/_/g, " ")}</Tag>{" "}
              <span className="text-slate-400">confidence {(Number(d.root_cause_confidence) * 100).toFixed(0)}%</span>
            </div>
          </Card>
          <Card>
            <SectionTitle>Incident timeline</SectionTitle>
            <ol className="relative border-l border-slate-700 ml-2 mt-1">
              {timeline.map((e, i) => (
                <li key={i} className="ml-4 mb-2">
                  <span className="absolute -left-[5px] w-2.5 h-2.5 rounded-full"
                    style={{ background: e.kind === "recovered" ? SEV.NORMAL.dot : e.kind.includes("incident") ? SEV.CRITICAL.dot : "#3b82f6" }} />
                  <div className="text-xs text-slate-400">t+{e.minute}m</div>
                  <div className="text-sm">{e.detail}</div>
                </li>
              ))}
            </ol>
          </Card>
        </div>

        <Card className="mt-4">
          <SectionTitle>Database query latency (cascade signal)</SectionTitle>
          <MetricChart data={focusMin} label="database · db_latency_ms" color="#f97316" />
        </Card>

        <div className="flex gap-2 mt-4">
          <Button variant="primary" onClick={runInvestigate} disabled={busy === "inv"}>
            {busy === "inv" ? "Investigating…" : "🔍 Investigate"}
          </Button>
          <Button onClick={runSimulate} disabled={busy === "sim"}>
            {busy === "sim" ? "Simulating…" : "🧪 Simulate actions"}
          </Button>
          <Button onClick={runApprove} disabled={busy === "act"}>
            {busy === "act" ? "Executing…" : `✅ Approve: ${d.recommended_action.replace(/_/g, " ")}`}
          </Button>
        </div>

        {inv && <InvestigationView inv={inv} />}
        {recs && <RecommendationView recs={recs} />}
        {action && <ActionResult action={action} />}
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] uppercase tracking-wider text-slate-400 mb-2">{children}</div>;
}
function Stat({ label, value, accent }: { label: string; value: React.ReactNode; accent?: string }) {
  return (
    <Card><div className="text-[11px] text-slate-400">{label}</div>
      <div className="text-lg font-bold" style={{ color: accent }}>{value}</div></Card>
  );
}

function InvestigationView({ inv }: { inv: Investigation }) {
  return (
    <Card className="mt-4">
      <SectionTitle>AI investigation ({inv.provider})</SectionTitle>
      <p className="text-sm mb-2">{inv.summary}</p>
      <div className="flex flex-wrap gap-2 mb-3 text-xs">
        <Tag>{inv.tool_calls} tool calls</Tag>
        <Tag>{inv.tools_failed} failed</Tag>
        <Tag>{inv.duration_ms.toFixed(0)} ms</Tag>
        <Tag color={inv.hallucination_flags.length ? "#fca5a5" : "#6ee7b7"}>
          hallucinations: {inv.hallucination_flags.length}
        </Tag>
      </div>
      <div className="text-xs text-slate-400 mb-1">Evidence gathered</div>
      <ul className="text-sm space-y-1">
        {inv.evidence.slice(0, 6).map((e, i) => (
          <li key={i}><Tag>{e.agent}</Tag> <span className="text-slate-300">{e.observation}</span></li>
        ))}
      </ul>
      {inv.similar_incidents.length > 0 && (
        <div className="mt-3 text-xs text-slate-400">
          Similar past incidents: {inv.similar_incidents.map((s) => `${s.incident_ref} (${(s.similarity * 100).toFixed(0)}%)`).join(", ")}
        </div>
      )}
    </Card>
  );
}

function RecommendationView({ recs }: { recs: Recommendation[] }) {
  return (
    <Card className="mt-4">
      <SectionTitle>Counterfactual recommendations (what-if estimates)</SectionTitle>
      <table className="w-full text-sm">
        <thead><tr className="text-slate-400 text-xs text-left">
          <th className="py-1">Action</th><th>Recovery</th><th>Downtime</th><th>Benefit</th><th>Risk</th></tr></thead>
        <tbody>
          {recs.map((r, i) => (
            <tr key={i} className="border-t border-slate-800">
              <td className="py-1.5 font-medium">{r.action.replace(/_/g, " ")}</td>
              <td><div className="w-16"><Meter value={r.recovery_probability} /></div></td>
              <td>{r.expected_downtime_min} min</td>
              <td style={{ color: "#6ee7b7" }}>{money(r.expected_benefit)}</td>
              <td><Tag color={r.risk === "MEDIUM" ? "#fde68a" : "#93c5fd"}>{r.risk}</Tag></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function ActionResult({ action }: { action: any }) {
  const ok = action.success;
  return (
    <Card className="mt-4" >
      <SectionTitle>Remediation outcome</SectionTitle>
      <div className="text-sm">
        Decision: <Tag>{action.decision}</Tag>{" "}
        {action.status && <>Status: <Tag color={ok ? "#6ee7b7" : "#fca5a5"}>{action.status}</Tag></>}
      </div>
      {action.actual && <div className="text-xs text-slate-400 mt-1">{action.actual.note} · recovery ~{action.actual.recovery_min} min</div>}
      {action.decision === "blocked" && <div className="text-xs mt-1" style={{ color: "#fca5a5" }}>Action is permanently blocked by policy.</div>}
    </Card>
  );
}
