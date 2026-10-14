import { useEffect, useState } from "react";
import { api, money, type IncidentCard, type Overview } from "./lib/api";
import { Button, Card, Meter, SeverityBadge, SEV, StatCard, Tag } from "./components/ui";
import { HealthTimeline } from "./components/Charts";
import { DependencyGraph } from "./components/DependencyGraph";
import { IncidentDetail } from "./components/IncidentDetail";

export default function App() {
  const [ov, setOv] = useState<Overview | null>(null);
  const [incidents, setIncidents] = useState<IncidentCard[]>([]);
  const [pred, setPred] = useState<{ minute: number; probability: number }[]>([]);
  const [edges, setEdges] = useState<{ source: string; target: string }[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.overview(), api.incidents(), api.predictions(), api.topology()])
      .then(([o, inc, p, topo]) => {
        setOv(o); setIncidents(inc.incidents);
        setPred(p.available ? p.points ?? [] : []);
        setEdges(topo.edges);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <Shell><Card className="text-sm">Could not reach the VEYRONIX API — start it with <code>make run</code>. <div className="text-slate-500 mt-1">{err}</div></Card></Shell>;
  if (!ov) return <Shell><div className="text-slate-400">Loading operations data…</div></Shell>;

  const prob = ov.latest_incident_probability;
  const worst = incidents[0];

  return (
    <Shell>
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-4">
        <StatCard label="System status"
          value={<span style={{ color: prob && prob >= 0.5 ? SEV.HIGH.dot : SEV.NORMAL.dot }}>
            {prob && prob >= 0.5 ? "AT RISK" : "HEALTHY"}</span>}
          sub={`${ov.services} services monitored`} />
        <StatCard label="Live incident prob." accent="#93c5fd"
          value={prob != null ? `${(prob * 100).toFixed(1)}%` : "NA"} sub="next 30 min" />
        <StatCard label="Predicted incidents" accent="#fde68a" value={ov.predicted_incidents}
          sub="prob ≥ 50%" />
        <StatCard label="Incidents (run)" value={ov.total_incidents}
          sub={Object.entries(ov.incidents_by_severity).map(([k, v]) => `${v} ${k[0]}`).join(" · ")} />
        <StatCard label="Model" accent={ov.model_loaded ? SEV.NORMAL.dot : "#fca5a5"}
          value={ov.model_loaded ? "ACTIVE" : "NOT LOADED"} sub={ov.model_loaded ? "RandomForest" : "run make train"} />
        <StatCard label="Environment" value="SIMULATED" sub={ov.run_id} />
      </div>

      <div className="grid md:grid-cols-3 gap-4 mb-4">
        <Card className="md:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[11px] uppercase tracking-wider text-slate-400">System health timeline · incident probability</div>
            <Tag>PREDICTION</Tag>
          </div>
          {pred.length > 0 ? <HealthTimeline data={pred} />
            : <div className="text-sm text-slate-400 py-8 text-center">Predictions unavailable — run <code>make train</code> to enable the model.</div>}
        </Card>
        <Card>
          <div className="text-[11px] uppercase tracking-wider text-slate-400 mb-2">Service dependency graph</div>
          <DependencyGraph edges={edges} affected={worst?.affected_services ?? []} rootCause={worst?.root_cause_service} />
        </Card>
      </div>

      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-slate-300">Incidents (most severe first)</h2>
        <Tag color="#93c5fd">{incidents.length} total</Tag>
      </div>
      <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
        {incidents.map((c) => <IncidentTile key={c.id} c={c} onOpen={() => setSelected(c.id)} />)}
      </div>

      {selected != null && <IncidentDetail id={selected} edges={edges} onClose={() => setSelected(null)} />}
    </Shell>
  );
}

function IncidentTile({ c, onOpen }: { c: IncidentCard; onOpen: () => void }) {
  return (
    <Card className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <SeverityBadge sev={c.severity} />
        {c.predicted_probability != null && (
          <span className="text-xs text-slate-400">p={(c.predicted_probability * 100).toFixed(0)}%
            {c.eta_minutes != null && <> · ETA {c.eta_minutes}m</>}</span>)}
      </div>
      <div className="font-semibold leading-snug">{c.title}</div>
      <div className="text-xs text-slate-400">
        Likely cause: <b className="text-slate-200">{c.root_cause_service ?? "—"}</b>
        {c.root_cause_confidence != null && <> · {(c.root_cause_confidence * 100).toFixed(0)}% conf</>}
      </div>
      {c.predicted_probability != null && <Meter value={c.predicted_probability}
        color={c.predicted_probability >= 0.8 ? SEV.CRITICAL.dot : SEV.MEDIUM.dot} />}
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">Est. impact</span>
        <span style={{ color: "#fca5a5" }}>{money(c.business_impact?.estimated_revenue_loss)}</span>
      </div>
      <div className="flex items-center justify-between mt-1">
        <Tag>rec: {c.recommended_action.replace(/_/g, " ")}</Tag>
        <Button variant="primary" onClick={onOpen}>Open</Button>
      </div>
    </Card>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-20 px-5 py-3 flex items-center justify-between"
        style={{ background: "rgba(11,16,32,.8)", borderBottom: "1px solid #26324d", backdropFilter: "blur(8px)" }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg grid place-items-center font-black"
            style={{ background: "linear-gradient(135deg,#3b82f6,#8b5cf6)" }}>V</div>
          <div>
            <div className="font-extrabold tracking-tight">VEYRONIX</div>
            <div className="text-[10px] text-slate-400 -mt-0.5">AI OPERATIONS CENTER</div>
          </div>
        </div>
        <span className="pill px-2.5 py-1" style={{ background: "rgba(52,211,153,.12)", color: "#6ee7b7" }}>
          ● SIMULATED ENVIRONMENT
        </span>
      </header>
      <main className="max-w-[1400px] mx-auto px-5 py-5">{children}</main>
      <footer className="text-center text-xs text-slate-600 py-6">
        VEYRONIX · every figure is real system output over a simulated environment · not financial truth
      </footer>
    </div>
  );
}
