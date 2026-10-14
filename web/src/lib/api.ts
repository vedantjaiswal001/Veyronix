const BASE = (import.meta as any).env?.VITE_API_BASE ?? "";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
async function post<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

export type Overview = {
  run_id: string; label: string; services: number; total_incidents: number;
  incidents_by_severity: Record<string, number>; predicted_incidents: number;
  model_loaded: boolean; latest_incident_probability: number | null;
};

export type IncidentCard = {
  id: number; fault_type: string; title: string; severity: string;
  start_min: number; end_min: number; start_time: string;
  predicted_probability: number | null; eta_minutes: number | null;
  root_cause_service: string | null; root_cause_type: string | null;
  root_cause_confidence: number | null; affected_services: string[];
  business_impact: Record<string, any>; recommended_action: string;
  recommendation_reason: string;
};

export type RootCause = {
  rank: number; cause_service: string; cause_type: string; confidence: number;
  explanation: string; evidence_chain: { type: string; service: string; detail: string; minute?: number }[];
};

export type Investigation = {
  provider: string; status: string; summary: string; recommended_action: string;
  top_finding: RootCause | null; ranked_findings: RootCause[];
  evidence: { agent: string; kind: string; service?: string; observation: string }[];
  similar_incidents: { incident_ref: string; similarity: number; root_cause: string }[];
  steps: number; tool_calls: number; tools_failed: number; duration_ms: number;
  hallucination_flags: string[];
};

export type Recommendation = {
  action: string; expected_benefit: number; net_score: number;
  recovery_probability: number; expected_downtime_min: number;
  residual_revenue_loss: number; cost: number; risk: string; confidence: number; reason: string;
};

export const api = {
  overview: () => get<Overview>("/api/overview"),
  topology: () => get<{ nodes: any[]; edges: any[] }>("/api/topology"),
  incidents: () => get<{ incidents: IncidentCard[] }>("/api/incidents"),
  incident: (id: number) => get<IncidentCard & { ranked_root_causes: RootCause[] }>(`/api/incidents/${id}`),
  timeline: (id: number) => get<{ events: { minute: number; kind: string; detail: string }[] }>(`/api/incidents/${id}/timeline`),
  investigate: (id: number) => post<Investigation>(`/api/incidents/${id}/investigate`),
  recommendations: (id: number) => get<{ recommendations: Recommendation[]; note: string }>(`/api/incidents/${id}/recommendations`),
  timeseries: (service: string, channel: string) =>
    get<{ points: { minute: number; value: number }[] }>(`/api/timeseries?service=${service}&channel=${channel}`),
  predictions: () => get<{ available: boolean; points?: { minute: number; probability: number }[] }>("/api/predictions"),
  business: () => get<{ points: { minute: number; revenue_per_min: number; checkout_error_rate: number }[] }>("/api/business"),
  modelHealth: () => get<any>("/api/model-health"),
  submitAction: (b: unknown) => post<any>("/api/actions/submit", b),
  approve: (id: number, b: unknown) => post<any>(`/api/actions/${id}/approve`, b),
  execute: (id: number, b: unknown) => post<any>(`/api/actions/${id}/execute`, b),
};

export const money = (n: number | undefined) =>
  n == null ? "—" : "₹" + Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(n);
