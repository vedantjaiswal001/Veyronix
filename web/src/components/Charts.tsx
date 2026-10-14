import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

const AX = { stroke: "#5b6b8c", fontSize: 11 };
const grid = <CartesianGrid stroke="#1e2842" strokeDasharray="3 3" />;
const tip = { contentStyle: { background: "#0f1629", border: "1px solid #26324d", borderRadius: 8, fontSize: 12 } };

export function HealthTimeline({ data }: { data: { minute: number; probability: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="p" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.6} />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        {grid}
        <XAxis dataKey="minute" {...AX} tickFormatter={(m) => `${Math.floor(m / 60)}h`} />
        <YAxis domain={[0, 1]} {...AX} />
        <Tooltip {...tip} />
        <ReferenceLine y={0.5} stroke="#f5c542" strokeDasharray="4 4" label={{ value: "WARN", fill: "#f5c542", fontSize: 10 }} />
        <ReferenceLine y={0.8} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "CRIT", fill: "#ef4444", fontSize: 10 }} />
        <Area type="monotone" dataKey="probability" stroke="#3b82f6" fill="url(#p)" strokeWidth={2} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function MetricChart({ data, color = "#f97316", label }:
  { data: { minute: number; value: number }[]; color?: string; label: string }) {
  return (
    <div>
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          {grid}
          <XAxis dataKey="minute" {...AX} tickFormatter={(m) => `${Math.floor(m / 60)}h`} />
          <YAxis {...AX} width={44} />
          <Tooltip {...tip} />
          <Line type="monotone" dataKey="value" stroke={color} dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
