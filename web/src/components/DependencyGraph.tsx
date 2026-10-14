import { SEV } from "./ui";

// Fixed layout for the 8-node topology; colored by involvement in an incident.
const POS: Record<string, [number, number]> = {
  client: [40, 40], api_gateway: [40, 120], user_service: [190, 60],
  order_service: [190, 130], payment_service: [190, 200],
  database: [360, 130], cache: [360, 55], message_queue: [360, 205],
};
const IDLE = "#3a4a6b";
const EDGE = "#2a3550";

export function DependencyGraph({ edges, affected = [], rootCause }:
  { edges: { source: string; target: string }[]; affected?: string[]; rootCause?: string | null }) {
  const nodes = Object.keys(POS);
  return (
    <svg viewBox="0 0 460 260" className="w-full" style={{ maxHeight: 260 }}>
      {edges.map((e, i) => {
        const a = POS[e.source], b = POS[e.target];
        if (!a || !b) return null;
        const hot = affected.includes(e.source) && affected.includes(e.target);
        return <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}
          stroke={hot ? SEV.HIGH.dot : EDGE} strokeWidth={hot ? 2 : 1}
          opacity={hot ? 0.9 : 0.5} />;
      })}
      {nodes.map((n) => {
        const [x, y] = POS[n];
        const c = n === rootCause ? SEV.CRITICAL.dot : affected.includes(n) ? SEV.HIGH.dot : IDLE;
        return (
          <g key={n}>
            <circle cx={x} cy={y} r={n === rootCause ? 11 : 8} fill={c}
              stroke={n === rootCause ? "#fff" : "none"} strokeWidth={1.5} />
            <text x={x + 14} y={y + 4} fontSize={10} fill="#c7d2e8">{n.replace("_", " ")}</text>
          </g>
        );
      })}
    </svg>
  );
}
