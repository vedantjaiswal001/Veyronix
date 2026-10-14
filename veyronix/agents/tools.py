"""Read-only investigation tools (spec §26/§28).

Every tool queries real telemetry (metrics/logs/traces/deployments/business) or
runs a real algorithm (anomaly profiling, similarity retrieval). Tools NEVER see
incident ground truth and NEVER mutate state — the AI layer can only read. The
same registry backs both the in-process orchestrator and the MCP server.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from veyronix.core.topology import downstream_dependencies, upstream_dependents
from veyronix.rootcause.detect import profile_services


@dataclass
class ToolContext:
    """Observable data for one run. Deliberately excludes incident labels."""

    run_id: str
    metrics: pd.DataFrame
    logs: pd.DataFrame = field(default_factory=pd.DataFrame)
    traces: pd.DataFrame = field(default_factory=pd.DataFrame)
    deployments: pd.DataFrame = field(default_factory=pd.DataFrame)
    business: pd.DataFrame = field(default_factory=pd.DataFrame)
    memory: Any = None  # IncidentMemory | None

    def window(self, df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
        if df is None or df.empty or "minute" not in df.columns:
            return df if df is not None else pd.DataFrame()
        return df[(df.minute >= start) & (df.minute <= end)]


@dataclass
class ToolResult:
    ok: bool
    data: dict
    summary: str
    latency_ms: float


ToolFn = Callable[..., dict]
TOOL_REGISTRY: dict[str, dict] = {}


def tool(name: str, description: str, params: dict):
    def deco(fn: ToolFn):
        TOOL_REGISTRY[name] = {"fn": fn, "description": description, "params": params}
        return fn
    return deco


def call_tool(name: str, ctx: ToolContext, **args) -> ToolResult:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return ToolResult(False, {"error": f"unknown tool {name}"}, "unknown tool", 0.0)
    t0 = time.perf_counter()
    try:
        data = spec["fn"](ctx, **args)
        ok = "error" not in data
        summary = data.get("_summary", "")
        return ToolResult(ok, data, summary, (time.perf_counter() - t0) * 1000)
    except Exception as exc:  # tools must fail safely (spec §38)
        return ToolResult(False, {"error": str(exc)}, f"tool error: {exc}",
                          (time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool("query_metrics", "Summarize a service's metrics over a time window.",
      {"service": "str", "start_min": "int", "end_min": "int"})
def query_metrics(ctx, service: str, start_min: int, end_min: int) -> dict:
    g = ctx.metrics[ctx.metrics.service == service]
    win = ctx.window(g, start_min, end_min)
    base = ctx.window(g, start_min - 45, start_min - 12)
    if win.empty:
        return {"error": "no data", "_summary": f"no metrics for {service}"}
    channels = ["cpu_pct", "memory_pct", "latency_p95_ms", "error_rate",
                "db_latency_ms", "queue_depth", "request_rate", "network_latency_ms"]
    stats = {}
    for c in channels:
        if c in win.columns:
            stats[c] = {"peak": round(float(win[c].max()), 3),
                        "mean": round(float(win[c].mean()), 3),
                        "baseline": round(float(base[c].mean()), 3) if not base.empty else None}
    hot = max(stats.items(), key=lambda kv: (kv[1]["peak"] / (abs(kv[1]["baseline"] or 1) + 1e-6)))
    return {"service": service, "stats": stats,
            "_summary": f"{service}: {hot[0]} peaked at {hot[1]['peak']}"}


@tool("query_logs", "Top log templates by level for a service/window.",
      {"service": "str|None", "start_min": "int", "end_min": "int", "level": "str|None"})
def query_logs(ctx, start_min: int, end_min: int, service: str | None = None,
               level: str | None = None) -> dict:
    if ctx.logs is None or ctx.logs.empty:
        return {"templates": [], "_summary": "no logs"}
    w = ctx.window(ctx.logs, start_min, end_min)
    if service:
        w = w[w.service == service]
    if level:
        w = w[w.level == level]
    if w.empty:
        return {"templates": [], "_summary": "no matching logs"}
    top = (w.groupby(["level", "template"])["count"].sum()
           .sort_values(ascending=False).head(6))
    templates = [{"level": lvl, "template": tpl, "count": int(c)}
                 for (lvl, tpl), c in top.items()]
    errs = [t for t in templates if t["level"] == "ERROR"]
    return {"templates": templates,
            "_summary": f"top errors: {', '.join(t['template'] for t in errs[:3]) or 'none'}"}


@tool("query_traces", "Distributed-trace error rate and slowest services.",
      {"start_min": "int", "end_min": "int"})
def query_traces(ctx, start_min: int, end_min: int) -> dict:
    if ctx.traces is None or ctx.traces.empty:
        return {"error_rate": None, "_summary": "no traces"}
    w = ctx.window(ctx.traces, start_min, end_min)
    if w.empty:
        return {"error_rate": None, "_summary": "no traces in window"}
    err_rate = round(float(w.error.mean()), 4)
    slow = round(float(w.duration_ms.quantile(0.95)), 2)
    return {"trace_error_rate": err_rate, "p95_duration_ms": slow, "n": int(len(w)),
            "_summary": f"trace error rate {err_rate}, p95 {slow}ms"}


@tool("get_deployments", "Deployments within a time window.",
      {"start_min": "int", "end_min": "int"})
def get_deployments(ctx, start_min: int, end_min: int) -> dict:
    if ctx.deployments is None or ctx.deployments.empty:
        return {"deployments": [], "_summary": "no deployments"}
    w = ctx.window(ctx.deployments, start_min - 20, end_min)
    deps = [{"service": r["service"], "version": r["version"], "minute": int(r["minute"]),
             "change": r["change_summary"]} for _, r in w.iterrows()]
    return {"deployments": deps,
            "_summary": f"{len(deps)} deployment(s) near window"}


@tool("get_service_dependencies", "Topology up/downstream of a service.",
      {"service": "str"})
def get_service_dependencies(ctx, service: str) -> dict:
    return {"service": service,
            "depends_on": list(downstream_dependencies(service)),
            "depended_on_by": list(upstream_dependents(service)),
            "_summary": f"{service} depends on {len(downstream_dependencies(service))} services"}


@tool("get_anomalies", "Detected per-service anomalies in a window (real detector).",
      {"start_min": "int", "end_min": "int"})
def get_anomalies(ctx, start_min: int, end_min: int) -> dict:
    prof = profile_services(ctx.metrics, start_min, end_min)
    anomalies = [{"service": a.service, "channel": a.dominant_channel,
                  "magnitude": a.magnitude, "onset_min": a.onset_min,
                  "implied_fault": a.implied_fault} for a in prof.values()]
    anomalies.sort(key=lambda x: (x["onset_min"] or 1e9, -x["magnitude"]))
    return {"anomalies": anomalies,
            "_summary": f"{len(anomalies)} anomalous service(s)"}


@tool("get_business_metrics", "Business KPI change over a window.",
      {"start_min": "int", "end_min": "int"})
def get_business_metrics(ctx, start_min: int, end_min: int) -> dict:
    if ctx.business is None or ctx.business.empty:
        return {"error": "no business metrics", "_summary": "no business data"}
    w = ctx.window(ctx.business, start_min, end_min)
    base = ctx.window(ctx.business, start_min - 45, start_min - 12)
    if w.empty or base.empty:
        return {"error": "no data", "_summary": "no business data in window"}
    out = {}
    for c in ["conversion_rate", "orders_per_min", "checkout_error_rate", "revenue_per_min"]:
        if c in w.columns:
            out[c] = {"window": round(float(w[c].mean()), 3),
                      "baseline": round(float(base[c].mean()), 3)}
    rev_drop = 1 - (out.get("revenue_per_min", {}).get("window", 0)
                    / (out.get("revenue_per_min", {}).get("baseline", 1) + 1e-9))
    return {"kpis": out, "revenue_drop_pct": round(rev_drop * 100, 1),
            "_summary": f"revenue down {round(rev_drop*100,1)}%"}


@tool("search_similar_incidents", "Retrieve similar past incidents (RAG memory).",
      {"signature": "str", "top_k": "int"})
def search_similar_incidents(ctx, signature: str, top_k: int = 3) -> dict:
    if ctx.memory is None:
        return {"matches": [], "_summary": "no incident memory"}
    matches = ctx.memory.search(signature, top_k=top_k)
    return {"matches": matches,
            "_summary": f"{len(matches)} similar past incident(s)"}


def tool_specs() -> list[dict]:
    """Machine-readable tool specs (for LLM tool-calling / MCP)."""
    return [{"name": n, "description": s["description"], "parameters": s["params"]}
            for n, s in TOOL_REGISTRY.items()]
