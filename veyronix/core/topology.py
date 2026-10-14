"""Service dependency graph and derived topology metrics.

A single directed graph is the backbone shared by the simulator (fault
propagation), the root-cause engine (blame propagation walks the reverse of
dependency edges), the severity model (dependency centrality), and the UI
(dependency visualization).
"""

from __future__ import annotations

from functools import lru_cache

import networkx as nx

from veyronix.core.constants import (
    BUSINESS_CRITICAL,
    SERVICE_DEPENDENCIES,
    TELEMETRY_SERVICES,
    ServiceName,
)


@lru_cache
def service_graph() -> nx.DiGraph:
    """Directed graph where edge (a, b) means 'a depends on b'.

    Faults therefore propagate along reversed edges: a problem in ``b`` impacts
    every service that (transitively) depends on it.
    """
    g = nx.DiGraph()
    for s in ServiceName:
        g.add_node(
            s.value,
            business_critical=s in BUSINESS_CRITICAL,
            emits_telemetry=s in TELEMETRY_SERVICES,
        )
    for src, dst in SERVICE_DEPENDENCIES:
        g.add_edge(src.value, dst.value)
    return g


@lru_cache
def dependency_centrality() -> dict[str, float]:
    """Blast-radius score per service in [0, 1].

    A service's blast radius = fraction of all services that transitively
    depend on it (i.e. would be affected if it fails). Computed on the reversed
    graph's descendants.
    """
    g = service_graph()
    rev = g.reverse(copy=True)
    n = g.number_of_nodes()
    out: dict[str, float] = {}
    for node in g.nodes:
        affected = nx.descendants(rev, node)
        out[node] = round(len(affected) / max(n - 1, 1), 4)
    return out


@lru_cache
def upstream_dependents(service: str) -> tuple[str, ...]:
    """Services that (transitively) depend on ``service`` — its blast radius."""
    rev = service_graph().reverse(copy=True)
    return tuple(sorted(nx.descendants(rev, service)))


@lru_cache
def downstream_dependencies(service: str) -> tuple[str, ...]:
    """Services ``service`` (transitively) depends on — candidate causes."""
    return tuple(sorted(nx.descendants(service_graph(), service)))


def topology_summary() -> dict:
    """Serializable topology snapshot for the API / UI."""
    g = service_graph()
    cent = dependency_centrality()
    return {
        "nodes": [
            {
                "id": n,
                "business_critical": bool(d.get("business_critical")),
                "emits_telemetry": bool(d.get("emits_telemetry")),
                "centrality": cent[n],
            }
            for n, d in g.nodes(data=True)
        ],
        "edges": [{"source": u, "target": v} for u, v in g.edges],
    }
