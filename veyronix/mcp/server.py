"""MCP server exposing VEYRONIX investigation tools (spec §28).

Cleanly separated from the core: it wraps the same read-only tool registry the
in-process orchestrator uses, so any MCP client (Claude Desktop, etc.) can query
metrics, logs, traces, deployments, the service graph, anomalies, business
impact, similar incidents — and request a full deterministic investigation.

Requires the optional ``mcp`` extra (``pip install -e '.[ai]'``). The tool layer
itself has no MCP dependency, so the rest of the system runs without it.
"""

from __future__ import annotations

import json

from veyronix.core.logging import get_logger

log = get_logger("mcp.server")

# Extra MCP-only tools layered on top of the shared registry.
_EXTRA_TOOLS = {
    "get_service_graph": "Return the full service dependency graph + centrality.",
    "investigate_incident": "Run a full deterministic root-cause investigation "
                            "for a time window and return the report.",
}


def _handle(name: str, arguments: dict) -> str:
    """Dispatch a tool call to the shared registry or an MCP-only tool."""
    from veyronix.agents.context import default_context
    from veyronix.agents.tools import TOOL_REGISTRY, call_tool
    from veyronix.core.topology import topology_summary

    ctx = default_context()
    if name == "get_service_graph":
        return json.dumps(topology_summary(), default=str)
    if name == "investigate_incident":
        from veyronix.agents.orchestrator import Investigator

        rep = Investigator().investigate(
            ctx, int(arguments["start_min"]), int(arguments["end_min"]))
        return rep.model_dump_json()
    if name in TOOL_REGISTRY:
        res = call_tool(name, ctx, **arguments)
        return json.dumps({"ok": res.ok, "data": res.data, "summary": res.summary},
                          default=str)
    return json.dumps({"error": f"unknown tool {name}"})


def build_server():  # pragma: no cover - requires the optional mcp extra
    """Construct the MCP server object (imports mcp lazily)."""
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    from veyronix.agents.tools import tool_specs

    server = Server("veyronix")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = [Tool(name=s["name"], description=s["description"],
                      inputSchema={"type": "object",
                                   "properties": {k: {"type": "string"}
                                                  for k in s["parameters"]}})
                 for s in tool_specs()]
        for name, desc in _EXTRA_TOOLS.items():
            tools.append(Tool(name=name, description=desc,
                              inputSchema={"type": "object",
                                           "properties": {"start_min": {"type": "integer"},
                                                          "end_min": {"type": "integer"}}}))
        return tools

    @server.call_tool()
    async def call_tool_handler(name: str, arguments: dict) -> list[TextContent]:
        return [TextContent(type="text", text=_handle(name, arguments or {}))]

    return server


def main() -> None:  # pragma: no cover - requires the optional mcp extra
    import anyio
    from mcp.server.stdio import stdio_server

    async def _run():
        server = build_server()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    log.info("mcp.server.start", tools=len(_EXTRA_TOOLS) + 9)
    anyio.run(_run)


def list_exposed_tools() -> list[str]:
    """Names of all tools this server exposes (testable without the mcp extra)."""
    from veyronix.agents.tools import TOOL_REGISTRY

    return sorted([*TOOL_REGISTRY.keys(), *_EXTRA_TOOLS.keys()])


if __name__ == "__main__":
    main()
