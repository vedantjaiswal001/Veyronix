"""Structured outputs for the AI investigator (spec §26/§27).

Pydantic models keep the agent's output typed and validated — the LLM proposes,
but its output must conform to these schemas or it is rejected. No free-form
conclusions leak through.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    agent: str
    tool: str
    args: dict = Field(default_factory=dict)
    ok: bool = True
    latency_ms: float = 0.0
    result_summary: str = ""


class Evidence(BaseModel):
    agent: str                     # which specialist produced it
    kind: str                      # metric | log | trace | deployment | database | memory
    service: str | None = None
    observation: str
    supports: str | None = None    # which hypothesis it supports
    strength: float = 0.5          # 0..1


class RootCauseFinding(BaseModel):
    rank: int
    cause_service: str
    cause_type: str
    confidence: float
    explanation: str
    evidence_chain: list[dict] = Field(default_factory=list)


class SimilarIncident(BaseModel):
    incident_ref: str
    similarity: float
    root_cause: str
    resolution: str


class InvestigationReport(BaseModel):
    incident_id: int | None = None
    run_id: str
    provider: str
    top_finding: RootCauseFinding | None = None
    ranked_findings: list[RootCauseFinding] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    similar_incidents: list[SimilarIncident] = Field(default_factory=list)
    recommended_action: str = "no_op"
    summary: str = ""
    # telemetry / guardrail accounting
    steps: int = 0
    tool_calls: int = 0
    tools_failed: int = 0
    tokens_used: int = 0
    duration_ms: float = 0.0
    status: str = "done"           # done | timeout | budget_exceeded | failed
    hallucination_flags: list[str] = Field(default_factory=list)
