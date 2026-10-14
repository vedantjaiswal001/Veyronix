"""LLM provider abstraction (spec §3/§27).

Three providers behind one interface:
  * offline  — a deterministic reasoner (no API key). It does NOT hit a network;
    it synthesizes a grounded narrative from the evidence the tools returned.
  * anthropic / openai — real models, used ONLY to narrate/synthesize over the
    evidence and the deterministic root-cause ranking. They are constrained to
    the candidate hypotheses the RCA engine produced, so the LLM can explain and
    prioritise but can never invent a cause or a number (spec §3).

The whole system runs end-to-end on ``offline`` with zero secrets; setting a key
upgrades synthesis quality without changing what is measured.
"""

from __future__ import annotations

import json

from veyronix.core.config import get_settings
from veyronix.core.logging import get_logger

log = get_logger("agents.llm")


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        s = get_settings()
        self.provider = provider or s.llm_provider
        self.model = model or s.llm_model
        self._settings = s
        self.tokens_used = 0

    def available(self) -> bool:
        """True if a live provider is configured with a key."""
        if self.provider == "anthropic":
            return bool(self._settings.anthropic_api_key)
        if self.provider == "openai":
            return bool(self._settings.openai_api_key)
        return False

    def synthesize(self, context: dict) -> dict:
        """Return {summary, findings:[{rank, explanation}]} grounded in evidence.

        Falls back to the deterministic reasoner on any provider error, so an
        investigation never fails because of the LLM.
        """
        if not self.available():
            return self._offline_synthesis(context)
        try:
            if self.provider == "anthropic":
                return self._anthropic_synthesis(context)
            if self.provider == "openai":
                return self._openai_synthesis(context)
        except Exception as exc:  # pragma: no cover - needs a live key
            log.warning("llm.fallback_offline", provider=self.provider, error=str(exc))
        return self._offline_synthesis(context)

    # ---- offline deterministic reasoner ----
    def _offline_synthesis(self, context: dict) -> dict:
        findings = context.get("hypotheses", [])
        ev = context.get("evidence", [])
        biz = context.get("business", {})
        top = findings[0] if findings else None
        lines = []
        if top:
            lines.append(
                f"Most likely root cause: {top['cause_service']} "
                f"({top['cause_type'].replace('_', ' ')}), confidence "
                f"{top['confidence']:.0%}.")
        anomaly_ev = [e for e in ev if e["kind"] in ("metric", "database", "trace")]
        if anomaly_ev:
            lines.append("Supporting signals: " + "; ".join(
                e["observation"] for e in anomaly_ev[:4]) + ".")
        deploy_ev = [e for e in ev if e["kind"] == "deployment"]
        if deploy_ev:
            lines.append(deploy_ev[0]["observation"] + ".")
        if biz.get("revenue_drop_pct", 0) > 1:
            lines.append(f"Business impact: revenue down ~{biz['revenue_drop_pct']}%.")
        per = [{"rank": f["rank"], "explanation": f.get("explanation", "")}
               for f in findings]
        return {"summary": " ".join(lines) or "No clear anomaly detected.",
                "findings": per}

    # ---- live providers (synthesis only; constrained to given hypotheses) ----
    def _prompt(self, context: dict) -> tuple[str, str]:
        system = (
            "You are VEYRONIX's incident synthesis agent. You are given verified "
            "evidence gathered by specialist tools and a ranked list of candidate "
            "root causes produced by a deterministic engine. Explain the most "
            "likely root cause and rank the candidates. You MUST choose only from "
            "the provided candidate cause_service values and MUST NOT invent "
            "metrics, services, or numbers. Respond as strict JSON with keys "
            "'summary' (string) and 'findings' (list of {rank:int, explanation:str}).")
        user = json.dumps(context, default=str)[:12000]
        return system, user

    def _anthropic_synthesis(self, context: dict) -> dict:  # pragma: no cover
        import anthropic

        system, user = self._prompt(context)
        client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        resp = client.messages.create(
            model=self.model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": user}])
        self.tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        return self._parse(text, context)

    def _openai_synthesis(self, context: dict) -> dict:  # pragma: no cover
        from openai import OpenAI

        system, user = self._prompt(context)
        client = OpenAI(api_key=self._settings.openai_api_key)
        resp = client.chat.completions.create(
            model=self.model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        if resp.usage:
            self.tokens_used += resp.usage.total_tokens
        return self._parse(resp.choices[0].message.content, context)

    def _parse(self, text: str, context: dict) -> dict:  # pragma: no cover
        try:
            data = json.loads(text)
            return {"summary": str(data.get("summary", ""))[:1000],
                    "findings": list(data.get("findings", []))}
        except Exception:
            return self._offline_synthesis(context)
