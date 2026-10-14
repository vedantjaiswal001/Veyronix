"""Incident memory / RAG (spec §29).

After each resolved incident we store its observable *signature* (affected
services + anomalous channels), its confirmed root cause, and its resolution.
New investigations retrieve similar past incidents by TF-IDF cosine similarity —
organizational memory, with no external embedding API required. Retrieval uses
only the observable signature, so it is not a channel for label leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from veyronix.core.config import ARTIFACT_DIR

MEMORY_PATH = ARTIFACT_DIR / "incident_memory.json"


@dataclass
class MemoryRecord:
    ref: str
    signature: str
    root_cause: str
    resolution: str
    meta: dict = field(default_factory=dict)


class IncidentMemory:
    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []
        self._vec: TfidfVectorizer | None = None
        self._matrix = None

    def add(self, ref: str, signature: str, root_cause: str, resolution: str,
            meta: dict | None = None) -> None:
        self.records.append(MemoryRecord(ref, signature, root_cause, resolution, meta or {}))
        self._vec = None  # invalidate index

    def _reindex(self) -> None:
        if not self.records:
            return
        self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = self._vec.fit_transform([r.signature for r in self.records])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.records:
            return []
        if self._vec is None:
            self._reindex()
        qv = self._vec.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = np.argsort(sims)[::-1][:top_k]
        out = []
        for i in order:
            if sims[i] <= 0:
                continue
            r = self.records[i]
            out.append({"incident_ref": r.ref, "similarity": round(float(sims[i]), 3),
                        "root_cause": r.root_cause, "resolution": r.resolution})
        return out

    def save(self, path: Path | None = None) -> None:
        p = path or MEMORY_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([r.__dict__ for r in self.records], indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> IncidentMemory:
        p = path or MEMORY_PATH
        mem = cls()
        if p.exists():
            for d in json.loads(p.read_text()):
                mem.add(d["ref"], d["signature"], d["root_cause"],
                        d["resolution"], d.get("meta"))
        return mem


def signature_from_anomalies(anomalies: list[dict], business_drop: float = 0.0) -> str:
    """Build a retrieval signature string from observable anomalies."""
    parts = [f"{a['service']}:{a['channel']}:{a['implied_fault']}" for a in anomalies]
    if business_drop > 5:
        parts.append("business_revenue_impact")
    return " ".join(parts) if parts else "no_anomaly"
