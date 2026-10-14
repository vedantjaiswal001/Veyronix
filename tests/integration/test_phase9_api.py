"""Phase 9 (backend): the API serves real system data and the remediation flow."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from veyronix.api.app import app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_and_metrics(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["label"] == "SIMULATED"
    assert "veyronix_" in client.get("/metrics").text


def test_topology_and_overview(client):
    topo = client.get("/api/topology").json()
    assert len(topo["nodes"]) == 8 and len(topo["edges"]) >= 10
    ov = client.get("/api/overview").json()
    assert ov["total_incidents"] > 0
    assert "SIMULATED" in ov["label"]


def test_incident_cards_have_expected_fields(client):
    cards = client.get("/api/incidents").json()["incidents"]
    assert cards
    c = cards[0]
    for key in ("id", "fault_type", "severity", "business_impact",
                "recommended_action", "affected_services"):
        assert key in c


def test_incident_detail_and_timeline(client):
    fid = client.get("/api/incidents").json()["incidents"][0]["id"]
    detail = client.get(f"/api/incidents/{fid}").json()
    assert detail["ranked_root_causes"]
    tl = client.get(f"/api/incidents/{fid}/timeline").json()
    kinds = [e["kind"] for e in tl["events"]]
    assert "incident_start" in kinds and "recovered" in kinds


def test_investigate_and_recommend(client):
    fid = client.get("/api/incidents").json()["incidents"][0]["id"]
    rep = client.post(f"/api/incidents/{fid}/investigate").json()
    assert rep["top_finding"] is not None
    assert rep["hallucination_flags"] == []
    recs = client.get(f"/api/incidents/{fid}/recommendations").json()["recommendations"]
    assert any(r["action"] == "no_op" for r in recs)  # do-nothing considered


def test_timeseries(client):
    ts = client.get("/api/timeseries", params={"service": "database",
                                               "channel": "db_latency_ms"}).json()
    assert ts["service"] == "database" and len(ts["points"]) > 100


def test_remediation_flow_via_api(client, db):
    sub = client.post("/api/actions/submit", json={
        "action_type": "rollback_deployment", "params": {"service": "order_service"},
        "reason": "bad deploy"}).json()
    assert sub["decision"] == "requires_approval"
    aid = sub["action_id"]
    # operator cannot approve a medium-risk action
    denied = client.post(f"/api/actions/{aid}/approve",
                         json={"name": "alice", "role": "operator"})
    assert denied.status_code == 403
    ok = client.post(f"/api/actions/{aid}/approve", json={"name": "bob", "role": "approver"})
    assert ok.status_code == 200
    ex = client.post(f"/api/actions/{aid}/execute", json={"name": "bob", "role": "operator"})
    assert ex.status_code == 200
