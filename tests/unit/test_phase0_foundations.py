"""Phase 0 foundations: config, constants, topology, schema."""

from __future__ import annotations

import pytest

from veyronix.core.config import Settings, get_settings
from veyronix.core.constants import (
    HELD_OUT_FAULTS,
    KNOWN_FAULTS,
    ActionType,
    FaultType,
    RiskTier,
    ServiceName,
    Severity,
)
from veyronix.core.topology import (
    dependency_centrality,
    downstream_dependencies,
    service_graph,
    upstream_dependents,
)

pytestmark = pytest.mark.unit


# ---- config ----
def test_settings_fractions_and_test_frac():
    s = get_settings()
    assert 0 < s.train_frac < 1 and 0 < s.val_frac < 1
    assert s.test_frac == pytest.approx(1 - s.train_frac - s.val_frac)
    assert s.test_frac > 0


def test_settings_rejects_bad_fraction():
    with pytest.raises(ValueError):
        Settings(train_frac=1.5)


def test_default_offline_llm_needs_no_key():
    s = get_settings()
    assert s.llm_provider == "offline"
    assert s.allow_auto_remediation is False  # safe default


# ---- constants ----
def test_thirteen_fault_types():
    assert len(list(FaultType)) == 13


def test_known_and_heldout_partition_is_clean():
    known = set(KNOWN_FAULTS)
    held = set(HELD_OUT_FAULTS)
    assert known.isdisjoint(held)
    # cascading is the held-out unknown-failure fault (spec §19)
    assert FaultType.CASCADING_FAILURE in held
    assert FaultType.CASCADING_FAILURE not in known


def test_severity_ordering():
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    assert [x.rank for x in order] == [0, 1, 2, 3]
    assert sorted(order, key=lambda s: s.rank) == order


def test_delete_database_action_exists_for_blocking():
    # It must exist so the policy layer can prove it is BLOCKED (spec §33).
    assert ActionType.DELETE_DATABASE in list(ActionType)
    assert RiskTier.BLOCKED in list(RiskTier)


# ---- topology ----
def test_graph_has_all_services_and_edges():
    g = service_graph()
    assert g.number_of_nodes() == len(list(ServiceName))
    assert g.number_of_edges() >= 10
    # database is a leaf dependency (nothing it depends on within app tier)
    assert downstream_dependencies("database") == ()


def test_database_has_highest_blast_radius():
    cent = dependency_centrality()
    # DB is depended on by the most services -> highest centrality.
    assert cent["database"] == max(cent.values())
    assert cent["client"] == 0.0  # nothing depends on the client


def test_blast_radius_of_database_covers_business_path():
    deps = set(upstream_dependents("database"))
    assert {"order_service", "payment_service", "api_gateway"}.issubset(deps)


# ---- schema ----
def test_schema_builds_all_tables(db):
    from veyronix.db.base import Base

    names = set(Base.metadata.tables)
    expected = {
        "users", "services", "metrics", "logs", "traces", "deployments",
        "incidents", "incident_events", "predictions", "anomalies",
        "root_cause_hypotheses", "causal_edges", "actions", "action_outcomes",
        "models", "model_versions", "agent_runs", "agent_tool_calls",
        "evaluation_runs",
    }
    assert expected.issubset(names)


def test_can_write_and_read_a_service(db):
    from veyronix.db.models import Service
    from veyronix.db.session import session_scope

    with session_scope() as s:
        s.add(Service(name="probe_svc", business_critical=True, centrality=0.5))
    from veyronix.db.session import session_scope as scope2

    with scope2() as s:
        got = s.query(Service).filter_by(name="probe_svc").one()
        assert got.business_critical is True
