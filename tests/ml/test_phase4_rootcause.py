"""Phase 4: root-cause detection + ranking (symptom vs. cause discrimination)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from veyronix.core.constants import FaultType, ServiceName
from veyronix.rootcause.detect import profile_services
from veyronix.rootcause.engine import rank_root_causes
from veyronix.rootcause.evaluate import evaluate_rootcause
from veyronix.simulator.faults import FAULT_CATALOG, duration_bounds
from veyronix.simulator.generator import SimulationEngine
from veyronix.simulator.models import FaultInstance
from veyronix.simulator.run import run_simulation

pytestmark = pytest.mark.ml
START = datetime(2026, 1, 1, tzinfo=UTC)
EMPTY_DEPLOYS = pd.DataFrame(columns=["minute", "service", "version", "change_summary"])


def _single(ft, minutes=320, seed=7, start_min=140):
    spec = FAULT_CATALOG[ft]
    lo, hi = duration_bounds(ft)
    inst = FaultInstance(0, ft, spec.default_target, start_min, (lo + hi) // 2, spec.lead)
    res = SimulationEngine(f"rc-{ft.value}", START, minutes, seed).run([inst])
    return inst, res


def test_no_anomalies_on_a_healthy_window():
    _, res = _single(FaultType.CPU_SPIKE)
    # a window far from the incident should surface no service anomalies
    prof = profile_services(res.metrics, start_min=70, end_min=100)
    assert prof == {}


@pytest.mark.parametrize("ft,truth", [
    (FaultType.DATABASE_OVERLOAD, ServiceName.DATABASE),
    (FaultType.CPU_SPIKE, ServiceName.USER_SERVICE),
    (FaultType.QUEUE_BACKLOG, ServiceName.MESSAGE_QUEUE),
    (FaultType.MEMORY_LEAK, ServiceName.ORDER_SERVICE),
    (FaultType.NETWORK_LATENCY, ServiceName.PAYMENT_SERVICE),
    (FaultType.DEPENDENCY_FAILURE, ServiceName.CACHE),
])
def test_intrinsic_faults_attributed_to_source(ft, truth):
    inst, res = _single(ft)
    hyps = rank_root_causes(res.metrics, EMPTY_DEPLOYS, inst.start_min, inst.end_min)
    assert hyps, f"{ft.value}: no hypothesis"
    assert hyps[0].cause_service == truth.value, \
        f"{ft.value}: got {hyps[0].cause_service}"


def test_bad_deploy_blamed_on_deployer_not_the_database():
    res = run_simulation(minutes=260, seed=5, scenario="bad_deploy")
    inc = res.incidents[0]
    hyps = rank_root_causes(res.metrics, res.deployments, inc.start_min, inc.end_min)
    top = hyps[0]
    # DB shows the biggest symptom, but the deploying service is the root cause
    assert top.cause_service == ServiceName.ORDER_SERVICE.value
    assert top.cause_type == FaultType.BAD_DEPLOYMENT.value
    assert any(s["type"] == "deployment" for s in top.evidence_chain)


def test_hypotheses_ranked_by_confidence_and_have_evidence():
    inst, res = _single(FaultType.DATABASE_OVERLOAD)
    hyps = rank_root_causes(res.metrics, EMPTY_DEPLOYS, inst.start_min, inst.end_min)
    confs = [h.confidence for h in hyps]
    assert confs == sorted(confs, reverse=True)
    assert all(h.evidence_chain for h in hyps)
    # the true cause appears somewhere in the top hypothesis' evidence chain
    assert ServiceName.DATABASE.value in {s.get("service") for s in hyps[0].evidence_chain}


def test_aggregate_top1_accuracy_is_strong():
    r = evaluate_rootcause(seeds=[42, 43, 44], minutes=1440)
    # excludes model_drift (no infra footprint by design); overall still strong
    assert r["top1_accuracy"] >= 0.6
    assert r["top3_accuracy"] >= r["top1_accuracy"]
    assert r["cause_in_evidence_rate"] >= 0.7
