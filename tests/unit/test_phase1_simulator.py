"""Phase 1: simulator realism, causal cascades, ground-truth integrity."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from veyronix.core.constants import ActionType, FaultType, Severity
from veyronix.simulator.faults import FAULT_CATALOG, duration_bounds
from veyronix.simulator.generator import SimulationEngine, _seasonal
from veyronix.simulator.models import FaultInstance
from veyronix.simulator.run import run_simulation

pytestmark = pytest.mark.unit
START = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def sim():
    return run_simulation(minutes=1440, seed=42, scenario="random")


def _single(ft: FaultType, minutes=300, seed=7, start_min=120):
    spec = FAULT_CATALOG[ft]
    lo, hi = duration_bounds(ft)
    inst = FaultInstance(0, ft, spec.default_target, start_min, (lo + hi) // 2, spec.lead)
    res = SimulationEngine(f"t-{ft.value}", START, minutes, seed).run([inst])
    return inst, res


def test_no_nans_or_infs(sim):
    num = sim.metrics.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    assert not np.isnan(num).any()
    assert not np.isinf(num).any()


def test_determinism_same_seed():
    a = run_simulation(minutes=300, seed=99)
    b = run_simulation(minutes=300, seed=99)
    assert a.metrics["cpu_pct"].round(6).equals(b.metrics["cpu_pct"].round(6))
    assert [i.fault_type for i in a.incidents] == [i.fault_type for i in b.incidents]


def test_seasonality_present(sim):
    g = sim.metrics[sim.metrics.service == "api_gateway"].sort_values("minute")
    season = np.array([_seasonal(int(x) % 1440) for x in g.minute])
    corr = np.corrcoef(g.request_rate.to_numpy(dtype=float), season)[0, 1]
    assert corr > 0.5  # strong diurnal pattern


def test_normal_periods_are_healthy(sim):
    inc = set()
    for i in sim.incidents:
        inc.update(range(i.start_min - 5, i.end_min + 5))
    normal = sim.metrics[~sim.metrics.minute.isin(inc)]
    assert float(normal.error_rate.mean()) < 0.02  # mostly healthy


@pytest.mark.parametrize("ft", list(FaultType))
def test_every_fault_type_is_detectable(ft):
    """During each fault, an affected service shows a clear deviation."""
    inst, res = _single(ft)
    m = res.metrics
    win = m[(m.minute >= inst.start_min) & (m.minute < inst.end_min)]
    pre = m[(m.minute >= 60) & (m.minute < inst.onset_min)]
    # incident window must exceed baseline on error OR latency OR a signature.
    sig = FAULT_CATALOG[ft].signature_metrics[0]
    peak_err = float(win.error_rate.max())
    base_err = float(pre.error_rate.mean())
    peak_lat = float(win.latency_p95_ms.max())
    base_lat = float(pre.latency_p95_ms.mean())
    sig_peak = float(win[sig].max())
    sig_base = float(pre[sig].mean())
    deviates = (
        peak_err > base_err + 0.03
        or peak_lat > base_lat * 1.5
        or abs(sig_peak - sig_base) > max(1.0, 0.25 * abs(sig_base))
    )
    assert deviates, f"{ft.value}: no clear deviation"


def test_bad_deploy_cascade():
    inst, res = _single(FaultType.BAD_DEPLOYMENT, minutes=300, seed=5)
    m, b = res.metrics, res.business

    def stat(svc, col, agg="max"):
        w = m[(m.service == svc) & (m.minute >= inst.start_min) & (m.minute < inst.end_min)]
        p = m[(m.service == svc) & (m.minute >= 60) & (m.minute < inst.onset_min)]
        return getattr(w[col], agg)(), p[col].mean()

    db_w, db_b = stat("database", "db_latency_ms")
    ord_w, ord_b = stat("order_service", "latency_p95_ms")
    err_w, _ = stat("order_service", "error_rate")
    assert db_w > db_b * 3          # DB query latency spikes
    assert ord_w > ord_b * 3        # propagates to the order service
    assert err_w > 0.05             # becomes timeout errors

    bw = b[(b.minute >= inst.start_min) & (b.minute < inst.end_min)]
    bp = b[(b.minute >= 60) & (b.minute < inst.onset_min)]
    assert bw.revenue_per_min.mean() < bp.revenue_per_min.mean() * 0.9  # revenue falls


def test_memory_leak_climbs_and_crashes():
    inst, res = _single(FaultType.MEMORY_LEAK, minutes=400, seed=3, start_min=180)
    o = res.metrics[res.metrics.service == inst.target.value]
    win = o[(o.minute >= inst.start_min) & (o.minute < inst.end_min)]
    assert win.memory_pct.max() > 90       # climbs toward OOM
    assert win.error_rate.max() > 0.5      # crash at the tail


def test_traffic_spike_is_fleetwide():
    inst, res = _single(FaultType.TRAFFIC_SPIKE, minutes=250, seed=4)
    m = res.metrics
    for svc in ("api_gateway", "order_service", "user_service"):
        w = m[(m.service == svc) & (m.minute >= inst.start_min) & (m.minute < inst.end_min)]
        p = m[(m.service == svc) & (m.minute >= 60) & (m.minute < inst.onset_min)]
        assert w.request_rate.max() > p.request_rate.mean() * 1.5


def test_ground_truth_integrity(sim):
    assert len(sim.incidents) > 0
    for i in sim.incidents:
        assert i.start_min < i.end_min
        assert i.affected_services  # non-empty
        assert i.root_cause_service.value in i.affected_services
        assert Severity(i.severity)  # valid enum
        assert ActionType(i.recovery_action)  # valid allowlisted action
        assert i.expected_impact["estimated_revenue_loss"] >= 0


def test_precursor_before_labelled_start():
    """A ramp fault shows rising signal during the lead window (early warning)."""
    inst, res = _single(FaultType.BAD_DEPLOYMENT, minutes=300, seed=5)
    db = res.metrics[res.metrics.service == "database"].sort_values("minute")
    lead_win = db[(db.minute >= inst.onset_min) & (db.minute < inst.start_min)]
    before = db[(db.minute >= inst.onset_min - 20) & (db.minute < inst.onset_min)]
    assert lead_win.db_latency_ms.mean() > before.db_latency_ms.mean() + 1.0
