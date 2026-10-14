"""Phase 1 integration: a simulation run persists to DB and parquet and reloads."""

from __future__ import annotations

import pytest

from veyronix.simulator.persist import (
    load_run_frames,
    persist_result,
    save_parquet,
)
from veyronix.simulator.run import run_simulation

pytestmark = pytest.mark.integration


def test_persist_to_db_and_parquet_roundtrip(db, tmp_path):
    result = run_simulation(minutes=240, seed=11, run_id="itest")
    counts = persist_result(result)
    assert counts["metrics"] == len(result.metrics)
    assert counts["incidents"] == len(result.incidents)
    assert counts["deployments"] >= 1

    # DB roundtrip
    from veyronix.db.models import Incident, Metric
    from veyronix.db.session import session_scope

    with session_scope() as s:
        assert s.query(Metric).filter_by(run_id="itest").count() == len(result.metrics)
        inc = s.query(Incident).filter_by(run_id="itest").first()
        assert inc is not None
        assert isinstance(inc.affected_services, list)
        # JSON metric values round-trip with all channels present
        m = s.query(Metric).filter_by(run_id="itest").first()
        assert "cpu_pct" in m.values and "error_rate" in m.values

    # parquet roundtrip
    frames = load_run_frames("itest", base_dir=save_parquet(result, base_dir=tmp_path).parent)
    assert len(frames["metrics"]) == len(result.metrics)
    assert not frames["incidents"].empty
