"""Fault scheduling: random realistic incident timelines and scripted scenarios.

Incidents are spaced with healthy gaps (so the dataset is mostly normal, as in
real ops) and cover all 13 fault types across a long run. Benign deployments are
scattered as decoys so the root-cause/deployment agents must discriminate, not
just blame the nearest deploy.
"""

from __future__ import annotations

import numpy as np

from veyronix.core.constants import KNOWN_FAULTS, FaultType, ServiceName
from veyronix.simulator.faults import FAULT_CATALOG, duration_bounds
from veyronix.simulator.models import FaultInstance

# Plausible alternative targets per fault, to add variety beyond the default.
_ALT_TARGETS: dict[FaultType, tuple[ServiceName, ...]] = {
    FaultType.CPU_SPIKE: (ServiceName.USER_SERVICE, ServiceName.ORDER_SERVICE,
                          ServiceName.PAYMENT_SERVICE),
    FaultType.MEMORY_LEAK: (ServiceName.ORDER_SERVICE, ServiceName.USER_SERVICE,
                            ServiceName.PAYMENT_SERVICE),
    FaultType.SERVICE_CRASH: (ServiceName.USER_SERVICE, ServiceName.ORDER_SERVICE),
    FaultType.NETWORK_LATENCY: (ServiceName.PAYMENT_SERVICE, ServiceName.USER_SERVICE),
}


def _pick_target(rng: np.random.Generator, ft: FaultType) -> ServiceName:
    alts = _ALT_TARGETS.get(ft)
    if alts and rng.random() < 0.5:
        return alts[int(rng.integers(len(alts)))]
    return FAULT_CATALOG[ft].default_target


def schedule_random_faults(
    minutes: int, seed: int, mean_gap_min: int = 75, warmup_min: int = 90,
) -> list[FaultInstance]:
    """Lay incidents across the timeline with healthy gaps between them.

    Every fault type is guaranteed at least once (if the run is long enough),
    then remaining slots are filled by random draws.
    """
    rng = np.random.default_rng(seed + 1000)
    all_types = list(FaultType)
    rng.shuffle(all_types)
    queue = list(all_types)  # guarantee coverage first

    faults: list[FaultInstance] = []
    fid = 0
    cursor = warmup_min
    while cursor < minutes - 90:
        ft = queue.pop(0) if queue else all_types[int(rng.integers(len(all_types)))]
        spec = FAULT_CATALOG[ft]
        lo, hi = duration_bounds(ft)
        duration = int(rng.integers(lo, hi + 1))
        lead = spec.lead
        start = cursor + lead  # leave room for the precursor before labelled start
        inst = FaultInstance(
            fault_id=fid, fault_type=ft, target=_pick_target(rng, ft),
            start_min=start, duration_min=duration, lead_min=lead,
        )
        if inst.end_min >= minutes - 30:
            break
        faults.append(inst)
        fid += 1
        gap = int(rng.exponential(mean_gap_min)) + 30
        cursor = inst.end_min + gap
    return faults


def build_bad_deploy_cascade(start_min: int = 120, duration_min: int = 55) -> list[FaultInstance]:
    """Flagship scenario: the Bad Deployment Cascade (spec §58)."""
    spec = FAULT_CATALOG[FaultType.BAD_DEPLOYMENT]
    return [FaultInstance(
        fault_id=0, fault_type=FaultType.BAD_DEPLOYMENT,
        target=ServiceName.ORDER_SERVICE, start_min=start_min,
        duration_min=duration_min, lead_min=spec.lead,
    )]


def build_unknown_cascade(start_min: int = 120, duration_min: int = 45) -> list[FaultInstance]:
    """Second scenario: an unknown cascading failure (spec §59)."""
    spec = FAULT_CATALOG[FaultType.CASCADING_FAILURE]
    return [FaultInstance(
        fault_id=0, fault_type=FaultType.CASCADING_FAILURE,
        target=ServiceName.ORDER_SERVICE, start_min=start_min,
        duration_min=duration_min, lead_min=spec.lead,
    )]


def is_known_fault(ft: FaultType) -> bool:
    return ft in KNOWN_FAULTS
