"""Remediation action models + fault/action efficacy (spec §30/§32).

Each action has an execution time, operational cost and risk tier. Efficacy —
the probability an action actually resolves a given fault — is modelled
explicitly and conservatively: the matching recovery action is highly effective,
some actions help related faults partially, and most are ineffective. These are
*estimates* used for counterfactual reasoning, never ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from veyronix.core.constants import ActionType, FaultType, RiskTier


@dataclass(frozen=True)
class ActionModel:
    action: ActionType
    exec_minutes: float      # time to take effect
    cost_units: float        # relative operational cost
    risk: RiskTier
    description: str


ACTION_MODELS: dict[ActionType, ActionModel] = {
    ActionType.NO_OP: ActionModel(ActionType.NO_OP, 0.0, 0.0, RiskTier.LOW,
                                  "Do nothing; let the incident run its course"),
    ActionType.RESTART_SERVICE: ActionModel(ActionType.RESTART_SERVICE, 3.0, 1.0,
                                            RiskTier.LOW, "Restart the affected service"),
    ActionType.FLUSH_CACHE: ActionModel(ActionType.FLUSH_CACHE, 1.0, 1.0, RiskTier.LOW,
                                        "Flush and warm the cache"),
    ActionType.SCALE_UP_SERVICE: ActionModel(ActionType.SCALE_UP_SERVICE, 4.0, 4.0,
                                             RiskTier.LOW, "Add service replicas"),
    ActionType.CLEAR_QUEUE: ActionModel(ActionType.CLEAR_QUEUE, 2.0, 1.5, RiskTier.MEDIUM,
                                        "Drain / clear the message queue"),
    ActionType.FAILOVER_DEPENDENCY: ActionModel(ActionType.FAILOVER_DEPENDENCY, 4.0, 3.0,
                                                RiskTier.MEDIUM, "Fail over to a healthy dependency"),
    ActionType.SCALE_UP_DATABASE: ActionModel(ActionType.SCALE_UP_DATABASE, 8.0, 6.0,
                                              RiskTier.MEDIUM, "Scale up the database tier"),
    ActionType.ROLLBACK_DEPLOYMENT: ActionModel(ActionType.ROLLBACK_DEPLOYMENT, 5.0, 3.0,
                                                RiskTier.MEDIUM, "Roll back the recent deployment"),
    ActionType.DELETE_DATABASE: ActionModel(ActionType.DELETE_DATABASE, 1.0, 100.0,
                                            RiskTier.BLOCKED, "Delete the database (never allowed)"),
}

# The correct recovery action per fault (mirrors the fault catalog).
_CORRECT: dict[FaultType, ActionType] = {
    FaultType.CPU_SPIKE: ActionType.SCALE_UP_SERVICE,
    FaultType.MEMORY_LEAK: ActionType.RESTART_SERVICE,
    FaultType.DATABASE_OVERLOAD: ActionType.SCALE_UP_DATABASE,
    FaultType.NETWORK_LATENCY: ActionType.FAILOVER_DEPENDENCY,
    FaultType.PACKET_LOSS: ActionType.FAILOVER_DEPENDENCY,
    FaultType.TRAFFIC_SPIKE: ActionType.SCALE_UP_SERVICE,
    FaultType.SERVICE_CRASH: ActionType.RESTART_SERVICE,
    FaultType.BAD_DEPLOYMENT: ActionType.ROLLBACK_DEPLOYMENT,
    FaultType.DEPENDENCY_FAILURE: ActionType.FAILOVER_DEPENDENCY,
    FaultType.QUEUE_BACKLOG: ActionType.CLEAR_QUEUE,
    FaultType.CASCADING_FAILURE: ActionType.ROLLBACK_DEPLOYMENT,
    FaultType.DATA_PIPELINE_FAILURE: ActionType.CLEAR_QUEUE,
    FaultType.MODEL_DRIFT: ActionType.NO_OP,
}

# Partial cross-efficacy: action -> {fault: probability it helps}.
_CROSS: dict[ActionType, dict[FaultType, float]] = {
    ActionType.RESTART_SERVICE: {FaultType.MEMORY_LEAK: 0.9, FaultType.SERVICE_CRASH: 0.9,
                                 FaultType.CPU_SPIKE: 0.35, FaultType.CASCADING_FAILURE: 0.4},
    ActionType.SCALE_UP_SERVICE: {FaultType.CPU_SPIKE: 0.9, FaultType.TRAFFIC_SPIKE: 0.9,
                                  FaultType.CASCADING_FAILURE: 0.3},
    ActionType.SCALE_UP_DATABASE: {FaultType.DATABASE_OVERLOAD: 0.9,
                                   FaultType.BAD_DEPLOYMENT: 0.45},  # treats symptom, not cause
    ActionType.ROLLBACK_DEPLOYMENT: {FaultType.BAD_DEPLOYMENT: 0.92,
                                     FaultType.CASCADING_FAILURE: 0.6},
    ActionType.CLEAR_QUEUE: {FaultType.QUEUE_BACKLOG: 0.9, FaultType.DATA_PIPELINE_FAILURE: 0.85},
    ActionType.FAILOVER_DEPENDENCY: {FaultType.DEPENDENCY_FAILURE: 0.9,
                                     FaultType.NETWORK_LATENCY: 0.85, FaultType.PACKET_LOSS: 0.8},
    ActionType.FLUSH_CACHE: {FaultType.DEPENDENCY_FAILURE: 0.4},
}

BASELINE_EFFICACY = 0.12  # an unrelated action rarely helps


def efficacy(action: ActionType, fault: FaultType) -> float:
    """Probability that ``action`` resolves ``fault`` (0..1)."""
    if action == ActionType.NO_OP:
        return 0.0
    if action in _CROSS and fault in _CROSS[action]:
        return _CROSS[action][fault]
    if _CORRECT.get(fault) == action:
        return 0.9
    return BASELINE_EFFICACY


def candidate_actions(fault: FaultType) -> list[ActionType]:
    """A focused, safe candidate set to simulate for a fault (excludes BLOCKED)."""
    actions = {ActionType.NO_OP, _CORRECT.get(fault, ActionType.NO_OP),
               ActionType.RESTART_SERVICE, ActionType.SCALE_UP_SERVICE,
               ActionType.SCALE_UP_DATABASE, ActionType.ROLLBACK_DEPLOYMENT}
    return [a for a in actions if ACTION_MODELS[a].risk != RiskTier.BLOCKED]
