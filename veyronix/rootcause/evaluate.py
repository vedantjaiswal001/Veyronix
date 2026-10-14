"""Root-cause evaluation (spec §24): Top-1 / Top-3 service accuracy, fault-type
accuracy, and whether the true cause appears anywhere in the evidence chain."""

from __future__ import annotations

from collections import defaultdict

from veyronix.core.logging import get_logger
from veyronix.rootcause.engine import rank_root_causes
from veyronix.simulator.run import run_simulation

log = get_logger("rootcause.eval")


def evaluate_rootcause(seeds: list[int] | None = None, minutes: int = 1440) -> dict:
    seeds = seeds or [42, 43, 44, 45, 46, 47, 48, 49]
    n = top1 = top3 = ftype = ev_hit = 0
    per_fault: dict[str, dict] = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    examples: list[dict] = []

    for seed in seeds:
        res = run_simulation(minutes=minutes, seed=seed, run_id=f"rc-{seed}")
        for inc in res.incidents:
            hyps = rank_root_causes(res.metrics, res.deployments,
                                    inc.start_min, inc.end_min)
            if not hyps:
                n += 1
                per_fault[inc.fault_type.value]["n"] += 1
                continue
            truth = inc.root_cause_service.value
            top_services = [h.cause_service for h in hyps]
            is1 = top_services[0] == truth
            is3 = truth in top_services[:3]
            evh = truth in {s.get("service") for s in hyps[0].evidence_chain}
            ftm = hyps[0].cause_type == inc.fault_type.value

            n += 1
            top1 += is1
            top3 += is3
            ftype += ftm
            ev_hit += evh
            pf = per_fault[inc.fault_type.value]
            pf["n"] += 1
            pf["top1"] += is1
            pf["top3"] += is3
            if len(examples) < 6:
                examples.append({
                    "fault": inc.fault_type.value, "truth": truth,
                    "predicted": hyps[0].cause_service,
                    "cause_type": hyps[0].cause_type,
                    "confidence": hyps[0].confidence, "correct": bool(is1),
                })

    result = {
        "incidents": n,
        "top1_accuracy": round(top1 / n, 4) if n else 0.0,
        "top3_accuracy": round(top3 / n, 4) if n else 0.0,
        "fault_type_accuracy": round(ftype / n, 4) if n else 0.0,
        "cause_in_evidence_rate": round(ev_hit / n, 4) if n else 0.0,
        "per_fault": {k: {"n": v["n"],
                          "top1": round(v["top1"] / v["n"], 3) if v["n"] else 0.0,
                          "top3": round(v["top3"] / v["n"], 3) if v["n"] else 0.0}
                      for k, v in sorted(per_fault.items())},
        "examples": examples,
    }
    log.info("rootcause.eval.done", incidents=n, top1=result["top1_accuracy"],
             top3=result["top3_accuracy"])
    return result


def main(seed: int = 42) -> None:
    """Write root-cause evaluation results to the ML results directory."""
    import json

    from veyronix.core.config import ARTIFACT_DIR

    out = evaluate_rootcause()
    path = ARTIFACT_DIR / "ml_results" / "rootcause.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("incidents", "top1_accuracy", "top3_accuracy",
                       "fault_type_accuracy", "cause_in_evidence_rate")}, indent=2))


if __name__ == "__main__":
    import typer

    typer.run(main)
