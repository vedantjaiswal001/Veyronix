"""Automated data-leakage audits (spec §16).

The strongest check is *truncation invariance*: a leakage-free feature computed
at minute t must be identical whether or not any rows after t exist. If a
feature ever peeks into the future, truncating the series at t changes it. This
module verifies that property directly, plus split disjointness / embargo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from veyronix.features.windows import build_feature_matrix, feature_columns


def truncation_invariance_report(
    metrics: pd.DataFrame, business: pd.DataFrame, probe_minutes: list[int],
) -> dict:
    """Recompute features on the series truncated at each probe minute and
    confirm the row at that minute is unchanged vs. the full-series features."""
    full = build_feature_matrix(metrics, business)
    cols = feature_columns(full)
    run_id = metrics["run_id"].iloc[0]
    full_run = full[full["run_id"] == run_id].set_index("minute")

    mismatches: list[dict] = []
    for t in probe_minutes:
        m_trunc = metrics[metrics["minute"] <= t]
        b_trunc = business[business["minute"] <= t]
        trunc = build_feature_matrix(m_trunc, b_trunc)
        trunc_run = trunc[trunc["run_id"] == run_id].set_index("minute")
        if t not in trunc_run.index or t not in full_run.index:
            continue
        a = full_run.loc[t, cols].to_numpy(dtype=float)
        b = trunc_run.loc[t, cols].to_numpy(dtype=float)
        diff = np.abs(a - b)
        bad = int((diff > 1e-9).sum())
        if bad:
            worst = int(np.argmax(diff))
            mismatches.append({"minute": int(t), "n_diff": bad,
                               "worst_feature": cols[worst],
                               "max_abs_diff": float(diff.max())})
    return {"probes": len(probe_minutes), "leaky_features": mismatches,
            "leak_free": len(mismatches) == 0}


def split_audit(df: pd.DataFrame, embargo: int) -> dict:
    """Verify splits are chronologically ordered and embargo-separated per run."""
    problems: list[str] = []
    for run_id, g in df.groupby("run_id", sort=False):
        for a, b in (("train", "val"), ("val", "test")):
            ga, gb = g[g.split == a], g[g.split == b]
            if ga.empty or gb.empty:
                continue
            gap = gb.minute.min() - ga.minute.max()
            if gb.minute.min() <= ga.minute.max():
                problems.append(f"{run_id}: {a}->{b} not chronological")
            elif gap < embargo:
                problems.append(f"{run_id}: {a}->{b} gap {gap} < embargo {embargo}")
    return {"ok": not problems, "problems": problems}
