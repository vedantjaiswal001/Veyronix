"""Leakage-safe temporal feature engineering (spec §12/§16).

Every feature at minute t uses ONLY telemetry at or before t (rolling windows,
lags, backward differences). No feature ever reads a future value. Features are
computed per run (each run is an independent timeline) so history never leaks
across runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from veyronix.core.constants import ServiceName

# Channels turned into temporal features, per service group.
GLOBAL_SIGNALS = [
    "cpu_pct", "memory_pct", "latency_p95_ms", "error_rate", "request_rate",
    "saturation", "http_5xx_rate", "gc_pause_ms", "network_latency_ms",
    "packet_loss_pct",
]
DB_SIGNALS = ["db_latency_ms", "db_cpu_pct", "db_connections"]
QUEUE_SIGNALS = ["queue_depth"]
CACHE_SIGNALS = ["cache_hit_rate"]
BUSINESS_SIGNALS = ["conversion_rate", "orders_per_min", "checkout_error_rate",
                    "revenue_per_min", "active_users"]

SHORT_W, LONG_W = 15, 30
LAG_DELTA = 5


def _signal_columns() -> dict[str, list[str]]:
    """Map each service -> the channels we build temporal features from."""
    out: dict[str, list[str]] = {}
    for s in ServiceName:
        if s == ServiceName.CLIENT:
            continue
        cols = list(GLOBAL_SIGNALS)
        if s == ServiceName.DATABASE:
            cols += DB_SIGNALS
        if s == ServiceName.MESSAGE_QUEUE:
            cols += QUEUE_SIGNALS
        if s == ServiceName.CACHE:
            cols += CACHE_SIGNALS
        out[s.value] = cols
    return out


def _temporal_block(g: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """Backward-only temporal features for one column of one run."""
    s = g[col]
    feats = {
        f"{prefix}__last": s,
        f"{prefix}__mean15": s.rolling(SHORT_W, min_periods=1).mean(),
        f"{prefix}__std15": s.rolling(SHORT_W, min_periods=1).std().fillna(0.0),
        f"{prefix}__max15": s.rolling(SHORT_W, min_periods=1).max(),
        f"{prefix}__mean30": s.rolling(LONG_W, min_periods=1).mean(),
        f"{prefix}__delta5": (s - s.shift(LAG_DELTA)).fillna(0.0),
        f"{prefix}__trend15": ((s - s.shift(SHORT_W)) / SHORT_W).fillna(0.0),
    }
    return pd.DataFrame(feats, index=g.index)


def build_feature_matrix(metrics: pd.DataFrame, business: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per (run_id, minute) with leakage-safe features."""
    sig_cols = _signal_columns()

    # wide: (run_id, minute, ts) x  service__channel
    keep_cols = ["run_id", "minute", "ts", "service", *sorted(
        {c for cols in sig_cols.values() for c in cols})]
    m = metrics[keep_cols].copy()
    wide = m.pivot(index=["run_id", "minute", "ts"], columns="service")
    wide.columns = [f"{svc}__{ch}" for ch, svc in wide.columns]
    wide = wide.sort_index().reset_index()

    # merge business (per run, minute)
    biz = business[["run_id", "minute", *BUSINESS_SIGNALS]].copy()
    wide = wide.merge(biz, on=["run_id", "minute"], how="left")
    wide = wide.sort_values(["run_id", "minute"]).reset_index(drop=True)

    feature_frames = [wide[["run_id", "minute", "ts"]]]

    # per-service temporal features (grouped by run so history never crosses runs)
    for svc, chans in sig_cols.items():
        for ch in chans:
            col = f"{svc}__{ch}"
            if col not in wide.columns:
                continue
            blocks = []
            for _, g in wide.groupby("run_id", sort=False):
                blocks.append(_temporal_block(g, col, prefix=col))
            feature_frames.append(pd.concat(blocks).sort_index())

    # business temporal features
    for ch in BUSINESS_SIGNALS:
        blocks = []
        for _, g in wide.groupby("run_id", sort=False):
            blocks.append(_temporal_block(g, ch, prefix=f"biz__{ch}"))
        feature_frames.append(pd.concat(blocks).sort_index())

    feats = pd.concat(feature_frames, axis=1)

    # ---- topology-aware, system-level features (dependency health) ----
    telem = [s.value for s in ServiceName if s != ServiceName.CLIENT]
    err_cols = [f"{s}__error_rate" for s in telem if f"{s}__error_rate" in wide]
    p95_cols = [f"{s}__latency_p95_ms" for s in telem if f"{s}__latency_p95_ms" in wide]
    sat_cols = [f"{s}__saturation" for s in telem if f"{s}__saturation" in wide]
    feats["sys__max_error_rate"] = wide[err_cols].max(axis=1)
    feats["sys__mean_error_rate"] = wide[err_cols].mean(axis=1)
    feats["sys__max_latency_p95"] = wide[p95_cols].max(axis=1)
    feats["sys__max_saturation"] = wide[sat_cols].max(axis=1)
    feats["sys__saturated_count"] = (wide[sat_cols] > 0.8).sum(axis=1)
    feats["sys__db_latency"] = wide.get("database__db_latency_ms", 0.0)
    feats["sys__unhealthy_services"] = (wide[err_cols] > 0.05).sum(axis=1)

    # minute-of-day cyclical encoding (seasonality context, known at time t)
    mod = (wide["ts"].dt.hour * 60 + wide["ts"].dt.minute).to_numpy()
    feats["ctx__tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    feats["ctx__tod_cos"] = np.cos(2 * np.pi * mod / 1440)

    feats = feats.loc[:, ~feats.columns.duplicated()]
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Model feature columns (everything except identifiers)."""
    ids = {"run_id", "minute", "ts"}
    return [c for c in feats.columns if c not in ids]
