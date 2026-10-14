"""Incident-prediction models with a common interface (spec §14).

Baselines to strong models, so we can show whether complexity is justified:
  * NaiveThreshold  — a static health-score threshold (the "traditional alert").
  * LogisticRegression — linear, scaled, class-balanced.
  * RandomForest — bagged trees, class-balanced.
  * XGBoost — gradient boosting with scale_pos_weight for imbalance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Predictor(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None) -> Predictor:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def feature_importances(self) -> dict | None:
        return None


class NaiveThreshold(Predictor):
    """Traditional static-threshold detector using a system health score.

    Score = max(system saturation, scaled system error rate). No learning of
    interactions — the baseline that predictive ML must beat (spec §51 #1).
    """

    name = "naive_threshold"

    def __init__(self) -> None:
        self.feat_sat = "sys__max_saturation"
        self.feat_err = "sys__max_error_rate"

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        sat = X[self.feat_sat].to_numpy(dtype=float) if self.feat_sat in X else np.zeros(len(X))
        err = X[self.feat_err].to_numpy(dtype=float) if self.feat_err in X else np.zeros(len(X))
        return np.clip(np.maximum(sat, np.minimum(1.0, err * 5.0)), 0, 1)

    def fit(self, X, y, X_val=None, y_val=None):
        return self

    def predict_proba(self, X):
        return self._score(X)


class _SklearnPredictor(Predictor):
    def __init__(self, model, scale: bool = False):
        self.model = model
        self.scaler = StandardScaler() if scale else None
        self.columns: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.columns = list(X.columns)
        Xt = X.to_numpy(dtype=float)
        if self.scaler is not None:
            Xt = self.scaler.fit_transform(Xt)
        self.model.fit(Xt, np.asarray(y).astype(int))
        return self

    def predict_proba(self, X):
        Xt = X[self.columns].to_numpy(dtype=float)
        if self.scaler is not None:
            Xt = self.scaler.transform(Xt)
        return self.model.predict_proba(Xt)[:, 1]

    def feature_importances(self):
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            return dict(sorted(zip(self.columns, map(float, imp), strict=True),
                               key=lambda kv: kv[1], reverse=True)[:25])
        if hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_[0])
            return dict(sorted(zip(self.columns, map(float, imp), strict=True),
                               key=lambda kv: kv[1], reverse=True)[:25])
        return None


class LogisticModel(_SklearnPredictor):
    name = "logistic_regression"

    def __init__(self, seed: int = 42):
        super().__init__(
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
            scale=True,
        )


class RandomForestModel(_SklearnPredictor):
    name = "random_forest"

    def __init__(self, seed: int = 42):
        super().__init__(RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5, n_jobs=-1,
            class_weight="balanced", random_state=seed))


class XGBoostModel(Predictor):
    name = "xgboost"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None
        self.columns: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None):
        from xgboost import XGBClassifier

        self.columns = list(X.columns)
        y = np.asarray(y).astype(int)
        pos = max(int(y.sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        self.model = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, scale_pos_weight=neg / pos, eval_metric="aucpr",
            tree_method="hist", random_state=self.seed, n_jobs=-1,
        )
        eval_set = [(X_val.to_numpy(dtype=float), np.asarray(y_val).astype(int))] \
            if X_val is not None and y_val is not None else None
        self.model.fit(X.to_numpy(dtype=float), y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns].to_numpy(dtype=float))[:, 1]

    def feature_importances(self):
        imp = self.model.feature_importances_
        return dict(sorted(zip(self.columns, map(float, imp), strict=True),
                           key=lambda kv: kv[1], reverse=True)[:25])


MODEL_REGISTRY: dict[str, type[Predictor]] = {
    NaiveThreshold.name: NaiveThreshold,
    LogisticModel.name: LogisticModel,
    RandomForestModel.name: RandomForestModel,
    XGBoostModel.name: XGBoostModel,
}
