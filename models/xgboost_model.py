"""
models/xgboost_model.py
─────────────────────────────────────────────────────────────
XGBoost + LightGBM models with Optuna hyperparameter tuning
and a stacking ensemble that combines LSTM + tree models.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger
import joblib

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, roc_auc_score
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed — using default hyperparameters")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models/saved"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostStockModel:
    """XGBoost wrapper with time-series CV and Optuna tuning."""

    def __init__(self, task: str = "regression", ticker: str = "AAPL", n_trials: int = 50):
        self.task      = task
        self.ticker    = ticker
        self.n_trials  = n_trials
        self.model     = None
        self.best_params = {}

    # ── Default params ────────────────────────────────────────
    def _default_params(self) -> dict:
        base = dict(
            n_estimators     = 500,
            learning_rate    = 0.05,
            max_depth        = 6,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            random_state     = 42,
            n_jobs           = -1,
            early_stopping_rounds = 30,
        )
        if self.task == "regression":
            base["objective"] = "reg:squarederror"
        else:
            base["objective"] = "binary:logistic"
            base["eval_metric"] = "auc"
        return base

    # ── Optuna tuning ─────────────────────────────────────────
    def _objective(self, trial, X, y, cv):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "random_state":      42,
            "n_jobs":            -1,
        }
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_tr, X_v = X[train_idx], X[val_idx]
            y_tr, y_v = y[train_idx], y[val_idx]
            if self.task == "regression":
                m = xgb.XGBRegressor(**params)
                m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
                scores.append(mean_squared_error(y_v, m.predict(X_v)))
            else:
                m = xgb.XGBClassifier(**params)
                m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
                scores.append(-roc_auc_score(y_v, m.predict_proba(X_v)[:, 1]))
        return np.mean(scores)

    def tune(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not OPTUNA_AVAILABLE:
            logger.info("Skipping tuning — Optuna not available")
            return self._default_params()

        cv = TimeSeriesSplit(n_splits=5)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: self._objective(t, X, y, cv),
                       n_trials=self.n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        logger.info(f"Best params: {self.best_params}")
        return self.best_params

    # ── Fit ───────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        tune:    bool = False,
    ):
        if tune:
            X_all = np.vstack([X_train, X_val])
            y_all = np.concatenate([y_train, y_val])
            params = self.tune(X_all, y_all)
        else:
            params = self._default_params()

        logger.info(f"[XGBoost] Training {self.task} model for {self.ticker}…")
        if self.task == "regression":
            self.model = xgb.XGBRegressor(**params)
        else:
            self.model = xgb.XGBClassifier(**params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        return self

    # ── Predict ───────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification task")
        return self.model.predict_proba(X)

    # ── Feature importance ────────────────────────────────────
    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        imp = self.model.feature_importances_
        df  = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Save / Load ───────────────────────────────────────────
    def save(self):
        path = MODEL_DIR / f"xgboost_{self.ticker}_{self.task}.pkl"
        joblib.dump(self, path)
        logger.info(f"XGBoost saved → {path}")

    @classmethod
    def load(cls, ticker: str, task: str) -> "XGBoostStockModel":
        path = MODEL_DIR / f"xgboost_{ticker}_{task}.pkl"
        obj  = joblib.load(path)
        logger.info(f"XGBoost loaded ← {path}")
        return obj


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LightGBMStockModel:
    """Faster LightGBM variant — useful for large feature sets."""

    def __init__(self, task: str = "regression", ticker: str = "AAPL"):
        self.task   = task
        self.ticker = ticker
        self.model  = None

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ):
        params = dict(
            n_estimators       = 500,
            learning_rate      = 0.05,
            max_depth          = 6,
            num_leaves         = 63,
            subsample          = 0.8,
            colsample_bytree   = 0.8,
            reg_alpha          = 0.1,
            reg_lambda         = 1.0,
            random_state       = 42,
            n_jobs             = -1,
            early_stopping_round=30,
            verbose            = -1,
        )
        if self.task == "regression":
            self.model = lgb.LGBMRegressor(**params)
        else:
            self.model = lgb.LGBMClassifier(**params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        logger.info(f"[LightGBM] Training complete for {self.ticker}")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ── Stacking Ensemble ─────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Meta-learner that stacks XGBoost + LightGBM predictions.
    Can also accept LSTM predictions as an additional input column.
    """

    def __init__(self, task: str = "regression", ticker: str = "AAPL"):
        self.task   = task
        self.ticker = ticker
        self.meta   = None
        self.xgb_model  = XGBoostStockModel(task=task, ticker=ticker)
        self.lgb_model  = LightGBMStockModel(task=task, ticker=ticker)

    def _make_meta_features(
        self,
        X: np.ndarray,
        lstm_preds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Combine base model predictions into meta-feature matrix."""
        xgb_p = self.xgb_model.predict(X).reshape(-1, 1)
        lgb_p = self.lgb_model.predict(X).reshape(-1, 1)
        parts = [xgb_p, lgb_p]
        if lstm_preds is not None:
            parts.append(lstm_preds.reshape(-1, 1))
        return np.hstack(parts)

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        lstm_train_preds: Optional[np.ndarray] = None,
        lstm_val_preds:   Optional[np.ndarray] = None,
    ):
        logger.info("[Ensemble] Training base models…")
        self.xgb_model.fit(X_train, y_train, X_val, y_val)
        self.lgb_model.fit(X_train, y_train, X_val, y_val)

        logger.info("[Ensemble] Building meta-features…")
        X_meta_train = self._make_meta_features(X_train, lstm_train_preds)
        X_meta_val   = self._make_meta_features(X_val,   lstm_val_preds)

        # Meta learner
        if self.task == "regression":
            self.meta = Ridge(alpha=1.0)
        else:
            self.meta = LogisticRegression(C=1.0, max_iter=500)

        X_meta_all = np.vstack([X_meta_train, X_meta_val])
        y_meta_all = np.concatenate([y_train, y_val])
        self.meta.fit(X_meta_all, y_meta_all)
        logger.info("[Ensemble] Meta-learner fitted ✓")
        return self

    def predict(
        self,
        X: np.ndarray,
        lstm_preds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_meta = self._make_meta_features(X, lstm_preds)
        return self.meta.predict(X_meta)

    def predict_proba(
        self,
        X: np.ndarray,
        lstm_preds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        X_meta = self._make_meta_features(X, lstm_preds)
        return self.meta.predict_proba(X_meta)

    def individual_predictions(self, X: np.ndarray) -> dict:
        """Return each model's raw prediction for transparency."""
        return {
            "xgboost":  self.xgb_model.predict(X),
            "lightgbm": self.lgb_model.predict(X),
            "ensemble": self.predict(X),
        }

    def save(self):
        path = MODEL_DIR / f"ensemble_{self.ticker}_{self.task}.pkl"
        joblib.dump(self, path)
        logger.info(f"Ensemble saved → {path}")

    @classmethod
    def load(cls, ticker: str, task: str) -> "StackingEnsemble":
        path = MODEL_DIR / f"ensemble_{ticker}_{task}.pkl"
        obj  = joblib.load(path)
        logger.info(f"Ensemble loaded ← {path}")
        return obj


# ── Evaluate helper ───────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, task="regression") -> dict:
    if task == "regression":
        return {
            "MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2":   1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-9),
            "DirAcc": np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))),
        }
    else:
        y_bin = (y_pred > 0.5).astype(int)
        return {
            "Accuracy": accuracy_score(y_true, y_bin),
            "AUC":      roc_auc_score(y_true, y_pred),
        }


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N, F = 600, 50
    X = np.random.randn(N, F).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)

    split = int(0.8 * N)
    xgb_m = XGBoostStockModel(task="regression", ticker="TEST")
    xgb_m.fit(X[:split], y[:split], X[split:], y[split:])
    preds = xgb_m.predict(X[split:])
    metrics = evaluate_model(y[split:], preds, "regression")
    print("XGBoost metrics:", metrics)
