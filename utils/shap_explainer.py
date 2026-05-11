"""
utils/shap_explainer.py
─────────────────────────────────────────────────────────────
SHAP (SHapley Additive exPlanations) for StockSense AI.
Explains WHY the XGBoost model made each prediction.

Provides:
  - Waterfall chart  : contribution of each feature to ONE prediction
  - Summary plot     : feature importance across ALL predictions
  - What-if analysis : change one feature value and see impact
  - Dependence plot  : how one feature affects prediction as it varies

Install: pip install shap
"""

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


# ── Core explainer ────────────────────────────────────────────────────────────

class StockSHAPExplainer:
    """
    Wraps XGBoost model with SHAP TreeExplainer.
    Provides all explanation types needed for the dashboard.
    """

    def __init__(self, xgb_model, feature_names: list[str]):
        if not SHAP_AVAILABLE:
            raise ImportError("Install SHAP first: pip install shap")

        self.feature_names = feature_names
        self.xgb_model     = xgb_model

        logger.info("Building SHAP TreeExplainer…")
        # TreeExplainer is exact (not approximate) for tree models — very fast
        self.explainer = shap.TreeExplainer(xgb_model.model)
        logger.info("SHAP explainer ready ✓")

    def explain_single(self, X_row: np.ndarray) -> dict:
        """
        Explain one prediction — returns data for waterfall chart.

        Args:
            X_row: shape (1, n_features) or (n_features,)

        Returns dict with:
            base_value    : model's average prediction (E[f(X)])
            shap_values   : contribution of each feature
            feature_values: actual feature values for this row
            feature_names : list of feature names
            prediction    : final predicted value
            top_positive  : top 5 features pushing prediction UP
            top_negative  : top 5 features pushing prediction DOWN
        """
        X = X_row.reshape(1, -1) if X_row.ndim == 1 else X_row
        sv = self.explainer.shap_values(X)[0]         # (n_features,)
        bv = float(self.explainer.expected_value)
        fv = X[0]
        pred = bv + sv.sum()

        # Sort features by absolute SHAP value
        idx_sorted = np.argsort(np.abs(sv))[::-1]

        top_pos = [(self.feature_names[i], float(sv[i]), float(fv[i]))
                   for i in idx_sorted if sv[i] > 0][:5]
        top_neg = [(self.feature_names[i], float(sv[i]), float(fv[i]))
                   for i in idx_sorted if sv[i] < 0][:5]

        return {
            "base_value":     bv,
            "shap_values":    sv.tolist(),
            "feature_values": fv.tolist(),
            "feature_names":  self.feature_names,
            "prediction":     float(pred),
            "top_positive":   top_pos,
            "top_negative":   top_neg,
            "idx_sorted":     idx_sorted[:15].tolist(),
        }

    def explain_batch(self, X: np.ndarray, sample_size: int = 200) -> dict:
        """
        Explain predictions across a dataset — for summary/importance plots.

        Returns dict with:
            shap_values      : (n_samples, n_features)
            feature_names    : list
            mean_abs_shap    : mean |SHAP| per feature (global importance)
            importance_df    : DataFrame sorted by importance
            shap_over_time   : dict of top-10 features → shap values over time
        """
        # Sample for speed
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        logger.info(f"Computing SHAP values for {len(X_sample)} samples…")
        sv = self.explainer.shap_values(X_sample)     # (n_samples, n_features)

        mean_abs = np.abs(sv).mean(axis=0)            # (n_features,)
        sorted_idx = np.argsort(mean_abs)[::-1]

        importance_df = pd.DataFrame({
            "feature":    [self.feature_names[i] for i in sorted_idx[:20]],
            "importance": mean_abs[sorted_idx[:20]],
            "mean_shap":  sv[:, sorted_idx[:20]].mean(axis=0),
        })

        # SHAP over time for top 8 features
        top8_idx = sorted_idx[:8]
        shap_over_time = {
            self.feature_names[i]: sv[:, i].tolist()
            for i in top8_idx
        }

        return {
            "shap_values":     sv,
            "feature_names":   self.feature_names,
            "mean_abs_shap":   mean_abs.tolist(),
            "importance_df":   importance_df,
            "shap_over_time":  shap_over_time,
            "n_samples":       len(X_sample),
        }

    def what_if_analysis(
        self,
        X_row: np.ndarray,
        feature_name: str,
        values_to_try: list,
    ) -> list[dict]:
        """
        What-if: vary one feature across a range and show how prediction changes.

        Args:
            X_row          : baseline feature vector
            feature_name   : name of feature to vary
            values_to_try  : list of values to test

        Returns list of {"value": x, "prediction": y, "shap": z}
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found")

        feat_idx = self.feature_names.index(feature_name)
        results  = []
        X_base   = X_row.reshape(1, -1).copy()

        for v in values_to_try:
            X_mod = X_base.copy()
            X_mod[0, feat_idx] = v
            sv   = self.explainer.shap_values(X_mod)[0]
            pred = float(self.explainer.expected_value + sv.sum())
            results.append({
                "value":      float(v),
                "prediction": pred,
                "shap":       float(sv[feat_idx]),
            })

        return results

    def get_interaction_features(self, X: np.ndarray, top_n: int = 5) -> pd.DataFrame:
        """
        Find which pairs of features interact most in predictions.
        Returns DataFrame of top feature pairs by interaction strength.
        """
        if len(X) > 100:
            X = X[:100]

        try:
            sv_inter = self.explainer.shap_interaction_values(X)
            # Average absolute interaction across samples
            mean_inter = np.abs(sv_inter).mean(axis=0)
            np.fill_diagonal(mean_inter, 0)

            pairs = []
            for i in range(len(self.feature_names)):
                for j in range(i+1, len(self.feature_names)):
                    pairs.append({
                        "feature_1":   self.feature_names[i],
                        "feature_2":   self.feature_names[j],
                        "interaction": float(mean_inter[i, j]),
                    })

            return pd.DataFrame(pairs).sort_values(
                "interaction", ascending=False
            ).head(top_n).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Interaction values failed: {e}")
            return pd.DataFrame()


# ── Plotly chart builders ─────────────────────────────────────────────────────

def build_waterfall_data(explanation: dict, max_features: int = 12) -> dict:
    """
    Build data for a Plotly waterfall chart from single-prediction explanation.
    Returns dict ready to pass to plotly go.Waterfall.
    """
    sv    = np.array(explanation["shap_values"])
    fv    = np.array(explanation["feature_values"])
    names = explanation["feature_names"]
    base  = explanation["base_value"]
    pred  = explanation["prediction"]

    # Sort by absolute value, take top features
    sorted_idx = np.argsort(np.abs(sv))[::-1][:max_features]

    # Remaining features lumped together
    shown_idx  = sorted_idx
    other_sv   = sv.sum() - sv[shown_idx].sum()

    measures = []
    x_labels = []
    y_values = []

    # Base value bar
    measures.append("absolute")
    x_labels.append(f"Base value\n(avg prediction)")
    y_values.append(round(float(base), 2))

    # Each feature contribution
    for i in shown_idx:
        label = names[i]
        val   = float(fv[i])
        # Shorten long names
        if len(label) > 18:
            label = label[:16] + "…"
        feat_label = f"{label}\n= {val:.2f}"
        measures.append("relative")
        x_labels.append(feat_label)
        y_values.append(round(float(sv[i]), 4))

    # Other features
    if abs(other_sv) > 0.001:
        measures.append("relative")
        x_labels.append(f"Other features\n({len(names)-max_features} more)")
        y_values.append(round(float(other_sv), 4))

    # Final prediction
    measures.append("total")
    x_labels.append(f"Prediction\n${pred:.2f}")
    y_values.append(round(float(pred), 2))

    colors = []
    for i, (m, v) in enumerate(zip(measures, y_values)):
        if m == "absolute":    colors.append("#C4A050")
        elif m == "total":     colors.append("#C4A050")
        elif v >= 0:           colors.append("#4ADE80")
        else:                  colors.append("#F87171")

    return {
        "measures":  measures,
        "x_labels":  x_labels,
        "y_values":  y_values,
        "colors":    colors,
        "base":      float(base),
        "prediction":float(pred),
    }


def build_importance_bar_data(importance_df: pd.DataFrame, top_n: int = 15) -> dict:
    """Build data for horizontal bar chart of global feature importance."""
    df = importance_df.head(top_n).copy()
    return {
        "features":    df["feature"].tolist(),
        "importance":  [round(v, 4) for v in df["importance"].tolist()],
        "mean_shap":   [round(v, 4) for v in df["mean_shap"].tolist()],
        "colors":      ["#4ADE80" if v >= 0 else "#F87171"
                        for v in df["mean_shap"].tolist()],
    }
