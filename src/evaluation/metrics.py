"""Model evaluation metrics and SHAP explainability.

Provides calibration analysis, custom cricket-specific metrics,
and SHAP-based model explanations.

Usage:
    from src.evaluation.metrics import evaluate_calibration, explain_prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import settings
from src.utils.logger import logger


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
) -> dict:
    """Evaluate probability calibration — critical for prediction markets.

    A well-calibrated model: when it says 70% win probability,
    that team should win ~70% of the time.

    Returns:
        Dictionary with Brier score and calibration curve data.
    """
    brier = brier_score_loss(y_true, y_prob)
    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(mean_predicted, fraction_pos, "s-", label="Model", color="#2196F3")
    ax.plot([0, 1], [0, 1], "--", label="Perfectly Calibrated", color="#9E9E9E")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Actual Win Fraction")
    ax.set_title(f"Calibration Plot (Brier Score: {brier:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Calibration plot saved to {save_path}")
    plt.close()

    return {
        "brier_score": round(brier, 4),
        "fraction_positive": fraction_pos.tolist(),
        "mean_predicted": mean_predicted.tolist(),
    }


def explain_prediction(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    match_label: str = "Match",
    save_path: Optional[Path] = None,
) -> dict:
    """Generate SHAP explanation for a prediction.

    Shows which features pushed the prediction up or down.

    Returns:
        Dictionary with top positive and negative contributing features.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping explanation")
        return {"error": "SHAP not installed"}

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if len(X) == 1:
        # Single prediction explanation
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        importance = pd.DataFrame({
            "feature": feature_names,
            "shap_value": sv,
            "abs_shap": np.abs(sv),
        }).sort_values("abs_shap", ascending=False)

        top_positive = importance[importance["shap_value"] > 0].head(3)
        top_negative = importance[importance["shap_value"] < 0].head(3)

        # SHAP waterfall plot
        if save_path:
            fig = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv,
                    base_values=explainer.expected_value,
                    data=X.iloc[0].values,
                    feature_names=feature_names,
                ),
                show=False,
            )
            plt.title(f"SHAP Explanation: {match_label}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return {
            "key_positive_factors": top_positive["feature"].tolist(),
            "key_negative_factors": top_negative["feature"].tolist(),
            "top_features": importance.head(10).to_dict("records"),
        }

    return {"shap_values": shap_values}
