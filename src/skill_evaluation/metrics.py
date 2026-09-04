"""
ASTRA — Skill Evaluation Module
Computes standard NWP verification metrics against ground truth.
"""
import numpy as np
import pandas as pd


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """
    Compute a suite of NWP skill metrics between a predicted field and ground truth.

    Parameters
    ----------
    pred   : 1-D or 2-D flattened array of model predictions.
    truth  : Matching array of observations / analysis values.

    Returns
    -------
    dict with keys: rmse, mae, bias, corr, skill_score_vs_climatology
    """
    pred = np.asarray(pred, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()

    valid = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[valid], truth[valid]

    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mae  = float(np.mean(np.abs(pred - truth)))
    bias = float(np.mean(pred - truth))

    corr = float(np.corrcoef(pred, truth)[0, 1]) if len(pred) > 1 else np.nan

    # Skill score relative to a naive climatology (truth mean)
    clim_rmse = float(np.sqrt(np.mean((truth.mean() - truth) ** 2)))
    skill_score = 1.0 - (rmse / clim_rmse) if clim_rmse > 0 else np.nan

    return {
        "RMSE":         rmse,
        "MAE":          mae,
        "Bias":         bias,
        "Correlation":  corr,
        "Skill Score":  skill_score,
    }


def compare_models(predictions: dict, truth: np.ndarray) -> pd.DataFrame:
    """
    Evaluate multiple model predictions against truth and return a comparison table.

    Parameters
    ----------
    predictions : dict of {model_name: np.ndarray}
    truth       : np.ndarray ground truth

    Returns
    -------
    pd.DataFrame with one row per model.
    """
    rows = []
    for name, pred in predictions.items():
        metrics = compute_metrics(pred, truth)
        metrics["Model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("Model")
    return df
