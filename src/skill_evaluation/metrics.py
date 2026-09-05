"""
ASTRA — Skill Evaluation Module
=================================
Computes standard NWP verification metrics against ground truth.

Continuous metrics (all variables):
  RMSE, MAE, Bias, Correlation, Skill Score vs climatology

Categorical / binary metrics (precipitation):
  POD  — Probability of Detection  (hits / (hits + misses))
  FAR  — False Alarm Ratio         (false alarms / (hits + false alarms))
  CSI  — Critical Success Index    (hits / (hits + misses + false alarms))
  BIAS — Frequency bias            (hits + false alarms) / (hits + misses)
  F1   — Harmonic mean of precision and recall
"""
import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Continuous metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """
    Compute a suite of NWP skill metrics between a predicted field and ground truth.

    Parameters
    ----------
    pred   : 1-D or 2-D flattened array of model predictions.
    truth  : Matching array of observations / analysis values.

    Returns
    -------
    dict with keys: RMSE, MAE, Bias, Correlation, Skill Score
    """
    pred  = np.asarray(pred,  dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()

    valid = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[valid], truth[valid]

    if len(pred) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "Bias": np.nan,
                "Correlation": np.nan, "Skill Score": np.nan}

    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mae  = float(np.mean(np.abs(pred - truth)))
    bias = float(np.mean(pred - truth))
    corr = float(np.corrcoef(pred, truth)[0, 1]) if len(pred) > 1 else np.nan

    # Skill score relative to naive climatology (truth mean)
    clim_rmse   = float(np.sqrt(np.mean((truth.mean() - truth) ** 2)))
    skill_score = 1.0 - (rmse / clim_rmse) if clim_rmse > 0 else np.nan

    return {
        "RMSE":        rmse,
        "MAE":         mae,
        "Bias":        bias,
        "Correlation": corr,
        "Skill Score": skill_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Categorical / binary metrics for precipitation
# ─────────────────────────────────────────────────────────────────────────────

def compute_categorical_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    threshold: float = 2.5,
) -> dict:
    """
    Compute categorical skill metrics for binary (event / no-event) forecasts.

    Threshold is applied to both pred and truth to create binary masks.
    Default threshold = 2.5 mm corresponds to "light rain" (IMD classification).

    Parameters
    ----------
    pred      : Forecast values (same units as truth; usually mm)
    truth     : Observed values (mm)
    threshold : Event threshold in the same units

    Returns
    -------
    dict with: POD, FAR, CSI, FBIAS, F1, threshold, n_events_obs
    """
    pred  = np.asarray(pred,  dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    valid = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[valid], truth[valid]

    if len(pred) == 0:
        return {"POD": np.nan, "FAR": np.nan, "CSI": np.nan,
                "FBIAS": np.nan, "F1": np.nan,
                "threshold_mm": threshold, "n_events_obs": 0}

    f_evt = pred  >= threshold   # forecast events
    o_evt = truth >= threshold   # observed events

    hits         = int(np.sum( f_evt &  o_evt))
    misses       = int(np.sum(~f_evt &  o_evt))
    false_alarms = int(np.sum( f_evt & ~o_evt))
    # correct negatives not needed for these metrics

    pod  = hits / (hits + misses)        if (hits + misses)        > 0 else np.nan
    far  = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    csi  = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan

    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    recall    = pod
    f1 = (2 * precision * recall / (precision + recall)
          if (precision is not np.nan and recall is not np.nan
              and (precision + recall) > 0) else np.nan)

    fbias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else np.nan

    return {
        "POD":           round(float(pod),   3) if np.isfinite(pod)   else None,
        "FAR":           round(float(far),   3) if np.isfinite(far)   else None,
        "CSI":           round(float(csi),   3) if np.isfinite(csi)   else None,
        "FBIAS":         round(float(fbias), 3) if np.isfinite(fbias) else None,
        "F1":            round(float(f1),    3) if np.isfinite(f1)    else None,
        "threshold_mm":  threshold,
        "n_events_obs":  int(np.sum(o_evt)),
        "hits":          hits,
        "misses":        misses,
        "false_alarms":  false_alarms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    predictions: dict,
    truth: np.ndarray,
    categorical_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Evaluate multiple model predictions against truth and return a comparison table.

    Parameters
    ----------
    predictions           : dict of {model_name: np.ndarray}
    truth                 : np.ndarray ground truth
    categorical_threshold : If provided, also compute POD/FAR/CSI above this threshold

    Returns
    -------
    pd.DataFrame with one row per model, continuous + optional categorical columns.
    """
    rows = []
    for name, pred in predictions.items():
        metrics = compute_metrics(pred, truth)
        metrics["Model"] = name
        if categorical_threshold is not None:
            cat = compute_categorical_metrics(pred, truth, threshold=categorical_threshold)
            metrics.update(cat)
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Model")
    return df
