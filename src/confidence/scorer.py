"""
ASTRA — Confidence Scoring Module
==================================
Computes a calibrated forecast confidence score (0–100%) for the blended output.

Confidence is derived from:
  1. Model Agreement  — how closely ECMWF HRES and GFS agree (lower spread → higher confidence)
  2. Ensemble Spread  — width of the ensemble uncertainty envelope
  3. Lead-time Decay  — confidence degrades predictably with forecast horizon
  4. Skill History    — if skill CSV available, past accuracy boosts confidence

Usage:
    from src.confidence.scorer import ConfidenceScorer
    scorer = ConfidenceScorer()
    score, breakdown = scorer.compute(ds, lead_hours=24, skill_df=df)
"""
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional


class ConfidenceScorer:
    """
    Produces a per-variable, per-lead-time forecast confidence score.

    Returns:
        score (float): 0–100 overall confidence
        level (str): 'HIGH' | 'MEDIUM' | 'LOW'
        breakdown (dict): individual component scores for explainability
    """

    # Lead-time decay constants (exponential): confidence at T+0 = 100%, at T+72 ≈ 55%
    _LEAD_DECAY_HALF = 72.0   # hours for ~50% decay

    def __init__(
        self,
        model_agree_weight: float = 0.40,
        spread_weight: float = 0.25,
        lead_decay_weight: float = 0.20,
        skill_weight: float = 0.15,
    ):
        total = model_agree_weight + spread_weight + lead_decay_weight + skill_weight
        self.w_agree  = model_agree_weight / total
        self.w_spread = spread_weight / total
        self.w_lead   = lead_decay_weight / total
        self.w_skill  = skill_weight / total

    # ─────────────────────────────────────────────────────────────────────────
    def _model_agreement(
        self,
        ds: xr.Dataset,
        var: str,
    ) -> float:
        """
        Score [0–100] measuring how closely HRES and GFS agree on the forecast.
        High agreement → high score. Normalised by the mean field magnitude.
        """
        hres_col = f"{var}_hres"
        gfs_col  = f"{var}_gfs"
        if hres_col not in ds or gfs_col not in ds:
            return 70.0  # neutral fallback

        hres = ds[hres_col].values.ravel()
        gfs  = ds[gfs_col].values.ravel()
        valid = np.isfinite(hres) & np.isfinite(gfs)
        if valid.sum() < 10:
            return 70.0

        diff   = np.abs(hres[valid] - gfs[valid])
        scale  = np.abs(hres[valid]) + np.abs(gfs[valid]) + 1e-9
        rel_diff = np.nanmean(diff / scale)          # 0 = perfect agreement

        score = float(np.clip(100.0 * (1.0 - rel_diff), 0, 100))
        return round(score, 1)

    def _spread_score(
        self,
        ds: xr.Dataset,
        var: str,
    ) -> float:
        """
        Score [0–100] based on ensemble spread size.
        Zero spread → 100; spread → ∞ → 0.
        Uses a soft normalisation so small spreads score well.
        """
        spread_col = f"{var}_spread"
        if spread_col not in ds:
            return 65.0  # neutral fallback (no ensemble available)

        spread = ds[spread_col].values.ravel()
        valid  = np.isfinite(spread)
        if valid.sum() < 10:
            return 65.0

        mean_spread  = float(np.nanmean(np.abs(spread[valid])))
        # Normalise by field magnitude
        blended_col = f"{var}_blended"
        if blended_col in ds:
            mean_field = float(np.nanmean(np.abs(ds[blended_col].values))) + 1e-9
        else:
            mean_field = 1.0

        rel_spread = mean_spread / mean_field
        score = float(np.clip(100.0 * np.exp(-2.0 * rel_spread), 0, 100))
        return round(score, 1)

    def _lead_time_score(self, lead_hours: int) -> float:
        """
        Score [0–100] that decays exponentially with forecast horizon.
        T+0h → 100,  T+72h → ~50
        """
        score = 100.0 * np.exp(-0.693 * lead_hours / self._LEAD_DECAY_HALF)
        return round(float(np.clip(score, 0, 100)), 1)

    def _skill_score(self, skill_df: Optional[pd.DataFrame], model: str = "ASTRA Blend") -> float:
        """
        Score [0–100] derived from historical RMSE skill score vs climatology.
        Skill Score close to 1.0 → 100.
        """
        if skill_df is None or model not in skill_df.index:
            return 70.0  # neutral fallback

        ss = skill_df.loc[model, "Skill Score"] if "Skill Score" in skill_df.columns else None
        if ss is None or not np.isfinite(ss):
            return 70.0

        # Skill score ∈ (−∞, 1]; map 0→50, 1→100, negative→0
        score = float(np.clip(50.0 + 50.0 * ss, 0, 100))
        return round(score, 1)

    # ─────────────────────────────────────────────────────────────────────────
    def compute(
        self,
        ds: xr.Dataset,
        var: str = "tp",
        lead_hours: int = 24,
        skill_df: Optional[pd.DataFrame] = None,
    ) -> tuple[float, str, dict]:
        """
        Compute blended confidence score.

        Parameters
        ----------
        ds          : Blended forecast xr.Dataset
        var         : Target variable ('tp', 't2m', 'wind')
        lead_hours  : Forecast horizon in hours
        skill_df    : Historical skill DataFrame from metrics.compare_models()

        Returns
        -------
        (score, level, breakdown)
          score     : float 0–100
          level     : 'HIGH' | 'MEDIUM' | 'LOW'
          breakdown : dict with component scores
        """
        agree  = self._model_agreement(ds, var)
        spread = self._spread_score(ds, var)
        lead   = self._lead_time_score(lead_hours)
        skill  = self._skill_score(skill_df)

        score = (
            self.w_agree  * agree  +
            self.w_spread * spread +
            self.w_lead   * lead   +
            self.w_skill  * skill
        )
        score = round(float(np.clip(score, 0, 100)), 1)

        if score >= 80:
            level = "HIGH"
        elif score >= 60:
            level = "MEDIUM"
        else:
            level = "LOW"

        breakdown = {
            "Model Agreement":  agree,
            "Ensemble Spread":  spread,
            "Lead-Time Factor": lead,
            "Historical Skill": skill,
            "Overall":          score,
            "Level":            level,
        }
        return score, level, breakdown
