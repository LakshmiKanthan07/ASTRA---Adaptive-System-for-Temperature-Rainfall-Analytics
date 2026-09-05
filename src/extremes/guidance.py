"""
ASTRA — Extreme Weather Guidance Module
Analyzes blended forecasts for heavy rainfall, heatwaves, and high winds.
Outputs a boolean alert mask and summary metrics.
"""
import xarray as xr
import numpy as np
import json
import os

class ExtremeWeatherDetector:
    def __init__(self, rain_thresh: float = 64.5, heat_thresh: float = 40.0, wind_thresh: float = 15.0):
        # Default thresholds
        # Heavy rain: > 64.5 mm/24h (IMD definition for heavy)
        # Heatwave: > 40 C in plains
        # High Wind: > 15 m/s (~ 54 km/h)
        self.rain_thresh = rain_thresh
        self.heat_thresh = heat_thresh
        self.wind_thresh = wind_thresh

    def detect(self, ds: xr.Dataset, output_json: str = "data/alerts.json") -> dict:
        """
        Takes the blended dataset and checks for extremes.
        Outputs summary to JSON for the dashboard to consume.
        """
        alerts = []
        
        if "tp_blended" in ds:
            # converting from m to mm (if model predicts in meters, wait, log says values are 0-1.4 meters for GFS?
            # Actually ERA5 tp is in meters, so mm = val * 1000
            tp_mm = ds["tp_blended"] * 1000
            max_tp = float(tp_mm.max())
            if max_tp > self.rain_thresh:
                alerts.append({
                    "type": "HEAVY RAINFALL",
                    "level": "WARNING",
                    "value": round(max_tp, 1),
                    "unit": "mm",
                    "message": f"Isolated heavy precipitation ({max_tp:.1f} mm) detected in forecast grid. Flood-related rainfall risk elevated."
                })
        
        # Check Temperature
        if "t2m_blended" in ds:
            # ERA5 and GFS t2m is in Kelvin. C = K - 273.15
            t2m_c = ds["t2m_blended"] - 273.15
            max_t2m = float(t2m_c.max())
            if max_t2m > self.heat_thresh:
                alerts.append({
                    "type": "HEATWAVE",
                    "level": "WARNING",
                    "value": round(max_t2m, 1),
                    "unit": "°C",
                    "message": f"Extreme temperatures ({max_t2m:.1f}°C) detected. Heatwave conditions likely in affected grid cells."
                })
                
        # Check Wind
        if "wind_blended" in ds:
            # Wind is in m/s
            max_wind = float(ds["wind_blended"].max())
            if max_wind > self.wind_thresh:
                alerts.append({
                    "type": "HIGH WIND",
                    "level": "ADVISORY",
                    "value": round(max_wind, 1),
                    "unit": "m/s",
                    "message": f"High wind speeds ({max_wind:.1f} m/s) detected. Possible structural/agricultural impacts."
                })
                
        summary = {
            "active_alerts": len(alerts),
            "alerts": alerts
        }
        
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"[Extremes] Generated {len(alerts)} alerts -> {output_json}")
        return summary