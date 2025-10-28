
from __future__ import annotations
import numpy as np, pandas as pd

def rules_score(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    trend = 0.0
    trend += 1.0 * (d["close_gt_sma50"] == 1)
    trend += 1.0 * (d["sma50_gt_sma200"] == 1)
    trend += 1.0 * (d["ret_20"] > 0)
    trend -= 1.0 * (d["ret_20"] < 0)
    mr = 0.0
    mr += 1.0 * (d["z_close_sma20"] < -2.0)
    mr -= 1.0 * (d["z_close_sma20"] > 2.0)
    raw = trend + mr
    score = np.tanh(raw / 3.0)
    return pd.Series(score, index=df.index, name="rules_score")
