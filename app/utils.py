
from __future__ import annotations
import numpy as np, pandas as pd, yaml

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def annualize_return(daily_ret: pd.Series) -> float:
    mu = daily_ret.mean()
    return float((1.0 + mu) ** 252 - 1.0)

def sharpe_ratio(daily_ret: pd.Series) -> float:
    mu = daily_ret.mean() * np.sqrt(252)
    sd = daily_ret.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(mu / sd)

def sortino_ratio(daily_ret: pd.Series) -> float:
    downside = daily_ret[daily_ret < 0.0]
    dd = downside.std(ddof=0)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(daily_ret.mean() * np.sqrt(252) / dd)

def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1.0
    return float(dd.min())
