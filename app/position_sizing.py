
from __future__ import annotations
import numpy as np

def volatility_target_size(equity: float, risk_budget: float, price: float, atr: float, atr_mult: float=2.0, max_name_exposure: float=0.05):
    per_share_risk = atr_mult * atr
    if per_share_risk <= 0 or np.isnan(per_share_risk):
        return 0
    dollar_risk = equity * risk_budget
    shares = int(dollar_risk / per_share_risk)
    max_shares_by_exposure = int((equity * max_name_exposure) / price)
    return int(max(0, min(shares, max_shares_by_exposure)))
