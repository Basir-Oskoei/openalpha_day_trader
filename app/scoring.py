
from __future__ import annotations
import pandas as pd

def blend_score(p_up: pd.Series, rules: pd.Series, w_model: float=0.6, w_rules: float=0.4) -> pd.Series:
    model_score = (p_up - 0.5) * 2.0
    final = w_model * model_score + w_rules * rules
    return final.clip(-1.0, 1.0).rename("final_score")
