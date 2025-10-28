# app/model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# model feature set must match features.compute_features
MODEL_FEATURES = [
    "ret_1","ret_5","ret_20","ret_60",
    "z_close_sma20","roll_sd20","up_down_vol_ratio20",
    "close_gt_sma50","sma50_gt_sma200",
]

def _future_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0

def _target_from_future_ret(fr: pd.Series) -> pd.Series:
    # 1 if up, 0 if down or flat
    return (fr > 0).astype(int)

def train_logistic(d: pd.DataFrame, horizon: int = 1) -> LogisticRegression:
    # build target from future return
    fut_ret = _future_return(d["Close"], horizon)
    y = _target_from_future_ret(fut_ret)

    X = d[MODEL_FEATURES].copy()
    # align and drop NaN
    m = pd.concat([X, y], axis=1).dropna()
    y = m.iloc[:, -1].astype(int).values
    X = m.iloc[:, :-1].values

    # bail if too small
    if len(m) < 200:
        # tiny fit with default to avoid crashes
        # fit on whatever is there
        pass

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=200,
        n_jobs=None,
    )
    clf.fit(X, y)
    return clf

def predict_prob(model: LogisticRegression, d: pd.DataFrame) -> pd.Series:
    X = d[MODEL_FEATURES].copy().dropna()
    if X.empty:
        # no predictions
        return pd.Series([], dtype=float, index=d.index)
    probs = model.predict_proba(X.values)[:, 1]
    out = pd.Series(index=X.index, data=probs)
    # reindex to full d so downstream .iloc[-1] works
    return out.reindex(d.index).ffill()
