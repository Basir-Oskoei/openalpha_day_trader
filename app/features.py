# app/features.py
import numpy as np
import pandas as pd

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with the exact columns the model expects:
      Close, sma20, sma50, sma200, atr14,
      ret_1, ret_5, ret_20, ret_60,
      z_close_sma20, roll_sd20, up_down_vol_ratio20,
      close_gt_sma50, sma50_gt_sma200
    Index is time.
    """
    d = bars.copy()

    # sanity on columns
    needed = {"Open","High","Low","Close","Volume"}
    missing = needed - set(d.columns)
    if missing:
        return pd.DataFrame()

    # moving averages
    d["sma20"] = d["Close"].rolling(20, min_periods=20).mean()
    d["sma50"] = d["Close"].rolling(50, min_periods=50).mean()
    d["sma200"] = d["Close"].rolling(200, min_periods=200).mean()

    # returns
    d["ret_1"]   = d["Close"].pct_change(1)
    d["ret_5"]   = d["Close"].pct_change(5)
    d["ret_20"]  = d["Close"].pct_change(20)
    d["ret_60"]  = d["Close"].pct_change(60)

    # z score of close vs sma20 using 20 day rolling std of close
    roll_sd20 = d["Close"].rolling(20, min_periods=20).std()
    d["roll_sd20"] = roll_sd20
    d["z_close_sma20"] = (d["Close"] - d["sma20"]) / roll_sd20

    # up vs down volume ratio over 20 bars
    up_vol = np.where(d["ret_1"] >= 0, d["Volume"], 0.0)
    dn_vol = np.where(d["ret_1"] <  0, d["Volume"], 0.0)
    up_20 = pd.Series(up_vol, index=d.index).rolling(20, min_periods=20).sum()
    dn_20 = pd.Series(dn_vol, index=d.index).rolling(20, min_periods=20).sum()
    d["up_down_vol_ratio20"] = up_20 / (up_20 + dn_20)

    # simple state flags
    d["close_gt_sma50"] = (d["Close"] > d["sma50"]).astype(float)
    d["sma50_gt_sma200"] = (d["sma50"] > d["sma200"]).astype(float)

    # ATR for sizing
    d["atr14"] = _atr(d, 14)

    # clean up
    cols = [
        "Close","sma20","sma50","sma200","atr14",
        "ret_1","ret_5","ret_20","ret_60",
        "z_close_sma20","roll_sd20","up_down_vol_ratio20",
        "close_gt_sma50","sma50_gt_sma200",
    ]
    d = d[cols].dropna()
    return d
