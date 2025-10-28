# app/data_providers.py
from __future__ import annotations
import os
from typing import Optional
import pandas as pd
import requests

# --- Base Provider ----------------------------------------------------------
class BaseProvider:
    name = "base"
    delayed = False

    def get_recent_bars(self, symbol: str, lookback: int, interval: str = "1d") -> pd.DataFrame:
        raise NotImplementedError


# --- Helpers ----------------------------------------------------------------
def _map_interval(interval: str) -> str:
    """Map internal intervals to Twelve Data-compatible ones."""
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1d": "1day"}
    return mapping.get(interval, "5min")


def _to_ohlcv(js: dict) -> pd.DataFrame:
    """Convert Twelve Data JSON into a clean OHLCV DataFrame."""
    if not js or "values" not in js:
        return pd.DataFrame()

    df = pd.DataFrame(js["values"])
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).set_index("Date")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


# --- Twelve Data Provider ---------------------------------------------------
class TwelveDataProvider(BaseProvider):
    name = "twelvedata"
    delayed = False

    def __init__(self, api_key: Optional[str] = None):
        # Use explicit key if passed, else fall back to environment
        self.api_key = api_key or os.getenv("TWELVEDATA_KEY")
        if not self.api_key:
            raise RuntimeError(
                'TWELVEDATA_KEY is not set. In PowerShell run:\n'
                '$env:TWELVEDATA_KEY = "YOUR_KEY_HERE"'
            )

    def get_recent_bars(self, symbol: str, lookback: int, interval: str = "5m") -> pd.DataFrame:
        url = "https://api.twelvedata.com/time_series"
        itv = _map_interval(interval)
        params = {
            "symbol": symbol,
            "interval": itv,
            "outputsize": min(max(lookback, 50), 5000),
            "apikey": self.api_key,
            "order": "asc",
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            js = r.json()
            df = _to_ohlcv(js)
            if df.empty and "message" in js:
                print(f"Twelve Data error for {symbol}: {js.get('message')}")
            return df
        except Exception as e:
            print(f"Twelve Data fetch failed for {symbol}: {e}")
            return pd.DataFrame()


# --- Provider Factory -------------------------------------------------------
def provider_from_config(cfg: dict) -> BaseProvider:
    """Factory: choose provider from config dict."""
    p = (cfg.get("provider") or {})
    name = (p.get("name") or "twelvedata").lower()
    api_key = p.get("api_key") or os.getenv("TWELVEDATA_KEY")

    if name == "twelvedata":
        return TwelveDataProvider(api_key=api_key)

    # Default fallback
    return TwelveDataProvider(api_key=api_key)
