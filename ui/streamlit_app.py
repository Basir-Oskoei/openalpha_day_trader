import time
import os
import pandas as pd
import streamlit as st

from app.utils import load_config
from app.data_providers import provider_from_config
from app.features import compute_features
from app.model import train_logistic, predict_prob
from app.rules import rules_score
from app.scoring import blend_score
from app.position_sizing import volatility_target_size
from app.backtest import simple_backtest

st.set_page_config(page_title="OpenAlpha Day Trader", layout="wide")

@st.cache_resource
def get_cfg():
    return load_config("configs/config.yaml")

cfg = get_cfg()
st.title("OpenAlpha Day Trader")
tab_live, tab_bt, tab_settings = st.tabs(["Live", "Backtest", "Settings"])

# -------------------- Settings --------------------
with tab_settings:
    interval = st.number_input(
        "Scan interval seconds", min_value=5, max_value=600,
        value=int(cfg["provider"]["interval_seconds"]), step=5
    )
    allow_short = st.checkbox("Allow short", value=bool(cfg["provider"]["allow_short"]))

    equity = st.number_input("Equity dollars (or GBP, same math)", min_value=1.0, value=89.0, step=1.0)
    risk_budget = st.number_input(
        "Risk budget per trade (fraction of equity)",
        min_value=0.001, max_value=1.0, value=0.50, step=0.01, format="%.3f",
        help="0.50 means you’re willing to risk 50% of equity on one trade. High risk."
    )
    max_name = st.number_input(
        "Max name exposure (fraction of equity)",
        min_value=0.0, max_value=1.0, value=1.00, step=0.05, format="%.2f"
    )

    prob_long = st.number_input("Prob long threshold", min_value=0.50, max_value=0.70,
                                value=float(cfg["signals"]["prob_long_threshold"]), step=0.01)
    prob_short = st.number_input("Prob short threshold", min_value=0.30, max_value=0.50,
                                 value=float(cfg["signals"]["prob_short_threshold"]), step=0.01)

    model_w = st.slider("Model weight", 0.0, 1.0, float(cfg["signals"]["model_weight"]), 0.05)
    rules_w = st.slider("Rules weight", 0.0, 1.0, float(cfg["signals"]["rules_weight"]), 0.05)
    st.caption("Tip: model_w + rules_w ≈ 1 is fine, but any combo is allowed.")

    bar_interval = st.selectbox("Bar interval", ["1m", "5m", "1d"], index=1)

    uploaded = st.file_uploader("Universe CSV with header symbol", type=["csv"])
    if uploaded:
        uni_df = pd.read_csv(uploaded)
    else:
        uni_df = pd.read_csv("configs/universe.csv")
    st.write("Universe size", len(uni_df))
    st.dataframe(uni_df.head(20))

# -------------------- Live --------------------
with tab_live:
    provider = provider_from_config({
        "provider": {
            "name": "twelvedata",
            "allow_short": allow_short,
            "api_key": os.getenv("TWELVEDATA_KEY")
        }
    })

    symbols = uni_df["symbol"].tolist()
    run = st.toggle("Run scheduler", value=False)
    holder = st.empty()

    def scan_once() -> pd.DataFrame:
        rows = []
        need_rows = 220 if bar_interval != "1d" else 60

        for s in symbols[:40]:
            bars = provider.get_recent_bars(s, lookback=2500, interval=bar_interval)
            if bars is None or bars.empty:
                continue

            feats = compute_features(bars)
            if feats is None or feats.empty or len(feats) < need_rows or "Close" not in feats.columns:
                continue

            model = train_logistic(feats, horizon=1)
            p_up = predict_prob(model, feats)
            rules = rules_score(feats)
            final = blend_score(p_up, rules, model_w, rules_w)

            atr = feats["atr14"].iloc[-1]
            price = feats["Close"].iloc[-1]
            size = volatility_target_size(equity, risk_budget, price, atr, 2.0, max_name)

            rows.append({
                "symbol": s,
                "price": float(price),
                "p_up": float(p_up.iloc[-1]),
                "rules": float(rules.iloc[-1]),
                "final": float(final.iloc[-1]),
                "size": float(size),
            })

        if not rows:
            return pd.DataFrame(columns=["symbol", "price", "p_up", "rules", "final", "size"])

        return pd.DataFrame(rows).sort_values("final", ascending=False).reset_index(drop=True)

    while True:
        df = scan_once()
        with holder.container():
            if df.empty:
                st.warning("No data right now. Provider may be delayed or interval has no values.")
            else:
                top = df.iloc[0]
                action = (
                    "buy" if (top["p_up"] >= prob_long and top["rules"] > 0)
                    else ("sell" if (allow_short and top["p_up"] <= prob_short and top["rules"] < 0)
                          else "hold")
                )
                st.metric("Top pick", f"{top['symbol']} {action.upper()}  size {top['size']:.2f}")
                st.caption(f"Confidence {top['final']:.2f} • Prob up {top['p_up']:.2f} • Rules {top['rules']:.2f}")
                st.dataframe(df.head(10), use_container_width=True)

                sel = st.selectbox("Chart ticker", df["symbol"].tolist(), index=0)
                bars = provider.get_recent_bars(sel, lookback=1500, interval=bar_interval)
                if bars is None or bars.empty:
                    st.warning(f"No chart data for {sel} (provider returned empty).")
                else:
                    feats = compute_features(bars)
                    if feats is None or feats.empty or "Close" not in feats.columns:
                        st.warning(f"Data for {sel} missing OHLCV columns. Try switching interval.")
                    else:
                        st.line_chart(feats[["Close", "sma20", "sma50", "sma200"]].tail(400), use_container_width=True)

        if not run:
            break
        time.sleep(interval)
        st.rerun()

# -------------------- Backtest --------------------
with tab_bt:
    bt_symbol = st.selectbox("Backtest symbol", ["SPY", "QQQ", "AAPL", "MSFT", "IWM", "DIA"])
    provider_bt = provider_from_config({
        "provider": {
            "name": "twelvedata",
            "allow_short": True,
            "api_key": os.getenv("TWELVEDATA_KEY")
        }
    })
    bars = provider_bt.get_recent_bars(bt_symbol, lookback=20000, interval="1d")
    if bars is None or bars.empty:
        st.warning("No data for backtest")
    else:
        res = simple_backtest(bars, cfg)
        st.write({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in res["report"].items()})
        st.line_chart(res["timeline"]["equity"])
        st.area_chart((res["timeline"]["equity"] / res["timeline"]["equity"].cummax() - 1.0))
        st.dataframe(res["timeline"].tail(20))
