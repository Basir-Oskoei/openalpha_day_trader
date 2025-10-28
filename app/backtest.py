
from __future__ import annotations
import pandas as pd
from .features import compute_features
from .model import train_logistic, predict_prob
from .rules import rules_score
from .scoring import blend_score
from .position_sizing import volatility_target_size
from .utils import annualize_return, sharpe_ratio, sortino_ratio, max_drawdown

def simple_backtest(price_df: pd.DataFrame, cfg: dict):
    sig = cfg["signals"]
    equity = cfg["risk"]["equity"]
    risk_budget = cfg["risk"]["risk_budget_per_trade"]
    max_name = cfg["risk"]["max_name_exposure"]

    d = compute_features(price_df)
    wf = cfg["backtest"]["walk_forward_window"]
    preds = []
    shares = 0
    eq = equity

    for i in range(wf, len(d)-1):
        window = d.iloc[i-wf:i].copy()
        model = train_logistic(window, horizon=1)
        cur = d.iloc[:i+1].copy()
        p_up = predict_prob(model, cur).iloc[-1]
        rules = rules_score(cur).iloc[-1]
        final = blend_score(p_up.rename(cur.index[-1]), rules.rename(cur.index[-1]), sig["model_weight"], sig["rules_weight"])

        price = cur["Close"].iloc[-1]
        atr = cur["atr14"].iloc[-1]
        target_shares = volatility_target_size(eq, risk_budget, price, atr, 2.0, max_name)

        action = "hold"
        if p_up >= sig["prob_long_threshold"] and rules > 0 and shares <= 0 and target_shares > 0:
            shares = target_shares
            action = "buy"
        elif p_up <= sig["prob_short_threshold"] and rules < 0 and shares >= 0 and target_shares > 0 and cfg["provider"]["allow_short"]:
            shares = -target_shares
            action = "sell"

        nxt = d.iloc[i+1]
        nxt_price = nxt["Close"]
        pnl = (nxt_price - price) * shares
        eq += pnl

        preds.append({"ts": cur.index[-1], "price": price, "action": action, "shares": shares,
                      "p_up": float(p_up), "rules": float(rules), "final": float(final.iloc[-1]), "equity": float(eq)})

        if len(preds) > sig["time_exit_bars"]:
            shares = 0

    dfp = pd.DataFrame(preds).set_index("ts")
    ret = dfp["equity"].pct_change().fillna(0.0)
    report = {
        "CAGR": annualize_return(ret),
        "Sharpe": sharpe_ratio(ret),
        "Sortino": sortino_ratio(ret),
        "MaxDrawdown": max_drawdown(dfp["equity"]),
        "WinRate": float((dfp["equity"].diff() > 0).mean()),
        "ProfitFactor": float(max(1e-9, dfp["equity"].diff().clip(lower=0).sum() / abs(dfp["equity"].diff().clip(upper=0).sum()))),
        "Turnover": float((dfp["shares"].abs().diff().fillna(0) > 0).mean()),
        "AvgTradeCost": float(0.0)
    }
    return {"timeline": dfp, "report": report}
