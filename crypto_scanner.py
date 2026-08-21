#!/usr/bin/env python3
"""
Crypto swing scanner — same six signals and trailing-stop strategy as the stock
scanner, on liquid coins, with its OWN separate paper book.

Data: daily OHLC from yfinance (BTC-USD, ETH-USD, ...). Works in GitHub Actions;
the sandbox here can't reach market data, so develop with --offline.

Why a separate book: crypto is far more volatile and trades 24/7, so mixing it
into the stock account would muddy both. Keeping it separate lets you see whether
crypto works on its own merits.

Honest note carried from the stock work: a signal firing is a statistical tilt,
not a prediction — and crypto history is short and cycle-driven, so its backtests
overfit even more easily than equities. Trust the backtest, not the folklore.

Usage:
  python crypto_scanner.py                       # scan liquid majors
  python crypto_scanner.py --offline data_crypto/
NOT financial advice.
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import portfolio as pf
from scanner import CONFIG, add_indicators, compute_signals
from stock_scanner import (download_universe, load_offline, plain_candidate,
                           pooled_stats, write_html)

# Liquid, non-stablecoin, non-wrapped coins with reliable yfinance -USD history.
# Stablecoins (USDT/USDC/DAI) and wrapped tokens are deliberately excluded.
MAJORS = ("BTC ETH SOL XRP BNB ADA DOGE AVAX LINK DOT LTC BCH XLM ATOM UNI ETC "
          "FIL APT NEAR ICP IMX INJ HBAR VET ALGO AAVE MKR GRT SAND MANA AXS "
          "XTZ EOS FLOW CHZ CRV SNX SUI PEPE SHIB").split()

CRYPTO_CFG = {
    "history_years": 8,
    "min_dollar_vol": 30e6,     # daily $ volume floor (majors clear this easily)
    "min_history_days": 200,    # newer coins allowed, but need some history
    "top_n": 10,
}


def get_crypto_universe() -> list[str]:
    return [f"{c}-USD" for c in MAJORS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", metavar="DIR", help="read <TICKER>.csv files from DIR")
    ap.add_argument("--html", default="crypto.html")
    ap.add_argument("--portfolio-html", default="crypto_portfolio.html")
    ap.add_argument("--portfolio-txt", default="crypto_portfolio_digest.txt")
    ap.add_argument("--no-portfolio", action="store_true")
    args = ap.parse_args()

    # separate paper book + shadow log for crypto
    pf.STATE_PATH = "state/crypto_paper.json"
    pf.SHADOW_PATH = "state/crypto_shadow.json"

    if args.offline:
        data = load_offline(args.offline)
    else:
        # temporarily relax the history floor for younger coins
        import stock_scanner
        stock_scanner.STOCK_CFG["min_history_days"] = CRYPTO_CFG["min_history_days"]
        data = download_universe(get_crypto_universe(), CRYPTO_CFG["history_years"])

    processed = {}
    for t, raw in data.items():
        df = add_indicators(raw)
        processed[t] = (df, compute_signals(df))
    if not processed:
        print("No crypto data available.")
        return

    stats = pooled_stats(processed)
    wmap = dict(zip(stats["signal"], stats["weight"]))
    amap = dict(zip(stats["signal"], stats[f"avg_{CONFIG['key_horizon']}d"]))
    total_w = sum(wmap.values()) or 1.0
    med_mom = np.nanmedian([float(df["MOM63"].iloc[-1]) for df, _ in processed.values()])

    candidates = []
    for t, (df, signals) in processed.items():
        fired = {n: int(s.iloc[-1]) for n, s in signals.items() if s.iloc[-1] != 0}
        if not fired:
            continue
        dvol = float((df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]) \
            if "Volume" in df else np.inf
        if np.isfinite(dvol) and 0 < dvol < CRYPTO_CFG["min_dollar_vol"]:
            continue
        score = sum(d * wmap.get(n, 0) for n, d in fired.items()) / total_w
        if pf.PORTFOLIO.get("long_only") and score <= 0:
            continue
        if score == 0:
            continue
        close = float(df["Close"].iloc[-1])
        a = float(df["ATR"].iloc[-1])
        candidates.append({
            "ticker": t.replace("-USD", ""), "side": "LONG" if score > 0 else "SHORT",
            "score": round(abs(score), 2), "close": round(close, 4),
            "signals": ", ".join(f"{n} ({'L' if d > 0 else 'S'})" for n, d in fired.items()),
            "RS_3m_%": round(100 * (float(df["MOM63"].iloc[-1]) - med_mom), 1),
            "ATR%": round(100 * a / close, 2),
            "stop": round(close - np.sign(score) * CONFIG["stop_atr_mult"] * a, 4),
            "min_KO_dist": round(CONFIG["ko_atr_mult"] * a, 4),
            "_fired": fired, "earnings": "n/a",
            "expected_pct": round(float(np.mean([abs(amap.get(n, 0.0)) for n in fired])), 3) or 0.1,
        })

    cand = pd.DataFrame(candidates)
    top = pd.DataFrame()
    if len(cand):
        cand = cand.sort_values("score", ascending=False)
        top = pd.concat([cand[cand["side"] == "LONG"].head(CRYPTO_CFG["top_n"]),
                         cand[cand["side"] == "SHORT"].head(CRYPTO_CFG["top_n"])])

    date = max(df.index[-1] for df, _ in processed.values()).strftime("%Y-%m-%d")
    display = top.drop(columns=["_fired"]) if len(top) else top
    print(f"\nCRYPTO CANDIDATES — {date} (universe: {len(processed)} coins)")
    print(display.to_string(index=False) if len(top) else "No candidates today.")

    write_html(args.html, date, len(processed), top, display, stats)
    print(f"Written {args.html}")

    if not args.no_portfolio:
        market = {c["ticker"]: {"close": c["close"]} for c in candidates}
        for t, (df, _) in processed.items():
            k = t.replace("-USD", "")
            market[k] = {"close": float(df["Close"].iloc[-1]),
                         "open": float(df["Open"].iloc[-1]),
                         "high": float(df["High"].iloc[-1]),
                         "low": float(df["Low"].iloc[-1]),
                         "atr": float(df["ATR"].iloc[-1]),
                         "rsi": float(df["RSI"].iloc[-1])}
        perf = pf.run(top if len(top) else None, market, date,
                      path_html=args.portfolio_html, path_txt=args.portfolio_txt,
                      all_candidates=cand if len(cand) else None)
        if perf.get("n"):
            print(f"Crypto paper: {perf['n']} closed, {perf['net']:+.2f} net")
        print(f"Written {args.portfolio_html}")


if __name__ == "__main__":
    main()
