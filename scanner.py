#!/usr/bin/env python3
"""
Daily swing-signal scanner for Nasdaq-100 (NDX) and Russell 2000 (RUT).

Horizon: days to a couple of weeks. Long and short.

What it does each run:
  1. Downloads ~15y of daily data (yfinance, fallback: stooq, fallback: local CSV).
  2. Computes a set of classic swing signals (mean reversion + breakout).
  3. For EVERY signal it shows you its historical edge on that index:
     how often it fired, win rate, and average forward return vs. baseline.
     A signal with no historical edge gets ~zero weight automatically.
  4. Combines fired signals into a net LONG / SHORT / FLAT bias per index.
  5. Prints a risk block: ATR, suggested stop distance, knock-out buffer,
     and a volatility-regime size multiplier.
  6. Writes an HTML report (signals_report.html) you can glance at daily.

Usage:
  pip install yfinance pandas numpy
  python scanner.py                     # fetch data and scan
  python scanner.py --offline data/     # use local CSVs (NDX.csv, RUT.csv, VIX.csv)

NOT financial advice. Historical stats are not a guarantee of anything.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG — everything you might want to tweak lives here
# ----------------------------------------------------------------------------
CONFIG = {
    "indices": {
        # name: (yfinance ticker, stooq symbol)
        "NDX": ("^NDX", "^ndx"),
        "RUT": ("^RUT", "^rut"),
    },
    "vix_ticker": "^VIX",          # regime gauge; falls back to realized vol
    "history_years": 15,           # lookback for signal statistics
    "horizons": [3, 5, 10],        # forward-return horizons (trading days)
    "key_horizon": 5,              # horizon used for weighting the composite
    "bias_threshold": 0.25,        # |score| above this => LONG/SHORT, else FLAT
    # signal parameters
    "rsi_len": 2,
    "rsi_low": 10, "rsi_high": 90,
    "ibs_low": 0.15, "ibs_high": 0.85,
    "nday_extreme": 7,             # n-day low/high pullback signal
    "donchian_len": 20,            # breakout channel length
    "sweep_len": 10,               # liquidity sweep: prior n-day high/low level
    "fvg_expiry": 10,              # FVG zone valid for n bars after creation
    "sma_fast": 50, "sma_slow": 200,
    "atr_len": 14,
    # risk
    "stop_atr_mult": 2.0,          # suggested stop distance for 1-2 week holds
    "ko_atr_mult": 3.0,            # minimum knock-out barrier distance
    "high_vol_pctile": 80,         # vol regime percentile => half size
    "risk_per_trade": 0.01,        # 1% of account risked per trade (for sizing hint)
}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_yfinance(ticker: str, years: int) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{years}y", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 300:
            return None
        if isinstance(df.columns, pd.MultiIndex):       # yfinance >= 0.2.x quirk
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.title)[["Open", "High", "Low", "Close"]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna()
    except Exception as e:
        print(f"  yfinance failed for {ticker}: {e}")
        return None


def load_stooq(symbol: str, years: int) -> pd.DataFrame | None:
    try:
        from urllib.parse import quote
        url = f"https://stooq.com/q/d/l/?s={quote(symbol)}&i=d"
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        if len(df) < 300:
            return None
        df = df[["Open", "High", "Low", "Close"]].dropna()
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        return df[df.index >= cutoff]
    except Exception as e:
        print(f"  stooq failed for {symbol}: {e}")
        return None


def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df[["Open", "High", "Low", "Close"]].dropna()


def get_data(offline_dir: str | None) -> tuple[dict[str, pd.DataFrame], pd.Series | None]:
    """Returns ({index_name: OHLC df}, vix_close_series_or_None)."""
    cfg = CONFIG
    data, vix = {}, None
    for name, (yf_t, stooq_t) in cfg["indices"].items():
        df = None
        if offline_dir:
            df = load_csv(os.path.join(offline_dir, f"{name}.csv"))
        else:
            df = load_yfinance(yf_t, cfg["history_years"]) or load_stooq(stooq_t, cfg["history_years"])
        if df is None:
            sys.exit(f"ERROR: no data for {name}. Try --offline with CSVs (Date,Open,High,Low,Close).")
        data[name] = df
    if offline_dir:
        v = load_csv(os.path.join(offline_dir, "VIX.csv"))
        vix = v["Close"] if v is not None else None
    else:
        v = load_yfinance(cfg["vix_ticker"], cfg["history_years"])
        vix = v["Close"] if v is not None else None
    return data, vix


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = CONFIG
    out = df.copy()
    out["SMA_F"] = out["Close"].rolling(c["sma_fast"]).mean()
    out["SMA_S"] = out["Close"].rolling(c["sma_slow"]).mean()
    out["RSI"] = rsi(out["Close"], c["rsi_len"])
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    out["IBS"] = ((out["Close"] - out["Low"]) / rng).fillna(0.5)
    out["ATR"] = atr(out, c["atr_len"])
    n = c["nday_extreme"]
    out["NLOW"] = out["Close"] == out["Close"].rolling(n).min()
    out["NHIGH"] = out["Close"] == out["Close"].rolling(n).max()
    d = c["donchian_len"]
    out["DON_HI"] = out["Close"].rolling(d).max().shift()
    out["DON_LO"] = out["Close"].rolling(d).min().shift()
    out["MOM63"] = out["Close"].pct_change(63)
    out["RVOL"] = out["Close"].pct_change().rolling(21).std() * np.sqrt(252)
    return out


# ----------------------------------------------------------------------------
# Signals — each returns a Series in {+1 long, -1 short, 0 nothing}
# ----------------------------------------------------------------------------
def compute_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    c = CONFIG
    up = df["Close"] > df["SMA_S"]           # long-term uptrend filter
    dn = df["Close"] < df["SMA_S"]
    sig = {}

    s = pd.Series(0, index=df.index)
    s[(df["RSI"] < c["rsi_low"]) & up] = 1                  # oversold dip in uptrend
    s[(df["RSI"] > c["rsi_high"]) & dn] = -1                # overbought pop in downtrend
    sig[f"RSI({c['rsi_len']}) mean-reversion"] = s

    s = pd.Series(0, index=df.index)
    s[(df["IBS"] < c["ibs_low"]) & up] = 1                  # close near day's low
    s[(df["IBS"] > c["ibs_high"]) & dn] = -1                # close near day's high
    sig["IBS mean-reversion"] = s

    s = pd.Series(0, index=df.index)
    s[df["NLOW"] & up] = 1                                  # n-day low pullback in uptrend
    s[df["NHIGH"] & dn] = -1                                # n-day high rally in downtrend
    sig[f"{c['nday_extreme']}-day extreme pullback"] = s

    trend_up = df["SMA_F"] > df["SMA_S"]
    s = pd.Series(0, index=df.index)
    s[(df["Close"] > df["DON_HI"]) & trend_up] = 1          # breakout with trend
    s[(df["Close"] < df["DON_LO"]) & ~trend_up] = -1        # breakdown with trend
    sig[f"Donchian {c['donchian_len']}d breakout"] = s

    # Liquidity sweep & reclaim ("stop run"): price takes out the prior n-day
    # low intraday but closes back above it -> failed breakdown -> long.
    n = c["sweep_len"]
    prior_lo = df["Low"].rolling(n).min().shift()
    prior_hi = df["High"].rolling(n).max().shift()
    s = pd.Series(0, index=df.index)
    s[(df["Low"] < prior_lo) & (df["Close"] > prior_lo) & up] = 1
    s[(df["High"] > prior_hi) & (df["Close"] < prior_hi) & dn] = -1
    sig[f"Sweep & reclaim ({n}d)"] = s

    # FVG retest: bullish fair-value gap = low[t] > high[t-2]; zone stays valid
    # for `fvg_expiry` bars. Long when price dips into the zone but closes
    # inside/above it (zone held). Mirror for bearish FVGs.
    exp = c["fvg_expiry"]
    bull = df["Low"] > df["High"].shift(2)
    bear = df["High"] < df["Low"].shift(2)
    b_top = pd.Series(np.where(bull, df["Low"], np.nan), index=df.index).ffill(limit=exp).shift()
    b_bot = pd.Series(np.where(bull, df["High"].shift(2), np.nan), index=df.index).ffill(limit=exp).shift()
    r_bot = pd.Series(np.where(bear, df["High"], np.nan), index=df.index).ffill(limit=exp).shift()
    r_top = pd.Series(np.where(bear, df["Low"].shift(2), np.nan), index=df.index).ffill(limit=exp).shift()
    s = pd.Series(0, index=df.index)
    s[(df["Low"] <= b_top) & (df["Close"] >= b_bot) & up] = 1
    s[(df["High"] >= r_bot) & (df["Close"] <= r_top) & dn] = -1
    sig["FVG retest"] = s

    return sig


# ----------------------------------------------------------------------------
# Historical validation of each signal
# ----------------------------------------------------------------------------
def signal_stats(df: pd.DataFrame, signals: dict[str, pd.Series]) -> pd.DataFrame:
    """Direction-adjusted forward returns for each signal vs. all-days baseline."""
    rows = []
    fwd = {h: df["Close"].shift(-h) / df["Close"] - 1 for h in CONFIG["horizons"]}
    kh = CONFIG["key_horizon"]
    baseline = {h: fwd[h].mean() for h in CONFIG["horizons"]}
    for name, s in signals.items():
        fired = s != 0
        n = int(fired.sum())
        row = {"signal": name, "N": n}
        for h in CONFIG["horizons"]:
            adj = (s[fired] * fwd[h][fired]).dropna()       # direction-adjusted
            base = baseline[h]                              # long-only drift reference
            row[f"win%_{h}d"] = round(100 * (adj > 0).mean(), 1) if len(adj) else np.nan
            row[f"avg_{h}d"] = round(100 * adj.mean(), 2) if len(adj) else np.nan
            row[f"edge_{h}d"] = round(100 * (adj.mean() - abs(base)), 2) if len(adj) else np.nan
        # weight for the composite: positive key-horizon edge only
        row["weight"] = max(row.get(f"edge_{kh}d") or 0, 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Composite + risk
# ----------------------------------------------------------------------------
def todays_view(df: pd.DataFrame, signals: dict[str, pd.Series],
                stats: pd.DataFrame, vix: pd.Series | None) -> dict:
    c = CONFIG
    last = df.index[-1]
    fired = {n: int(s.iloc[-1]) for n, s in signals.items() if s.iloc[-1] != 0}

    wmap = dict(zip(stats["signal"], stats["weight"]))
    total_w = sum(wmap.values()) or 1.0
    score = sum(d * wmap.get(n, 0) for n, d in fired.items()) / total_w
    bias = "LONG" if score > c["bias_threshold"] else "SHORT" if score < -c["bias_threshold"] else "FLAT"

    close = float(df["Close"].iloc[-1])
    a = float(df["ATR"].iloc[-1])

    # volatility regime: VIX percentile over 1y, else realized-vol percentile
    if vix is not None and len(vix) > 260:
        v = vix.dropna()
        pct = float((v.iloc[-252:] <= v.iloc[-1]).mean() * 100)
        gauge = f"VIX {v.iloc[-1]:.1f} ({pct:.0f}th pctile 1y)"
    else:
        rv = df["RVOL"].dropna()
        pct = float((rv.iloc[-252:] <= rv.iloc[-1]).mean() * 100)
        gauge = f"realized vol {rv.iloc[-1]*100:.1f}% ({pct:.0f}th pctile 1y)"
    size_mult = 0.5 if pct >= c["high_vol_pctile"] else 1.0

    return {
        "date": last.strftime("%Y-%m-%d"),
        "close": close,
        "fired": fired,
        "score": round(score, 2),
        "bias": bias,
        "atr": a,
        "atr_pct": 100 * a / close,
        "stop_dist": c["stop_atr_mult"] * a,
        "ko_dist": c["ko_atr_mult"] * a,
        "vol_gauge": gauge,
        "size_mult": size_mult,
        "trend": "UP" if df["Close"].iloc[-1] > df["SMA_S"].iloc[-1] else "DOWN",
        "mom63": 100 * float(df["MOM63"].iloc[-1]),
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_report(name: str, view: dict, stats: pd.DataFrame):
    print(f"\n{'='*72}\n{name}  |  {view['date']}  |  close {view['close']:,.1f}  "
          f"|  trend {view['trend']}  |  3m mom {view['mom63']:+.1f}%\n{'='*72}")
    print(f"BIAS: {view['bias']}  (score {view['score']:+.2f})")
    if view["fired"]:
        for n, d in view["fired"].items():
            print(f"  fired: {n} -> {'LONG' if d > 0 else 'SHORT'}")
    else:
        print("  no signals fired today")
    print(f"Risk: ATR({CONFIG['atr_len']}) = {view['atr']:,.1f} ({view['atr_pct']:.2f}%)"
          f" | stop ~{view['stop_dist']:,.0f} pts | KO barrier >= {view['ko_dist']:,.0f} pts away")
    print(f"Regime: {view['vol_gauge']} | size multiplier x{view['size_mult']}")
    print(f"Sizing hint: risk {CONFIG['risk_per_trade']*100:.0f}% of account / "
          f"stop distance = position size, then x{view['size_mult']}")
    cols = ["signal", "N"] + [f"{p}_{h}d" for h in CONFIG["horizons"] for p in ("win%", "avg", "edge")]
    print("\nSignal history on this index (direction-adjusted fwd returns, % — "
          "'edge' is vs. buy-and-hold drift):")
    print(stats[cols].to_string(index=False))


def html_report(results: dict, path: str):
    css = ("body{font-family:system-ui;margin:24px;max-width:900px}"
           "table{border-collapse:collapse;width:100%;margin:8px 0 24px}"
           "td,th{border:1px solid #ddd;padding:6px 10px;font-size:13px;text-align:right}"
           "th{background:#f5f5f5}td:first-child,th:first-child{text-align:left}"
           ".long{color:#0a7a2f;font-weight:700}.short{color:#b3261e;font-weight:700}"
           ".flat{color:#666;font-weight:700}h2{margin-bottom:2px}.meta{color:#555;font-size:14px}")
    parts = [f"<style>{css}</style><h1>Swing Scanner — {datetime.now():%Y-%m-%d %H:%M}</h1>",
             "<p class=meta>Days-to-weeks horizon. Not financial advice.</p>"]
    for name, (view, stats) in results.items():
        cls = view["bias"].lower()
        fired = ", ".join(f"{n} ({'L' if d>0 else 'S'})" for n, d in view["fired"].items()) or "none"
        parts.append(
            f"<h2>{name} — <span class={cls}>{view['bias']}</span> (score {view['score']:+.2f})</h2>"
            f"<p class=meta>{view['date']} | close {view['close']:,.1f} | trend {view['trend']} | "
            f"3m mom {view['mom63']:+.1f}% | signals fired: {fired}<br>"
            f"ATR {view['atr']:,.1f} ({view['atr_pct']:.2f}%) | stop ~{view['stop_dist']:,.0f} | "
            f"KO barrier &ge; {view['ko_dist']:,.0f} away | {view['vol_gauge']} | size x{view['size_mult']}</p>"
            + stats.to_html(index=False))
    with open(path, "w") as f:
        f.write("".join(parts))
    print(f"\nHTML report written to {path}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", metavar="DIR", help="read NDX.csv / RUT.csv / VIX.csv from DIR")
    ap.add_argument("--html", default="signals_report.html")
    args = ap.parse_args()

    print("Loading data...")
    data, vix = get_data(args.offline)

    results = {}
    for name, raw in data.items():
        df = add_indicators(raw)
        signals = compute_signals(df)
        stats = signal_stats(df, signals)
        view = todays_view(df, signals, stats, vix)
        print_report(name, view, stats)
        results[name] = (view, stats)

    html_report(results, args.html)


if __name__ == "__main__":
    main()
