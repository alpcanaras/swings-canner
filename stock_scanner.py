#!/usr/bin/env python3
"""
Individual-stock swing scanner (days to a couple of weeks, long/short).

Scans a universe of stocks (default: Nasdaq-100 constituents) with the same
six signals as scanner.py, then ranks today's candidates.

Differences vs. the index scanner — important for single names:
  * Signal edge is measured POOLED across the whole universe (per-stock stats
    are too noisy); weights come from that pooled edge.
  * Liquidity filter: min average dollar volume.
  * Earnings warning: candidates reporting within N days are flagged —
    a 2x-ATR stop means nothing against an earnings gap.
  * Relative strength vs. universe median shown for each candidate.

Usage:
  pip install yfinance pandas numpy lxml
  python stock_scanner.py                        # Nasdaq-100 universe
  python stock_scanner.py --universe r2k         # Russell 2000 (via IWM holdings)
  python stock_scanner.py --universe all         # both
  python stock_scanner.py --tickers my_list.txt  # your own universe
  python stock_scanner.py --offline data_stocks/ # local CSVs, one per ticker
  python stock_scanner.py --no-earnings          # skip earnings-date lookups (faster)

Run it together with scanner.py: only take stock longs when the index bias
isn't SHORT, and vice versa. NOT financial advice.
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from scanner import CONFIG, add_indicators, compute_signals  # reuse signal logic

STOCK_CFG = {
    "history_years": 10,
    "min_dollar_vol": 20e6,     # 20-day avg dollar volume floor
    "top_n": 10,                # candidates shown per side
    "earnings_warn_days": 7,    # flag if earnings within this many days
    "min_history_days": 300,    # skip stocks with less data (recent IPOs)
}

# Fallback if the Wikipedia fetch fails. May drift from the actual index —
# the live fetch is always preferred.
NDX_FALLBACK = """AAPL MSFT NVDA AMZN GOOGL GOOG META AVGO TSLA COST NFLX AMD PEP ADBE CSCO
QCOM TMUS INTU AMAT TXN CMCSA AMGN ISRG HON BKNG VRTX PANW ADP SBUX GILD MU INTC ADI LRCX
MDLZ REGN KLAC SNPS PYPL CDNS MAR CRWD MRVL ORLY CSX ABNB FTNT DASH ADSK CTAS PCAR ROP NXPI
WDAY AEP CPRT PAYX MNST ROST ODFL FAST KDP EA BKR VRSK CTSH XEL EXC TEAM GEHC IDXX CCEP TTWO
ZS DDOG ON FANG CSGP WBD GFS BIIB DXCM KHC AZN LULU CDW ARM MELI PDD""".split()


# ----------------------------------------------------------------------------
# Universe + data
# ----------------------------------------------------------------------------
def _fetch(url: str) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


def get_ndx() -> list[str]:
    try:  # live Nasdaq-100 constituents
        import io
        tables = pd.read_html(io.StringIO(_fetch("https://en.wikipedia.org/wiki/Nasdaq-100")))
        for t in tables:
            for col in ("Ticker", "Symbol"):
                if col in t.columns and len(t) > 80:
                    return sorted(t[col].astype(str).str.replace(".", "-", regex=False))
        raise ValueError("no constituent table found")
    except Exception as e:
        print(f"  Wikipedia constituent fetch failed ({e}); using built-in list.")
    return NDX_FALLBACK


def get_r2000() -> list[str]:
    """Russell 2000 constituents via iShares IWM daily holdings CSV (free).
    Returns [] on failure so the scan continues with the other universes."""
    url = ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
           "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")
    try:
        import io
        raw = _fetch(url)
        lines = raw.splitlines()
        hdr = next((i for i, ln in enumerate(lines)
                    if ln.replace('"', "").startswith("Ticker,")), None)
        if hdr is None:
            raise ValueError("no 'Ticker' header found in holdings file")
        df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        if "Asset Class" in df.columns:   # keep equities, drop cash/futures lines
            df = df[df["Asset Class"].astype(str).str.contains("Equity", na=False)]
        ticks = df["Ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
        ticks = sorted({t for t in ticks if t.replace("-", "").isalpha() and 1 <= len(t) <= 5})
        if len(ticks) < 500:
            raise ValueError(f"only {len(ticks)} tickers parsed — source layout changed?")
        print(f"  Russell 2000 via IWM holdings: {len(ticks)} tickers")
        return ticks
    except Exception as e:
        print(f"  WARNING: Russell 2000 fetch failed ({e}) — continuing without it. "
              "You can supply names via --tickers.")
        return []


def get_universe(args) -> list[str]:
    if args.tickers:
        with open(args.tickers) as f:
            return sorted({t.strip().upper() for t in f.read().split() if t.strip()})
    u = args.universe.lower()
    parts: list[str] = []
    if u in ("ndx", "all"):
        parts += get_ndx()
    if u in ("r2k", "all"):
        parts += get_r2000()
    if not parts:
        sys.exit(f"Unknown universe '{args.universe}' (use ndx, r2k, or all)")
    return sorted(set(parts))


def download_universe(tickers: list[str], years: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    print(f"Downloading {len(tickers)} tickers ({years}y daily)...")
    if len(tickers) > 500:
        print("  large universe — this can take 10-20 minutes on first run")
    data = {}
    start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    chunk_size = 200
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(chunk, start=start, interval="1d", auto_adjust=True,
                              group_by="ticker", progress=False, threads=True)
        except Exception as e:
            print(f"  chunk {i//chunk_size + 1} failed: {e}")
            continue
        for t in chunk:
            try:
                df = (raw[t] if len(chunk) > 1 else raw)[
                    ["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) >= STOCK_CFG["min_history_days"]:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    data[t] = df
            except Exception:
                pass
        if len(tickers) > chunk_size:
            print(f"  {min(i + chunk_size, len(tickers))}/{len(tickers)} done")
    print(f"  usable: {len(data)} tickers")
    return data


def load_offline(directory: str) -> dict[str, pd.DataFrame]:
    data = {}
    for path in glob.glob(os.path.join(directory, "*.csv")):
        t = os.path.splitext(os.path.basename(path))[0].upper()
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").dropna()
        if len(df) >= STOCK_CFG["min_history_days"]:
            data[t] = df
    if not data:
        sys.exit(f"No usable CSVs in {directory}")
    return data


def next_earnings_days(ticker: str) -> int | None:
    """Days until next earnings, or None if unknown. One request per candidate."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            d = pd.Timestamp(dates[0])
            return (d - pd.Timestamp.now().normalize()).days
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Pooled signal validation across the universe
# ----------------------------------------------------------------------------
def pooled_stats(processed: dict[str, tuple[pd.DataFrame, dict]]) -> pd.DataFrame:
    horizons = CONFIG["horizons"]
    kh = CONFIG["key_horizon"]
    adj_pool: dict[str, dict[int, list]] = {}
    base_pool: dict[int, list] = {h: [] for h in horizons}
    for t, (df, signals) in processed.items():
        fwd = {h: df["Close"].shift(-h) / df["Close"] - 1 for h in horizons}
        for h in horizons:
            base_pool[h].append(fwd[h].dropna())
        for name, s in signals.items():
            fired = s != 0
            d = adj_pool.setdefault(name, {h: [] for h in horizons})
            for h in horizons:
                d[h].append((s[fired] * fwd[h][fired]).dropna())
    baseline = {h: pd.concat(base_pool[h]).mean() for h in horizons}
    rows = []
    for name, per_h in adj_pool.items():
        row = {"signal": name}
        for h in horizons:
            adj = pd.concat(per_h[h])
            row["N"] = len(pd.concat(per_h[horizons[0]]))
            row[f"win%_{h}d"] = round(100 * (adj > 0).mean(), 1)
            row[f"avg_{h}d"] = round(100 * adj.mean(), 2)
            row[f"edge_{h}d"] = round(100 * (adj.mean() - abs(baseline[h])), 2)
        row["weight"] = max(row[f"edge_{kh}d"], 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ndx", help="ndx | r2k | all (default ndx)")
    ap.add_argument("--tickers", help="text file with tickers (space/newline separated)")
    ap.add_argument("--offline", metavar="DIR", help="read <TICKER>.csv files from DIR")
    ap.add_argument("--no-earnings", action="store_true")
    ap.add_argument("--html", default="stock_candidates.html")
    args = ap.parse_args()

    if args.offline:
        data = load_offline(args.offline)
    else:
        data = download_universe(get_universe(args), STOCK_CFG["history_years"])

    # indicators + signals for every stock
    processed = {}
    for t, raw in data.items():
        df = add_indicators(raw)
        processed[t] = (df, compute_signals(df))

    stats = pooled_stats(processed)
    wmap = dict(zip(stats["signal"], stats["weight"]))
    total_w = sum(wmap.values()) or 1.0
    if total_w == 1.0 and not any(wmap.values()):
        print("\nNOTE: no signal shows positive pooled edge on this universe — "
              "candidates are suppressed by design. Believe the data.")

    # universe median 3-month return for relative strength
    med_mom = np.nanmedian([float(df["MOM63"].iloc[-1]) for df, _ in processed.values()])

    candidates = []
    for t, (df, signals) in processed.items():
        fired = {n: int(s.iloc[-1]) for n, s in signals.items() if s.iloc[-1] != 0}
        if not fired:
            continue
        dvol = float((df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]) \
            if "Volume" in df else np.inf
        if dvol < STOCK_CFG["min_dollar_vol"]:
            continue
        score = sum(d * wmap.get(n, 0) for n, d in fired.items()) / total_w
        if score == 0:
            continue
        close = float(df["Close"].iloc[-1])
        a = float(df["ATR"].iloc[-1])
        candidates.append({
            "ticker": t, "side": "LONG" if score > 0 else "SHORT",
            "score": round(abs(score), 2), "close": round(close, 2),
            "signals": ", ".join(f"{n} ({'L' if d > 0 else 'S'})" for n, d in fired.items()),
            "RS_3m_%": round(100 * (float(df["MOM63"].iloc[-1]) - med_mom), 1),
            "ATR%": round(100 * a / close, 2),
            "stop": round(close - np.sign(score) * CONFIG["stop_atr_mult"] * a, 2),
            "min_KO_dist": round(CONFIG["ko_atr_mult"] * a, 2),
        })

    cand = pd.DataFrame(candidates)
    top = pd.DataFrame()
    if len(cand):
        cand = cand.sort_values("score", ascending=False)
        top = pd.concat([cand[cand["side"] == "LONG"].head(STOCK_CFG["top_n"]),
                         cand[cand["side"] == "SHORT"].head(STOCK_CFG["top_n"])])
        if not args.no_earnings and not args.offline:
            print("Checking earnings dates for candidates...")
            warn = STOCK_CFG["earnings_warn_days"]
            top["earnings"] = [
                (f"in {d}d !" if d is not None and 0 <= d <= warn else
                 (f"in {d}d" if d is not None and d >= 0 else "?"))
                for d in (next_earnings_days(t) for t in top["ticker"])]

    date = max(df.index[-1] for df, _ in processed.values()).strftime("%Y-%m-%d")
    print(f"\n{'='*72}\nSTOCK CANDIDATES — {date}  (universe: {len(processed)} names)\n{'='*72}")
    print(top.to_string(index=False) if len(top) else "No candidates today.")
    print("\nPooled signal history across universe (direction-adjusted fwd returns, %):")
    cols = ["signal", "N"] + [f"{p}_{h}d" for h in CONFIG["horizons"] for p in ("win%", "avg", "edge")]
    print(stats[cols].to_string(index=False))
    print("\nReminder: check index bias (scanner.py) — avoid stock longs when the "
          "index bias is SHORT, and vice versa. Flagged earnings = consider skipping.")

    css = ("body{font-family:system-ui;margin:24px;max-width:1000px}"
           "table{border-collapse:collapse;width:100%;margin:8px 0 24px}"
           "td,th{border:1px solid #ddd;padding:6px 10px;font-size:13px;text-align:right}"
           "th{background:#f5f5f5}td:first-child,th:first-child{text-align:left}")
    with open(args.html, "w") as f:
        f.write(f"<style>{css}</style><h1>Stock candidates — {datetime.now():%Y-%m-%d %H:%M}</h1>"
                + (top.to_html(index=False) if len(top) else "<p>No candidates today.</p>")
                + "<h2>Pooled signal stats</h2>" + stats.to_html(index=False))
    print(f"HTML report written to {args.html}")


if __name__ == "__main__":
    main()
