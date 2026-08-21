#!/usr/bin/env python3
"""
Fundamental (FA) scanner — longer-term stock ideas, a DIFFERENT product from the
swing sheet.

Timeframe: months to years, not days. Fundamentals barely move day to day, so
this is meant to run WEEKLY. It ranks companies cross-sectionally on three
pillars and surfaces quality businesses at reasonable valuations that are also
in a long-term uptrend (the well-known "quality + value + trend" combination).

Honest framing:
  * This is not a trading signal. It's a shortlist of businesses to research.
  * Ranks are RELATIVE within the universe scanned — "cheaper than peers", not
    "cheap in absolute terms".
  * Free fundamental data (yfinance) is incomplete and sometimes stale; names
    with too little data are skipped rather than guessed.
  * A high score is not a recommendation. Do your own due diligence.

Usage:
  python fa_scanner.py --universe spx        # S&P 500 (default NDX+SPX large caps)
  python fa_scanner.py --offline fa_data.json
NOT financial advice. Not a licensed adviser.
"""

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from stock_scanner import _fetch, get_ndx, get_spx  # reuse universe fetchers

# Metric -> (pillar, higher_is_better). Cross-sectional percentile ranks are
# averaged per pillar, so absolute thresholds never bake in a fixed worldview.
METRICS = {
    "trailingPE":                   ("value", False),
    "forwardPE":                    ("value", False),
    "priceToSalesTrailing12Months": ("value", False),
    "pegRatio":                     ("value", False),
    "fcf_yield":                    ("value", True),
    "profitMargins":                ("quality", True),
    "returnOnEquity":               ("quality", True),
    "operatingMargins":             ("quality", True),
    "debtToEquity":                 ("quality", False),
    "currentRatio":                 ("quality", True),
    "revenueGrowth":                ("growth", True),
    "earningsGrowth":               ("growth", True),
}
PILLARS = ("value", "quality", "growth", "momentum")


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """One yfinance .info call per name. Slow and flaky by nature — hence weekly."""
    import yfinance as yf
    rows = []
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info or {}
            mc = info.get("marketCap")
            fcf = info.get("freeCashflow")
            row = {"ticker": t, "name": info.get("shortName", t),
                   "marketCap": mc, "price": info.get("currentPrice"),
                   "sector": info.get("sector", "?"),
                   "fcf_yield": (fcf / mc) if (fcf and mc) else None}
            for m in METRICS:
                if m not in row:
                    row[m] = info.get(m)
            rows.append(row)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  fundamentals {i + 1}/{len(tickers)}")
    return pd.DataFrame(rows)


def trend_and_momentum(tickers: list[str]) -> tuple[dict, dict]:
    """One bulk download gives both the 200-day trend flag and 12-1 momentum
    (return over the last year excluding the most recent month — the classic
    academic momentum definition, the best-documented cross-sectional factor)."""
    import yfinance as yf
    up, mom = {}, {}
    try:
        px = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                         progress=False, threads=True)["Close"]
        for t in tickers:
            s = px[t].dropna() if t in px else pd.Series(dtype=float)
            if len(s) > 200:
                up[t] = bool(s.iloc[-1] > s.rolling(200).mean().iloc[-1])
            if len(s) > 260:
                mom[t] = float(s.iloc[-21] / s.iloc[-252] - 1)
    except Exception as e:
        print(f"  trend download failed: {e}")
    return up, mom


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks -> pillar scores -> composite (0-100)."""
    df = df.copy()
    for m, (_, higher) in METRICS.items():
        col = pd.to_numeric(df.get(m), errors="coerce")
        # guard against nonsense (negative PE, absurd ratios)
        if m in ("trailingPE", "forwardPE", "pegRatio", "priceToSalesTrailing12Months"):
            col = col.where(col > 0)
        r = col.rank(pct=True)
        df[f"r_{m}"] = (r if higher else 1 - r) * 100
    for pillar in ("value", "quality", "growth"):
        cols = [f"r_{m}" for m, (p, _) in METRICS.items() if p == pillar]
        df[pillar] = df[cols].mean(axis=1, skipna=True)
    # momentum pillar: percentile rank of 12-1 return
    df["momentum"] = pd.to_numeric(df.get("mom_12_1"), errors="coerce").rank(pct=True) * 100
    df["quality_value"] = df[["quality", "value"]].mean(axis=1)
    df["composite"] = df[list(PILLARS)].mean(axis=1, skipna=True)
    # require a minimum amount of real data behind the score
    df["coverage"] = df[[f"r_{m}" for m in METRICS]].notna().mean(axis=1)
    return df


def thesis(row) -> str:
    bits = []
    pe = row.get("trailingPE")
    if pd.notna(pe) and pe:
        bits.append(f"P/E {pe:.0f}")
    roe = row.get("returnOnEquity")
    if pd.notna(roe) and roe:
        bits.append(f"ROE {roe*100:.0f}%")
    marg = row.get("profitMargins")
    if pd.notna(marg) and marg:
        bits.append(f"net margin {marg*100:.0f}%")
    g = row.get("revenueGrowth")
    if pd.notna(g) and g:
        bits.append(f"rev growth {g*100:.0f}%")
    de = row.get("debtToEquity")
    if pd.notna(de) and de:
        bits.append(f"debt/equity {de/100:.1f}" if de > 5 else f"debt/equity {de:.1f}")
    strengths = [p for p in PILLARS if row.get(p, 0) >= 66]
    lead = ("strong " + " & ".join(strengths)) if strengths else "balanced profile"
    trend = "in an uptrend" if row.get("uptrend") else "below its 200-day (out of favour)"
    return f"{lead}; {trend}. " + ", ".join(bits)


def flags(row) -> list[str]:
    out = []
    if (row.get("profitMargins") or 0) < 0:
        out.append("loss-making")
    de = row.get("debtToEquity")
    if de is not None and pd.notna(de) and de > 200:
        out.append("high debt")
    g = row.get("revenueGrowth")
    if g is not None and pd.notna(g) and g < 0:
        out.append("shrinking revenue")
    pe = row.get("trailingPE")
    if pe is not None and pd.notna(pe) and pe > 60:
        out.append("very expensive")
    if (row.get("coverage") or 0) < 0.5:
        out.append("thin data")
    return out


def render(df: pd.DataFrame, path_html: str, path_txt: str, universe_n: int):
    def sector_cap(d, per_sector, total):
        rows, cnt = [], {}
        for _, r in d.iterrows():
            sec = r.get("sector") or "?"
            if cnt.get(sec, 0) >= per_sector:
                continue
            cnt[sec] = cnt.get(sec, 0) + 1
            rows.append(r)
            if len(rows) >= total:
                break
        return pd.DataFrame(rows)

    ranked = df[df["coverage"] >= 0.5].sort_values("composite", ascending=False)
    buys = sector_cap(ranked[ranked["uptrend"] == True], 3, 12)      # noqa: E712
    watch = sector_cap(ranked[ranked["uptrend"] != True], 2, 8)

    def line(r):
        f = flags(r)
        fl = f" ⚠ {', '.join(f)}" if f else ""
        mc = f"${r['marketCap']/1e9:.0f}B" if pd.notna(r.get("marketCap")) else ""
        mo = r.get("momentum")
        mo_s = f"/M{mo:.0f}" if pd.notna(mo) else ""
        return (f"{r['ticker']} ({r['name']}, {mc}) — score {r['composite']:.0f}"
                f" [V{r['value']:.0f}/Q{r['quality']:.0f}/G{r['growth']:.0f}{mo_s}]. "
                f"{thesis(r)}{fl}")

    # plain-text weekly bulletin
    L = [f"# Long-term ideas (FA) — week of {datetime.now():%d %b %Y}", "",
         f"Ranked {universe_n} large caps on value, quality, growth and momentum, max 3 per sector. "
         "Relative ranks, not absolute value. Research these — they are not buy "
         "signals, and this is a months-to-years view, separate from the swing sheet.",
         "", "## Quality at a reasonable price, and in an uptrend", ""]
    L += [f"- {line(r)}" for _, r in buys.iterrows()] or ["- Nothing qualifies."]
    L += ["", "## Strong businesses currently out of favour (watch, don't chase)", ""]
    L += [f"- {line(r)}" for _, r in watch.iterrows()] or ["- None."]
    L += ["", "_Scores are relative to the scanned universe. Free fundamental data "
          "is imperfect. Not financial advice; not a licensed adviser._"]
    with open(path_txt, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L[:8]))

    css = ("body{font-family:system-ui;margin:24px;max-width:960px;line-height:1.55}"
           "table{border-collapse:collapse;width:100%;margin:10px 0}"
           "td,th{border:1px solid #ddd;padding:6px 9px;font-size:13px;text-align:right}"
           "th{background:#f5f5f5}td:first-child,th:first-child{text-align:left}"
           ".note{background:#f7f7f9;border-radius:8px;padding:12px 16px;font-size:14px;color:#444}")
    def tbl(d):
        h = ("<table><tr><th>ticker</th><th>score</th><th>V</th><th>Q</th><th>G</th>"
             "<th>P/E</th><th>ROE</th><th>rev g</th><th>trend</th><th>flags</th></tr>")
        for _, r in d.iterrows():
            pe = f"{r['trailingPE']:.0f}" if pd.notna(r.get("trailingPE")) else "–"
            roe = f"{r['returnOnEquity']*100:.0f}%" if pd.notna(r.get("returnOnEquity")) else "–"
            g = f"{r['revenueGrowth']*100:.0f}%" if pd.notna(r.get("revenueGrowth")) else "–"
            h += (f"<tr><td>{r['ticker']}</td><td>{r['composite']:.0f}</td>"
                  f"<td>{r['value']:.0f}</td><td>{r['quality']:.0f}</td><td>{r['growth']:.0f}</td>"
                  f"<td>{pe}</td><td>{roe}</td><td>{g}</td>"
                  f"<td>{'up' if r.get('uptrend') else 'down'}</td>"
                  f"<td>{', '.join(flags(r)) or '–'}</td></tr>")
        return h + "</table>"
    html = [f"<style>{css}</style><h1>Long-term ideas (FA)</h1>",
            f"<p>Week of {datetime.now():%Y-%m-%d}. {universe_n} large caps ranked on "
            "value, quality and growth. Months-to-years horizon.</p>",
            "<h2>Quality + value + uptrend</h2>", tbl(buys),
            "<h2>Out of favour (watch)</h2>", tbl(watch),
            "<div class=note><b>How to read this.</b> Scores are percentile ranks "
            "<i>within this universe</i> — a 90 means cheaper/higher-quality/faster-"
            "growing than ~90% of peers, not cheap in absolute terms. V/Q/G are the "
            "three pillars. This is a research shortlist on a months-to-years horizon, "
            "deliberately separate from the swing trades. Not financial advice; "
            "I am not a licensed adviser.</div>"]
    with open(path_html, "w") as f:
        f.write("".join(html))
    print(f"\nWritten {path_html} and {path_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="all", help="ndx | spx | all")
    ap.add_argument("--offline", help="JSON list of fundamental rows (for testing)")
    ap.add_argument("--html", default="docs/fa.html")
    ap.add_argument("--txt", default="docs/fa_digest.txt")
    args = ap.parse_args()

    if args.offline:
        df = pd.DataFrame(json.load(open(args.offline)))
        if "uptrend" not in df:
            df["uptrend"] = True
        n = len(df)
    else:
        uni = sorted(set((get_ndx() if args.universe in ("ndx", "all") else [])
                         + (get_spx() if args.universe in ("spx", "all") else [])))
        print(f"Fundamentals for {len(uni)} names (weekly, this takes a while)...")
        df = fetch_fundamentals(uni)
        if len(df) < 20:
            sys.exit("Too little fundamental data returned.")
        up, mom = trend_and_momentum(list(df["ticker"]))
        df["uptrend"] = df["ticker"].map(up).fillna(False)
        df["mom_12_1"] = df["ticker"].map(mom)
        n = len(df)

    df = score(df)
    render(df, args.html, args.txt, n)


if __name__ == "__main__":
    main()
