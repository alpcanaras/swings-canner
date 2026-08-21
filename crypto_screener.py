#!/usr/bin/env python3
"""
Crypto red-flag screener + small-cap ("moonshot") discovery.

TWO honest tools, deliberately NOT a "buy these gems" list:

  1. SCREENER — for coins you already care about (watchlist), pull objective
     danger signals from free market data: liquidity, distance from all-time
     high, supply dilution risk, market-cap tier, momentum. Output is a RISK
     read, not a recommendation.

  2. DISCOVERY — surface small-caps that merely pass basic liquidity/size
     filters, as a RESEARCH watchlist. Wrapped in warnings because this corner
     is where most money is lost: by the time a coin shows on a public feed the
     insiders are already positioned, and the overwhelming majority of small
     tokens go to zero. There is no proven edge here. This tool cannot detect
     rug-pulls, honeypots, or fake volume — treat every name as guilty until you
     have done deep due diligence (contract audit, holder concentration, team,
     lockups) with proper on-chain tools.

Data: CoinGecko public API (no key). Works in GitHub Actions.

Usage:
  python crypto_screener.py                 # majors watchlist + discovery
  python crypto_screener.py --watch btc,eth,sol
  python crypto_screener.py --offline markets.json
NOT financial advice. Not a licensed adviser.
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime

CG = "https://api.coingecko.com/api/v3"
STABLES = {"usdt", "usdc", "dai", "tusd", "usde", "fdusd", "usdd", "pyusd",
           "busd", "gusd", "usdp", "frax", "lusd"}
WRAPPED_HINT = ("wrapped", "staked", "bridged", "wormhole", "-peg", "liquid staked")

# Discovery band: small enough to have upside, big enough to not be pure dust.
DISCOVERY = {"mcap_min": 5e6, "mcap_max": 250e6, "vol_min": 1e6,
             "liq_min": 0.03, "max_results": 12}


def fetch(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=45).read())


def markets(pages: int = 2) -> list:
    """Top coins by market cap with the fields we need."""
    out = []
    for p in range(1, pages + 1):
        url = (f"{CG}/coins/markets?vs_currency=usd&order=market_cap_desc"
               f"&per_page=250&page={p}&price_change_percentage=7d,30d")
        out += fetch(url)
        time.sleep(2)          # be gentle with the free endpoint
    return out


def is_junk(c: dict) -> bool:
    sym = (c.get("symbol") or "").lower()
    name = (c.get("name") or "").lower()
    return sym in STABLES or any(h in name for h in WRAPPED_HINT)


def risk_read(c: dict) -> dict:
    """Objective danger signals for one coin. Higher score = more concerning."""
    mc = c.get("market_cap") or 0
    vol = c.get("total_volume") or 0
    liq = (vol / mc) if mc else 0
    ath_chg = c.get("ath_change_percentage")          # % below ATH (negative)
    circ = c.get("circulating_supply") or 0
    total = c.get("total_supply") or c.get("max_supply") or 0
    float_pct = (circ / total * 100) if total else None

    flags, sc = [], 0
    if mc and mc < 50e6:
        flags.append("micro-cap (fragile, easily manipulated)"); sc += 2
    elif mc and mc < 300e6:
        flags.append("small-cap"); sc += 1
    if liq < 0.02:
        flags.append(f"thin liquidity (vol/mcap {liq:.1%} — hard to exit)"); sc += 2
    elif liq < 0.05:
        flags.append(f"modest liquidity ({liq:.1%})"); sc += 1
    if float_pct is not None and float_pct < 40:
        flags.append(f"only {float_pct:.0f}% of supply circulating (future dilution)"); sc += 1
    if ath_chg is not None and ath_chg > -15:
        flags.append(f"near all-time high ({ath_chg:.0f}% from ATH — chasing risk)"); sc += 1
    if vol and vol < 1e6:
        flags.append("very low absolute volume"); sc += 1
    level = "HIGH" if sc >= 4 else "MEDIUM" if sc >= 2 else "LOWER"
    return {"level": level, "score": sc, "flags": flags, "liq": liq,
            "mcap": mc, "vol": vol, "ath_chg": ath_chg, "float_pct": float_pct}


def screen(coins: list, watch: list[str]) -> list:
    want = {w.lower() for w in watch}
    rows = []
    for c in coins:
        if (c.get("symbol") or "").lower() in want or (c.get("id") or "") in want:
            rows.append((c, risk_read(c)))
    return rows


def discover(coins: list) -> list:
    d = DISCOVERY
    out = []
    for c in coins:
        if is_junk(c):
            continue
        mc = c.get("market_cap") or 0
        vol = c.get("total_volume") or 0
        if not (d["mcap_min"] <= mc <= d["mcap_max"]):
            continue
        if vol < d["vol_min"] or (mc and vol / mc < d["liq_min"]):
            continue
        out.append((c, risk_read(c)))
    # rank by 7d momentum, but momentum here is a talking point, not an edge
    out.sort(key=lambda x: -(x[0].get("price_change_percentage_7d_in_currency") or -999))
    return out[:d["max_results"]]


def fmt_mc(x):
    return f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.0f}M" if x else "?"


def render(screened: list, discovered: list, path_html: str, path_txt: str):
    def coin_line(c, r):
        chg = c.get("price_change_percentage_7d_in_currency")
        chg_s = f"{chg:+.0f}%/7d" if chg is not None else ""
        fl = "; ".join(r["flags"]) or "no major red flags in market data"
        return (f"{c['symbol'].upper()} ({c['name']}, {fmt_mc(r['mcap'])}, {chg_s}) "
                f"— risk {r['level']}: {fl}")

    L = [f"# Crypto risk screen — {datetime.now():%d %b %Y}", ""]
    L += ["## Watchlist risk read", ""]
    L += [f"- {coin_line(c, r)}" for c, r in screened] or ["- (no watchlist coins matched)"]
    L += ["", "## Small-cap discovery — RESEARCH ONLY, NOT PICKS", "",
          "> Most of these will fail. A public feed cannot find gems before "
          "insiders. This tool cannot see rug-pulls, honeypots or fake volume. "
          "Treat every name as guilty until proven safe with real due diligence.", ""]
    L += [f"- {coin_line(c, r)}" for c, r in discovered] or ["- Nothing passed the filters."]
    L += ["", "_Objective market-data signals only — no contract audit, no holder "
          "analysis. Not financial advice; not a licensed adviser._"]
    with open(path_txt, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L[:6]))

    css = ("body{font-family:system-ui;margin:24px;max-width:960px;line-height:1.55}"
           ".card{border:1px solid #e2e2e2;border-radius:10px;padding:10px 16px;margin:8px 0}"
           ".HIGH{border-left:6px solid #b3261e}.MEDIUM{border-left:6px solid #d9822b}"
           ".LOWER{border-left:6px solid #0a7a2f}"
           ".warn{background:#fff4f4;border:1px solid #f0c0c0;border-radius:8px;"
           "padding:12px 16px;font-size:14px}.meta{color:#666;font-size:13px}")
    def card(c, r):
        chg = c.get("price_change_percentage_7d_in_currency")
        chg_s = f" · {chg:+.0f}%/7d" if chg is not None else ""
        fl = "".join(f"<li>{x}</li>" for x in r["flags"]) or "<li>no major red flags in market data</li>"
        return (f"<div class='card {r['level']}'><b>{c['symbol'].upper()}</b> "
                f"{c['name']} · {fmt_mc(r['mcap'])}{chg_s} · <b>risk {r['level']}</b>"
                f"<ul>{fl}</ul></div>")
    html = [f"<style>{css}</style><h1>Crypto risk screen</h1>",
            f"<p class=meta>{datetime.now():%Y-%m-%d %H:%M}</p>",
            "<h2>Watchlist</h2>"] + [card(c, r) for c, r in screened]
    html += ["<h2>Small-cap discovery</h2>",
             "<div class=warn><b>Research only — not picks.</b> Most small tokens go "
             "to zero. A public feed cannot surface gems ahead of insiders, and this "
             "tool cannot detect rug-pulls, honeypots or fake volume. Every name here "
             "is a starting point for due diligence, never a buy. Not financial "
             "advice; I am not a licensed adviser.</div>"]
    html += [card(c, r) for c, r in discovered] or ["<p>Nothing passed the filters.</p>"]
    with open(path_html, "w") as f:
        f.write("".join(html))
    print(f"\nWritten {path_html} and {path_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", default="btc,eth,sol,xrp,bnb,doge",
                    help="comma-separated symbols to risk-screen")
    ap.add_argument("--offline", help="JSON file of CoinGecko /markets output")
    ap.add_argument("--html", default="docs/crypto_screen.html")
    ap.add_argument("--txt", default="docs/crypto_screen_digest.txt")
    args = ap.parse_args()

    coins = json.load(open(args.offline)) if args.offline else markets(2)
    watch = [w.strip() for w in args.watch.split(",") if w.strip()]
    render(screen(coins, watch), discover(coins), args.html, args.txt)


if __name__ == "__main__":
    main()
