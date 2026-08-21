#!/usr/bin/env python3
"""
Event-driven backtest of the ACTUAL strategy, not fixed-horizon statistics.

Replays history day by day with exactly the live rules:
  * signals fire on the close of day T; you can only buy at the OPEN of T+1
  * 5 concurrent slots, 20% of current equity each, trimmed by the risk guardrail
  * trailing ATR stop, checked against intraday low/high, gap fills with slippage
  * exit when the bounce is done (RSI target) or at the 20-day backstop
  * spread costs on both sides, compounding equity

Deliberate choices to avoid fooling ourselves:
  * signals are EQUALLY weighted — using weights fitted on the whole sample would
    leak the future into the past
  * entries at the next open, never the signal-day close
  * results compared against equal-weight buy-and-hold of the same universe

Known limitations, stated plainly:
  * survivorship bias — the universe is today's constituents, so companies that
    were delisted or collapsed are missing. This flatters the results.
  * no historical earnings calendar, so the live earnings filter isn't applied
  * one fill per day per name, at open or stop; no intraday path modelling

Usage:
  python backtest.py --universe all --years 10
  python backtest.py --offline data_stocks/ --years 5
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import portfolio as pf
from scanner import CONFIG, add_indicators, compute_signals
from stock_scanner import STOCK_CFG, download_universe, get_universe, load_offline


def prepare(data: dict) -> dict:
    """Indicators + signals for every ticker."""
    out = {}
    for t, raw in data.items():
        df = add_indicators(raw)
        sig = compute_signals(df)
        # net direction per day, equally weighted (no lookahead)
        net = sum(sig.values())
        out[t] = {"df": df, "sig": sig, "net": net}
    return out


VARIANTS = [
    {"name": "baseline (current live rules)", "cfg": {}},
    {"name": "trailing stop only (no RSI exit)", "cfg": {"use_rsi_target": False}},
    {"name": "long-only", "cfg": {}, "long_only": True},
    {"name": "long-only + trailing only",
     "cfg": {"use_rsi_target": False}, "long_only": True},
    {"name": "long-only, trailing only, mean-reversion signals only",
     "cfg": {"use_rsi_target": False}, "long_only": True,
     "families": {"mean reversion"}},
    {"name": "long-only, trailing only, 2 ideas, 3xATR",
     "cfg": {"use_rsi_target": False, "atr_stop_mult": 3.0},
     "long_only": True, "min_families": 2},
    {"name": "long+trailing + regime gate (breadth>50%)",
     "cfg": {"use_rsi_target": False}, "long_only": True, "regime_gate": True},
    {"name": "long+trailing + RS>median gate",
     "cfg": {"use_rsi_target": False}, "long_only": True, "rs_gate": True},
    {"name": "long+trailing + regime + RS gates",
     "cfg": {"use_rsi_target": False}, "long_only": True,
     "regime_gate": True, "rs_gate": True},
    {"name": "momentum/breakouts only, long, trailing",
     "cfg": {"use_rsi_target": False}, "long_only": True, "families": {"momentum"}},
    {"name": "failed-breakout only, long, trailing",
     "cfg": {"use_rsi_target": False}, "long_only": True, "families": {"failed breakout"}},
]


def variant_net(prepared: dict, families: set | None) -> dict:
    """Recompute the daily net signal using only the chosen signal families."""
    if not families:
        return {t: v["net"] for t, v in prepared.items()}
    out = {}
    for t, v in prepared.items():
        keep = [s for n, s in v["sig"].items() if pf.family_of(n) in families]
        out[t] = sum(keep) if keep else v["net"] * 0
    return out


def run_backtest(prepared: dict, start: pd.Timestamp, equity0: float,
                 long_only: bool = False, min_families: int = 1,
                 families: set | None = None,
                 regime_gate: bool = False, rs_gate: bool = False) -> dict:
    p = pf.PORTFOLIO
    nets = variant_net(prepared, families)
    # market breadth (fraction of names above their 200-day) and cross-sectional
    # momentum, both computable without lookahead
    above = pd.DataFrame({t: v["df"]["Close"] > v["df"]["SMA_S"]
                          for t, v in prepared.items()})
    breadth = above.mean(axis=1)
    momdf = pd.DataFrame({t: v["df"]["MOM63"] for t, v in prepared.items()})
    mom_med = momdf.median(axis=1)
    dates = sorted({d for v in prepared.values() for d in v["df"].index if d >= start})
    equity = equity0
    state = {"open": [], "closed": []}
    curve, exposure_days = [], 0

    for i, day in enumerate(dates[:-1]):
        nxt = dates[i + 1]

        # 1) manage open positions using today's bar
        market = {}
        for t, v in prepared.items():
            df = v["df"]
            if day in df.index:
                r = df.loc[day]
                market[t] = {"close": float(r["Close"]), "open": float(r["Open"]),
                             "high": float(r["High"]), "low": float(r["Low"]),
                             "atr": float(r["ATR"]), "rsi": float(r["RSI"])}
        before = len(state["closed"])
        pf.update_positions(state, market, str(day.date()))
        for tr in state["closed"][before:]:
            equity += tr["net"]

        if state["open"]:
            exposure_days += 1

        # 2) rank today's signals, enter at TOMORROW's open
        free = p["max_positions"] - len(state["open"])
        if regime_gate and day in breadth.index and breadth.loc[day] < 0.5:
            free = 0                      # risk-off: manage exits, no new entries
        if free > 0:
            held = {q["ticker"] for q in state["open"]}
            cands = []
            for t, v in prepared.items():
                net = nets[t]
                if t in held or day not in net.index:
                    continue
                score = int(net.loc[day])
                if score == 0 or nxt not in v["df"].index:
                    continue
                if long_only and score < 0:
                    continue
                if rs_gate:
                    m = momdf.at[day, t] if (day in momdf.index and t in momdf) else np.nan
                    md = mom_med.get(day, np.nan)
                    if not (pd.notna(m) and pd.notna(md) and m > md):
                        continue
                df = v["df"]
                dvol = float((df["Close"] * df["Volume"]).rolling(20).mean().loc[day]) \
                    if "Volume" in df else np.inf
                if np.isfinite(dvol) and dvol > 0 and dvol < STOCK_CFG["min_dollar_vol"]:
                    continue
                fired = {n: int(s.loc[day]) for n, s in v["sig"].items()
                         if s.loc[day] != 0
                         and (not families or pf.family_of(n) in families)}
                if len({pf.family_of(n) for n in fired}) < min_families:
                    continue
                cands.append((abs(score), t, np.sign(score), fired))

            cands.sort(key=lambda x: -x[0])
            for _, t, direction, fired in cands[:free]:
                df = prepared[t]["df"]
                entry = float(df.loc[nxt, "Open"])          # realistic fill
                atr = float(df.loc[day, "ATR"])
                if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr):
                    continue
                stop = entry - direction * CONFIG["stop_atr_mult"] * atr
                stop_pct = 100 * abs(entry - stop) / entry
                notional = equity * p["alloc_pct"] / 100
                max_risk = equity * p["max_risk_pct"] / 100
                if notional * stop_pct / 100 > max_risk:
                    notional = max_risk / (stop_pct / 100)
                if notional <= 0:
                    continue
                state["open"].append({
                    "ticker": t, "side": "LONG" if direction > 0 else "SHORT",
                    "entry": entry, "stop": stop, "shares": notional / entry,
                    "notional": notional,
                    "cost": 2 * notional * (p["spread_pct"] + p["fx_pct"]) / 100
                            + 2 * p["cost_per_order_eur"],
                    "date": str(nxt.date()), "days": 0,
                    "families": sorted({pf.family_of(n) for n in fired}),
                    "n_signals": len(fired),
                })

        curve.append({"date": day, "equity": equity})

    return {"state": state, "curve": pd.DataFrame(curve), "equity": equity,
            "exposure": 100 * exposure_days / max(len(dates) - 1, 1),
            "dates": dates}


def benchmark(prepared: dict, start: pd.Timestamp) -> float:
    """Equal-weight buy-and-hold of the same universe, % total return."""
    rets = []
    for v in prepared.values():
        df = v["df"]
        s = df[df.index >= start]["Close"]
        if len(s) > 20:
            rets.append(100 * (float(s.iloc[-1]) / float(s.iloc[0]) - 1))
    return float(np.mean(rets)) if rets else float("nan")


def stats(res: dict, equity0: float, bench_pct: float, years: float) -> dict:
    closed = res["state"]["closed"]
    curve = res["curve"]
    out = {"trades": len(closed), "final": res["equity"],
           "total_pct": 100 * (res["equity"] / equity0 - 1),
           "bench_pct": bench_pct, "exposure": res["exposure"]}
    if closed:
        pcts = [t["pct"] for t in closed]
        out |= {"win_pct": 100 * sum(1 for x in pcts if x > 0) / len(pcts),
                "avg_pct": float(np.mean(pcts)),
                "best": max(pcts), "worst": min(pcts),
                "avg_days": float(np.mean([t["days"] for t in closed])),
                "fees": sum(t["cost"] for t in closed)}
        reasons = {}
        for t in closed:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        out["reasons"] = reasons
        fam = {}
        for f in sorted({f for t in closed for f in t.get("families", [])}):
            rows = [t for t in closed if f in t["families"]]
            if len(rows) >= 20:
                fam[f] = {"n": len(rows),
                          "win": 100 * sum(1 for t in rows if t["pct"] > 0) / len(rows),
                          "avg": float(np.mean([t["pct"] for t in rows]))}
        out["families"] = fam
    if len(curve) > 2:
        eq = curve["equity"].values
        peak = np.maximum.accumulate(eq)
        out["max_dd"] = float(np.min((eq - peak) / peak) * 100)
        if years > 0 and eq[-1] > 0:
            out["cagr"] = float(((eq[-1] / equity0) ** (1 / years) - 1) * 100)
    return out


def report(s: dict, years: float, universe_n: int, path: str):
    verdict = ("The strategy beat buy-and-hold on this sample."
               if s.get("total_pct", 0) > s.get("bench_pct", 0) else
               "The strategy did NOT beat simply buying and holding the same stocks.")
    lines = [
        f"Period: {years:.1f} years · {universe_n} stocks · "
        f"{s['trades']} trades · {s['exposure']:.0f}% of days invested",
        f"Strategy return: {s['total_pct']:+.1f}%"
        + (f" ({s['cagr']:+.1f}%/yr)" if "cagr" in s else ""),
        f"Buy & hold the same universe: {s['bench_pct']:+.1f}%",
        f"Worst drawdown: {s.get('max_dd', float('nan')):.1f}%",
    ]
    if s["trades"]:
        lines += [
            f"Win rate: {s['win_pct']:.1f}% · average trade {s['avg_pct']:+.2f}% · "
            f"held {s['avg_days']:.1f} days · best {s['best']:+.1f}% / "
            f"worst {s['worst']:+.1f}%",
            "Exits: " + ", ".join(f"{k} {v}" for k, v in s["reasons"].items()),
        ]
        if s.get("families"):
            lines.append("By signal family: " + " | ".join(
                f"{k}: {v['n']} trades, {v['win']:.0f}% win, {v['avg']:+.2f}% avg"
                for k, v in s["families"].items()))
    lines.append(verdict)

    for ln in lines:
        print(ln)

    css = ("body{font-family:system-ui;margin:24px;max-width:860px;line-height:1.6}"
           ".card{border:1px solid #e2e2e2;border-radius:10px;padding:14px 20px;margin:12px 0}"
           ".big{font-size:20px;font-weight:700}.meta{color:#666;font-size:13px}"
           ".note{background:#f7f7f9;border-radius:8px;padding:12px 16px;font-size:14px;color:#444}")
    html = [f"<style>{css}</style><h1>Backtest — the actual strategy</h1>",
            f"<p class=meta>Run {datetime.now():%Y-%m-%d %H:%M}</p>",
            f"<div class=card><p class=big>{verdict}</p>"
            + "".join(f"<p>{ln}</p>" for ln in lines[:-1]) + "</div>",
            "<div class=note><b>How to read this.</b> This replays the real rules: "
            "signals on the close, entry at the next open, trailing stops checked "
            "against intraday lows, gap slippage, spread costs, five slots, "
            "compounding.<br><br>"
            "<b>The benchmark is the point.</b> If the strategy returns less than "
            "buying and holding the same stocks, the work isn't paying for itself — "
            "and it carries more risk and effort.<br><br>"
            "<b>Biases that flatter these numbers:</b> the universe is today's "
            "constituents, so dead companies are missing (survivorship bias); "
            "no earnings filter is applied historically; fills assume you get the "
            "open or the stop. Real results would be somewhat worse.</div>"]
    with open(path, "w") as f:
        f.write("".join(html))
    print(f"\nWritten to {path}")


def sweep(prepared: dict, start: pd.Timestamp, equity0: float, years: float,
          bench: float, path: str):
    """Run every variant over identical data and rank them honestly."""
    base = dict(pf.PORTFOLIO)
    rows = []
    for v in VARIANTS:
        pf.PORTFOLIO.clear()
        pf.PORTFOLIO.update(base)
        pf.PORTFOLIO.update(v.get("cfg", {}))
        print(f"\n--- {v['name']}")
        res = run_backtest(prepared, start, equity0,
                           long_only=v.get("long_only", False),
                           min_families=v.get("min_families", 1),
                           families=v.get("families"),
                           regime_gate=v.get("regime_gate", False),
                           rs_gate=v.get("rs_gate", False))
        s = stats(res, equity0, bench, years)
        s["name"] = v["name"]
        rows.append(s)
        print(f"    {s['trades']:>5} trades | return {s['total_pct']:+8.1f}% | "
              f"win {s.get('win_pct', 0):.1f}% | avg {s.get('avg_pct', 0):+.2f}% | "
              f"maxDD {s.get('max_dd', float('nan')):.1f}%")
    pf.PORTFOLIO.clear()
    pf.PORTFOLIO.update(base)

    rows.sort(key=lambda r: -r["total_pct"])
    best = rows[0]
    beat = best["total_pct"] > bench

    print("\n" + "=" * 78)
    print(f"BENCHMARK — buy & hold the same stocks: {bench:+.1f}%")
    for r in rows:
        flag = "BEATS" if r["total_pct"] > bench else "loses to"
        print(f"{r['total_pct']:+9.1f}%  {flag:>8} buy&hold  | "
              f"{r['trades']:>5} trades | maxDD {r.get('max_dd', float('nan')):6.1f}% "
              f"| {r['name']}")
    print("=" * 78)
    print("VERDICT: " + ("a variant beats buy-and-hold — worth a closer look."
                         if beat else
                         "no variant beats buy-and-hold. The idea does not work."))

    css = ("body{font-family:system-ui;margin:24px;max-width:920px;line-height:1.6}"
           "table{border-collapse:collapse;width:100%;margin:12px 0}"
           "td,th{border:1px solid #ddd;padding:7px 10px;font-size:13px;text-align:right}"
           "th{background:#f5f5f5}td:first-child,th:first-child{text-align:left}"
           ".win{background:#eaf7ee}.note{background:#f7f7f9;border-radius:8px;"
           "padding:12px 16px;font-size:14px;color:#444}")
    html = [f"<style>{css}</style><h1>Strategy variants — {years:.0f}-year backtest</h1>",
            f"<p>Benchmark: buy &amp; hold the same stocks = <b>{bench:+.1f}%</b>. "
            f"Run {datetime.now():%Y-%m-%d %H:%M}.</p>",
            "<table><tr><th>variant</th><th>return</th><th>vs buy&amp;hold</th>"
            "<th>trades</th><th>win%</th><th>avg trade</th><th>max DD</th></tr>"]
    for r in rows:
        cls = " class=win" if r["total_pct"] > bench else ""
        html.append(
            f"<tr{cls}><td>{r['name']}</td><td>{r['total_pct']:+.1f}%</td>"
            f"<td>{r['total_pct'] - bench:+.1f}pp</td><td>{r['trades']}</td>"
            f"<td>{r.get('win_pct', 0):.1f}%</td><td>{r.get('avg_pct', 0):+.2f}%</td>"
            f"<td>{r.get('max_dd', float('nan')):.1f}%</td></tr>")
    html.append("</table>")
    html.append("<div class=note><b>Read this carefully.</b> Eight variants were "
                "tested on the same data. If one looks good, remember that testing "
                "many versions and keeping the winner is how people fool themselves — "
                "the winner needs to survive a different period and a different "
                "universe before it means anything.<br><br>"
                "Survivorship bias flatters every row here, and the benchmark most "
                "of all.</div>")
    with open(path, "w") as f:
        f.write("".join(html))
    print(f"\nWritten to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ndx")
    ap.add_argument("--tickers")
    ap.add_argument("--offline")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--html", default="docs/backtest.html")
    ap.add_argument("--sweep", action="store_true", help="test all variants")
    args = ap.parse_args()

    data = load_offline(args.offline) if args.offline else \
        download_universe(get_universe(args), int(args.years) + 2)
    prepared = prepare(data)
    start = pd.Timestamp.now().normalize() - pd.DateOffset(years=int(args.years))
    print(f"Replaying {len(prepared)} tickers from {start.date()}...")
    bench = benchmark(prepared, start)

    if args.sweep:
        sweep(prepared, start, args.equity, args.years, bench, args.html)
    else:
        res = run_backtest(prepared, start, args.equity)
        s = stats(res, args.equity, bench, args.years)
        report(s, args.years, len(prepared), args.html)


if __name__ == "__main__":
    main()
