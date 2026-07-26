#!/usr/bin/env python3
"""
Portfolio layer: position sizing, cost reality-check, and paper tracking.

Three jobs:
  1. SIZE       — turn a candidate into a concrete position for YOUR account,
                  using fixed-fractional risk (never more than `risk_pct` at stake).
  2. COST CHECK — compare the expected edge against real trading costs and say
                  plainly whether the trade is worth taking at this size.
  3. TRACK      — run a paper portfolio automatically: open positions from the
                  day's candidates, exit on stop or after `hold_days`, and report
                  P&L both gross and net of costs.

Everything you'd want to change lives in PORTFOLIO below.
Paper trading only. Not financial advice.
"""

import json
import os
from datetime import date, datetime

from scanner import CONFIG

PORTFOLIO = {
    "account_eur": 500.0,        # your starting balance
    "risk_pct": 1.0,             # % of account risked per trade (the stop distance)
    "cost_per_order_eur": 1.0,   # Trade Republic style flat fee, per side
    "spread_pct": 0.05,          # half-spread paid on each side, %
    "max_positions": 3,          # never hold more than this many at once
    "max_position_pct": 40.0,    # cap one position at % of account
    "hold_days": CONFIG["key_horizon"],   # time exit: close after N trading days
    "min_edge_cost_ratio": 3.0,  # expected profit must be >= 3x costs to "take"
    "paper": True,               # paper mode: track trades regardless of verdict,
                                 # so you can see gross vs. net over time
}

STATE_PATH = "state/paper.json"


# ----------------------------------------------------------------------------
# 1 + 2. Sizing and the cost reality-check
# ----------------------------------------------------------------------------
def size_and_verdict(close: float, stop: float, expected_pct: float) -> dict:
    """Concrete position for this account, plus whether costs make it worth it.

    expected_pct: the signal's historical average move over the hold, in %.
    """
    p = PORTFOLIO
    stop_dist = abs(close - stop)
    if stop_dist <= 0 or close <= 0:
        return {"viable": False, "reason": "bad prices"}

    stop_pct = 100 * stop_dist / close
    risk_eur = p["account_eur"] * p["risk_pct"] / 100
    notional = risk_eur / (stop_pct / 100)                    # risk-based size
    cap = p["account_eur"] * p["max_position_pct"] / 100
    capped = notional > cap
    notional = min(notional, cap)
    shares = notional / close

    cost = 2 * p["cost_per_order_eur"] + 2 * notional * p["spread_pct"] / 100
    gross = notional * expected_pct / 100
    net = gross - cost
    ratio = gross / cost if cost > 0 else float("inf")

    # what this trade would need to be worth doing
    fixed = 2 * p["cost_per_order_eur"]
    per_eur_edge = expected_pct / 100 - 2 * p["spread_pct"] / 100
    breakeven = fixed / per_eur_edge if per_eur_edge > 0 else float("inf")
    worthwhile = breakeven * p["min_edge_cost_ratio"] if per_eur_edge > 0 else float("inf")

    return {
        "viable": ratio >= p["min_edge_cost_ratio"],
        "stop_pct": stop_pct, "risk_eur": risk_eur, "notional": notional,
        "shares": shares, "capped": capped, "cost": cost, "gross": gross,
        "net": net, "ratio": ratio, "breakeven_notional": breakeven,
        "worthwhile_notional": worthwhile, "expected_pct": expected_pct,
    }


def sizing_sentence(v: dict) -> str:
    """The sizing result, in words."""
    if not v.get("viable") and not v.get("notional"):
        return "Position could not be sized."
    s = (f"Size: <b>€{v['notional']:,.0f}</b> ({v['shares']:.4g} shares), "
         f"risking €{v['risk_eur']:.2f} to a stop {v['stop_pct']:.1f}% away.")
    if v["capped"]:
        s += " (Capped by the max-position limit — the stop is tight enough that "
        "risk-based sizing wanted more.)"
    if v["viable"]:
        s += (f" Expected profit €{v['gross']:.2f} vs €{v['cost']:.2f} costs "
              f"({v['ratio']:.1f}× — worth doing).")
    else:
        s += (f" <b>Costs eat it:</b> expected profit €{v['gross']:.2f} against "
              f"€{v['cost']:.2f} in fees. At these fees a position needs to be "
              f"about €{v['breakeven_notional']:,.0f} just to break even, and "
              f"~€{v['worthwhile_notional']:,.0f} to be worth the risk.")
    return s


# ----------------------------------------------------------------------------
# 3. Paper portfolio
# ----------------------------------------------------------------------------
def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"open": [], "closed": [], "started": str(date.today())}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=1)


def update_positions(state: dict, prices: dict, today: str) -> list[str]:
    """Advance open positions one day; close on stop or time exit."""
    p = PORTFOLIO
    events, still_open = [], []
    for pos in state["open"]:
        px = prices.get(pos["ticker"])
        if px is None:                     # no data today: carry it unchanged
            still_open.append(pos)
            continue
        pos["days"] = pos.get("days", 0) + 1
        long = pos["side"] == "LONG"
        hit_stop = px <= pos["stop"] if long else px >= pos["stop"]
        timed_out = pos["days"] >= p["hold_days"]

        if hit_stop or timed_out:
            gross = (px - pos["entry"]) * pos["shares"] * (1 if long else -1)
            net = gross - pos["cost"]
            pct = 100 * gross / pos["notional"] if pos["notional"] else 0
            reason = "stop hit" if hit_stop else f"{p['hold_days']}-day time exit"
            state["closed"].append({**pos, "exit": px, "exit_date": today,
                                    "gross": gross, "net": net, "pct": pct,
                                    "reason": reason})
            events.append(
                f"<b>{pos['ticker']} — CLOSE ({reason}).</b> "
                f"In at {pos['entry']:,.2f}, out at {px:,.2f} — "
                f"{pct:+.1f}%, €{gross:+.2f} gross, <b>€{net:+.2f} after fees</b>.")
        else:
            left = p["hold_days"] - pos["days"]
            move = 100 * (px - pos["entry"]) / pos["entry"] * (1 if long else -1)
            events.append(
                f"<b>{pos['ticker']} — hold.</b> Day {pos['days']} of "
                f"{p['hold_days']} ({left} to go), {move:+.1f}% so far. "
                f"Stop stays at {pos['stop']:,.2f}.")
            still_open.append(pos)
    state["open"] = still_open
    return events


def open_positions(state: dict, candidates, today: str) -> list[str]:
    """Fill free slots from today's candidates (best score first)."""
    p = PORTFOLIO
    events = []
    held = {pos["ticker"] for pos in state["open"]}
    free = p["max_positions"] - len(state["open"])
    if free <= 0 or candidates is None or not len(candidates):
        return events

    for _, c in candidates.iterrows():
        if free <= 0:
            break
        if c["ticker"] in held:
            continue
        if str(c.get("earnings", "")).endswith("!"):        # earnings too close
            continue
        v = size_and_verdict(float(c["close"]), float(c["stop"]),
                             float(c.get("expected_pct", 0.3)))
        if not v.get("notional"):
            continue
        state["open"].append({
            "ticker": c["ticker"], "side": c["side"], "entry": float(c["close"]),
            "stop": float(c["stop"]), "shares": v["shares"], "notional": v["notional"],
            "cost": v["cost"], "date": today, "days": 0, "viable": v["viable"],
        })
        events.append(
            f"<b>{c['ticker']} — OPEN {c['side'].lower()}</b> at {c['close']:,.2f}. "
            + sizing_sentence(v))
        free -= 1
        held.add(c["ticker"])
    return events


def performance(state: dict) -> dict:
    closed = state["closed"]
    if not closed:
        return {"n": 0}
    gross = sum(t["gross"] for t in closed)
    net = sum(t["net"] for t in closed)
    wins = sum(1 for t in closed if t["gross"] > 0)
    return {"n": len(closed), "gross": gross, "net": net,
            "win_pct": 100 * wins / len(closed),
            "fees": sum(t["cost"] for t in closed),
            "equity": PORTFOLIO["account_eur"] + net}


def render(state: dict, events: list[str], path_html: str, path_txt: str):
    p, perf = PORTFOLIO, performance(state)
    css = ("body{font-family:system-ui;margin:24px;max-width:860px;line-height:1.5;color:#111}"
           ".card{border:1px solid #e2e2e2;border-left:5px solid #888;border-radius:10px;"
           "padding:12px 18px;margin:10px 0}.meta{color:#666;font-size:13px}"
           ".note{background:#f7f7f9;border-radius:8px;padding:12px 16px;font-size:14px;color:#444}")
    parts = [f"<style>{css}</style><h1>Paper portfolio</h1>",
             f"<p class=meta>Updated {datetime.now():%Y-%m-%d %H:%M} · "
             f"€{p['account_eur']:,.0f} simulated account · {p['risk_pct']}% risk per trade · "
             f"€{p['cost_per_order_eur']:.2f}/order + {p['spread_pct']}% spread · "
             "no real money · not financial advice.</p>"]

    if perf["n"]:
        parts.append(
            f"<div class=card><b>Results so far:</b> {perf['n']} closed trades, "
            f"{perf['win_pct']:.0f}% winners. "
            f"<b>€{perf['gross']:+.2f} gross</b>, paid €{perf['fees']:.2f} in fees, "
            f"<b>€{perf['net']:+.2f} net</b>. "
            f"Simulated balance: €{perf['equity']:,.2f}.</div>")
    else:
        parts.append("<div class=card>No closed trades yet — results appear here "
                     "once the first positions run their course.</div>")

    parts.append("<h2>Today</h2>")
    parts += [f"<div class=card>{e}</div>" for e in events] or \
             ["<div class=card>Nothing to do today.</div>"]

    parts.append(
        "<div class=note><b>What this is.</b> A simulation that takes the daily "
        "candidates automatically, sizes each one to risk "
        f"{p['risk_pct']}% of a €{p['account_eur']:,.0f} account, and closes it on the stop "
        f"or after {p['hold_days']} trading days — whichever comes first. "
        "No money is involved and nothing here is a recommendation.<br><br>"
        "<b>Watch the gap between gross and net.</b> Gross is whether the signals "
        "work. Net is what would actually reach you after fees. If net stays negative "
        "while gross is positive, the strategy is fine and the account is simply too "
        "small for these costs — the fix is bigger positions or cheaper trading, not "
        "different signals.<br><br>"
        "Settings live at the top of <code>portfolio.py</code>.</div>")

    with open(path_html, "w") as f:
        f.write("".join(parts))

    # plain-text version for the daily email digest
    import re
    lines = ["## Paper portfolio", ""]
    if perf["n"]:
        lines.append(f"- Results: {perf['n']} closed, {perf['win_pct']:.0f}% winners, "
                     f"EUR {perf['gross']:+.2f} gross / {perf['net']:+.2f} net "
                     f"after EUR {perf['fees']:.2f} fees.")
    lines += [f"- {re.sub('<[^>]+>', '', e)}" for e in events] or ["- Nothing today."]
    with open(path_txt, "w") as f:
        f.write("\n".join(lines))


def run(candidates, prices: dict, today: str,
        path_html="docs/portfolio.html", path_txt="docs/portfolio_digest.txt") -> dict:
    """Called once per scan: advance, then fill free slots."""
    state = load_state()
    events = update_positions(state, prices, today)
    events += open_positions(state, candidates, today)
    save_state(state)
    render(state, events, path_html, path_txt)
    return performance(state)
