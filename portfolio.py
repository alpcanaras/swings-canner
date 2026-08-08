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
    "account_eur": 1000.0,       # your starting balance
    # --- how big is one position? -------------------------------------------
    # "allocation": put a fixed % of capital into each position (simple, and what
    #               you want when flat fees punish small positions).
    # "risk":       size so the distance to the stop equals `risk_pct` of capital
    #               (classic, but produces tiny positions on small accounts).
    "sizing_mode": "allocation",
    "alloc_pct": 20.0,           # allocation mode: % of capital per position
    "risk_pct": 1.0,             # risk mode: % of capital risked per trade
    "max_risk_pct": 3.0,         # guardrail: never let one trade risk more than this
    "max_positions": 5,          # always aim to hold this many; refilled as they close
    # --- costs ---------------------------------------------------------------
    # Commission-free broker (Midas): no per-order fee, you pay the spread.
    # Costs are then proportional to size, so many small positions cost the same
    # as one big one — diversification becomes free.
    "cost_per_order_eur": 0.0,   # flat fee per side (Trade Republic would be 1.0)
    "spread_pct": 0.05,          # half-spread paid on each side, %
    "fx_pct": 0.0,               # currency conversion each way, % (set if applicable)
    "min_edge_cost_ratio": 3.0,  # expected profit must be >= 3x costs to "take"
    # --- exits: follow the price, don't just count days ----------------------
    # "trailing": initial stop at entry - 2*ATR; it ratchets up (never down) as
    #             price moves your way, and the trade also closes when the bounce
    #             is done (RSI target) — no fixed holding period.
    # "time":     the old behaviour, close after `hold_days`.
    "exit_mode": "trailing",
    "atr_stop_mult": CONFIG["stop_atr_mult"],   # trail distance, in ATRs
    "target_rsi": 70.0,          # long exit when RSI(2) recovers above this
    "max_hold_days": 20,         # backstop so nothing becomes a zombie
    "hold_days": CONFIG["key_horizon"],   # only used in "time" mode
    "paper": True,               # paper mode: track trades regardless of verdict,
                                 # so you can see gross vs. net over time
}

STATE_PATH = "state/paper.json"
SHADOW_PATH = "state/shadow.json"

# Several signals measure nearly the same thing, so counting them as independent
# confirmations overstates conviction. Group them into families instead.
FAMILIES = {
    "rsi": "mean reversion", "ibs": "mean reversion", "extreme": "mean reversion",
    "donchian": "momentum", "sweep": "failed breakout", "fvg": "gap",
}


def family_of(signal_name: str) -> str:
    n = signal_name.lower()
    for key, fam in FAMILIES.items():
        if key in n:
            return fam
    return "other"


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

    if p["sizing_mode"] == "allocation":
        notional = p["account_eur"] * p["alloc_pct"] / 100
    else:                                                     # risk-based sizing
        notional = (p["account_eur"] * p["risk_pct"] / 100) / (stop_pct / 100)

    # guardrail: cap the position so one trade can't risk more than max_risk_pct
    max_risk_eur = p["account_eur"] * p["max_risk_pct"] / 100
    risk_at_size = notional * stop_pct / 100
    capped = risk_at_size > max_risk_eur
    if capped:
        notional = max_risk_eur / (stop_pct / 100)

    risk_eur = notional * stop_pct / 100
    shares = notional / close

    cost = (2 * p["cost_per_order_eur"]
            + 2 * notional * (p["spread_pct"] + p["fx_pct"]) / 100)
    gross = notional * expected_pct / 100
    net = gross - cost
    ratio = gross / cost if cost > 0 else float("inf")

    # what this trade would need to be worth doing
    fixed = 2 * p["cost_per_order_eur"]
    per_eur_edge = expected_pct / 100 - 2 * (p["spread_pct"] + p["fx_pct"]) / 100
    if per_eur_edge <= 0:                       # spread alone swallows the edge
        breakeven = worthwhile = float("inf")
    elif fixed == 0:                            # commission-free: size-independent
        breakeven = worthwhile = 0.0
    else:
        breakeven = fixed / per_eur_edge
        worthwhile = breakeven * p["min_edge_cost_ratio"]

    return {
        "viable": ratio >= p["min_edge_cost_ratio"],
        "alloc_pct": 100 * notional / p["account_eur"],
        "risk_pct_actual": 100 * risk_eur / p["account_eur"],
        "stop_pct": stop_pct, "risk_eur": risk_eur, "notional": notional,
        "shares": shares, "capped": capped, "cost": cost, "gross": gross,
        "net": net, "ratio": ratio, "breakeven_notional": breakeven,
        "worthwhile_notional": worthwhile, "expected_pct": expected_pct,
    }


def sizing_sentence(v: dict) -> str:
    """The sizing result, in words."""
    if not v.get("viable") and not v.get("notional"):
        return "Position could not be sized."
    s = (f"Size: <b>€{v['notional']:,.0f}</b> = {v['alloc_pct']:.0f}% of capital "
         f"({v['shares']:.4g} shares). If the stop hits you lose "
         f"€{v['risk_eur']:.2f} ({v['risk_pct_actual']:.1f}% of the account), "
         f"the stop being {v['stop_pct']:.1f}% away.")
    if v["capped"]:
        s += (" (Trimmed below the normal allocation — this one is volatile enough "
              "that a full-size position would breach the risk guardrail.)")
    if v["viable"]:
        s += (f" Expected profit €{v['gross']:.2f} vs €{v['cost']:.2f} costs "
              f"({v['ratio']:.1f}× — worth doing).")
    elif PORTFOLIO["cost_per_order_eur"] == 0:
        # commission-free: the ratio doesn't depend on size, so a bigger
        # position wouldn't help — the edge itself is just thin here.
        s += (f" Thin edge: expected profit €{v['gross']:.2f} against "
              f"€{v['cost']:.2f} in spread ({v['ratio']:.1f}×, below the "
              f"{PORTFOLIO['min_edge_cost_ratio']:.0f}× bar). Sizing up wouldn't "
              "change that — costs scale with the position.")
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


def update_positions(state: dict, market: dict, today: str) -> list[str]:
    """Advance open positions one day.

    market[ticker] = {"close": float, "atr": float, "rsi": float}
    In trailing mode the stop ratchets in your favour and never loosens; the
    trade closes when price hits that stop, when the move has played out
    (RSI target), or at the max-hold backstop.
    """
    p = PORTFOLIO
    events, still_open = [], []
    for pos in state["open"]:
        m = market.get(pos["ticker"])
        if not m:                          # no data today: carry it unchanged
            still_open.append(pos)
            continue
        px, atr, rsi = m["close"], m.get("atr"), m.get("rsi")
        lo, hi, op = m.get("low", px), m.get("high", px), m.get("open", px)
        pos["days"] = pos.get("days", 0) + 1
        long = pos["side"] == "LONG"
        trailing = p["exit_mode"] == "trailing"

        # --- was the stop touched INTRADAY? ----------------------------------
        # A real stop order triggers the moment price trades through it, not at
        # the close. If the day opened beyond the stop, the fill is the open
        # (gap), otherwise it fills at the stop itself.
        stop_now = pos["stop"]
        if long and lo <= stop_now:
            hit_intraday, fill = True, min(op, stop_now)
        elif not long and hi >= stop_now:
            hit_intraday, fill = True, max(op, stop_now)
        else:
            hit_intraday, fill = False, None

        if hit_intraday:
            gross = (fill - pos["entry"]) * pos["shares"] * (1 if long else -1)
            net = gross - pos["cost"]
            pct = 100 * gross / pos["notional"] if pos["notional"] else 0
            gapped = (op < stop_now) if long else (op > stop_now)
            reason = ("gapped through the stop" if gapped else
                      "trailing stop" if pos.get("trailed") else "stop hit")
            state["closed"].append({**pos, "exit": fill, "exit_date": today,
                                    "gross": gross, "net": net, "pct": pct,
                                    "reason": reason})
            events.append(
                f"<b>{pos['ticker']} — CLOSE ({reason}).</b> "
                f"In at {pos['entry']:,.2f}, out at {fill:,.2f} — "
                f"{pct:+.1f}%, €{gross:+.2f} gross, <b>€{net:+.2f} after fees</b>.")
            continue

        # --- trail the stop (never against you) ------------------------------
        if trailing and atr:
            trail = (px - p["atr_stop_mult"] * atr if long
                     else px + p["atr_stop_mult"] * atr)
            new_stop = max(pos["stop"], trail) if long else min(pos["stop"], trail)
            if new_stop != pos["stop"]:
                pos["stop"] = round(new_stop, 4)
                pos["trailed"] = True

        hit_stop = False            # intraday check above already handled this
        if trailing:
            target_hit = rsi is not None and (
                rsi >= p["target_rsi"] if long else rsi <= 100 - p["target_rsi"])
            timed_out = pos["days"] >= p["max_hold_days"]
        else:
            target_hit = False
            timed_out = pos["days"] >= p["hold_days"]

        if hit_stop or timed_out or target_hit:
            gross = (px - pos["entry"]) * pos["shares"] * (1 if long else -1)
            net = gross - pos["cost"]
            pct = 100 * gross / pos["notional"] if pos["notional"] else 0
            reason = ("trailing stop" if hit_stop and pos.get("trailed") else
                      "stop hit" if hit_stop else
                      "move played out (RSI target)" if target_hit else
                      f"{p['max_hold_days' if p['exit_mode'] == 'trailing' else 'hold_days']}"
                      "-day backstop")
            state["closed"].append({**pos, "exit": px, "exit_date": today,
                                    "gross": gross, "net": net, "pct": pct,
                                    "reason": reason})
            events.append(
                f"<b>{pos['ticker']} — CLOSE ({reason}).</b> "
                f"In at {pos['entry']:,.2f}, out at {px:,.2f} — "
                f"{pct:+.1f}%, €{gross:+.2f} gross, <b>€{net:+.2f} after fees</b>.")
        else:
            move = 100 * (px - pos["entry"]) / pos["entry"] * (1 if long else -1)
            if p["exit_mode"] == "trailing":
                locked = 100 * (pos["stop"] - pos["entry"]) / pos["entry"] * (1 if long else -1)
                state_txt = (f"stop trailed up to {pos['stop']:,.2f}"
                             if pos.get("trailed") else f"stop at {pos['stop']:,.2f}")
                risk_txt = (f", which now locks in {locked:+.1f}%"
                            if locked > 0 else "")
                events.append(
                    f"<b>{pos['ticker']} — hold.</b> Day {pos['days']}, "
                    f"{move:+.1f}% so far; {state_txt}{risk_txt}.")
            else:
                left = p["hold_days"] - pos["days"]
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


def shadow_run(candidates, market: dict, today: str) -> dict:
    """Track EVERY candidate to its exit, with no capital limit.

    Same entry and exit rules as the real portfolio, but each position is a
    notional €100 so results read as percentages. This exists to answer
    'does the strategy work' quickly — the 5-slot portfolio answers
    'what would it have done to my account'.
    """
    p = PORTFOLIO
    global STATE_PATH
    real, STATE_PATH = STATE_PATH, SHADOW_PATH        # reuse load/save/update
    try:
        state = load_state()
        update_positions(state, market, today)        # events not needed here
        held = {pos["ticker"] for pos in state["open"]}
        if candidates is not None and len(candidates):
            for _, c in candidates.iterrows():
                if c["ticker"] in held:
                    continue
                close = float(c["close"])
                fams = sorted({family_of(n) for n in c["_fired"]})
                state["open"].append({
                    "ticker": c["ticker"], "side": c["side"], "entry": close,
                    "stop": float(c["stop"]), "shares": 100.0 / close,
                    "notional": 100.0,
                    "cost": 100.0 * 2 * (p["spread_pct"] + p["fx_pct"]) / 100,
                    "date": today, "days": 0,
                    "families": fams, "n_signals": len(c["_fired"]),
                })
                held.add(c["ticker"])
        save_state(state)
        return shadow_stats(state)
    finally:
        STATE_PATH = real


def shadow_stats(state: dict) -> dict:
    """Aggregate the shadow log overall and per signal family."""
    closed = state.get("closed", [])
    if not closed:
        return {"n": 0, "open": len(state.get("open", []))}

    def agg(rows):
        n = len(rows)
        return {
            "n": n,
            "win_pct": 100 * sum(1 for t in rows if t["gross"] > 0) / n,
            "avg_gross": sum(t["pct"] for t in rows) / n,
            "avg_net": sum(100 * t["net"] / t["notional"] for t in rows) / n,
            "days": sum(t["days"] for t in rows) / n,
        }

    per_family = {}
    for fam in sorted({f for t in closed for f in t.get("families", [])}):
        rows = [t for t in closed if fam in t.get("families", [])]
        if len(rows) >= 5:                       # ignore tiny samples
            per_family[fam] = agg(rows)
    out = agg(closed)
    out.update({"open": len(state.get("open", [])), "by_family": per_family})
    return out


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


def shadow_block(sh: dict) -> str:
    """The 'does the strategy work' panel, in words."""
    if not sh or not sh.get("n"):
        return ("<div class=card><b>Signal log:</b> tracking "
                f"{sh.get('open', 0)} signals; results appear once the first ones "
                "close.</div>")
    s = (f"<div class=card><b>Every signal tracked ({sh['n']} closed, "
         f"{sh['open']} still open):</b> {sh['win_pct']:.0f}% were profitable, "
         f"averaging <b>{sh['avg_gross']:+.2f}%</b> per trade "
         f"({sh['avg_net']:+.2f}% after spread), held {sh['days']:.1f} days on average.")
    if sh.get("by_family"):
        rows = "".join(
            f"<tr><td>{fam}</td><td>{d['n']}</td><td>{d['win_pct']:.0f}%</td>"
            f"<td>{d['avg_gross']:+.2f}%</td><td>{d['avg_net']:+.2f}%</td>"
            f"<td>{d['days']:.1f}</td></tr>"
            for fam, d in sh["by_family"].items())
        s += ("<table><tr><th>signal family</th><th>trades</th><th>win rate</th>"
              "<th>avg gross</th><th>avg net</th><th>avg days</th></tr>"
              + rows + "</table>")
    return s + "</div>"


def render(state: dict, events: list[str], path_html: str, path_txt: str,
           shadow: dict | None = None):
    p, perf = PORTFOLIO, performance(state)
    css = ("body{font-family:system-ui;margin:24px;max-width:860px;line-height:1.5;color:#111}"
           ".card{border:1px solid #e2e2e2;border-left:5px solid #888;border-radius:10px;"
           "padding:12px 18px;margin:10px 0}.meta{color:#666;font-size:13px}"
           ".note{background:#f7f7f9;border-radius:8px;padding:12px 16px;font-size:14px;color:#444}")
    parts = [f"<style>{css}</style><h1>Paper portfolio</h1>",
             f"<p class=meta>Updated {datetime.now():%Y-%m-%d %H:%M} · "
             f"€{p['account_eur']:,.0f} simulated account · "
             + (f"{p['alloc_pct']:.0f}% of capital per position"
                if p["sizing_mode"] == "allocation"
                else f"{p['risk_pct']}% risk per trade")
             + f", up to {p['max_positions']} at once · "
             + ("commission-free, "
                if p["cost_per_order_eur"] == 0 else
                f"€{p['cost_per_order_eur']:.2f}/order, ")
             + f"{p['spread_pct']}% spread · "
             + ("price-following exits" if p["exit_mode"] == "trailing"
                else f"{p['hold_days']}-day exits")
             + " · no real money · not financial advice.</p>"]

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

    parts.append("<h2>Does the strategy work?</h2>")
    parts.append(shadow_block(shadow or {}))

    parts.append(f"<h2>Today (portfolio: {len(state['open'])}/"
                 f"{p['max_positions']} slots filled)</h2>")
    parts += [f"<div class=card>{e}</div>" for e in events] or \
             ["<div class=card>Nothing to do today.</div>"]

    parts.append(
        "<div class=note><b>What this is.</b> A simulation that takes the daily "
        f"candidates automatically, puts {p['alloc_pct']:.0f}% of a "
        f"€{p['account_eur']:,.0f} account into each (up to {p['max_positions']} at a time, "
        f"trimmed if a trade would risk more than {p['max_risk_pct']}%). "
        + (f"Exits follow the price: the stop starts {p['atr_stop_mult']}×ATR away and "
           "ratchets up as the trade works, never down. A position closes when price "
           "hits that trailing stop, when the bounce has played out "
           f"(RSI(2) back above {p['target_rsi']:.0f}), or at a "
           f"{p['max_hold_days']}-day backstop — there is no fixed holding period."
           if p["exit_mode"] == "trailing" else
           f"Positions close on the stop or after {p['hold_days']} trading days.")
        + " "
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
    if shadow and shadow.get("n"):
        lines.append(f"- All signals tracked: {shadow['n']} closed, "
                     f"{shadow['win_pct']:.0f}% profitable, "
                     f"{shadow['avg_gross']:+.2f}% average ({shadow['avg_net']:+.2f}% "
                     f"after spread), {shadow['days']:.1f} days held on average.")
    if perf["n"]:
        lines.append(f"- Portfolio: {perf['n']} closed, {perf['win_pct']:.0f}% winners, "
                     f"EUR {perf['gross']:+.2f} gross / {perf['net']:+.2f} net "
                     f"after EUR {perf['fees']:.2f} fees.")
    lines += [f"- {re.sub('<[^>]+>', '', e)}" for e in events] or ["- Nothing today."]
    with open(path_txt, "w") as f:
        f.write("\n".join(lines))


def run(candidates, market: dict, today: str,
        path_html="docs/portfolio.html", path_txt="docs/portfolio_digest.txt",
        all_candidates=None) -> dict:
    """Called once per scan: advance, then fill free slots.

    market[ticker] = {"close": .., "atr": .., "rsi": ..}
    """
    state = load_state()
    events = update_positions(state, market, today)
    events += open_positions(state, candidates, today)
    save_state(state)
    shadow = shadow_run(all_candidates if all_candidates is not None else candidates,
                        market, today)
    render(state, events, path_html, path_txt, shadow)
    perf = performance(state)
    perf["shadow"] = shadow
    return perf
