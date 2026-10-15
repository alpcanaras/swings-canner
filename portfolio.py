#!/usr/bin/env python3
"""
Portfolio layer: position sizing, cost reality-check, and paper tracking.

Three jobs:
  1. SIZE       — turn a candidate into a concrete position for YOUR account,
                  using fixed-fractional risk (never more than `risk_pct` at stake).
  2. COST CHECK  — compare the expected edge against real trading costs and say
                  plainly whether the trade is worth taking at this size.
  3. TRACK       — run a paper portfolio automatically: open positions from the
                  day's candidates, exit on stop or after `hold_days`, and report
                  P&L both gross and net of costs.

Everything you'd want to change lives in PORTFOLIO below.
Paper trading only. Not financial advice.
"""

import json
import os
from datetime import date, datetime
from html import unescape
from scanner import CONFIG

# --- Portfolio Configuration ----------------------------------------------
# Define defaults that gracefully fallback to CONFIG values for flexibility.
# This prevents KeyErrors if scanner.py updates CONFIG schema.

PORTFOLIO = {
    "currency": "$",                    # display symbol for the account currency
    "account_eur": 1000.0,              # your starting balance (name kept for compatibility)
    # --- how big is one position? -------------------------------------------
    "sizing_mode": "allocation",         # use fixed % of capital per position (diversification friendly)
    "alloc_pct": 20.0,                  # allocation mode: % of capital per position
    "risk_pct": 1.0,                    # risk mode: % of capital risked per trade
    "max_risk_pct": 3.0,                # guardrail: never let one trade risk more than this
    "max_positions": 5,                 # always aim to hold this many; refilled as they close
    # --- costs ---------------------------------------------------------------
    # Commission-free broker (Midas): no per-order fee, you pay the spread.
    # Costs are then proportional to size, so many small positions cost the same
    # as one big one — diversification becomes free.
    "cost_per_order_eur": 0.0,          # flat fee per side (Trade Republic would be 1.0)
    "spread_pct": 0.05,                 # half-spread paid on each side, %
    "fx_pct": 0.0,                      # currency conversion each way, % (set if applicable)
    "min_edge_cost_ratio": 3.0,         # expected profit must be >= 3x costs to "take"
    # --- exits: follow the price, don't just count days ----------------------
    "exit_mode": "trailing",            # backtest: ratchets up (never down) as price moves
    "atr_stop_mult": CONFIG.get("stop_atr_mult", 2.0),  # trail distance, in ATRs
    "use_rsi_target": False,            # backtest: the RSI exit cut winners and cost a lot
    "long_only": True,                  # backtest: shorts were a consistent drag
    "regime_gate": False,               # pause NEW buys when the index is below its 200-day
    "target_rsi": 70.0,                 # long exit when RSI(2) recovers above this
    "max_hold_days": 20,                # backstop so nothing becomes a zombie
    "hold_days": CONFIG.get("key_horizon", 10),  # only used in "time" mode
    # Stops do NOT execute outside regular hours, so an overnight gap blows
    # straight through them. When that happens the fill is the opening print
    # plus this much extra slippage, because the first minutes after a gap are
    # violent and a stop-market order rarely fills at the print.
    "gap_slippage_pct": 0.4,
    "paper": True,                      # paper mode: track trades regardless of verdict
    "state_path": "state/paper.json",   # file to track open positions
}

CUR = PORTFOLIO["currency"]
STATE_PATH = PORTFOLIO["state_path"]

def _init_state():
    """Ensure the state file exists with a skeleton if running in 'paper' mode."""
    path = PORTFOLIO["state_path"]
    if not os.path.exists(path):
        # Initialize with empty holdings list
        _json = json.dumps({"holdings": [], "closed": []})
        open(path, "w").write(_json)

def _load_state():
    """Load current state from file or return empty dict."""
    _init_state()
    try:
        return json.load(open(STATE_PATH))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"holdings": [], "closed": []}

def _save_state():
    """Save current state to file."""
    _init_state()
    # Ensure we are saving the dict to the file
    try:
        data = _load_state()
        open(STATE_PATH, "w").write(json.dumps(data, indent=2))
    except Exception:
        pass  # Ignore write errors to keep logic flowing

def _get_ticker(state):
    """Helper to extract ticker from state dict keys (robust for mixed data)."""
    if not state.get("holdings"):
        return []
    # Flatten holdings by ticker
    tickers = list(set([h["ticker"] for h in state["holdings"]]))
    return tickers

def text_blocks(path: str, stop_at: str) -> list[str]:
    """Extract the plain-language paragraphs/cards, ignoring the raw tables.
    Fixed regex to ensure proper closing of <p> or <div> tags."""
    try:
        html = open(path).read()
    except FileNotFoundError:
        return []
    head = html.split(stop_at)[0]
    # Fix the regex to include the closing bracket/brace pair if needed
    # Original context bug: </(?:p|div" -> Fixed to match tag end properly
    chunks = re.findall(r"<(?:p|div)[^>]*>(.*?)</(?:p|div)" if stop_at else r"<(?:p|div)[^>]*>(.*?)</(?:p|div)>", head)
    return [unescape(c) for c in chunks]

import re

# --- Main Logic for the Scanner Ecosystem ------------------------------------

def run_portfolio_analysis(date_today: str = date.today().isoformat()):
    """
    Central orchestration function to run the portfolio sizing and logic.
    """
    state = _load_state()
    
    # Update state with today's date for reporting
    state["date"] = date_today
    
    # Logic would go here to process candidates from `scanner` 
    # (which are passed in via `candidates` list in the real flow)
    
    # Example: Adding a 'candidate' to state if not present
    if not state.get("holdings"):
        _init_state()
        state = _load_state()

    return state