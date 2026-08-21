# Market scanner — three products, three emails

Runs itself on GitHub Actions (free). You get three separate emails, each posted
as a GitHub issue so it lands in your inbox:

| Email | Cadence | What it is | Honest status |
|---|---|---|---|
| **[Swing]** stocks | daily | Days-to-weeks trade sheet: picks, signal price, stop $/%, trailing rule, updated stops | Best of 8 backtested variants (long-only, trailing). Still lost to buy-and-hold over 10y — traded eyes-open. |
| **[Long-term]** FA | weekly (Mon) | Quality + value + trend ranking of large caps; a research shortlist | Months-to-years. Not signals. `fa_scanner.py` |
| **[Crypto]** | daily | **Regime brief** (risk-on/off via 200-day trend) + risk screener + small-cap discovery | Swing-trading crypto lost badly to holding; a trend filter roughly halved drawdown, so the brief leads. `crypto_scanner.py`, `crypto_screener.py` |

Key finding from replaying real history: on both stocks and crypto, mechanical
swing-trading **underperformed simply buying and holding** — but a trend filter
(stay long above the 200-day, step aside below) cut crypto's drawdown about in
half. So the products are framed around what the evidence supports, not hype.
The "gem finder" is deliberately a **red-flag screener**, not a picker: a public
feed has no edge finding gems before insiders, and that corner is scam-heavy.

Not financial advice; I am not a licensed adviser.

---

## Swing engine (the original)

Three parts, one system:

| File | Where it runs | What you get |
|---|---|---|
| `scanner.py` | Your computer (Python) | **Market regime**: NDX/RUT index bias + historical edge stats per signal |
| `stock_scanner.py` | Your computer (Python) | **Trade candidates**: ranked individual stocks (default: Nasdaq-100) |
| `swing_scanner.pine` | TradingView | On-chart signals + server-side alerts on any symbol |

## Daily workflow

```
pip install yfinance pandas numpy lxml     # once
python scanner.py                          # 1. what's the index regime?
python stock_scanner.py                    # 2. which stocks are set up?
```

Run after US close (22:00 CET). Take stock **longs only when the index bias isn't SHORT**, and vice versa — trading against the market regime is the main way single-stock setups fail.

`stock_scanner.py` universes:

```
python stock_scanner.py                   # Nasdaq-100 (live constituents from Wikipedia)
python stock_scanner.py --universe r2k    # Russell 2000 via iShares IWM holdings;
                                          # falls back to S&P SmallCap 600 when that
                                          # source is bot-walled (always is on GitHub runners)
python stock_scanner.py --universe all    # large + small caps; first run takes a while
python stock_scanner.py --tickers my.txt  # any custom list
```

Small caps note: the $20M dollar-volume floor drops the illiquid tail of the Russell — intentional. Spreads on illiquid small caps eat swing-trade edges alive.

## Position sizing and the cost problem (read this once)

`portfolio.py` sizes every candidate for a real account and runs an automatic **paper portfolio** — it opens positions from the daily list, exits on the stop or after 5 trading days, and reports P&L both gross and net of fees. Settings are the block at the top of the file.

Sizing is **allocation-based** by default: a fixed % of capital per position, trimmed automatically if a trade would risk more than `max_risk_pct` of the account. That suits flat-fee brokers, where small positions are punished.

**The uncomfortable arithmetic**, at €1,000 with €1 per order and a typical 0.3% edge per trade:

| positions | % each | € per position | expected profit | costs | net |
|---|---|---|---|---|---|
| 1 | 100% | €1,000 | €3.00 | €3.00 | €0.00 |
| 2 | 50% | €500 | €1.50 | €2.50 | −€1.00 |
| 3 | 33% | €330 | €0.99 | €2.33 | −€1.34 |
| 5 | 20% | €200 | €0.60 | €2.20 | −€1.60 |

Diversifying makes it *worse*, because each flat fee is charged per position regardless of size. At this account size the choice is between concentration (break-even, and your whole account riding on one small-cap) and diversification (guaranteed to lose to fees).

What actually changes the picture, in order of effect:

1. **Commission-free trading.** With spread only (~0.1% round trip vs a 0.3% edge), positions of any size clear the bar. This is the single biggest lever.
2. **A larger account.** At €1/order, roughly €2,000+ per position makes fees a minor drag.
3. **Fewer, bigger trades.** Taking only the strongest setups rather than everything.

Until one of those is true, this stays a paper exercise — which is exactly what the tracker is for. It costs nothing and produces real evidence.

## Access from your phone (free, no server needed)

This sandbox can't reach market-data APIs, so the scanner can't run scheduled here — GitHub Actions is the right home. One-time setup, ~10 min:

1. Create a GitHub repo and push everything in this folder (including `.github/workflows/daily-scan.yml`). Repo must be **public** for free GitHub Pages (the reports contain no personal data).
2. Repo → Settings → Pages → Source: "Deploy from a branch" → branch `main`, folder `/docs`.
3. Done. Every weekday at 22:15 UTC the workflow runs both scanners and updates:
   - `https://<your-username>.github.io/<repo>/` — index bias
   - `https://<your-username>.github.io/<repo>/stocks.html` — ranked stock candidates
4. Bookmark those on your phone. To trigger a run manually (also works from the GitHub mobile app): repo → Actions → daily-scan → Run workflow.

### Getting it emailed to you

After each run the workflow posts the plain-language digest as a **GitHub issue**, and GitHub emails watchers. No SMTP setup, no passwords.

To receive it: on the repo page click **Watch → All Activity**, and make sure email delivery is on at [github.com/settings/notifications](https://github.com/settings/notifications) ("Email" checked under Subscriptions → Watching). Old digests auto-close so only the latest five stay open.

For instant push alerts on specific symbols, TradingView (`swing_scanner.pine`) is still the better channel — Pages and email are pull/daily, TradingView is push/immediate.

### Stock-specific safety rails

- **Pooled validation**: signal edge is measured across the entire universe (~250k signal-days), which is far more statistically robust than per-stock stats.
- **Earnings flag**: candidates reporting within 7 days get flagged — a 2×ATR stop is worthless against an earnings gap. Default: skip those trades.
- **Liquidity floor**: names below $20M average daily dollar volume are dropped.
- **Relative strength**: each candidate shows its 3-month return vs. the universe median — prefer longs in strong names, shorts in weak ones.

## Quick start (TradingView)

Open a **daily** chart of NDX/QQQ or RUT/IWM → Pine Editor → paste `swing_scanner.pine` → Add to chart → right-click the indicator → Add alert → trigger **"Once per bar close"**. Signals only confirm at close; intraday values are provisional.

## The signals

Two families, because they win in different conditions:

**Mean reversion** (indices snap back after short-term extremes — the classic edge on NDX/SPX dailies):
- **RSI(2) < 10 in an uptrend** → long (mirror: RSI(2) > 90 in downtrend → short)
- **IBS < 0.15 in an uptrend** → long (close near day's low; mirror for shorts)
- **7-day closing low in an uptrend** → long (mirror for shorts)

**Momentum**:
- **20-day Donchian breakout** in the direction of the 50/200 SMA trend → hold 1–2 weeks

**ICT-style, made mechanical** (so their edge can actually be measured):
- **Sweep & reclaim**: price takes out the prior 10-day low intraday but closes back above it (a stop run / failed breakdown) → long in uptrend; mirror for shorts. This is ICT's "liquidity sweep" as a falsifiable rule.
- **FVG retest**: a bullish fair-value gap (low > high of two bars back) creates a zone valid for 10 bars; price dipping into the zone and closing inside/above it → long in uptrend; mirror for bearish FVGs.

Both go through the same validation as everything else — if they show no historical edge on NDX/RUT, they get zero weight in the bias. Let the stats table settle the ICT debate for you.

The 200-day SMA acts as the trend filter throughout: only fade dips in uptrends, only fade pops in downtrends. Fighting the primary trend is where mean reversion blows up.

## How the self-validation works (Python version)

For every signal, the scanner computes direction-adjusted forward returns (3/5/10 days) over every historical occurrence, and compares to buy-and-hold drift. A signal's weight in the composite bias = its 5-day edge, floored at zero. **A signal with no historical edge on that index influences nothing.** This is why NDX and RUT can weigh the same signal differently — RUT is choppier and mean-reverts differently than NDX.

Sanity check: run on random synthetic data, the mean-reversion signals correctly show ~zero edge and get zero weight. On real index data you should see meaningfully positive edge for RSI(2)/IBS longs — if you don't, believe the data, not the folklore.

## Risk block (matters more than entries)

- **Stop**: ~2×ATR(14) for a 1–2 week hold. Tighter than that and normal noise stops you out.
- **Knock-outs (Trade Republic)**: barrier at least **3×ATR** from entry, ideally more. A KO barrier inside 2×ATR is a coin flip on getting knocked out by noise before the trade works. Remember the barrier is watched ~24h — overnight/pre-market moves can kill a KO while the index chart looks fine.
- **Sizing**: risk a fixed ~1% of account per trade: `size = (1% × account) / stop distance`. Halve it when volatility is in its top quintile (the scanner flags this).
- **Costs**: KO financing + spread eats multi-week holds. For holds >1 week, lower-leverage KOs or spot (QQQ/IWM equivalents) usually price better than high-leverage certificates.

## Honest caveats

- Fired signal ≠ trade. It's a statistical tilt: expect ~52–60% win rates and small average edges. The money is made by consistent sizing and stops over many trades, not any single signal.
- Past edge can decay. Rerun the stats occasionally; the table tells you if a signal stopped working.
- These are well-known public signals — the edge that survives is small. Treat the first 2–3 months as paper trading; log every signal and what you did.
- Parameters at the top of both files are deliberately un-optimized round numbers. Resist tuning them until backtest stats look great — that's how you overfit.

*Not financial advice — educational tooling.*
