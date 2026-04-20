"""Donchian Channel breakout — classic trend-following strategy.

Reference: Richard Donchian (1950s), popularized by the Turtle Traders
(Dennis / Eckhardt 1983). The strategy has 50+ years of track record
across commodities, FX, equities, with well-known live performance
from CTAs (Dunn, Chesapeake, Man AHL, Winton).

Core idea:

- **Entry (LONG)**: today's close > highest high of the prior N bars
  (typical N=20 for short-term, 55 for long-term)
- **Exit**: today's close < lowest low of the prior M bars (M < N,
  typically M = N/2). This creates an asymmetric "keep profits running,
  cut losses" profile WITHOUT a fixed take-profit.
- **Stop-loss**: ATR-based trailing stop, widens to avoid noise stops.

Why this might work when the old indicator-voting didn't:

1. **No fixed take-profit** — the old bot locked in small wins and
   missed 200-400% moves (META, TSLA, NFLX). Donchian lets profits
   compound until the trend actually breaks.
2. **ATR-wide stops** — the old 1.5×ATR stop got noise-chopped out.
   A trailing stop of 2-3×ATR with tightening-only logic is more
   patient with normal volatility.
3. **Clean entry signal** — "new 20-day high on close" is
   unambiguous. The old scoring system (RSI+MACD+BB+...) was a
   committee vote that rarely agreed cleanly.

No guarantee this works on US equities 2020-2024; the backtest will
say. But it has a clear theoretical basis and decades of evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from backtest.models import Bar


Signal = Literal["LONG_ENTRY", "LONG_EXIT", "NONE"]


@dataclass
class DonchianReading:
    """One day's Donchian state for a ticker."""
    trade_date: date
    close: float
    high: float
    low: float
    upper_entry: float | None  # highest high over entry lookback
    lower_exit: float | None   # lowest low over exit lookback
    atr: float | None          # ATR for position sizing + trailing stop
    signal: Signal


def compute_channel_bounds(
    bars: list[Bar],
    entry_lookback: int = 20,
    exit_lookback: int = 10,
) -> tuple[float | None, float | None]:
    """Return (highest_high, lowest_low) over the prior N / M bars,
    EXCLUDING the current bar (so the signal is based on prior highs
    only — avoids "today's own high makes today's breakout" tautology).

    Returns ``(None, None)`` if not enough bars.
    """
    if len(bars) < max(entry_lookback, exit_lookback) + 1:
        return None, None
    # Bars are assumed chronological; "prior N" = bars[-(N+1):-1].
    prior_entry = bars[-(entry_lookback + 1):-1]
    prior_exit = bars[-(exit_lookback + 1):-1]
    if len(prior_entry) < entry_lookback or len(prior_exit) < exit_lookback:
        return None, None
    return (max(b.high for b in prior_entry), min(b.low for b in prior_exit))


def compute_atr(bars: list[Bar], period: int = 14) -> float | None:
    """Wilder's ATR over the given period. Requires ``period + 1`` bars
    minimum because TR needs a previous close reference."""
    if len(bars) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(bars)):
        hi, lo, prev_close = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
    if len(trs) < period:
        return None
    # Simple mean-of-last-N for the MVP (not Wilder smoothing). Close
    # enough for strategy signals; tests pin the exact behaviour.
    atr = sum(trs[-period:]) / period
    return atr


def analyze_bar(
    bars: list[Bar],
    entry_lookback: int = 20,
    exit_lookback: int = 10,
    atr_period: int = 14,
    held_long: bool = False,
) -> DonchianReading:
    """Decide whether today's bar is a LONG_ENTRY (new high breakout)
    or, if already holding, a LONG_EXIT (fell through the exit channel).

    A position in-progress cannot also generate an ENTRY — the first
    call with ``held_long=False`` that sees a breakout gets ENTRY;
    subsequent calls until an exit get NONE (or LONG_EXIT when due).
    """
    last = bars[-1]
    upper, lower = compute_channel_bounds(bars, entry_lookback, exit_lookback)
    atr = compute_atr(bars, atr_period)

    signal: Signal = "NONE"
    if held_long:
        if lower is not None and last.close < lower:
            signal = "LONG_EXIT"
    else:
        if upper is not None and last.close > upper:
            signal = "LONG_ENTRY"

    return DonchianReading(
        trade_date=last.trade_date,
        close=last.close, high=last.high, low=last.low,
        upper_entry=upper, lower_exit=lower, atr=atr,
        signal=signal,
    )


def position_size(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    risk_per_trade: float,
) -> int:
    """Classic ATR-risk sizing: risk X% of equity on the distance from
    entry to stop. Same formula as ``core/risk_manager.py`` but inlined
    so the backtester stays independent of the live code.
    """
    import math

    if account_balance <= 0 or entry_price <= 0:
        return 0
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        return 0
    risk_dollars = account_balance * risk_per_trade
    shares_by_risk = math.floor(risk_dollars / risk_per_share)
    shares_by_cash = math.floor(account_balance / entry_price)
    return max(0, min(shares_by_risk, shares_by_cash))


def atr_initial_stop(entry_price: float, atr: float, atr_multiple: float = 2.0) -> float:
    """Initial stop N ATRs below entry (long-only for now)."""
    return round(entry_price - atr * atr_multiple, 2)


def ratchet_trailing_stop(
    current_stop: float,
    high_since_entry: float,
    atr: float,
    atr_multiple: float = 2.0,
) -> float:
    """Never lowers a stop. Raises it to ``high_since_entry - N×ATR``
    only if that's above the current stop (classic "chandelier exit")."""
    proposed = round(high_since_entry - atr * atr_multiple, 2)
    return max(current_stop, proposed)
