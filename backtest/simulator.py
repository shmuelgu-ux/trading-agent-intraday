"""Backtest simulator — replays the scanner's per-day decisions against
historical bars, simulates fills and SL/TP exits, tracks the full equity
curve.

Design notes:

- **Re-uses live code paths where practical**: the same
  ``technical_analysis.analyze`` + ``analyze_macro`` used by the live
  scanner produces signals here, guaranteeing any future tweak to the
  scoring math shows up in backtest without a second implementation
  drifting out of sync.
- **Does NOT re-use AlpacaClient or JournalService**: the simulator is
  offline. Trades are tracked in plain-Python lists, metrics come out
  of the simulator itself.
- **Entry timing**: a signal fires on the close of day D, entry fills
  at the OPEN of D+1. That's honest about the timing — you couldn't
  have known D's close at 9:31am on D itself.
- **Exit precedence on a conflict day**: if a single bar's high reaches
  the take-profit AND its low reaches the stop-loss, we assume the SL
  was hit first (conservative). The real market can go either way;
  picking the bad one avoids overstating returns.
- **Commissions**: per-share rate, applied on both entry and exit legs.
  Matches the IBKR Pro tier roughly; tune via ``BacktestConfig``.
- **No slippage / short borrow costs yet** — explicit MVP call-out.
  Backtests will therefore be slightly optimistic vs. live trading.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Optional

from backtest.data_loader import BarCache
from backtest.models import (
    BacktestConfig,
    BacktestResult,
    Bar as DomainBar,
    ClosedTrade,
    EquityPoint,
    OpenPosition,
    Side,
)
from core.technical_analysis import (
    Bar as TABar,
    analyze,
    analyze_macro,
    MacroContext,
    AnalysisResult,
)


# Minimum daily bars needed for all indicators in ``analyze`` to be
# computable. The full-history scanner fetches 90; we mirror that.
MIN_DAILY_BARS = 60
# Minimum weekly bars for ``analyze_macro``. The live scanner grabs
# ~105; 50 is the floor inside the macro function itself.
MIN_WEEKLY_BARS = 50
# How many business days ≈ one week, for the D→weekly resampling below.
BUSINESS_DAYS_PER_WEEK = 5


TRADING_DAYS_PER_YEAR = 252


def _to_ta_bars(bars: list[DomainBar]) -> list[TABar]:
    """Convert our date-tagged bars into the dateless bars used by
    ``technical_analysis``. The TA functions don't care about dates;
    they only need OHLCV in chronological order."""
    return [TABar(open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume) for b in bars]


def daily_to_weekly(daily: list[DomainBar]) -> list[DomainBar]:
    """Resample daily bars to weekly (Mon-Fri → one bar).

    Uses ISO calendar week (year + week) as the grouping key. Open is
    the week's first bar's open; close is the week's last bar's close;
    high/low are extremes across the week; volume sums.

    Partial weeks at the boundaries are included — better to have a
    small under-sized bar at the edges than to shorten the history.
    """
    if not daily:
        return []
    groups: dict[tuple[int, int], list[DomainBar]] = {}
    for b in daily:
        key = b.trade_date.isocalendar()[:2]  # (year, week)
        groups.setdefault(key, []).append(b)
    weeks = []
    for key in sorted(groups.keys()):
        group = groups[key]
        weeks.append(DomainBar(
            trade_date=group[-1].trade_date,  # week's "label" = last day
            open=group[0].open,
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,
            volume=sum(b.volume for b in group),
        ))
    return weeks


@dataclass
class _Portfolio:
    """In-memory portfolio state for the simulator. Deliberately simple
    — a dict-keyed list of OpenPosition and cash."""
    cash: float
    open_positions: list[OpenPosition]

    def total_risk_percent(self) -> float:
        return sum(p.risk_percent for p in self.open_positions)

    def has_ticker(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self.open_positions)


def _size_position(
    signal_price: float,
    stop_loss: float,
    account_balance: float,
    risk_pct: float,
) -> int:
    """Same formula as production RiskManager.calculate_position_size, inlined
    so the simulator doesn't need a live RiskManager (which uses ``settings``).
    """
    risk_amount = account_balance * risk_pct
    risk_per_share = abs(signal_price - stop_loss)
    if risk_per_share <= 0 or signal_price <= 0 or account_balance <= 0:
        return 0
    shares = math.floor(risk_amount / risk_per_share)
    max_affordable = math.floor(account_balance / signal_price)
    return max(0, min(shares, max_affordable))


def _calc_stop_and_target(
    entry: float, atr: float, side: Side, atr_mult: float, rr: float
) -> tuple[float, float]:
    """Mirror of RiskManager.calculate_stop_loss + calculate_take_profit."""
    distance = atr * atr_mult
    if side == "buy":
        sl = round(entry - distance, 2)
        tp = round(entry + distance * rr, 2)
    else:
        sl = round(entry + distance, 2)
        tp = round(entry - distance * rr, 2)
    return sl, tp


def _check_exit(
    pos: OpenPosition,
    bar: DomainBar,
) -> tuple[str, float] | None:
    """Decide whether a position's SL or TP was hit by ``bar``.

    Conservative convention: if both the high reaches TP AND the low
    reaches SL on the same bar, we assume SL was hit first. Real
    intraday ordering is unknowable from daily OHLC alone.
    """
    if pos.side == "buy":
        hit_sl = bar.low <= pos.stop_loss
        hit_tp = bar.high >= pos.take_profit
        if hit_sl and hit_tp:
            return "stop_loss", pos.stop_loss
        if hit_sl:
            return "stop_loss", pos.stop_loss
        if hit_tp:
            return "take_profit", pos.take_profit
    else:  # short
        hit_sl = bar.high >= pos.stop_loss
        hit_tp = bar.low <= pos.take_profit
        if hit_sl and hit_tp:
            return "stop_loss", pos.stop_loss
        if hit_sl:
            return "stop_loss", pos.stop_loss
        if hit_tp:
            return "take_profit", pos.take_profit
    return None


class Backtester:
    """Runs one backtest over ``config.start..config.end`` against the
    historical bars in ``cache``.

    Simple API:

        result = Backtester(cache, config).run()

    The heavy lifting (``analyze`` / ``analyze_macro``) is done
    per-ticker per-day; O(universe × days) analysis calls per backtest.
    On a modern laptop this is tens of thousands of lightweight
    numpy-free list computations — fast enough that we don't bother
    parallelising.
    """

    def __init__(self, cache: BarCache, config: BacktestConfig):
        self.cache = cache
        self.config = config

    # ---- public entry ----------------------------------------------------
    def run(self) -> BacktestResult:
        config = self.config
        portfolio = _Portfolio(cash=config.initial_capital, open_positions=[])
        closed: list[ClosedTrade] = []
        equity_curve: list[EquityPoint] = []

        # Collect the union of trading dates from the universe's cached
        # bars so we only step through days the market was open.
        trading_dates = self._collect_trading_dates()
        if not trading_dates:
            return BacktestResult(config=config)

        # Pending entries are queued when a signal fires on day D. They
        # execute at the OPEN of the next trading date, with SL/TP checks
        # beginning from that same next-day bar.
        pending_entries: list[dict] = []

        for idx, today in enumerate(trading_dates):
            # 1. Execute any queued entries at today's open (if within universe).
            self._execute_pending_entries(pending_entries, portfolio, today)

            # 2. Check exits on all open positions using today's OHLC.
            self._check_and_close_positions(portfolio, closed, today)

            # 3. Generate signals on today's bars (close-of-day analysis),
            #    queue new entries for tomorrow's open.
            if idx < len(trading_dates) - 1:  # no point scanning on the last day — no "tomorrow" to fill
                self._scan_and_queue(pending_entries, portfolio, today, closed)

            # 4. Mark-to-market equity snapshot.
            equity_curve.append(self._mark(portfolio, today))

            # 5. Enforce time stop — any position held > max_hold_days
            #    gets force-closed at today's close.
            self._enforce_time_stop(portfolio, closed, today)

        # At the end, flip any still-open positions to ClosedTrade at the
        # last day's close for metric consistency (don't carry unrealized
        # past the backtest window).
        self._close_remaining(portfolio, closed, trading_dates[-1])

        result = BacktestResult(
            config=config,
            closed_trades=closed,
            equity_curve=equity_curve,
            open_at_end=list(portfolio.open_positions),
        )
        _attach_metrics(result)
        return result

    # ---- internals -------------------------------------------------------
    def _collect_trading_dates(self) -> list[date]:
        """Union of trading dates across the universe, restricted to the
        backtest window. Dates are ordered ascending."""
        dates: set[date] = set()
        for t in self.config.universe:
            bars = self.cache.get_bars(t, "day", self.config.start, self.config.end)
            for b in bars:
                dates.add(b.trade_date)
        return sorted(dates)

    def _get_history(self, ticker: str, upto: date) -> list[DomainBar]:
        """Return ALL cached daily bars for ``ticker`` up to and INCLUDING
        ``upto``. Not truncated to any lookback — ``analyze`` only needs
        enough bars to compute indicators, and taking the full window is
        cheap and avoids window-size bugs.
        """
        # Fetch a wide window; SQLite handles ORDER BY efficiently.
        wide_start = date(self.config.start.year - 3, 1, 1)
        bars = self.cache.get_bars(ticker, "day", wide_start, upto)
        return bars

    def _scan_and_queue(
        self,
        pending: list[dict],
        portfolio: _Portfolio,
        today: date,
        closed: list[ClosedTrade],
    ) -> None:
        """Run the scanner's analyze() over every ticker in the universe,
        queue executable entries for next trading day. Applies the same
        portfolio-level checks (max positions, max total risk, duplicate
        ticker) the live DecisionEngine runs at signal time.
        """
        cfg = self.config
        # Snapshot risk + positions at scan-start, same pattern the live
        # engine uses so we don't double-count within a single scan.
        start_positions = len(portfolio.open_positions)
        start_risk = portfolio.total_risk_percent()
        start_tickers = {p.ticker for p in portfolio.open_positions}

        # Also include queued-but-not-yet-filled entries in the dedup /
        # slot counts (otherwise we'd happily queue 30 trades in one day).
        pending_tickers = {q["ticker"] for q in pending if q["fire_before"] is None or today <= q["fire_before"]}
        pending_risk = sum(q["risk_percent"] for q in pending if q["fire_before"] is None or today <= q["fire_before"])

        for ticker in cfg.universe:
            if (start_positions + len(pending_tickers) - len(start_tickers & pending_tickers)
                    >= cfg.max_open_positions):
                break
            if ticker in start_tickers or ticker in pending_tickers:
                continue

            hist = self._get_history(ticker, today)
            if len(hist) < MIN_DAILY_BARS:
                continue

            # Weekly macro context from the same daily series.
            weekly = daily_to_weekly(hist)
            macro = analyze_macro(_to_ta_bars(weekly)) if len(weekly) >= MIN_WEEKLY_BARS else None

            result = analyze(ticker, _to_ta_bars(hist), macro=macro)
            if result is None:
                continue
            if result.signal == "NONE":
                continue
            if result.strength < cfg.min_signal_strength:
                continue

            side: Side = "buy" if result.signal == "BUY" else "sell"
            entry_price = hist[-1].close  # scan fires at close-of-day
            if not result.atr or result.atr <= 0:
                continue
            sl, tp = _calc_stop_and_target(
                entry_price, result.atr, side, cfg.atr_sl_multiplier, cfg.default_rr_ratio,
            )
            # Re-check that SL/TP aren't equal to entry (rounding can collapse)
            if sl == entry_price or tp == entry_price:
                continue
            # RR floor — same hard gate the risk_manager enforces.
            rr_actual = abs(tp - entry_price) / abs(entry_price - sl)
            if rr_actual < 1.5:
                continue
            # RSI extreme filter — same as live.
            if side == "buy" and result.rsi is not None and result.rsi > 75:
                continue
            if side == "sell" and result.rsi is not None and result.rsi < 25:
                continue

            # Position sizing: use ``min(cash, live-equity)`` as balance cap,
            # same idea as live ``balance = min(account.equity, max_capital)``.
            balance_cap = min(
                portfolio.cash,
                cfg.initial_capital,
            )
            shares = _size_position(entry_price, sl, balance_cap, cfg.max_risk_per_trade)
            if shares <= 0:
                continue
            risk_amount = abs(entry_price - sl) * shares
            risk_pct = risk_amount / cfg.initial_capital if cfg.initial_capital > 0 else 0
            if start_risk + pending_risk + risk_pct > cfg.max_total_risk + 1e-9:
                continue

            pending.append({
                "ticker": ticker,
                "side": side,
                "signal_date": today,
                "entry_price_indicative": entry_price,
                "stop_loss": sl,
                "take_profit": tp,
                "shares": shares,
                "risk_percent": risk_pct,
                # Entries expire if they can't fill in the next 2 trading
                # days (data gap / delisting). Prevents phantom signals.
                "fire_before": today + timedelta(days=4),
            })
            pending_tickers.add(ticker)
            pending_risk += risk_pct

    def _execute_pending_entries(
        self, pending: list[dict], portfolio: _Portfolio, today: date,
    ) -> None:
        """Fill queued entries at today's OPEN if bar exists.

        Cash accounting by side:
        - **long**: cash -= entry*shares + commission (we paid for the shares)
        - **short**: cash += entry*shares - commission (we received proceeds
          from selling borrowed shares; the liability to buy back lives
          on the ``OpenPosition`` side="sell" flag and is marked-to-market
          by ``_mark``).
        """
        still_pending: list[dict] = []
        for q in pending:
            if q["fire_before"] is not None and today > q["fire_before"]:
                continue  # expired
            bars = self.cache.get_bars(q["ticker"], "day", today, today)
            if not bars:
                # No bar today for this ticker — try again next day (still_pending).
                still_pending.append(q)
                continue
            if portfolio.has_ticker(q["ticker"]):
                continue  # ticker already entered since queuing
            open_price = bars[0].open
            if open_price <= 0:
                continue
            commission = q["shares"] * self.config.commission_per_share
            notional = open_price * q["shares"]
            if q["side"] == "buy":
                if portfolio.cash < notional + commission:
                    continue  # can't afford at today's open after a gap up
                portfolio.cash -= notional + commission
            else:  # short
                portfolio.cash += notional - commission
            portfolio.open_positions.append(OpenPosition(
                ticker=q["ticker"],
                side=q["side"],
                entry_date=today,
                entry_price=open_price,
                stop_loss=q["stop_loss"],
                take_profit=q["take_profit"],
                shares=q["shares"],
                risk_percent=q["risk_percent"],
            ))
        # Drop the executed/expired queue, keep genuine "no-bar-today" retries.
        pending.clear()
        pending.extend(still_pending)

    def _check_and_close_positions(
        self, portfolio: _Portfolio, closed: list[ClosedTrade], today: date,
    ) -> None:
        """For each open position with a bar today, check SL/TP triggers."""
        survivors: list[OpenPosition] = []
        for pos in portfolio.open_positions:
            if pos.entry_date == today:
                # Don't check exit on the same bar we entered at the open.
                survivors.append(pos)
                continue
            bars = self.cache.get_bars(pos.ticker, "day", today, today)
            if not bars:
                survivors.append(pos)
                continue
            bar = bars[0]
            exit_info = _check_exit(pos, bar)
            if exit_info is None:
                survivors.append(pos)
                continue
            reason, exit_price = exit_info
            self._close_position(pos, exit_price, today, reason, portfolio, closed)
        portfolio.open_positions = survivors

    def _enforce_time_stop(
        self, portfolio: _Portfolio, closed: list[ClosedTrade], today: date,
    ) -> None:
        cfg = self.config
        survivors: list[OpenPosition] = []
        for pos in portfolio.open_positions:
            age = (today - pos.entry_date).days
            if age < cfg.max_hold_days:
                survivors.append(pos)
                continue
            bars = self.cache.get_bars(pos.ticker, "day", today, today)
            if not bars:
                survivors.append(pos)
                continue
            close_price = bars[0].close
            self._close_position(pos, close_price, today, "time_stop", portfolio, closed)
        portfolio.open_positions = survivors

    def _close_position(
        self,
        pos: OpenPosition,
        exit_price: float,
        today: date,
        reason: str,
        portfolio: _Portfolio,
        closed: list[ClosedTrade],
    ) -> None:
        """Unwind a position, credit/debit cash, record the trade.

        Cash accounting on exit:
        - **long**: cash += exit*shares - commission (we sold the asset back)
        - **short**: cash -= exit*shares + commission (we bought shares to
          cover the short liability)

        ``pnl`` is the net of commissions on BOTH legs — the entry leg
        is baked in via the differential between entry cash flow and
        exit cash flow, plus the explicit exit commission here.
        """
        commission = pos.shares * self.config.commission_per_share
        notional = exit_price * pos.shares
        sign = 1 if pos.side == "buy" else -1
        gross = sign * (exit_price - pos.entry_price) * pos.shares
        # Both legs' commissions (entry leg was subtracted from cash at entry)
        pnl = gross - 2 * commission
        if pos.side == "buy":
            portfolio.cash += notional - commission
        else:
            portfolio.cash -= notional + commission
        closed.append(ClosedTrade(
            ticker=pos.ticker,
            side=pos.side,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=today,
            exit_price=exit_price,
            shares=pos.shares,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            pnl=round(pnl, 2),
            exit_reason=reason,  # type: ignore[arg-type]
        ))

    def _close_remaining(
        self, portfolio: _Portfolio, closed: list[ClosedTrade], last_date: date,
    ) -> None:
        """Close every still-open position at last bar's close."""
        for pos in list(portfolio.open_positions):
            bars = self.cache.get_bars(pos.ticker, "day", pos.entry_date, last_date)
            last_close = bars[-1].close if bars else pos.entry_price
            self._close_position(pos, last_close, last_date, "end_of_backtest", portfolio, closed)
        portfolio.open_positions.clear()

    def _mark(self, portfolio: _Portfolio, today: date) -> EquityPoint:
        """Mark-to-market. Equity = cash + mark value of every position.

        - **long**: position's mark value = ``price * shares``
          (positive contribution to equity)
        - **short**: mark value = ``-price * shares``
          (the account owes that much — a liability marked down as the
          stock rises, up as it falls; when added to the cash received
          at entry, the net matches the unrealized P&L)
        """
        mark_value = 0.0
        for pos in portfolio.open_positions:
            bars = self.cache.get_bars(pos.ticker, "day", today, today)
            price = bars[0].close if bars else pos.entry_price
            if pos.side == "buy":
                mark_value += price * pos.shares
            else:
                mark_value -= price * pos.shares
        return EquityPoint(
            trade_date=today,
            equity=round(portfolio.cash + mark_value, 2),
            cash=round(portfolio.cash, 2),
            open_position_count=len(portfolio.open_positions),
        )


def _attach_metrics(result: BacktestResult) -> None:
    """Populate Sharpe / Sortino / Max DD / Calmar on the result.

    Same formulas the live MetricsService uses, duplicated here because
    the simulator is offline and shouldn't depend on DB-backed services.
    Kept in sync manually (tests in both places).
    """
    eq_curve = result.equity_curve
    n = len(eq_curve)
    result.sample_size = n
    if n < 2:
        return

    equities = [e.equity for e in eq_curve]
    dates = [e.trade_date for e in eq_curve]
    returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, n)
        if equities[i - 1] > 0
    ]
    if len(returns) < 1:
        return

    mean_ret = statistics.mean(returns)

    if len(returns) >= 2:
        std_ret = statistics.stdev(returns)
        if std_ret > 0:
            result.sharpe = round(
                (mean_ret / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR), 3
            )

    downside_var = sum((r * r) for r in returns if r < 0) / len(returns)
    downside_std = math.sqrt(downside_var)
    if downside_std > 0:
        result.sortino = round(
            (mean_ret / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR), 3
        )

    peak = equities[0]
    max_dd = 0.0
    for e in equities:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
    result.max_drawdown = round(max_dd, 4)
    result.max_drawdown_pct = round(max_dd * 100, 2)

    days = max((dates[-1] - dates[0]).days, 1)
    total_return = (equities[-1] / equities[0]) - 1 if equities[0] > 0 else 0.0
    result.total_return = round(total_return, 4)
    result.total_return_pct = round(total_return * 100, 2)
    try:
        annualized = (equities[-1] / equities[0]) ** (365.25 / days) - 1
        result.annualized_return = round(annualized, 4)
        result.annualized_return_pct = round(annualized * 100, 2)
        if max_dd > 0:
            result.calmar = round(annualized / max_dd, 3)
    except (ValueError, OverflowError):
        pass
