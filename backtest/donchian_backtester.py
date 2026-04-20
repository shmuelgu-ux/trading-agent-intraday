"""Backtester for the Donchian breakout strategy.

Per trading day:

1. **Update open positions**: bump ``high_since_entry`` on today's high,
   ratchet the trailing stop higher (never lower). If today's LOW
   pierced the trailing stop, close at the stop price (conservative
   fill — the low can go lower than the trigger intra-bar).
2. **Check for channel-exit**: if still open after the stop check and
   today's CLOSE fell below the exit channel (lowest low of prior M
   bars), close at the close.
3. **Execute queued entries** at today's OPEN (queued from yesterday's
   signal). Honours portfolio checks: max_positions, total_risk,
   ticker-dedup.
4. **Scan for new entries**: for each ticker not held, run
   ``analyze_bar``. If LONG_ENTRY, queue for tomorrow's open.
5. **Mark-to-market** equity = cash + ΣΣ price_i * shares_i.

Intentionally separate from ``simulator.Backtester`` even though they
share ~70% of the scaffolding. The simulator is tied to the old
indicator-voting model; mixing Donchian-only logic in there would make
both harder to reason about. A third backtester for a third strategy
style is cheap; entangled ones are expensive.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from backtest.data_loader import BarCache
from backtest.donchian_strategy import (
    analyze_bar,
    atr_initial_stop,
    position_size,
    ratchet_trailing_stop,
)
from backtest.models import (
    Bar,
    BacktestConfig,
    BacktestResult,
    ClosedTrade,
    EquityPoint,
    OpenPosition,
)


TRADING_DAYS_PER_YEAR = 252


@dataclass
class DonchianConfig:
    start: date
    end: date
    initial_capital: float
    universe: list[str]
    # Strategy parameters
    entry_lookback: int = 20          # N-day high breakout
    exit_lookback: int = 10           # M-day low exit (N/2 is classical)
    atr_period: int = 14
    atr_stop_multiple: float = 2.0    # initial + trailing stop distance
    # Portfolio limits
    risk_per_trade: float = 0.01      # 1% of equity per position
    max_total_risk: float = 0.20      # 20% portfolio risk cap
    max_open_positions: int = 20
    # Costs
    commission_per_share: float = 0.0035
    # Short-swing variant: hard cap on holding period (None = unlimited).
    # The intraday bot uses ``max_hold_days=3`` to simulate short-term swings.
    max_hold_days: int | None = None
    # EOD force-close: if True, close all positions at the LAST trading
    # day's close regardless of trailing stop. Used by the intraday
    # variant; swing leaves this False.
    force_close_at_end: bool = False


@dataclass
class _PositionState:
    """Extension of OpenPosition with trailing-stop bookkeeping."""
    pos: OpenPosition
    trailing_stop: float
    high_since_entry: float


@dataclass
class _Portfolio:
    cash: float
    by_ticker: dict[str, _PositionState] = field(default_factory=dict)

    def total_risk(self) -> float:
        return sum(s.pos.risk_percent for s in self.by_ticker.values())

    def has(self, ticker: str) -> bool:
        return ticker in self.by_ticker


class DonchianBacktester:
    def __init__(self, cache: BarCache, config: DonchianConfig):
        self.cache = cache
        self.config = config

    def run(self) -> BacktestResult:
        cfg = self.config
        # Preload bars with enough pre-roll for the lookback windows.
        preload_start = cfg.start - timedelta(days=cfg.entry_lookback * 2 + 30)
        bars_by_ticker: dict[str, list[Bar]] = {}
        for t in cfg.universe:
            bars_by_ticker[t] = self.cache.get_bars(t, "day", preload_start, cfg.end)
        trading_dates = sorted({
            b.trade_date
            for bars in bars_by_ticker.values()
            for b in bars
            if cfg.start <= b.trade_date <= cfg.end
        })
        if not trading_dates:
            return self._empty()

        portfolio = _Portfolio(cash=cfg.initial_capital)
        closed: list[ClosedTrade] = []
        equity_curve: list[EquityPoint] = []
        pending_entries: list[dict] = []

        for idx, today in enumerate(trading_dates):
            # 1. Execute any queued entries at today's OPEN.
            self._execute_pending(pending_entries, portfolio, today, bars_by_ticker)

            # 2. Update stops + check exits on open positions.
            self._process_open_positions(portfolio, today, bars_by_ticker, closed)

            # 3. Scan for new signals on today's close. Queue for tomorrow's open.
            if idx < len(trading_dates) - 1 or not cfg.force_close_at_end:
                self._scan_for_entries(portfolio, today, bars_by_ticker, pending_entries, trading_dates)

            # 4. Mark equity.
            equity_curve.append(self._mark(portfolio, today, bars_by_ticker))

            # 5. Time stop (MVP: simple max-hold).
            if cfg.max_hold_days is not None:
                self._enforce_time_stop(portfolio, today, bars_by_ticker, closed)

            # 6. EOD force close (intraday variant).
            if cfg.force_close_at_end and idx == len(trading_dates) - 1:
                self._close_all_at_close(portfolio, today, bars_by_ticker, closed, "end_of_backtest")

        # End-of-backtest cleanup: flush any still-open positions at the last bar's close.
        self._close_all_at_close(portfolio, trading_dates[-1], bars_by_ticker, closed, "end_of_backtest")

        result = BacktestResult(
            config=self._compat_config(), closed_trades=closed,
            equity_curve=equity_curve, open_at_end=[],
        )
        _attach_metrics(result)
        return result

    # -------- phase helpers ---------------------------------------------
    def _scan_for_entries(
        self,
        portfolio: _Portfolio,
        today: date,
        bars_by_ticker: dict[str, list[Bar]],
        pending: list[dict],
        trading_dates: list[date],
    ) -> None:
        cfg = self.config
        current_risk = portfolio.total_risk()
        open_count = len(portfolio.by_ticker)
        pending_tickers = {q["ticker"] for q in pending}
        pending_risk = sum(q["risk_percent"] for q in pending)

        for ticker in cfg.universe:
            if open_count + len(pending_tickers) >= cfg.max_open_positions:
                break
            if portfolio.has(ticker) or ticker in pending_tickers:
                continue
            # History up to and including today.
            history = [b for b in bars_by_ticker.get(ticker, []) if b.trade_date <= today]
            if len(history) < cfg.entry_lookback + cfg.atr_period + 1:
                continue
            reading = analyze_bar(
                history,
                entry_lookback=cfg.entry_lookback,
                exit_lookback=cfg.exit_lookback,
                atr_period=cfg.atr_period,
                held_long=False,
            )
            if reading.signal != "LONG_ENTRY":
                continue
            if reading.atr is None or reading.atr <= 0:
                continue

            # Provisional sizing based on TODAY's close (actual fill on tomorrow's open).
            entry_provisional = reading.close
            stop_provisional = atr_initial_stop(
                entry_provisional, reading.atr, cfg.atr_stop_multiple
            )
            if stop_provisional >= entry_provisional:  # Sanity — ATR could be huge on outlier
                continue
            shares = position_size(
                account_balance=min(portfolio.cash + self._mark_value(portfolio, today, bars_by_ticker), cfg.initial_capital),
                entry_price=entry_provisional,
                stop_loss=stop_provisional,
                risk_per_trade=cfg.risk_per_trade,
            )
            if shares <= 0:
                continue
            risk_dollars = (entry_provisional - stop_provisional) * shares
            risk_pct = risk_dollars / cfg.initial_capital if cfg.initial_capital > 0 else 0
            if current_risk + pending_risk + risk_pct > cfg.max_total_risk + 1e-9:
                continue

            pending.append({
                "ticker": ticker,
                "signal_date": today,
                "provisional_entry": entry_provisional,
                "provisional_stop": stop_provisional,
                "atr": reading.atr,
                "shares": shares,
                "risk_percent": risk_pct,
                # Expire a stale pending after 3 days (data gaps / delistings).
                "fire_before": today + timedelta(days=4),
            })
            pending_tickers.add(ticker)
            pending_risk += risk_pct

    def _execute_pending(
        self,
        pending: list[dict],
        portfolio: _Portfolio,
        today: date,
        bars_by_ticker: dict[str, list[Bar]],
    ) -> None:
        cfg = self.config
        still_pending: list[dict] = []
        for q in pending:
            if today > q["fire_before"]:
                continue  # expired
            if q["signal_date"] == today:
                # Can't enter on the same bar the signal fired on —
                # signal is at close, entry is at NEXT open.
                still_pending.append(q)
                continue
            bars_today = [b for b in bars_by_ticker.get(q["ticker"], []) if b.trade_date == today]
            if not bars_today:
                still_pending.append(q)
                continue
            if portfolio.has(q["ticker"]):
                continue  # already entered somehow
            open_price = bars_today[0].open
            if open_price <= 0:
                continue
            # Recompute the stop around the actual open (not the prior-close-based provisional).
            stop_at_open = atr_initial_stop(open_price, q["atr"], cfg.atr_stop_multiple)
            if stop_at_open >= open_price:
                continue
            shares = q["shares"]
            commission = shares * cfg.commission_per_share
            cost = open_price * shares + commission
            if portfolio.cash < cost:
                continue  # can't afford after overnight gap
            portfolio.cash -= cost
            pos = OpenPosition(
                ticker=q["ticker"], side="buy",
                entry_date=today, entry_price=open_price,
                stop_loss=stop_at_open,
                take_profit=0.0,  # Donchian has no fixed TP
                shares=shares, risk_percent=q["risk_percent"],
            )
            portfolio.by_ticker[q["ticker"]] = _PositionState(
                pos=pos, trailing_stop=stop_at_open, high_since_entry=open_price,
            )
        pending.clear()
        pending.extend(still_pending)

    def _process_open_positions(
        self,
        portfolio: _Portfolio,
        today: date,
        bars_by_ticker: dict[str, list[Bar]],
        closed: list[ClosedTrade],
    ) -> None:
        cfg = self.config
        for ticker in list(portfolio.by_ticker.keys()):
            st = portfolio.by_ticker[ticker]
            if st.pos.entry_date == today:
                continue  # don't process the bar we just entered on
            bars = bars_by_ticker.get(ticker, [])
            today_bar = next((b for b in bars if b.trade_date == today), None)
            if today_bar is None:
                continue  # no data today — carry forward

            # a. Trailing stop hit?
            if today_bar.low <= st.trailing_stop:
                self._close_position(ticker, today, st.trailing_stop, "stop_loss", portfolio, closed)
                continue

            # b. Channel exit? (close below prior M-day low)
            history = [b for b in bars if b.trade_date <= today]
            reading = analyze_bar(
                history,
                entry_lookback=cfg.entry_lookback, exit_lookback=cfg.exit_lookback,
                atr_period=cfg.atr_period, held_long=True,
            )
            if reading.signal == "LONG_EXIT":
                self._close_position(ticker, today, today_bar.close, "take_profit", portfolio, closed)
                # We report channel-exit as "take_profit" so the win/loss
                # split maps cleanly in downstream tooling — it's the
                # profit-taking side of the Donchian rule.
                continue

            # c. Otherwise ratchet the trailing stop higher.
            if today_bar.high > st.high_since_entry:
                st.high_since_entry = today_bar.high
            if reading.atr:
                st.trailing_stop = ratchet_trailing_stop(
                    st.trailing_stop, st.high_since_entry, reading.atr,
                    atr_multiple=cfg.atr_stop_multiple,
                )

    def _enforce_time_stop(
        self,
        portfolio: _Portfolio,
        today: date,
        bars_by_ticker: dict[str, list[Bar]],
        closed: list[ClosedTrade],
    ) -> None:
        cfg = self.config
        assert cfg.max_hold_days is not None
        for ticker in list(portfolio.by_ticker.keys()):
            st = portfolio.by_ticker[ticker]
            age = (today - st.pos.entry_date).days
            if age < cfg.max_hold_days:
                continue
            bars = bars_by_ticker.get(ticker, [])
            bar = next((b for b in bars if b.trade_date == today), None)
            if bar is None:
                continue
            self._close_position(ticker, today, bar.close, "time_stop", portfolio, closed)

    def _close_all_at_close(
        self,
        portfolio: _Portfolio,
        today: date,
        bars_by_ticker: dict[str, list[Bar]],
        closed: list[ClosedTrade],
        reason: str,
    ) -> None:
        for ticker in list(portfolio.by_ticker.keys()):
            bars = bars_by_ticker.get(ticker, [])
            bar = next((b for b in bars if b.trade_date == today), None)
            price = bar.close if bar else portfolio.by_ticker[ticker].pos.entry_price
            self._close_position(ticker, today, price, reason, portfolio, closed)

    def _close_position(
        self,
        ticker: str,
        today: date,
        exit_price: float,
        reason: str,
        portfolio: _Portfolio,
        closed: list[ClosedTrade],
    ) -> None:
        st = portfolio.by_ticker.pop(ticker, None)
        if st is None:
            return
        pos = st.pos
        commission = pos.shares * self.config.commission_per_share
        gross = (exit_price - pos.entry_price) * pos.shares
        pnl = gross - 2 * commission  # entry + exit commissions
        portfolio.cash += exit_price * pos.shares - commission
        closed.append(ClosedTrade(
            ticker=ticker, side="buy",
            entry_date=pos.entry_date, entry_price=pos.entry_price,
            exit_date=today, exit_price=exit_price,
            shares=pos.shares,
            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
            pnl=round(pnl, 2),
            exit_reason=reason,  # type: ignore[arg-type]
        ))

    # -------- marking ----------------------------------------------------
    def _mark_value(
        self, portfolio: _Portfolio, today: date, bars_by_ticker: dict[str, list[Bar]],
    ) -> float:
        mark = 0.0
        for ticker, st in portfolio.by_ticker.items():
            bars = bars_by_ticker.get(ticker, [])
            bar = next((b for b in bars if b.trade_date == today), None)
            price = bar.close if bar else st.pos.entry_price
            mark += price * st.pos.shares
        return mark

    def _mark(
        self, portfolio: _Portfolio, today: date, bars_by_ticker: dict[str, list[Bar]],
    ) -> EquityPoint:
        mark = self._mark_value(portfolio, today, bars_by_ticker)
        return EquityPoint(
            trade_date=today,
            equity=round(portfolio.cash + mark, 2),
            cash=round(portfolio.cash, 2),
            open_position_count=len(portfolio.by_ticker),
        )

    def _compat_config(self) -> BacktestConfig:
        cfg = self.config
        return BacktestConfig(
            start=cfg.start, end=cfg.end,
            initial_capital=cfg.initial_capital,
            universe=cfg.universe,
            max_risk_per_trade=cfg.risk_per_trade,
            max_total_risk=cfg.max_total_risk,
            max_open_positions=cfg.max_open_positions,
            default_rr_ratio=0.0,
            atr_sl_multiplier=cfg.atr_stop_multiple,
            min_signal_strength=0,
            commission_per_share=cfg.commission_per_share,
            max_hold_days=cfg.max_hold_days or 99999,
        )

    def _empty(self) -> BacktestResult:
        return BacktestResult(config=self._compat_config())


def _attach_metrics(result: BacktestResult) -> None:
    """Sharpe / Sortino / Max DD / Calmar on the equity curve.
    Same formulas as the other backtesters — duplicated so this module
    is self-contained."""
    eq = result.equity_curve
    n = len(eq)
    result.sample_size = n
    if n < 2:
        return

    equities = [e.equity for e in eq]
    dates = [e.trade_date for e in eq]
    returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, n) if equities[i - 1] > 0
    ]
    if not returns:
        return
    mean_ret = statistics.mean(returns)
    if len(returns) >= 2:
        std_ret = statistics.stdev(returns)
        if std_ret > 0:
            result.sharpe = round((mean_ret / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR), 3)
    downside_var = sum((r * r) for r in returns if r < 0) / len(returns)
    downside_std = math.sqrt(downside_var)
    if downside_std > 0:
        result.sortino = round((mean_ret / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR), 3)
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
