"""Tests for backtest.simulator — the core math + end-to-end run."""
from datetime import date, timedelta

import pytest

from backtest.data_loader import BarCache
from backtest.models import Bar, BacktestConfig, OpenPosition
from backtest.simulator import (
    Backtester,
    _attach_metrics,
    _calc_stop_and_target,
    _check_exit,
    _size_position,
    daily_to_weekly,
)
from backtest.models import BacktestResult


def _bar(d, o, h, lo, c, v=1_500_000):
    return Bar(trade_date=d, open=o, high=h, low=lo, close=c, volume=v)


class TestCheckExit:
    def test_long_sl_hit(self):
        pos = OpenPosition(
            ticker="X", side="buy", entry_date=date(2024, 1, 2),
            entry_price=100, stop_loss=95, take_profit=110, shares=10, risk_percent=0.03,
        )
        bar = _bar(date(2024, 1, 3), 99, 102, 94, 101)  # low=94 hits 95
        reason, price = _check_exit(pos, bar)
        assert reason == "stop_loss"
        assert price == 95

    def test_long_tp_hit(self):
        pos = OpenPosition(
            ticker="X", side="buy", entry_date=date(2024, 1, 2),
            entry_price=100, stop_loss=95, take_profit=110, shares=10, risk_percent=0.03,
        )
        bar = _bar(date(2024, 1, 3), 105, 112, 104, 108)  # high=112 hits 110
        reason, price = _check_exit(pos, bar)
        assert reason == "take_profit"
        assert price == 110

    def test_long_both_hit_prefers_sl(self):
        """Conservative: if high>=TP AND low<=SL on the same bar, SL wins."""
        pos = OpenPosition(
            ticker="X", side="buy", entry_date=date(2024, 1, 2),
            entry_price=100, stop_loss=95, take_profit=110, shares=10, risk_percent=0.03,
        )
        bar = _bar(date(2024, 1, 3), 99, 115, 90, 100)
        reason, _ = _check_exit(pos, bar)
        assert reason == "stop_loss"

    def test_long_no_hit(self):
        pos = OpenPosition(
            ticker="X", side="buy", entry_date=date(2024, 1, 2),
            entry_price=100, stop_loss=95, take_profit=110, shares=10, risk_percent=0.03,
        )
        bar = _bar(date(2024, 1, 3), 99, 103, 97, 101)
        assert _check_exit(pos, bar) is None

    def test_short_sl_hit(self):
        """Short SL is ABOVE entry — fires when high >= SL."""
        pos = OpenPosition(
            ticker="X", side="sell", entry_date=date(2024, 1, 2),
            entry_price=100, stop_loss=105, take_profit=90, shares=10, risk_percent=0.03,
        )
        bar = _bar(date(2024, 1, 3), 101, 106, 99, 100)
        reason, price = _check_exit(pos, bar)
        assert reason == "stop_loss"
        assert price == 105

    def test_short_tp_hit(self):
        """Short TP is BELOW entry — fires when low <= TP."""
        pos = OpenPosition(
            ticker="X", side="sell", entry_date=date(2024, 1, 2),
            entry_price=100, stop_loss=105, take_profit=90, shares=10, risk_percent=0.03,
        )
        bar = _bar(date(2024, 1, 3), 95, 96, 89, 91)
        reason, price = _check_exit(pos, bar)
        assert reason == "take_profit"
        assert price == 90


class TestSizePosition:
    def test_basic(self):
        # $2000, 3% risk = $60, risk/share = $5 → 12 shares, affordable=20
        assert _size_position(100, 95, 2000, 0.03) == 12

    def test_unaffordable_caps_at_max(self):
        # $1000 account at $500/share can buy only 2 shares regardless of risk math
        assert _size_position(500, 495, 1000, 0.03) == 2

    def test_zero_balance_returns_zero(self):
        assert _size_position(100, 95, 0, 0.03) == 0
        assert _size_position(100, 95, -50, 0.03) == 0

    def test_zero_risk_per_share_returns_zero(self):
        assert _size_position(100, 100, 2000, 0.03) == 0


class TestCalcStopAndTarget:
    def test_long(self):
        # entry 100, ATR 2, atr_mult 1.5 → SL = 97, distance=3, rr=2 → TP=106
        sl, tp = _calc_stop_and_target(100, 2, "buy", 1.5, 2.0)
        assert sl == 97.0
        assert tp == 106.0

    def test_short(self):
        sl, tp = _calc_stop_and_target(100, 2, "sell", 1.5, 2.0)
        assert sl == 103.0
        assert tp == 94.0


class TestDailyToWeekly:
    def test_aggregates_by_iso_week(self):
        # Mon 2024-01-01 through Fri 2024-01-05 = one ISO week
        daily = [
            _bar(date(2024, 1, 1), 100, 102, 99, 101),
            _bar(date(2024, 1, 2), 101, 103, 100, 102),
            _bar(date(2024, 1, 3), 102, 104, 101, 103),
            _bar(date(2024, 1, 4), 103, 106, 102, 105),
            _bar(date(2024, 1, 5), 105, 108, 104, 107),
        ]
        weeks = daily_to_weekly(daily)
        assert len(weeks) == 1
        w = weeks[0]
        assert w.open == 100  # first day's open
        assert w.close == 107  # last day's close
        assert w.high == 108   # week's high
        assert w.low == 99     # week's low
        assert w.volume == 5 * 1_500_000

    def test_multiple_weeks(self):
        daily = []
        for d in [1, 2, 3, 4, 5, 8, 9, 10]:  # 5 days week1, 3 days week2
            daily.append(_bar(date(2024, 1, d), 100 + d, 110, 90, 100 + d))
        weeks = daily_to_weekly(daily)
        assert len(weeks) == 2


class TestAttachMetrics:
    def test_insufficient_data_stays_none(self):
        r = BacktestResult(config=BacktestConfig(
            start=date(2024, 1, 1), end=date(2024, 1, 2),
            initial_capital=1000, universe=[],
        ))
        r.equity_curve = []
        _attach_metrics(r)
        assert r.sharpe is None
        assert r.max_drawdown is None

    def test_max_dd_simple(self):
        from backtest.models import EquityPoint
        r = BacktestResult(config=BacktestConfig(
            start=date(2024, 1, 1), end=date(2024, 1, 5),
            initial_capital=2000, universe=[],
        ))
        r.equity_curve = [
            EquityPoint(trade_date=date(2024, 1, d), equity=e, cash=e, open_position_count=0)
            for d, e in zip([1, 2, 3, 4, 5], [2000, 2500, 2200, 2000, 2100])
        ]
        _attach_metrics(r)
        assert abs(r.max_drawdown - 0.20) < 5e-5
        assert r.max_drawdown_pct == 20.0


# ---------- END-TO-END -------------------------------------------------

def _seed_straight_uptrend(cache: BarCache, ticker: str, start: date, days: int):
    """Seed a cache with a ticker that climbs 1% a day, very stable.

    Used to prove the simulator can load bars, run analyze, execute an
    entry, and mark-to-market across a horizon. Doesn't care whether
    a trade ever fires — that depends on the indicator scoring.
    """
    price = 100.0
    bars = []
    d = start
    while len(bars) < days:
        if d.weekday() < 5:  # skip weekends
            bars.append(Bar(trade_date=d, open=price, high=price * 1.01,
                            low=price * 0.99, close=price * 1.01, volume=2_000_000))
            price *= 1.01
        d += timedelta(days=1)
    cache.put(ticker, "day", bars)


def test_end_to_end_runs_without_error(tmp_path):
    """Smoke test: simulator processes ~6 months of bars for one ticker
    without raising. Whether it trades depends on whether the indicators
    score over threshold on synthetic data; what we assert is that the
    equity curve is populated."""
    cache = BarCache(str(tmp_path / "cache.db"))
    _seed_straight_uptrend(cache, "TEST", date(2024, 1, 1), days=130)
    cfg = BacktestConfig(
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        initial_capital=2000,
        universe=["TEST"],
    )
    result = Backtester(cache, cfg).run()
    # At least populated
    assert result.sample_size == len(result.equity_curve) >= 100
    # No open positions should leak past the end (all closed in _close_remaining)
    assert result.open_at_end == []
    # Every equity point has a date within the range
    for p in result.equity_curve:
        assert cfg.start <= p.trade_date <= cfg.end


def test_end_to_end_flat_market_no_drawdown(tmp_path):
    """A perfectly flat market = no trades AND no drawdown."""
    cache = BarCache(str(tmp_path / "cache.db"))
    bars = []
    d = date(2024, 1, 1)
    for _ in range(130):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        bars.append(Bar(trade_date=d, open=100, high=100, low=100, close=100, volume=1_500_000))
        d += timedelta(days=1)
    cache.put("FLAT", "day", bars)
    cfg = BacktestConfig(
        start=date(2024, 1, 1), end=date(2024, 6, 30),
        initial_capital=2000, universe=["FLAT"],
    )
    result = Backtester(cache, cfg).run()
    # Equity should never move from initial_capital (no trades, no marks)
    for p in result.equity_curve:
        assert p.equity == pytest.approx(2000, abs=0.01)
    assert result.max_drawdown == 0.0
    assert len(result.closed_trades) == 0
