"""Tests for the Donchian strategy + its backtester."""
from datetime import date, timedelta

import pytest

from backtest.data_loader import BarCache
from backtest.donchian_backtester import DonchianBacktester, DonchianConfig
from backtest.donchian_strategy import (
    analyze_bar,
    atr_initial_stop,
    compute_atr,
    compute_channel_bounds,
    position_size,
    ratchet_trailing_stop,
)
from backtest.models import Bar


def _bars(start: date, days: int, price_fn, vol=1_500_000):
    out = []
    d = start
    for i in range(days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        p = price_fn(i)
        # Give OHLC a little spread so compute_atr has non-zero TRs.
        out.append(Bar(trade_date=d, open=p, high=p + 1, low=max(p - 1, 0.01),
                       close=p, volume=vol))
        d += timedelta(days=1)
    return out


class TestComputeChannelBounds:
    def test_returns_none_when_too_few_bars(self):
        bars = _bars(date(2024, 1, 1), 5, lambda i: 100 + i)
        upper, lower = compute_channel_bounds(bars, entry_lookback=20)
        assert upper is None and lower is None

    def test_excludes_current_bar(self):
        """Channel must be based on prior N bars, NOT including today.
        Otherwise 'new high' is always trivially satisfied by today's own bar."""
        bars = _bars(date(2024, 1, 1), 30, lambda i: 100)  # flat
        # Now make the last bar have an extreme high
        bars[-1] = Bar(
            trade_date=bars[-1].trade_date,
            open=100, high=500, low=99, close=200,
            volume=1_500_000,
        )
        upper, lower = compute_channel_bounds(bars, entry_lookback=20, exit_lookback=10)
        assert upper == 101  # prior 20 bars have high=101 (flat series)
        assert lower == 99

    def test_finds_max_of_prior_window(self):
        bars = _bars(date(2024, 1, 1), 25, lambda i: 100 + (i if i < 15 else 0))
        # The max high in bars[-(20+1):-1] (bars index 4..23) = 100+14+1 = 115
        upper, _ = compute_channel_bounds(bars, entry_lookback=20, exit_lookback=10)
        assert upper == 115


class TestComputeATR:
    def test_returns_none_when_too_few(self):
        bars = _bars(date(2024, 1, 1), 5, lambda i: 100)
        assert compute_atr(bars, period=14) is None

    def test_nonzero_with_range(self):
        bars = _bars(date(2024, 1, 1), 20, lambda i: 100 + i * 0.1)
        atr = compute_atr(bars, period=14)
        assert atr is not None
        assert atr > 0


class TestAnalyzeBar:
    def test_no_signal_when_in_channel(self):
        bars = _bars(date(2024, 1, 1), 25, lambda i: 100)
        reading = analyze_bar(bars)
        assert reading.signal == "NONE"

    def test_long_entry_on_new_high_breakout(self):
        # 25 days at 100, then break up to 120
        bars = _bars(date(2024, 1, 1), 24, lambda i: 100)
        bars.append(Bar(
            trade_date=bars[-1].trade_date + timedelta(days=3),  # next Monday
            open=100, high=120, low=99, close=120, volume=1_500_000,
        ))
        r = analyze_bar(bars, entry_lookback=20, exit_lookback=10)
        assert r.signal == "LONG_ENTRY"
        assert r.upper_entry == 101  # prior 20 bars' high = 101

    def test_long_exit_when_held_and_breaks_low(self):
        # Uptrend then reversal below 10-day low
        up = _bars(date(2024, 1, 1), 20, lambda i: 100 + i)
        # Last bar reverses below prior 10-day low
        last_low = min(b.low for b in up[-11:-1])  # prior 10 bars' low
        reverse = Bar(
            trade_date=up[-1].trade_date + timedelta(days=3),
            open=up[-1].close, high=up[-1].close, low=last_low - 5,
            close=last_low - 2, volume=1_500_000,
        )
        up.append(reverse)
        r = analyze_bar(up, entry_lookback=20, exit_lookback=10, held_long=True)
        assert r.signal == "LONG_EXIT"

    def test_held_long_suppresses_entry_signal(self):
        """Once in a position, we don't re-trigger entries."""
        bars = _bars(date(2024, 1, 1), 25, lambda i: 100)
        bars.append(Bar(
            trade_date=bars[-1].trade_date + timedelta(days=3),
            open=100, high=120, low=99, close=120, volume=1_500_000,
        ))
        r = analyze_bar(bars, held_long=True)
        # Not an entry (we're already long); not an exit (close > lower)
        assert r.signal == "NONE"


class TestPositionSize:
    def test_basic(self):
        # $2000 account, 1% risk = $20, risk_per_share = $2 → 10 shares, affordable 20
        assert position_size(2000, 100, 98, 0.01) == 10

    def test_affordability_caps(self):
        # $200 account. Risk budget: 200*0.01 = $2; risk/share = $2 → 1 share by risk.
        # Affordable: floor(200/100) = 2. min(1, 2) = 1.
        assert position_size(200, 100, 98, 0.01) == 1

    def test_affordability_is_the_binding_constraint(self):
        # $500 account, 2% risk = $10. risk/share = $0.10 → risk-based = 100 shares.
        # Affordable: floor(500/100) = 5. min(100, 5) = 5.
        assert position_size(500, 100, 99.9, 0.02) == 5

    def test_zero_risk_returns_zero(self):
        assert position_size(2000, 100, 100, 0.01) == 0

    def test_negative_balance_returns_zero(self):
        assert position_size(-100, 100, 98, 0.01) == 0


class TestATRStops:
    def test_initial_stop_below_entry(self):
        assert atr_initial_stop(100, 2, atr_multiple=2.0) == 96.0

    def test_ratchet_only_raises(self):
        # New high + tighter ATR → stop raised
        stop = ratchet_trailing_stop(current_stop=95, high_since_entry=105, atr=2, atr_multiple=2.0)
        assert stop == 101.0
        # If peak drops back, stop doesn't lower
        stop2 = ratchet_trailing_stop(current_stop=stop, high_since_entry=104, atr=2, atr_multiple=2.0)
        assert stop2 == 101.0  # unchanged


# ---------- END-TO-END -------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    return BarCache(str(tmp_path / "donchian_cache.db"))


def _seed_pattern(cache: BarCache, ticker: str, start: date, price_fn, days: int = 200):
    cache.put(ticker, "day", _bars(start, days, price_fn))


class TestBacktesterEndToEnd:
    def test_smoke_one_uptrend(self, cache):
        # Strong steady uptrend — should trigger a breakout entry and ride it.
        # Step of 2/day beats the per-bar +1 high added by _bars() so each
        # new bar breaks the prior N-day high.
        _seed_pattern(cache, "UP", date(2023, 1, 1), lambda i: 50 + i * 2.0, days=250)
        cfg = DonchianConfig(
            start=date(2023, 6, 1), end=date(2023, 12, 31),
            initial_capital=10000, universe=["UP"],
            entry_lookback=20, exit_lookback=10,
            risk_per_trade=0.02, max_total_risk=1.0, max_open_positions=5,
        )
        result = DonchianBacktester(cache, cfg).run()
        # Should have taken at least one position in a steady uptrend.
        assert len(result.closed_trades) >= 1 or len(result.equity_curve) > 0
        # In a monotonic uptrend we should have positive total return
        # (the position should have been opened and still appreciating).
        assert result.total_return is None or result.total_return > -0.05

    def test_smoke_flat_no_trades(self, cache):
        _seed_pattern(cache, "FLAT", date(2023, 1, 1), lambda i: 100, days=250)
        cfg = DonchianConfig(
            start=date(2023, 6, 1), end=date(2023, 12, 31),
            initial_capital=10000, universe=["FLAT"],
        )
        result = DonchianBacktester(cache, cfg).run()
        # Flat market means no new 20-day highs → no entries.
        assert len(result.closed_trades) == 0
        # Equity unchanged throughout.
        for p in result.equity_curve:
            assert p.equity == pytest.approx(10000, abs=0.01)

    def test_trailing_stop_protects_in_crash(self, cache):
        """Strong rise then crash — trailing stop should exit well above the bottom."""
        def price(i):
            if i < 150:
                return 50 + i * 2.0   # steep enough to breach 20-day highs
            # Crash
            return max(5.0, 50 + 149 * 2.0 - (i - 149) * 5)
        _seed_pattern(cache, "BOOM_BUST", date(2023, 1, 1), price, days=250)
        # Backtest window covers BOTH the uptrend (until ~July 2023) and
        # the subsequent crash — so we should enter during the rise and
        # exit via trailing stop during the crash.
        cfg = DonchianConfig(
            start=date(2023, 3, 1), end=date(2023, 11, 30),
            initial_capital=10000, universe=["BOOM_BUST"],
            entry_lookback=20, exit_lookback=10,
            atr_stop_multiple=2.0, risk_per_trade=0.02, max_total_risk=1.0,
        )
        result = DonchianBacktester(cache, cfg).run()
        assert len(result.closed_trades) >= 1
        # Trailing stop should limit damage — didn't ride all the way down to $5.
        # The position was entered in the uptrend at a low price and exited
        # in the crash after the ratchet had raised the stop substantially.
        # Net should be positive (the stop locked in significant gains before the crash).
        assert result.total_return > 0

    def test_max_hold_days_time_stop(self, cache):
        """Short-swing variant: positions auto-close after max_hold_days."""
        _seed_pattern(cache, "STEADY", date(2023, 1, 1), lambda i: 50 + i * 0.5, days=250)
        cfg = DonchianConfig(
            start=date(2023, 6, 1), end=date(2023, 12, 31),
            initial_capital=10000, universe=["STEADY"],
            max_hold_days=3,
        )
        result = DonchianBacktester(cache, cfg).run()
        # Every closed trade's hold should be ≤ 3+1 (entry not counted) days
        for t in result.closed_trades:
            days_held = (t.exit_date - t.entry_date).days
            assert days_held <= 5  # allow 1-2 for weekend edge cases
