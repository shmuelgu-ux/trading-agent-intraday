"""Tests for backtest.data_loader.BarCache."""
from datetime import date

import pytest

from backtest.data_loader import BarCache, alpaca_bars_to_domain
from backtest.models import Bar


@pytest.fixture
def cache(tmp_path):
    return BarCache(str(tmp_path / "cache.db"))


def _bar(d, close=100.0, o=None, h=None, lo=None, v=1_000_000):
    return Bar(
        trade_date=d,
        open=o if o is not None else close,
        high=h if h is not None else close + 1,
        low=lo if lo is not None else close - 1,
        close=close,
        volume=v,
    )


class TestPutGet:
    def test_put_and_get_roundtrip(self, cache):
        bars = [_bar(date(2024, 1, 2), 100), _bar(date(2024, 1, 3), 101)]
        n = cache.put("AAPL", "day", bars)
        assert n == 2
        got = cache.get_bars("AAPL", "day", date(2024, 1, 1), date(2024, 1, 31))
        assert len(got) == 2
        assert got[0].trade_date == date(2024, 1, 2)
        assert got[0].close == 100.0
        assert got[1].trade_date == date(2024, 1, 3)

    def test_put_replaces_existing(self, cache):
        cache.put("AAPL", "day", [_bar(date(2024, 1, 2), 100)])
        cache.put("AAPL", "day", [_bar(date(2024, 1, 2), 105)])  # same PK
        got = cache.get_bars("AAPL", "day", date(2024, 1, 2), date(2024, 1, 2))
        assert len(got) == 1
        assert got[0].close == 105.0  # upserted

    def test_tickers_are_separate(self, cache):
        cache.put("AAPL", "day", [_bar(date(2024, 1, 2), 100)])
        cache.put("MSFT", "day", [_bar(date(2024, 1, 2), 300)])
        aapl = cache.get_bars("AAPL", "day", date(2024, 1, 1), date(2024, 1, 10))
        msft = cache.get_bars("MSFT", "day", date(2024, 1, 1), date(2024, 1, 10))
        assert aapl[0].close == 100.0
        assert msft[0].close == 300.0

    def test_timeframes_are_separate(self, cache):
        cache.put("AAPL", "day", [_bar(date(2024, 1, 2), 100)])
        cache.put("AAPL", "week", [_bar(date(2024, 1, 5), 105)])
        day = cache.get_bars("AAPL", "day", date(2024, 1, 1), date(2024, 1, 31))
        week = cache.get_bars("AAPL", "week", date(2024, 1, 1), date(2024, 1, 31))
        assert len(day) == 1 and day[0].close == 100.0
        assert len(week) == 1 and week[0].close == 105.0

    def test_get_respects_date_range(self, cache):
        bars = [_bar(date(2024, 1, d + 1), 100 + d) for d in range(10)]
        cache.put("AAPL", "day", bars)
        got = cache.get_bars("AAPL", "day", date(2024, 1, 3), date(2024, 1, 5))
        assert len(got) == 3
        assert got[0].trade_date == date(2024, 1, 3)
        assert got[-1].trade_date == date(2024, 1, 5)


class TestCoverage:
    def test_empty_cache(self, cache):
        first, last, count = cache.coverage("AAPL", "day")
        assert first is None and last is None
        assert count == 0

    def test_reports_actual_range(self, cache):
        cache.put("AAPL", "day", [
            _bar(date(2024, 1, 2)), _bar(date(2024, 3, 15)), _bar(date(2024, 6, 1)),
        ])
        first, last, count = cache.coverage("AAPL", "day")
        assert first == date(2024, 1, 2)
        assert last == date(2024, 6, 1)
        assert count == 3


class TestClear:
    def test_clear_ticker_only(self, cache):
        cache.put("AAPL", "day", [_bar(date(2024, 1, 2))])
        cache.put("MSFT", "day", [_bar(date(2024, 1, 2))])
        cache.clear(ticker="AAPL")
        assert cache.coverage("AAPL", "day")[2] == 0
        assert cache.coverage("MSFT", "day")[2] == 1

    def test_clear_all(self, cache):
        cache.put("AAPL", "day", [_bar(date(2024, 1, 2))])
        cache.put("MSFT", "day", [_bar(date(2024, 1, 2))])
        cache.clear()
        assert cache.coverage("AAPL", "day")[2] == 0
        assert cache.coverage("MSFT", "day")[2] == 0


class TestAlpacaBarsConversion:
    def test_converts_with_datetime_timestamp(self):
        from datetime import datetime, timezone

        class FakeBar:
            def __init__(self, ts, o, h, l, c, v):
                self.timestamp = ts
                self.open = o; self.high = h; self.low = l
                self.close = c; self.volume = v

        fakes = [FakeBar(datetime(2024, 1, 2, 16, tzinfo=timezone.utc), 100, 102, 99, 101, 1_000_000)]
        out = alpaca_bars_to_domain(fakes)
        assert len(out) == 1
        assert out[0].trade_date == date(2024, 1, 2)
        assert out[0].open == 100.0 and out[0].close == 101.0

    def test_skips_bars_with_no_timestamp(self):
        class NoTs:
            timestamp = None
            open = 1; high = 1; low = 1; close = 1; volume = 1
        assert alpaca_bars_to_domain([NoTs()]) == []
