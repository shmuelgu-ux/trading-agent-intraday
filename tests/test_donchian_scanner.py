"""Tests for DonchianStockScanner (intraday variant).

The intraday bot does NOT use a ChannelExitMonitor — EOD force close
at 15:55 ET already closes every position same-day, so channel-exit
logic would be redundant. Only the scanner is tested here.
"""
from types import SimpleNamespace
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from core.donchian_scanner import DonchianStockScanner, _BreakoutReading


def _alpaca_bar(close, high=None, low=None, volume=1_500_000):
    return SimpleNamespace(
        open=close, high=high if high is not None else close + 1,
        low=low if low is not None else close - 1,
        close=close, volume=volume,
    )


def _series(prices):
    return [_alpaca_bar(p) for p in prices]


class TestReadOne:
    def _scanner(self):
        alpaca = MagicMock()
        alpaca._client = MagicMock()
        alpaca._data_client = MagicMock()
        # Intraday default lookback is 10 (vs 20 for swing).
        return DonchianStockScanner(
            alpaca, entry_lookback=10, atr_period=14,
            min_price=5.0, min_volume=500_000,
        )

    def test_no_breakout_on_flat_series(self):
        bars = _series([100] * 30)
        r = self._scanner()._read_one("X", bars)
        # Flat close=100, prior high=101, so close NOT > prior high.
        assert r is None

    def test_breakout_detected(self):
        # Intraday needs at least 10 + 14 + 2 = 26 bars.
        bars = _series([100] * 30 + [120])
        r = self._scanner()._read_one("X", bars)
        assert r is not None
        assert r.symbol == "X"
        assert r.close == 120
        assert r.upper_entry == 101

    def test_price_below_min_price_rejected(self):
        bars = _series([3.0] * 30)
        r = self._scanner()._read_one("X", bars)
        assert r is None

    def test_too_few_bars_rejected(self):
        bars = _series([100] * 10)
        r = self._scanner()._read_one("X", bars)
        assert r is None

    def test_atr_computed(self):
        # Flat series + breakout at end — plus decent OHLC spread so
        # ATR is non-trivial. Each bar: close=100, high=103, low=97 → TR=6.
        bars = [_alpaca_bar(close=100, high=103, low=97) for _ in range(30)]
        bars.append(_alpaca_bar(close=120, high=122, low=99))
        r = self._scanner()._read_one("X", bars)
        assert r is not None
        assert r.atr > 0


class TestBreakoutReading:
    def test_breakout_pct(self):
        r = _BreakoutReading(symbol="X", close=110, upper_entry=100, atr=2, volume_ratio=1.0)
        assert abs(r.breakout_pct - 0.0909) < 1e-3

    def test_breakout_pct_handles_zero_close(self):
        r = _BreakoutReading(symbol="X", close=0, upper_entry=100, atr=2, volume_ratio=1.0)
        assert r.breakout_pct == 0.0
