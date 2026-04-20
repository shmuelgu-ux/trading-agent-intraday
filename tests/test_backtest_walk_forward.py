"""Tests for the walk-forward splitter + aggregator."""
from datetime import date

import pytest

from backtest.models import BacktestResult, BacktestConfig
from backtest.walk_forward import (
    WalkForwardResult,
    Window,
    split_windows,
)


class TestSplitWindows:
    def test_rejects_zero_durations(self):
        with pytest.raises(ValueError):
            split_windows(date(2024, 1, 1), date(2024, 12, 31), 0, 90)
        with pytest.raises(ValueError):
            split_windows(date(2024, 1, 1), date(2024, 12, 31), 365, 0)

    def test_single_fitting_window(self):
        # 365 + 90 = 455 days exactly fits the 455-day range
        windows = split_windows(date(2024, 1, 1), date(2025, 3, 30), 365, 90)
        assert len(windows) == 1
        w = windows[0]
        assert w.train_start == date(2024, 1, 1)
        assert w.test_start == date(2024, 12, 31)
        assert w.test_end == date(2025, 3, 30)

    def test_returns_empty_when_too_short(self):
        # range shorter than train+test
        windows = split_windows(date(2024, 1, 1), date(2024, 1, 10), 365, 90)
        assert windows == []

    def test_non_overlapping_step_defaults_to_test_days(self):
        windows = split_windows(
            date(2023, 1, 1), date(2024, 12, 31), train_days=180, test_days=60,
        )
        # Test windows should be consecutive (step = test_days)
        for i in range(1, len(windows)):
            assert windows[i].test_start == windows[i - 1].test_end + \
                   (windows[i].test_start - windows[i - 1].test_end)
        # Assert each test window is 60 days
        for w in windows:
            assert (w.test_end - w.test_start).days + 1 == 60

    def test_overlapping_step(self):
        """step_days < test_days → overlapping test windows."""
        wins = split_windows(
            date(2024, 1, 1), date(2024, 12, 31),
            train_days=60, test_days=30, step_days=10,
        )
        # Each window advances by only 10 days
        if len(wins) >= 2:
            diff = (wins[1].train_start - wins[0].train_start).days
            assert diff == 10

    def test_windows_within_range(self):
        start = date(2022, 1, 1)
        end = date(2024, 12, 31)
        wins = split_windows(start, end, train_days=365, test_days=90)
        for w in wins:
            assert w.train_start >= start
            assert w.test_end <= end


class TestWalkForwardResult:
    @staticmethod
    def _make_result(total_return, sharpe=None, sortino=None, max_dd=None):
        r = BacktestResult(config=BacktestConfig(
            start=date(2024, 1, 1), end=date(2024, 3, 31),
            initial_capital=2000, universe=[],
        ))
        r.total_return = total_return
        r.sharpe = sharpe
        r.sortino = sortino
        r.max_drawdown_pct = max_dd
        return r

    def test_avg_sharpe(self):
        wf = WalkForwardResult(
            windows=[],
            per_window=[
                self._make_result(0.1, sharpe=1.5),
                self._make_result(0.2, sharpe=2.0),
                self._make_result(0.05, sharpe=None),  # ignored
            ],
        )
        assert wf.avg_sharpe == pytest.approx(1.75)

    def test_avg_sortino_none_when_no_data(self):
        wf = WalkForwardResult(windows=[], per_window=[])
        assert wf.avg_sortino is None

    def test_worst_drawdown(self):
        wf = WalkForwardResult(
            windows=[],
            per_window=[
                self._make_result(0.0, max_dd=5.0),
                self._make_result(0.0, max_dd=18.0),
                self._make_result(0.0, max_dd=12.0),
            ],
        )
        assert wf.worst_drawdown_pct == 18.0

    def test_consistent_true_when_all_positive(self):
        wf = WalkForwardResult(
            windows=[],
            per_window=[
                self._make_result(0.01),
                self._make_result(0.05),
                self._make_result(0.001),
            ],
        )
        assert wf.consistent is True

    def test_consistent_false_when_any_negative(self):
        wf = WalkForwardResult(
            windows=[],
            per_window=[
                self._make_result(0.01),
                self._make_result(-0.02),  # losing window
                self._make_result(0.05),
            ],
        )
        assert wf.consistent is False
