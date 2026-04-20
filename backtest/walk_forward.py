"""Walk-forward validation.

Why it matters: a single backtest over one window is easy to overfit
(strategy tuned until the numbers look good on that window). Walk-
forward splits the history into rolling (train, test) windows and runs
the strategy on each TEST window — without re-fitting between windows.
If the strategy's edge is real, the per-window metrics should be
consistent; if it's overfit, one window will look great and the rest
will be noise.

For this bot, which has no trainable parameters yet (the scoring
weights are hard-coded), "train" doesn't actually adjust anything —
but the window structure is still useful for forward-testing a
fixed config across multiple out-of-sample periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from backtest.data_loader import BarCache
from backtest.models import BacktestConfig, BacktestResult
from backtest.simulator import Backtester


@dataclass
class Window:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class WalkForwardResult:
    windows: list[Window]
    per_window: list[BacktestResult]

    @property
    def avg_sharpe(self) -> float | None:
        vals = [r.sharpe for r in self.per_window if r.sharpe is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def avg_sortino(self) -> float | None:
        vals = [r.sortino for r in self.per_window if r.sortino is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def worst_drawdown_pct(self) -> float | None:
        vals = [r.max_drawdown_pct for r in self.per_window if r.max_drawdown_pct is not None]
        return max(vals) if vals else None

    @property
    def consistent(self) -> bool:
        """Heuristic consistency check — every window was positive.
        Imperfect but useful as a traffic light: ``False`` means the
        strategy lost money in at least one out-of-sample period."""
        returns = [r.total_return for r in self.per_window if r.total_return is not None]
        return all(r > 0 for r in returns) if returns else False


def split_windows(
    total_start: date,
    total_end: date,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[Window]:
    """Generate rolling (train, test) windows across [total_start, total_end].

    Each window advances by ``step_days`` (defaults to ``test_days`` —
    no overlap between consecutive test windows). Windows that would
    extend past ``total_end`` are skipped.

    Example: total = 2021-01-01..2024-12-31, train=365, test=90 →
    roughly 14 test windows of 3 months each, each preceded by the
    12 months that came before it.
    """
    if train_days < 1 or test_days < 1:
        raise ValueError("train_days and test_days must be positive")
    step = step_days if step_days is not None else test_days
    windows: list[Window] = []
    # Earliest possible TRAIN_START so that train_end + test_days fits.
    cursor = total_start
    while True:
        train_end = cursor + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)
        if test_end > total_end:
            break
        windows.append(Window(
            train_start=cursor,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
        cursor = cursor + timedelta(days=step)
    return windows


def run_walk_forward(
    cache: BarCache,
    base_config: BacktestConfig,
    windows: list[Window],
) -> WalkForwardResult:
    """Run a backtest for each TEST window using the base config's risk
    settings. The train window is ignored by the simulator for now
    (no trainable parameters); it's kept in the data model so adding
    per-window hyperparameter tuning later doesn't require a schema
    change.
    """
    per_window: list[BacktestResult] = []
    for w in windows:
        cfg = BacktestConfig(
            start=w.test_start,
            end=w.test_end,
            initial_capital=base_config.initial_capital,
            universe=base_config.universe,
            max_risk_per_trade=base_config.max_risk_per_trade,
            max_total_risk=base_config.max_total_risk,
            max_open_positions=base_config.max_open_positions,
            default_rr_ratio=base_config.default_rr_ratio,
            atr_sl_multiplier=base_config.atr_sl_multiplier,
            min_signal_strength=base_config.min_signal_strength,
            commission_per_share=base_config.commission_per_share,
            max_hold_days=base_config.max_hold_days,
        )
        result = Backtester(cache, cfg).run()
        per_window.append(result)
    return WalkForwardResult(windows=windows, per_window=per_window)
