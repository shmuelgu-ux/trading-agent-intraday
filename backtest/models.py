"""Data classes for the backtest engine.

Intentionally minimal — these are plain dataclasses, not SQLAlchemy
models. Backtest results are ephemeral; the production DB is off-limits
to the simulator so a wrong backtest run can't corrupt live data.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal


Side = Literal["buy", "sell"]


@dataclass
class Bar:
    """One OHLCV bar. Dates stored naive (date-level) — we only backtest on
    daily bars for now so intraday timestamps aren't needed."""
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OpenPosition:
    """A trade that's been entered but not yet closed."""
    ticker: str
    side: Side
    entry_date: date
    entry_price: float
    stop_loss: float
    take_profit: float
    shares: int
    risk_percent: float  # captured at entry for portfolio-risk checks

    def unrealized_pnl(self, mark_price: float) -> float:
        sign = 1 if self.side == "buy" else -1
        return sign * (mark_price - self.entry_price) * self.shares


@dataclass
class ClosedTrade:
    """A completed trade with its exit details."""
    ticker: str
    side: Side
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: int
    stop_loss: float
    take_profit: float
    pnl: float
    exit_reason: Literal["stop_loss", "take_profit", "end_of_backtest", "time_stop"]


@dataclass
class EquityPoint:
    """One day's equity snapshot. Written at the end of each simulated day."""
    trade_date: date
    equity: float  # cash + unrealized P&L of all open positions
    cash: float
    open_position_count: int


@dataclass
class BacktestConfig:
    """Parameters for a single backtest run."""
    start: date
    end: date
    initial_capital: float
    universe: list[str]
    max_risk_per_trade: float = 0.03
    max_total_risk: float = 0.20
    max_open_positions: int = 20
    default_rr_ratio: float = 2.0
    atr_sl_multiplier: float = 1.5
    min_signal_strength: int = 45
    commission_per_share: float = 0.0035  # IBKR Pro tiered, entry leg only (exit adds on close)
    # Max bars to hold a position before force-closing at market. Prevents
    # "forever-open" trades that never touch SL/TP in the sample window.
    max_hold_days: int = 60


@dataclass
class BacktestResult:
    """Aggregated output of a single backtest window."""
    config: BacktestConfig
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    open_at_end: list[OpenPosition] = field(default_factory=list)
    # Populated post-run by _attach_metrics in simulator.py
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    max_drawdown_pct: float | None = None
    calmar: float | None = None
    total_return: float | None = None
    total_return_pct: float | None = None
    annualized_return: float | None = None
    annualized_return_pct: float | None = None
    sample_size: int = 0

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.closed_trades if t.pnl > 0)

    @property
    def loss_count(self) -> int:
        return sum(1 for t in self.closed_trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        n = len(self.closed_trades)
        return (self.win_count / n) if n > 0 else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)
