import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from services.reconciliation_service import ReconciliationService


def _now_naive() -> datetime:
    """Tz-naive UTC now — matches how trade_log.timestamp is stored."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def make_trade(
    trade_id: int,
    ticker: str,
    side: str,
    entry_price: float,
    position_size: int,
    minutes_ago: int = 60,
):
    """Build a fake TradeLog ORM row."""
    t = MagicMock()
    t.id = trade_id
    t.ticker = ticker
    t.side = side
    t.entry_price = entry_price
    t.position_size = position_size
    t.timestamp = _now_naive() - timedelta(minutes=minutes_ago)
    t.status = "OPEN"
    return t


def make_fill(exit_price: float, minutes_ago: int = 5):
    return {
        "exit_price": exit_price,
        "exit_timestamp": _now_naive() - timedelta(minutes=minutes_ago),
        "filled_qty": 100,
    }


@pytest.mark.asyncio
async def test_reconciles_closed_long_with_profit():
    """Long position that hit its take-profit: pnl = (exit - entry) * qty."""
    trade = make_trade(1, "AAPL", "buy", entry_price=100.0, position_size=10)

    journal = MagicMock()
    journal.get_open_trades = AsyncMock(return_value=[trade])
    journal.mark_trade_closed = AsyncMock(return_value=True)

    alpaca = MagicMock()
    # AAPL is no longer in current positions -> it was closed
    alpaca.get_open_positions = MagicMock(return_value=[])
    alpaca.get_last_closing_fill = MagicMock(return_value=make_fill(exit_price=106.0))

    svc = ReconciliationService(alpaca, journal)
    reconciled = await svc.reconcile_closed_trades()

    assert reconciled == 1
    journal.mark_trade_closed.assert_awaited_once()
    call_args = journal.mark_trade_closed.await_args
    assert call_args.args[0] == 1               # trade_id
    assert call_args.args[1] == 106.0           # exit_price
    # pnl = (106 - 100) * 10 = 60.0
    assert call_args.args[3] == 60.0


@pytest.mark.asyncio
async def test_reconciles_closed_short_with_profit():
    """Short position that hit its take-profit (exit < entry): pnl = (entry - exit) * qty."""
    trade = make_trade(2, "TSLA", "sell", entry_price=200.0, position_size=5)

    journal = MagicMock()
    journal.get_open_trades = AsyncMock(return_value=[trade])
    journal.mark_trade_closed = AsyncMock(return_value=True)

    alpaca = MagicMock()
    alpaca.get_open_positions = MagicMock(return_value=[])
    alpaca.get_last_closing_fill = MagicMock(return_value=make_fill(exit_price=190.0))

    svc = ReconciliationService(alpaca, journal)
    reconciled = await svc.reconcile_closed_trades()

    assert reconciled == 1
    # pnl = (200 - 190) * 5 = 50.0
    assert journal.mark_trade_closed.await_args.args[3] == 50.0


@pytest.mark.asyncio
async def test_skips_positions_still_open():
    """Trades whose ticker is still in Alpaca positions should not be touched."""
    trade = make_trade(3, "MSFT", "buy", entry_price=300.0, position_size=2)

    journal = MagicMock()
    journal.get_open_trades = AsyncMock(return_value=[trade])
    journal.mark_trade_closed = AsyncMock(return_value=True)

    alpaca = MagicMock()
    alpaca.get_open_positions = MagicMock(return_value=[{"symbol": "MSFT"}])
    alpaca.get_last_closing_fill = MagicMock()

    svc = ReconciliationService(alpaca, journal)
    reconciled = await svc.reconcile_closed_trades()

    assert reconciled == 0
    journal.mark_trade_closed.assert_not_awaited()
    alpaca.get_last_closing_fill.assert_not_called()


@pytest.mark.asyncio
async def test_skips_when_no_closing_fill_found():
    """Manual close via Alpaca UI leaves no filled order — row stays OPEN."""
    trade = make_trade(4, "NFLX", "buy", entry_price=400.0, position_size=1)

    journal = MagicMock()
    journal.get_open_trades = AsyncMock(return_value=[trade])
    journal.mark_trade_closed = AsyncMock(return_value=True)

    alpaca = MagicMock()
    alpaca.get_open_positions = MagicMock(return_value=[])
    alpaca.get_last_closing_fill = MagicMock(return_value=None)

    svc = ReconciliationService(alpaca, journal)
    reconciled = await svc.reconcile_closed_trades()

    assert reconciled == 0
    journal.mark_trade_closed.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_batch_only_reconciles_actually_closed():
    """Three trades: one still open, one closed with fill, one closed without fill."""
    still_open = make_trade(10, "AAPL", "buy", 100.0, 10)
    closed_with_fill = make_trade(11, "TSLA", "sell", 200.0, 5)
    closed_no_fill = make_trade(12, "MSFT", "buy", 300.0, 2)

    journal = MagicMock()
    journal.get_open_trades = AsyncMock(
        return_value=[still_open, closed_with_fill, closed_no_fill]
    )
    journal.mark_trade_closed = AsyncMock(return_value=True)

    alpaca = MagicMock()
    # Only AAPL still in positions
    alpaca.get_open_positions = MagicMock(return_value=[{"symbol": "AAPL"}])

    def fake_fill(symbol, opposite_side, after):
        if symbol == "TSLA":
            return make_fill(exit_price=190.0)
        return None
    alpaca.get_last_closing_fill = MagicMock(side_effect=fake_fill)

    svc = ReconciliationService(alpaca, journal)
    reconciled = await svc.reconcile_closed_trades()

    assert reconciled == 1
    journal.mark_trade_closed.assert_awaited_once()
    # Only TSLA should have been closed
    assert journal.mark_trade_closed.await_args.args[0] == 11


@pytest.mark.asyncio
async def test_empty_open_trades_returns_zero():
    journal = MagicMock()
    journal.get_open_trades = AsyncMock(return_value=[])

    alpaca = MagicMock()
    svc = ReconciliationService(alpaca, journal)
    reconciled = await svc.reconcile_closed_trades()

    assert reconciled == 0
    # Should not even try to query positions when there's nothing to reconcile
    alpaca.get_open_positions.assert_not_called()


@pytest.mark.asyncio
async def test_skips_trade_with_missing_position_size():
    trade = make_trade(5, "AMZN", "buy", 150.0, position_size=0)
    trade.position_size = None

    journal = MagicMock()
    journal.get_open_trades = AsyncMock(return_value=[trade])
    journal.mark_trade_closed = AsyncMock(return_value=True)

    alpaca = MagicMock()
    alpaca.get_open_positions = MagicMock(return_value=[])
    alpaca.get_last_closing_fill = MagicMock()

    svc = ReconciliationService(alpaca, journal)
    reconciled = await svc.reconcile_closed_trades()

    assert reconciled == 0
    journal.mark_trade_closed.assert_not_awaited()
    alpaca.get_last_closing_fill.assert_not_called()


@pytest.mark.asyncio
async def test_skips_when_exit_ts_before_entry_ts():
    """Stale order data: exit timestamp is before entry timestamp."""
    trade = make_trade(6, "GOOG", "buy", 120.0, 5, minutes_ago=30)

    journal = MagicMock()
    journal.get_open_trades = AsyncMock(return_value=[trade])
    journal.mark_trade_closed = AsyncMock(return_value=True)

    alpaca = MagicMock()
    alpaca.get_open_positions = MagicMock(return_value=[])
    # Exit timestamp is 90 minutes ago, but entry was 30 minutes ago
    stale_fill = {
        "exit_price": 130.0,
        "exit_timestamp": _now_naive() - timedelta(minutes=90),
        "filled_qty": 5,
    }
    alpaca.get_last_closing_fill = MagicMock(return_value=stale_fill)

    svc = ReconciliationService(alpaca, journal)
    reconciled = await svc.reconcile_closed_trades()

    assert reconciled == 0
    journal.mark_trade_closed.assert_not_awaited()


@pytest.mark.asyncio
async def test_idempotent_second_call_reconciles_nothing():
    """Second reconcile returns 0 because the row is now CLOSED (not in get_open_trades)."""
    trade = make_trade(7, "AAPL", "buy", 100.0, 10)

    journal = MagicMock()
    # First call returns the open trade
    journal.get_open_trades = AsyncMock(side_effect=[[trade], []])
    journal.mark_trade_closed = AsyncMock(return_value=True)

    alpaca = MagicMock()
    alpaca.get_open_positions = MagicMock(return_value=[])
    alpaca.get_last_closing_fill = MagicMock(return_value=make_fill(exit_price=106.0))

    svc = ReconciliationService(alpaca, journal)
    first = await svc.reconcile_closed_trades()
    second = await svc.reconcile_closed_trades()

    assert first == 1
    assert second == 0
