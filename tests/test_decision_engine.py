import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.decision_engine import DecisionEngine
from core.risk_manager import RiskManager
from models.signals import TradingViewSignal, SignalAction, Indicators
from models.orders import DecisionAction


def make_signal(
    ticker="AAPL",
    action=SignalAction.BUY,
    price=185.0,
    atr=3.5,
    rsi=45.0,
    ema_trend="up",
) -> TradingViewSignal:
    return TradingViewSignal(
        ticker=ticker,
        action=action,
        price=price,
        indicators=Indicators(
            rsi=rsi, macd_signal="bullish_cross", ema_trend=ema_trend, atr=atr, volume_ratio=1.5
        ),
    )


def make_engine():
    risk_manager = RiskManager()
    alpaca = MagicMock()
    alpaca.get_account.return_value = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "buying_power": 200_000.0,
        "portfolio_value": 100_000.0,
        "pattern_day_trader": False,
    }
    alpaca.get_open_positions.return_value = []
    alpaca.get_position.return_value = None
    alpaca.submit_bracket_order.return_value = {
        "order_id": "test-001",
        "status": "accepted",
    }

    journal = MagicMock()
    journal.log_trade = AsyncMock(return_value=1)

    return DecisionEngine(risk_manager, alpaca, journal)


@pytest.mark.asyncio
async def test_valid_signal_executes():
    engine = make_engine()
    signal = make_signal()
    decision = await engine.process_signal(signal)

    assert decision.action == DecisionAction.EXECUTE
    assert decision.ticker == "AAPL"
    assert decision.risk_params is not None
    assert decision.risk_params.position_size > 0
    engine.alpaca.submit_bracket_order.assert_called_once()


@pytest.mark.asyncio
async def test_signal_against_trend_no_longer_rejected():
    """EMA-trend filter removed for the Donchian strategy — the breakout
    itself establishes the trend on its horizon. (The pre-existing
    after-15:00-ET time filter may still reject this if the suite runs
    during that window — see ``test_valid_signal_executes`` for the
    same issue.)"""
    engine = make_engine()
    signal = make_signal(ema_trend="down")
    decision = await engine.process_signal(signal)
    # Should NOT reject on trend-alignment specifically.
    assert not any("נגד הטרנד" in r for r in decision.reasoning)


@pytest.mark.asyncio
async def test_duplicate_position_rejected():
    engine = make_engine()
    engine.alpaca.get_open_positions.return_value = [
        {"symbol": "AAPL", "unrealized_pnl": -50}
    ]
    signal = make_signal(ticker="AAPL")
    decision = await engine.process_signal(signal)

    assert decision.action == DecisionAction.REJECT
    assert any("כבר יש פוזיציה" in r for r in decision.reasoning)


@pytest.mark.asyncio
async def test_journal_is_called():
    engine = make_engine()
    signal = make_signal()
    await engine.process_signal(signal)

    engine.journal.log_trade.assert_called_once()


# ---------------------------------------------------------------------------
# _is_market_open — holiday-aware via Alpaca clock
# ---------------------------------------------------------------------------


def test_market_open_prefers_alpaca_clock_true():
    """When Alpaca's clock says open, trust it regardless of weekday/time."""
    engine = make_engine()
    engine.alpaca.is_market_open.return_value = True
    assert engine._is_market_open() is True


def test_market_open_prefers_alpaca_clock_false_on_holiday():
    """The real Memorial Day 2026 bug: a regular Monday at 14:00 ET would
    pass the weekday/time fallback, but Alpaca's clock correctly returns
    False on the holiday. We must trust Alpaca, not the fallback.
    """
    engine = make_engine()
    engine.alpaca.is_market_open.return_value = False
    assert engine._is_market_open() is False


def test_market_open_falls_back_when_clock_returns_none_weekday_open():
    """If Alpaca's clock is unreachable, fall back to weekday+time check.
    Patch _now_et so the test is deterministic regardless of when it runs.
    """
    import datetime as _dt, zoneinfo
    engine = make_engine()
    engine.alpaca.is_market_open.return_value = None
    monday_noon_et = _dt.datetime(2026, 6, 1, 12, 0, tzinfo=zoneinfo.ZoneInfo("America/New_York"))
    with patch.object(engine, "_now_et", return_value=monday_noon_et):
        assert engine._is_market_open() is True


def test_market_open_falls_back_when_clock_returns_none_weekend():
    """Fallback correctly recognises weekends even without Alpaca."""
    import datetime as _dt, zoneinfo
    engine = make_engine()
    engine.alpaca.is_market_open.return_value = None
    saturday_noon_et = _dt.datetime(2026, 5, 30, 12, 0, tzinfo=zoneinfo.ZoneInfo("America/New_York"))
    with patch.object(engine, "_now_et", return_value=saturday_noon_et):
        assert engine._is_market_open() is False


def test_market_open_falls_back_when_clock_returns_none_after_hours():
    """Fallback rejects 18:00 ET (after the 16:00 close) on a weekday."""
    import datetime as _dt, zoneinfo
    engine = make_engine()
    engine.alpaca.is_market_open.return_value = None
    monday_evening = _dt.datetime(2026, 6, 1, 18, 0, tzinfo=zoneinfo.ZoneInfo("America/New_York"))
    with patch.object(engine, "_now_et", return_value=monday_evening):
        assert engine._is_market_open() is False
