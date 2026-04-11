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
async def test_signal_against_trend_rejected():
    engine = make_engine()
    signal = make_signal(ema_trend="down")
    decision = await engine.process_signal(signal)

    assert decision.action == DecisionAction.REJECT
    assert any("נגד הטרנד" in r for r in decision.reasoning)
    engine.alpaca.submit_bracket_order.assert_not_called()


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
