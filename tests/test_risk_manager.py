import pytest
from core.risk_manager import RiskManager
from models.signals import TradingViewSignal, SignalAction, Indicators


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


class TestStopLoss:
    def test_buy_stop_loss_below_entry(self):
        rm = RiskManager(atr_sl_multiplier=1.5)
        sl = rm.calculate_stop_loss(100.0, 2.0, SignalAction.BUY)
        assert sl == 97.0

    def test_sell_stop_loss_above_entry(self):
        rm = RiskManager(atr_sl_multiplier=1.5)
        sl = rm.calculate_stop_loss(100.0, 2.0, SignalAction.SELL)
        assert sl == 103.0


class TestTakeProfit:
    def test_buy_take_profit(self):
        rm = RiskManager(default_rr_ratio=2.0)
        tp = rm.calculate_take_profit(100.0, 97.0, SignalAction.BUY)
        assert tp == 106.0

    def test_sell_take_profit(self):
        rm = RiskManager(default_rr_ratio=2.0)
        tp = rm.calculate_take_profit(100.0, 103.0, SignalAction.SELL)
        assert tp == 94.0


class TestPositionSizing:
    def test_basic_sizing(self):
        rm = RiskManager(max_risk_per_trade=0.03)
        # $2000 account, 3% risk = $60, risk_per_share = $5
        shares = rm.calculate_position_size(2000, 100.0, 95.0)
        assert shares == 12  # 60 / 5

    def test_cannot_afford(self):
        rm = RiskManager(max_risk_per_trade=0.03)
        shares = rm.calculate_position_size(1000, 500.0, 495.0)
        assert shares == 2  # min(12, floor(1000/500)=2)

    def test_zero_risk_per_share(self):
        rm = RiskManager()
        shares = rm.calculate_position_size(100_000, 100.0, 100.0)
        assert shares == 0


class TestValidation:
    def test_valid_trade(self):
        rm = RiskManager()
        signal = make_signal()
        is_valid, reasons = rm.validate_trade(signal, 2000, [], 0.0)
        assert is_valid
        assert len(reasons) == 0

    def test_max_positions_reached(self):
        rm = RiskManager(max_open_positions=2)
        signal = make_signal()
        positions = [{"symbol": "MSFT"}, {"symbol": "GOOG"}]
        is_valid, reasons = rm.validate_trade(signal, 2000, positions, 0.0)
        assert not is_valid
        assert any("מקסימום פוזיציות" in r for r in reasons)

    def test_duplicate_ticker(self):
        rm = RiskManager()
        signal = make_signal(ticker="AAPL")
        positions = [{"symbol": "AAPL"}]
        is_valid, reasons = rm.validate_trade(signal, 2000, positions, 0.0)
        assert not is_valid
        assert any("כבר יש פוזיציה" in r for r in reasons)

    def test_total_risk_exceeded(self):
        rm = RiskManager(max_total_risk=0.20)
        signal = make_signal()
        # At intraday's 1% per-trade default the signal consumes ~0.8% risk,
        # so push current total close enough that the new trade tips us over.
        is_valid, reasons = rm.validate_trade(signal, 2000, [], 0.195)
        assert not is_valid
        assert any("סיכון כולל" in r for r in reasons)

    def test_buy_against_downtrend(self):
        rm = RiskManager()
        signal = make_signal(ema_trend="down")
        is_valid, reasons = rm.validate_trade(signal, 2000, [], 0.0)
        assert not is_valid
        assert any("נגד הטרנד" in r for r in reasons)

    def test_sell_against_uptrend(self):
        rm = RiskManager()
        signal = make_signal(action=SignalAction.SELL, ema_trend="up")
        is_valid, reasons = rm.validate_trade(signal, 2000, [], 0.0)
        assert not is_valid
        assert any("נגד הטרנד" in r for r in reasons)

    def test_rsi_too_high_for_buy(self):
        rm = RiskManager()
        signal = make_signal(rsi=80.0)
        is_valid, reasons = rm.validate_trade(signal, 2000, [], 0.0)
        assert not is_valid
        assert any("RSI" in r for r in reasons)

    def test_no_atr_rejects(self):
        rm = RiskManager()
        signal = make_signal(atr=None)
        is_valid, reasons = rm.validate_trade(signal, 2000, [], 0.0)
        assert not is_valid


class TestRiskParams:
    def test_full_calculation(self):
        rm = RiskManager(atr_sl_multiplier=1.5, default_rr_ratio=2.0, max_risk_per_trade=0.03)
        signal = make_signal(price=100.0, atr=2.0)
        params = rm.calculate_risk_params(signal, 2000)

        assert params is not None
        assert params.entry_price == 100.0
        assert params.stop_loss == 97.0
        assert params.take_profit == 106.0
        assert params.position_size > 0
        assert params.risk_percent <= 0.03
        assert params.reward_risk_ratio == 2.0
