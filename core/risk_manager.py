import math
from loguru import logger
from config import settings
from models.signals import TradingViewSignal, SignalAction
from models.orders import RiskParams


class RiskManager:
    """Central risk management engine.

    Enforces position sizing, stop-loss/take-profit calculation,
    and portfolio-level risk constraints before any trade is executed.
    """

    def __init__(
        self,
        max_risk_per_trade: float = settings.max_risk_per_trade,
        max_total_risk: float = settings.max_total_risk,
        max_open_positions: int = settings.max_open_positions,
        default_rr_ratio: float = settings.default_rr_ratio,
        atr_sl_multiplier: float = settings.atr_sl_multiplier,
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_risk = max_total_risk
        self.max_open_positions = max_open_positions
        self.default_rr_ratio = default_rr_ratio
        self.atr_sl_multiplier = atr_sl_multiplier

    def calculate_stop_loss(
        self, entry_price: float, atr: float, action: SignalAction
    ) -> float:
        """Dynamic stop-loss based on ATR."""
        distance = atr * self.atr_sl_multiplier
        if action == SignalAction.BUY:
            return round(entry_price - distance, 2)
        else:  # SELL (short)
            return round(entry_price + distance, 2)

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        action: SignalAction,
        rr_ratio: float | None = None,
    ) -> float:
        """Take-profit based on Risk:Reward ratio."""
        rr = rr_ratio or self.default_rr_ratio
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = risk_distance * rr

        if action == SignalAction.BUY:
            return round(entry_price + reward_distance, 2)
        else:
            return round(entry_price - reward_distance, 2)

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: float | None = None,
    ) -> int:
        """How many shares to buy given our risk tolerance."""
        risk = risk_pct or self.max_risk_per_trade
        risk_amount = account_balance * risk
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return 0

        shares = math.floor(risk_amount / risk_per_share)
        # Never buy more than we can afford
        max_affordable = math.floor(account_balance / entry_price)
        return min(shares, max_affordable)

    def calculate_risk_params(
        self,
        signal: TradingViewSignal,
        account_balance: float,
    ) -> RiskParams | None:
        """Full risk parameter calculation for a signal."""
        atr = signal.indicators.atr
        if not atr or atr <= 0:
            logger.warning(f"[{signal.ticker}] No ATR provided, cannot calculate risk")
            return None

        # A non-positive balance can't size a trade and previously produced
        # a bogus 100%-risk fallback that slipped past downstream checks.
        if account_balance <= 0:
            logger.warning(
                f"[{signal.ticker}] Non-positive account balance ({account_balance}), "
                f"cannot size trade"
            )
            return None

        stop_loss = self.calculate_stop_loss(signal.price, atr, signal.action)
        take_profit = self.calculate_take_profit(signal.price, stop_loss, signal.action)
        position_size = self.calculate_position_size(
            account_balance, signal.price, stop_loss
        )

        if position_size <= 0:
            logger.warning(f"[{signal.ticker}] Position size is 0, trade too risky")
            return None

        risk_per_share = abs(signal.price - stop_loss)
        if risk_per_share <= 0:
            # Defensive: position_size>0 guard above normally protects this,
            # but belt-and-braces in case rounding collapses entry == SL.
            logger.warning(
                f"[{signal.ticker}] risk_per_share collapsed to 0 "
                f"(entry={signal.price}, SL={stop_loss})"
            )
            return None

        risk_amount = risk_per_share * position_size
        risk_percent = risk_amount / account_balance
        rr_ratio = abs(take_profit - signal.price) / risk_per_share

        return RiskParams(
            entry_price=signal.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=round(risk_amount, 2),
            risk_percent=round(risk_percent, 4),
            reward_risk_ratio=round(rr_ratio, 2),
        )

    def validate_trade(
        self,
        signal: TradingViewSignal,
        account_balance: float,
        open_positions: list[dict],
        current_total_risk: float,
    ) -> tuple[bool, list[str]]:
        """Validate whether a trade passes all risk checks.

        Returns (is_valid, list_of_reasons).
        """
        reasons: list[str] = []

        # 1) Max open positions
        if len(open_positions) >= self.max_open_positions:
            reasons.append(
                f"הגעת למקסימום פוזיציות פתוחות ({self.max_open_positions})"
            )

        # 2) Already have a position in this ticker
        for pos in open_positions:
            if pos.get("symbol") == signal.ticker:
                reasons.append(f"כבר יש פוזיציה פתוחה ב-{signal.ticker}")
                break

        # 3) Calculate risk params
        risk_params = self.calculate_risk_params(signal, account_balance)
        if risk_params is None:
            reasons.append("לא ניתן לחשב פרמטרי סיכון (חסר ATR או גודל פוזיציה 0)")
            return False, reasons

        # 4) Single trade risk check
        if risk_params.risk_percent > self.max_risk_per_trade:
            reasons.append(
                f"סיכון העסקה {risk_params.risk_percent:.2%} חורג מהמקסימום "
                f"{self.max_risk_per_trade:.2%}"
            )

        # 5) Portfolio total risk check. Tiny epsilon so float drift at the
        # exact cap boundary doesn't falsely reject a trade (e.g. 20 x 0.01
        # coming out to 0.20000000000000004 > 0.2 in IEEE754).
        new_total_risk = current_total_risk + risk_params.risk_percent
        if new_total_risk > self.max_total_risk + 1e-9:
            reasons.append(
                f"סיכון כולל של התיק יגיע ל-{new_total_risk:.2%}, "
                f"חורג מהמקסימום {self.max_total_risk:.2%}"
            )

        # 6) Minimum R:R ratio check — derive the floor from the configured
        # default_rr_ratio so lowering the default in env doesn't accidentally
        # reject every trade. Gives us ~10% tolerance under the default.
        min_rr = max(1.0, self.default_rr_ratio - 0.1)
        if risk_params.reward_risk_ratio < min_rr:
            reasons.append(
                f"יחס סיכוי/סיכון {risk_params.reward_risk_ratio:.2f} "
                f"נמוך מהמינימום {min_rr:.2f}"
            )

        # 7) RSI extremes filter
        rsi = signal.indicators.rsi
        if rsi is not None:
            if signal.action == SignalAction.BUY and rsi > 75:
                reasons.append(f"RSI גבוה מדי לקנייה ({rsi:.1f} > 75) - קניית יתר")
            elif signal.action == SignalAction.SELL and rsi < 25:
                reasons.append(f"RSI נמוך מדי למכירה ({rsi:.1f} < 25) - מכירת יתר")

        # 8) Trend alignment check
        ema_trend = signal.indicators.ema_trend
        if ema_trend:
            if signal.action == SignalAction.BUY and ema_trend == "down":
                reasons.append("קנייה נגד הטרנד - EMA מצביע על מגמה יורדת")
            elif signal.action == SignalAction.SELL and ema_trend == "up":
                reasons.append("מכירה נגד הטרנד - EMA מצביע על מגמה עולה")

        is_valid = len(reasons) == 0
        return is_valid, reasons
