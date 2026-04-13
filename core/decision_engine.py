from loguru import logger
from config import settings
from models.signals import TradingViewSignal, SignalAction
from models.orders import TradeDecision, DecisionAction, RiskParams
from models.journal import TradeJournalEntry
from core.risk_manager import RiskManager
from services.alpaca_client import AlpacaClient
from services.journal_service import JournalService


# Alpaca order statuses that indicate the broker refused or killed the
# order up-front. Any of these means we should treat the trade as
# rejected and NOT bump the pending counters — otherwise the scanner
# will think there's a phantom position holding a slot + risk budget.
_ALPACA_FAIL_STATUSES = frozenset({
    "rejected",
    "canceled",
    "cancelled",
    "expired",
    "suspended",
    "stopped",
    "held",  # held indefinitely by Alpaca risk checks
})


class DecisionEngine:
    """The 'brain' - receives signals, evaluates, decides, executes.

    Phase 1: Technical analysis only (news/sentiment/fundamentals disabled).
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        alpaca: AlpacaClient,
        journal: JournalService,
        news=None,
        fundamentals=None,
    ):
        self.risk = risk_manager
        self.alpaca = alpaca
        self.journal = journal
        # Phase 2 disabled - kept for future use
        self.news = None
        self.fundamentals = None
        # Internal counters to track what we've sent during current scan
        # (Alpaca has delay - orders may not show up as positions immediately)
        self._pending_count = 0
        self._pending_risk = 0.0
        self._pending_tickers: set[str] = set()
        # Snapshot of positions at scan start (to avoid double-counting)
        self._scan_start_positions = 0
        self._scan_start_risk = 0.0
        # Tickers that existed BEFORE scan started
        self._scan_start_tickers: set[str] = set()
        # Tickers that lost money today — blocked from re-entry until
        # tomorrow. Populated by mark_ticker_lost(), reset by
        # reset_daily_blacklist(). Prevents the scanner from repeatedly
        # entering and getting stopped out on the same stock.
        self._daily_losers: set[str] = set()
        self._daily_losers_date: str = ""

    def mark_ticker_lost(self, ticker: str):
        """Called by reconciliation when a trade closes with a loss.
        Blocks re-entry on the same ticker for the rest of the day."""
        today = self._now_et().strftime("%Y-%m-%d")
        if self._daily_losers_date != today:
            self._daily_losers.clear()
            self._daily_losers_date = today
        self._daily_losers.add(ticker)
        logger.info(f"Blocked {ticker} for rest of day (lost today)")

    def _check_daily_blacklist(self, ticker: str) -> str | None:
        """Returns a rejection reason if ticker lost today, else None."""
        today = self._now_et().strftime("%Y-%m-%d")
        if self._daily_losers_date != today:
            self._daily_losers.clear()
            self._daily_losers_date = today
            return None
        if ticker in self._daily_losers:
            return f"המניה {ticker} כבר הפסידה היום — חסום כניסה חוזרת"
        return None

    async def reset_pending(self):
        """Reset pending counters - call at start of each scan cycle.

        Snapshots current positions so we don't double-count when orders
        transition from pending to filled during the scan. Pulls each
        position's ACTUAL risk_percent from the journal (fallback to
        max_risk_per_trade if not found).
        """
        self._pending_count = 0
        self._pending_risk = 0.0
        self._pending_tickers.clear()
        # Snapshot existing positions from Alpaca
        try:
            positions = self.alpaca.get_open_positions()
            self._scan_start_positions = len(positions)
            self._scan_start_tickers = {p["symbol"] for p in positions}

            # Look up ACTUAL risk_percent per position from the journal.
            # Query by ticker directly so reject noise in the journal can't
            # hide old EXECUTE rows (was broken with limit=100 approach).
            trade_risks: dict[str, float] = {}
            try:
                trade_risks = await self.journal.get_risk_for_tickers(
                    list(self._scan_start_tickers)
                )
            except Exception as e:
                logger.warning(f"Failed to read journal for risk snapshot: {e}")

            self._scan_start_risk = sum(
                trade_risks.get(p["symbol"], settings.max_risk_per_trade)
                for p in positions
            )
            missing = [
                p["symbol"] for p in positions if p["symbol"] not in trade_risks
            ]
            logger.info(
                f"Scan snapshot: {self._scan_start_positions} existing positions, "
                f"{self._scan_start_risk:.2%} existing risk (actual from journal"
                + (f", fallback for: {missing}" if missing else "")
                + ")"
            )
        except Exception as e:
            logger.warning(f"Failed to snapshot positions: {e}")
            self._scan_start_positions = 0
            self._scan_start_risk = 0.0
            self._scan_start_tickers = set()

    def _now_et(self):
        """Current time in US Eastern (handles DST automatically)."""
        from datetime import datetime
        import zoneinfo
        try:
            return datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        except Exception:
            import pytz
            return datetime.now(pytz.timezone("America/New_York"))

    def _is_market_open(self) -> bool:
        """Check if US market is currently open."""
        et = self._now_et()
        if et.weekday() >= 5:  # Weekend
            return False
        mins = et.hour * 60 + et.minute
        return 9 * 60 + 30 <= mins < 16 * 60  # 9:30 AM - 4:00 PM ET

    def _is_past_no_new_trades_cutoff(self) -> bool:
        """True once it's too close to the bell to open a fresh intraday
        trade. The cutoff is configurable — default is 15:00 ET so a 15m
        trade has at least one full bar before we force-close at 15:55.
        """
        et = self._now_et()
        if et.weekday() >= 5:
            return True
        cutoff = settings.intraday_no_new_trades_hour * 60 + settings.intraday_no_new_trades_minute
        return (et.hour * 60 + et.minute) >= cutoff

    def _build_context_reasoning(
        self,
        signal: TradingViewSignal,
        risk_params: "RiskParams | None",
    ) -> list[str]:
        """Build the full context of WHY this signal came in — always the same
        shape for both executed and rejected trades, so the journal always
        shows the full picture.
        """
        side_heb = "קנייה" if signal.action == SignalAction.BUY else "מכירה"
        lines: list[str] = [
            f"איתות {side_heb} ב-{signal.ticker} במחיר ${signal.price}"
        ]

        ind = signal.indicators
        if ind.ema_trend == "up":
            lines.append("טרנד עולה - EMA 9 מעל EMA 21 מעל EMA 50")
        elif ind.ema_trend == "down":
            lines.append("טרנד יורד - EMA 9 מתחת ל-EMA 21 מתחת ל-EMA 50")
        elif ind.ema_trend:
            lines.append(f"טרנד {ind.ema_trend}")

        if ind.macd_signal:
            macd_map = {
                "bullish_cross": "MACD חצה למעלה - מומנטום שורי",
                "bearish_cross": "MACD חצה למטה - מומנטום דובי",
                "bullish": "MACD חיובי - מומנטום שורי",
                "bearish": "MACD שלילי - מומנטום דובי",
            }
            macd_text = macd_map.get(ind.macd_signal)
            if macd_text:
                lines.append(macd_text)

        if ind.rsi is not None:
            if ind.rsi < 35:
                lines.append(f"RSI = {ind.rsi} - אזור מכירת יתר (הזדמנות קנייה)")
            elif ind.rsi > 65:
                lines.append(f"RSI = {ind.rsi} - אזור קניית יתר (הזדמנות מכירה)")
            else:
                lines.append(f"RSI = {ind.rsi} - טווח נייטרלי")
        else:
            lines.append("RSI לא זמין")

        if ind.volume_ratio and ind.volume_ratio > 1.3:
            lines.append(f"ווליום גבוה - פי {ind.volume_ratio} מהממוצע (אישור חוזק)")
        elif ind.volume_ratio:
            lines.append(f"ווליום רגיל (פי {ind.volume_ratio} מהממוצע)")

        if ind.atr:
            lines.append(f"ATR = {ind.atr} - תנודתיות {'גבוהה' if ind.atr > 5 else 'סבירה'}")
        else:
            lines.append("ATR לא זמין - לא ניתן לחשב סטופ דינמי")

        if risk_params is not None:
            lines.append(
                f"חישוב פוזיציה: {risk_params.position_size} מניות | "
                f"סטופ לוס ${risk_params.stop_loss} | טייק פרופיט ${risk_params.take_profit} | "
                f"יחס סיכוי/סיכון {risk_params.reward_risk_ratio}"
            )
            lines.append(
                f"סיכון מחושב: {risk_params.risk_percent:.2%} מהתיק "
                f"(${risk_params.risk_amount:.2f})"
            )
        else:
            lines.append("חישוב סטופ/טייק/גודל פוזיציה נכשל")

        return lines

    async def process_signal(self, signal: TradingViewSignal) -> TradeDecision:
        """Full decision pipeline for an incoming signal."""
        logger.info(f"Processing signal: {signal.action} {signal.ticker} @ {signal.price}")

        # Auto-fill missing indicators from Alpaca market data
        if not signal.indicators.atr or signal.indicators.atr <= 0:
            atr = self.alpaca.get_atr(signal.ticker)
            if atr:
                signal.indicators.atr = atr
                logger.info(f"Auto-filled ATR for {signal.ticker}: {atr}")
        if signal.indicators.rsi is None:
            rsi = self.alpaca.get_rsi(signal.ticker)
            if rsi:
                signal.indicators.rsi = rsi
                logger.info(f"Auto-filled RSI for {signal.ticker}: {rsi}")

        # Fetch account state (balance used for risk calc + validation)
        account = self.alpaca.get_account()
        positions = self.alpaca.get_open_positions()
        balance = min(account["equity"], settings.max_capital)

        # Compute risk params upfront so both EXECUTE and REJECT entries
        # include the would-be SL/TP/position size in the reasoning
        risk_params = self.risk.calculate_risk_params(signal, balance)

        # Build the full context that every journal entry will carry
        reasoning_lines = self._build_context_reasoning(signal, risk_params)

        side = "buy" if signal.action == SignalAction.BUY else "sell"

        # Run ALL validation checks and collect reject reasons
        reject_reasons: list[str] = []

        # 0) Intraday pre-close blackout — no new entries too close to the bell
        if self._is_past_no_new_trades_cutoff():
            cutoff_str = (
                f"{settings.intraday_no_new_trades_hour:02d}:"
                f"{settings.intraday_no_new_trades_minute:02d}"
            )
            reject_reasons.append(
                f"חלון כניסה סגור — לא פותחים עסקאות חדשות אחרי {cutoff_str} ET"
            )

        # 1) Max open positions (snapshot + pending, no double count)
        total_positions = self._scan_start_positions + self._pending_count
        if total_positions >= settings.max_open_positions:
            reject_reasons.append(
                f"הגעת למקסימום פוזיציות ({total_positions}/{settings.max_open_positions})"
            )

        # 2) Duplicate ticker (existing position OR pending order from this scan)
        if signal.ticker in self._pending_tickers or signal.ticker in self._scan_start_tickers:
            reject_reasons.append(f"כבר יש פוזיציה או פקודה פתוחה ב-{signal.ticker}")

        # 2b) Same-day re-entry after a loss — don't keep hitting the same wall
        daily_block = self._check_daily_blacklist(signal.ticker)
        if daily_block:
            reject_reasons.append(daily_block)

        # 3) Total portfolio risk cap
        current_risk = self._scan_start_risk + self._pending_risk
        logger.info(
            f"Risk check: {current_risk:.2%} "
            f"(start: {self._scan_start_positions} positions, "
            f"pending: {self._pending_count})"
        )
        # Tiny epsilon so float drift at the exact cap boundary (e.g. 20 x 0.01
        # coming out to 0.19999999999999998 or 0.20000000000000004 instead of
        # an exact 0.2) doesn't falsely reject a trade that mathematically fits.
        if current_risk >= settings.max_total_risk + 1e-9:
            reject_reasons.append(
                f"סיכון כולל בתיק {current_risk:.2%} הגיע לתקרה של {settings.max_total_risk:.0%}"
            )

        # 4) Risk params must be calculable (needs ATR, non-zero position size)
        if risk_params is None:
            reject_reasons.append("לא ניתן לחשב פרמטרי סיכון (חסר ATR או גודל פוזיציה יוצא 0)")
        else:
            # 5) Full per-trade validation (RSI extremes, trend alignment, RR ratio, etc.)
            is_valid, validation_reasons = self.risk.validate_trade(
                signal=signal,
                account_balance=balance,
                open_positions=positions,
                current_total_risk=current_risk,
            )
            if not is_valid:
                reject_reasons.extend(validation_reasons)

        # REJECT path — log full context + all failing checks
        if reject_reasons:
            decision = TradeDecision(
                action=DecisionAction.REJECT,
                ticker=signal.ticker,
                side=side,
                risk_params=risk_params,
            )
            for line in reasoning_lines:
                decision.add_reason(line)
            decision.add_reason(f"החלטה: נדחה — {reject_reasons[0]}")
            for extra in reject_reasons[1:]:
                decision.add_reason(f"סיבה נוספת: {extra}")
            logger.warning(f"REJECTED {signal.ticker}: {reject_reasons}")
            await self._log_decision(signal, decision)
            return decision

        # EXECUTE path — all checks passed
        decision = TradeDecision(
            action=DecisionAction.EXECUTE,
            ticker=signal.ticker,
            side=side,
            risk_params=risk_params,
        )
        for line in reasoning_lines:
            decision.add_reason(line)

        try:
            order_result = self.alpaca.submit_bracket_order(
                symbol=signal.ticker,
                side=side,
                risk_params=risk_params,
            )
            order_status = (order_result.get("status") or "").lower()

            if order_status in _ALPACA_FAIL_STATUSES:
                # Alpaca said no — treat as REJECT and do NOT bump pending
                # counters, so the scanner's next iteration sees the correct
                # slot / risk availability.
                decision.action = DecisionAction.REJECT
                decision.add_reason(
                    f"החלטה: נדחה — Alpaca סירבה לפקודה מיידית (סטטוס: {order_status})"
                )
                logger.warning(
                    f"Alpaca rejected {signal.ticker} on submit: {order_result}"
                )
            else:
                decision.add_reason(
                    f"החלטה: בוצע — כל הבדיקות עברו | פקודה נשלחה: {order_status}"
                )
                logger.info(f"EXECUTED {signal.ticker}: {order_result}")
                # Update pending counters with ACTUAL risk
                self._pending_count += 1
                self._pending_risk += risk_params.risk_percent  # Actual risk, not max
                self._pending_tickers.add(signal.ticker)
                logger.info(
                    f"Pending: {self._pending_count} positions, "
                    f"{self._pending_risk:.2%} risk"
                )
        except Exception as e:
            decision.action = DecisionAction.REJECT
            decision.add_reason(f"החלטה: נדחה — שליחת הפקודה ל-Alpaca נכשלה: {e}")
            logger.error(f"Execution failed for {signal.ticker}: {e}")

        await self._log_decision(signal, decision)
        return decision

    async def _log_decision(
        self, signal: TradingViewSignal, decision: TradeDecision
    ) -> None:
        """Log the decision to the trade journal."""
        signal_data = {"ticker": signal.ticker, "action": signal.action.value, "price": signal.price}

        entry = TradeJournalEntry(
            ticker=signal.ticker,
            side=decision.side or signal.action.value.lower(),
            action_taken=decision.action.value,
            entry_price=decision.risk_params.entry_price if decision.risk_params else signal.price,
            stop_loss=decision.risk_params.stop_loss if decision.risk_params else None,
            take_profit=decision.risk_params.take_profit if decision.risk_params else None,
            position_size=decision.risk_params.position_size if decision.risk_params else None,
            risk_params=decision.risk_params,
            signal_data=signal_data,
            reasoning=decision.reasoning,
            status="OPEN" if decision.action == DecisionAction.EXECUTE else "REJECTED",
        )
        await self.journal.log_trade(entry)
