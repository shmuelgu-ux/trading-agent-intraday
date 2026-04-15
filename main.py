import os
import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from loguru import logger

from config import settings
from core.risk_manager import RiskManager
from core.decision_engine import DecisionEngine
from core.scanner import StockScanner
from services.alpaca_client import AlpacaClient
from services.journal_service import JournalService
from services.reconciliation_service import ReconciliationService
from services.learning_service import LearningService
from services.system_state_service import SystemStateService
from services.email_service import EmailService
from api import dashboard

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
if os.path.exists("logs"):
    logger.add("logs/trading_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup."""
    logger.info("=" * 50)
    logger.info("TRADING AGENT — INTRADAY — STARTING")
    logger.info("=" * 50)

    # Initialize services
    alpaca = AlpacaClient()
    journal = JournalService()
    await journal.initialize()

    risk_manager = RiskManager()
    # Phase 2 (news/sentiment/fundamentals) disabled - technical analysis only
    engine = DecisionEngine(risk_manager, alpaca, journal)

    # Reconciles closed trades back into the journal (keeps the stats panel accurate)
    reconciler = ReconciliationService(
        alpaca, journal,
        on_loss_callback=engine.mark_ticker_lost,
    )

    # Learning service — analyzes every 10 closed trades
    learner = LearningService()

    # Safety rails: manual kill switch + daily circuit breaker
    system_state = SystemStateService()
    email_svc = EmailService()

    # Create scanner early so we can pass to dashboard
    scanner = StockScanner(alpaca)

    # Inject into routers
    dashboard.set_services(alpaca, journal, scanner, system_state, email_svc)

    # Startup catch-up: any positions that closed while the server was
    # down won't have been reconciled yet. Do it once before anything else.
    try:
        n = await reconciler.reconcile_closed_trades()
        if n:
            logger.info(f"Startup reconcile: closed {n} stale trades")
    except Exception as e:
        logger.warning(f"Startup reconcile failed (non-fatal): {e}")

    # Startup learning check: if 10+ closed trades accumulated since the
    # last report (or ever), generate one now without waiting for the
    # next position to close.
    try:
        report = await learner.check_and_learn()
        if report:
            logger.info(f"Startup learning: generated report #{report['id']}")
    except Exception as e:
        logger.warning(f"Startup learning failed (non-fatal): {e}")

    try:
        account = alpaca.get_account()
        logger.info(f"Account equity: ${account['equity']:,.2f}")
        logger.info(f"Alpaca connected: {alpaca.is_connected}")
    except Exception as e:
        logger.warning(f"Alpaca connection failed: {e}")
        logger.warning("Running in DRY-RUN mode (no real trades)")
        alpaca._client = None  # Force dry-run mode

    logger.info(f"Trading capital: ${settings.max_capital:,.2f}")
    logger.info(f"Risk per trade: {settings.max_risk_per_trade:.1%}")
    logger.info(f"Max total risk: {settings.max_total_risk:.1%}")
    logger.info(f"Max positions: {settings.max_open_positions}")

    # Start event-driven scanner in background
    scanner_task = None
    if settings.scanner_enabled and alpaca.is_connected:
        async def fill_available_slots():
            """Scan and fill all available position slots until full.

            Uses snapshot-based counting to avoid double-counting when orders
            transition from pending to filled during the scan.
            """
            # Reset pending counters and take snapshot of starting state
            await engine.reset_pending()
            start_positions = engine._scan_start_positions
            start_risk = engine._scan_start_risk

            slots_available = settings.max_open_positions - start_positions
            risk_available = settings.max_total_risk - start_risk

            if slots_available <= 0 or risk_available <= 0:
                logger.info(
                    f"No slots available (positions: {start_positions}/{settings.max_open_positions}, "
                    f"risk: {start_risk:.1%}/{settings.max_total_risk:.1%})"
                )
                return 0

            logger.info(
                f"Slots available: {slots_available}, risk budget: {risk_available:.1%}. "
                f"Starting full market scan..."
            )

            # Scan ALL tradeable stocks
            signals = await asyncio.to_thread(scanner.scan, True)

            if not signals:
                logger.info("Scanner found no signals")
                return 0

            logger.info(f"Processing {len(signals)} signals...")
            executed = 0
            for signal in signals:
                # Check limits using SNAPSHOT + pending (no double counting)
                total_positions = start_positions + engine._pending_count
                if total_positions >= settings.max_open_positions:
                    logger.info(f"Max positions ({total_positions}) reached, stopping")
                    break

                total_risk = start_risk + engine._pending_risk
                if total_risk >= settings.max_total_risk:
                    logger.info(f"Max risk ({total_risk:.2%}) reached, stopping")
                    break

                decision = await engine.process_signal(signal)
                if decision.action.value == "EXECUTE":
                    executed += 1
                await asyncio.sleep(1.5)  # Spacing between orders

            logger.info(f"Filled {executed} slots this round")
            return executed

        def _is_past_force_close():
            """True once the force-close wall time has passed. All positions
            MUST be flat at this point — day trading invariant."""
            et = engine._now_et()
            if et.weekday() >= 5:
                return False
            cutoff = (
                settings.intraday_force_close_hour * 60
                + settings.intraday_force_close_minute
            )
            return (et.hour * 60 + et.minute) >= cutoff

        async def force_close_all_positions():
            """End-of-day safety net: close every open position at market.
            Runs once per session when the force-close time passes.
            """
            logger.info("=" * 40)
            logger.info("END-OF-DAY FORCE CLOSE")
            logger.info("=" * 40)
            try:
                result = await asyncio.to_thread(alpaca.close_all_positions)
                logger.info(f"close_all_positions result: {result}")
            except Exception as e:
                logger.error(f"Force-close failed: {type(e).__name__}: {e}")
            # Reconcile shortly after so the journal rows flip to CLOSED
            await asyncio.sleep(5)
            try:
                reconciled = await reconciler.reconcile_closed_trades()
                if reconciled:
                    logger.info(f"Post-close reconcile: {reconciled} trades")
            except Exception as e:
                logger.error(f"Post-close reconcile failed: {e}")

        async def run_scanner():
            """Intraday loop — scans on a fixed interval during market hours,
            force-closes everything before the bell, and goes idle overnight.
            """
            from datetime import datetime as dt
            dashboard.scanner_state["started"] = True
            dashboard.scanner_state["last_action"] = "waiting 10s"
            logger.info("Scanner task started, waiting 10s...")
            await asyncio.sleep(10)  # Let server start

            last_position_count = -1
            last_market_check = False
            force_close_fired = False  # one-shot per day
            last_alert_sent: dict[str, dt] = {}  # error-type -> last alert time

            interval = max(60, int(settings.scanner_interval_seconds))

            while True:
                try:
                    dashboard.scanner_state["last_loop_time"] = dt.now().isoformat()
                    market_open = engine._is_market_open()
                    dashboard.scanner_state["market_open"] = market_open

                    # Outside market hours: reset the one-shot flag so
                    # tomorrow's force-close fires again, and idle.
                    if not market_open:
                        if last_market_check:
                            logger.info("Market closed. Scanner idle.")
                        last_market_check = False
                        force_close_fired = False
                        last_position_count = -1
                        dashboard.scanner_state["last_action"] = "market closed - idle"
                        await asyncio.sleep(60)
                        continue

                    # First tick after market open
                    if not last_market_check:
                        logger.info("=" * 40)
                        logger.info("MARKET OPEN: intraday scanner active")
                        logger.info("=" * 40)
                        last_market_check = True
                        dashboard.scanner_state["market_open_seen"] = True
                        # Capture today's equity baseline for the circuit breaker.
                        # ensure_daily_snapshot also clears yesterday's breaker.
                        try:
                            acc = alpaca.get_account()
                            await system_state.ensure_daily_snapshot(acc.get("equity", 0.0) or 0.0)
                        except Exception as e:
                            logger.warning(f"Daily snapshot failed (non-fatal): {e}")

                    # End-of-day force close — retry until positions are
                    # actually flat. The flag is only set AFTER Alpaca
                    # confirms zero open positions, so a failed first
                    # close attempt (halted stock, transient 5xx, etc.)
                    # will be retried on the next loop iteration with a
                    # short sleep instead of rolling overnight.
                    if not force_close_fired and _is_past_force_close():
                        dashboard.scanner_state["last_action"] = "end of day - closing all"
                        await force_close_all_positions()
                        # Give Alpaca a few seconds to actually fill the orders
                        await asyncio.sleep(5)
                        try:
                            remaining = len(alpaca.get_open_positions())
                        except Exception as e:
                            logger.error(f"Force-close verify failed: {e}")
                            remaining = -1  # unknown — stay in retry mode
                        if remaining == 0:
                            force_close_fired = True
                            # Reset last_position_count so the next iteration's
                            # "positions dropped" reconcile branch doesn't fire
                            # redundantly (force_close_all_positions already
                            # reconciled).
                            last_position_count = 0
                            logger.info("Force close complete — all positions flat")
                            dashboard.scanner_state["last_action"] = "end of day - all closed"
                            await asyncio.sleep(interval)
                        else:
                            logger.warning(
                                f"Force close incomplete: {remaining} positions "
                                f"still open. Retrying in 20s."
                            )
                            dashboard.scanner_state["last_action"] = (
                                f"force close retrying ({remaining} open)"
                            )
                            await asyncio.sleep(20)
                        continue

                    # Current positions (used for both reconcile trigger and slot math)
                    try:
                        current_positions = len(alpaca.get_open_positions())
                    except Exception as e:
                        logger.error(f"Failed to get positions: {e}")
                        dashboard.scanner_state["error"] = f"get_positions: {e}"
                        await asyncio.sleep(30)
                        continue

                    # If a position dropped (SL/TP hit), reconcile the journal row first
                    if last_position_count != -1 and current_positions < last_position_count:
                        closed = last_position_count - current_positions
                        dashboard.scanner_state["last_action"] = f"{closed} positions closed - reconciling"
                        try:
                            reconciled = await reconciler.reconcile_closed_trades()
                            if reconciled:
                                logger.info(f"Reconciled {reconciled} closed trade(s)")
                                # Check if we hit 10 closed trades for a learning cycle
                                try:
                                    report = await learner.check_and_learn()
                                    if report:
                                        logger.info(f"Learning cycle completed: report #{report['id']}")
                                except Exception as le:
                                    logger.debug(f"Learning check failed (non-fatal): {le}")
                        except Exception as e:
                            logger.error(f"Reconcile failed: {e}")

                    # Block new scans once we're past the no-new-trades cutoff
                    if engine._is_past_no_new_trades_cutoff():
                        last_position_count = current_positions
                        dashboard.scanner_state["last_action"] = "no-new-trades window"
                        await asyncio.sleep(interval)
                        continue

                    # Safety rail check: stop here if kill switch is set or the
                    # circuit breaker has fired for today. Open positions keep
                    # their own SL/TP — we only block NEW entries.
                    enabled, reason = await system_state.is_trading_enabled()
                    if not enabled:
                        dashboard.scanner_state["last_action"] = f"safety stop: {reason}"
                        last_position_count = current_positions
                        await asyncio.sleep(interval)
                        continue

                    # Evaluate the daily loss threshold on the latest equity.
                    try:
                        acc = alpaca.get_account()
                        current_equity = acc.get("equity", 0.0) or 0.0
                        fired = await system_state.check_circuit_breaker(current_equity)
                        if fired:
                            if email_svc:
                                try:
                                    snapshot = (await system_state.get_status()).get("daily_equity_snapshot") or 0
                                    await email_svc.send_alert(
                                        "Circuit breaker fired — 10% daily loss",
                                        f"The daily loss threshold has been crossed.\n\n"
                                        f"Snapshot equity (market open): ${snapshot:,.2f}\n"
                                        f"Current equity: ${current_equity:,.2f}\n\n"
                                        f"Scanner will not open new trades for the rest of today. "
                                        f"It will resume automatically at the next market open."
                                    )
                                except Exception:
                                    pass
                            dashboard.scanner_state["last_action"] = "circuit breaker fired (-10%)"
                            last_position_count = current_positions
                            await asyncio.sleep(interval)
                            continue
                    except Exception as e:
                        logger.error(f"Circuit breaker check failed (continuing): {e}")

                    # Time-based scan: every tick, try to fill open slots
                    slots_available = settings.max_open_positions - current_positions
                    if slots_available > 0:
                        dashboard.scanner_state["last_action"] = "scanning..."
                        try:
                            filled = await fill_available_slots()
                            dashboard.scanner_state["last_action"] = f"filled {filled} slots"
                        except Exception as e:
                            error_msg = f"fill_slots: {type(e).__name__}: {e}"
                            logger.error(error_msg)
                            import traceback
                            logger.error(traceback.format_exc())
                            dashboard.scanner_state["error"] = error_msg
                            dashboard.scanner_state["last_action"] = "error during scan"
                        try:
                            last_position_count = len(alpaca.get_open_positions())
                        except Exception:
                            last_position_count = current_positions
                    else:
                        last_position_count = current_positions
                        dashboard.scanner_state["last_action"] = f"all slots full ({current_positions})"

                    await asyncio.sleep(interval)

                except Exception as e:
                    error_msg = f"loop: {type(e).__name__}: {e}"
                    logger.error(f"Scanner loop error: {error_msg}")
                    import traceback
                    tb_text = traceback.format_exc()
                    logger.error(tb_text)
                    dashboard.scanner_state["error"] = error_msg
                    # Rate-limit to one email per scanner crash type per hour
                    # so a tight crash loop doesn't spam the inbox.
                    now = dt.now()
                    last = last_alert_sent.get(type(e).__name__)
                    if email_svc and (not last or (now - last).total_seconds() > 3600):
                        last_alert_sent[type(e).__name__] = now
                        try:
                            await email_svc.send_alert(
                                f"Scanner error: {type(e).__name__}",
                                f"Scanner hit an unhandled exception:\n\n{error_msg}\n\n{tb_text[:3000]}",
                            )
                        except Exception:
                            pass
                    await asyncio.sleep(60)

        scanner_task = asyncio.create_task(run_scanner())
        logger.info(
            f"Intraday scanner started: interval={settings.scanner_interval_seconds}s, "
            f"force-close={settings.intraday_force_close_hour:02d}:"
            f"{settings.intraday_force_close_minute:02d} ET"
        )

    logger.info("=" * 50)

    yield

    if scanner_task:
        scanner_task.cancel()
    logger.info("Trading Agent shutting down")


app = FastAPI(
    title="Trading Agent — Intraday",
    description="Autonomous intraday day-trading decision engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(dashboard.router)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    return html_path.read_text(encoding="utf-8")
