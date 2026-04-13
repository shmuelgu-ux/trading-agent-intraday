from fastapi import APIRouter
from services.alpaca_client import AlpacaClient
from services.journal_service import JournalService
from config import settings

router = APIRouter(prefix="/api", tags=["dashboard"])

# Will be injected from main.py
alpaca: AlpacaClient | None = None
journal: JournalService | None = None
scanner = None
scanner_state = {
    "started": False,
    "market_open_seen": False,
    "last_loop_time": None,
    "last_action": "not started",
    "error": None,
}


def set_services(alpaca_client: AlpacaClient, journal_service: JournalService, scanner_obj=None):
    global alpaca, journal, scanner
    alpaca = alpaca_client
    journal = journal_service
    if scanner_obj:
        scanner = scanner_obj


@router.get("/status")
async def system_status():
    """Overall system status."""
    account = alpaca.get_account() if alpaca else {}
    positions = alpaca.get_open_positions() if alpaca else []
    # Calculate live P&L and equity.
    # Must include BOTH unrealized (open positions) AND realized (closed
    # trades) to show accurate equity. Without the realized component a
    # closed loss like INTC -$13.77 would vanish from the display.
    total_unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
    realized_pnl = 0.0
    if journal:
        try:
            stats = await journal.get_stats()
            realized_pnl = stats.get("total_pnl", 0.0)
        except Exception:
            realized_pnl = 0.0
    trading_capital = settings.max_capital
    total_pnl = realized_pnl + total_unrealized_pnl
    live_equity = trading_capital + total_pnl
    if account:
        account["trading_capital"] = trading_capital
        account["live_equity"] = round(live_equity, 2)
        account["buying_power_live"] = round(live_equity * 8, 2)
        account["unrealized_pnl"] = round(total_unrealized_pnl, 2)
        account["realized_pnl"] = round(realized_pnl, 2)
        account["pnl_percent"] = round((total_pnl / trading_capital) * 100, 2) if trading_capital > 0 else 0
    return {
        "status": "running",
        "alpaca_connected": alpaca.is_connected if alpaca else False,
        "account": account,
        "open_positions_count": len(positions),
        "risk_config": {
            "max_risk_per_trade": settings.max_risk_per_trade,
            "max_total_risk": settings.max_total_risk,
            "max_open_positions": settings.max_open_positions,
            "default_rr_ratio": settings.default_rr_ratio,
        },
    }


@router.get("/positions")
async def get_positions():
    """Current open positions."""
    if not alpaca:
        return {"positions": []}
    return {"positions": alpaca.get_open_positions()}


@router.get("/journal")
async def get_journal(
    page: int = 1,
    per_page: int = 20,
    ticker: str | None = None,
    decision: str | None = None,
    since_hours: int | None = None,
    status: str | None = None,
):
    """Paginated trade journal entries.

    Query params:
        ticker: case-insensitive substring match (e.g. "TSL" matches "TSLA")
        decision: "EXECUTE" or "REJECT"
        since_hours: only rows from the last N hours
        status: "OPEN", "CLOSED", or "REJECTED"
    """
    if not journal:
        return {"trades": [], "page": 1, "total_pages": 0, "total": 0}

    since = None
    if since_hours is not None and since_hours > 0:
        from datetime import timedelta
        from db.database import _utc_naive_now
        since = _utc_naive_now() - timedelta(hours=since_hours)

    # Normalize decision to expected values
    decision_norm = decision.upper() if decision else None
    if decision_norm and decision_norm not in ("EXECUTE", "REJECT"):
        decision_norm = None

    # Normalize status
    status_norm = status.upper() if status else None
    if status_norm and status_norm not in ("OPEN", "CLOSED", "REJECTED"):
        status_norm = None

    result = await journal.get_paginated_trades(
        page=page,
        per_page=per_page,
        ticker=ticker,
        decision=decision_norm,
        since=since,
        status=status_norm,
    )
    return result


@router.get("/stats")
async def get_stats():
    """Trading performance statistics."""
    if not journal:
        return {"stats": {}}
    stats = await journal.get_stats()
    return {"stats": stats}


@router.get("/learning")
async def get_learning_report():
    """Latest learning report from the self-improvement cycle."""
    from services.learning_service import LearningService
    svc = LearningService()
    report = await svc.get_latest_report()
    if not report:
        return {"report": None, "message": "עדיין אין מספיק עסקאות סגורות (צריך לפחות 10)"}
    return {"report": report}


@router.get("/chart/{symbol}")
async def get_chart_data(symbol: str, interval: str = "1d", range_days: int = 365):
    """Return OHLCV bar data for the lightweight-charts widget.

    Args:
        symbol: ticker symbol (e.g. AAPL)
        interval: bar size — "5m", "15m", "1h", "1d", "1wk"
        range_days: how far back to fetch (max 1825 = 5 years for daily)
    """
    if not alpaca or not alpaca._data_client:
        return {"bars": []}

    import asyncio
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    from datetime import datetime, timedelta

    # Map interval string to TimeFrame
    tf_map = {
        "5m":  TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h":  TimeFrame(1, TimeFrameUnit.Hour),
        "1d":  TimeFrame.Day,
        "1wk": TimeFrame.Week,
    }
    timeframe = tf_map.get(interval, TimeFrame.Day)

    # Cap range to 5 years for daily, 60 days for intraday
    if interval in ("5m", "15m", "30m"):
        range_days = min(range_days, 60)
    elif interval == "1h":
        range_days = min(range_days, 730)
    else:
        range_days = min(range_days, 1825)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=timeframe,
            start=datetime.now() - timedelta(days=range_days),
            end=datetime.now(),
            limit=5000,
        )
        request.feed = DataFeed.IEX
        bars_data = await asyncio.to_thread(
            alpaca._data_client.get_stock_bars, request
        )
        raw_bars = bars_data.data.get(symbol.upper(), [])
        bars = [
            {
                "time": int(b.timestamp.timestamp()),
                "open": round(float(b.open), 2),
                "high": round(float(b.high), 2),
                "low": round(float(b.low), 2),
                "close": round(float(b.close), 2),
            }
            for b in raw_bars
        ]
        return {"bars": bars, "symbol": symbol.upper(), "interval": interval}
    except Exception as e:
        return {"bars": [], "error": str(e)}


@router.get("/risk")
async def get_risk_status():
    """Current risk exposure."""
    if not alpaca:
        return {"account_equity": 0, "total_risk_percent": 0, "max_total_risk": settings.max_total_risk, "open_positions": 0, "max_positions": settings.max_open_positions, "positions": [], "can_take_new_trade": True}

    try:
        account = alpaca.get_account()
        positions = alpaca.get_open_positions()
    except Exception:
        return {"account_equity": 0, "total_risk_percent": 0, "max_total_risk": settings.max_total_risk, "open_positions": 0, "max_positions": settings.max_open_positions, "positions": [], "can_take_new_trade": True}

    balance = min(account.get("equity", 0), settings.max_capital)

    # Get actual risk_percent per position via direct ticker lookup so
    # the journal's REJECT noise cannot hide older EXECUTE rows.
    trade_risks: dict[str, float] = {}
    if journal and positions:
        try:
            trade_risks = await journal.get_risk_for_tickers(
                [p["symbol"] for p in positions]
            )
        except Exception:
            trade_risks = {}

    total_theoretical_risk = 0.0
    position_risks = []
    for p in positions:
        pnl = p["unrealized_pnl"]
        entry = p.get("entry_price", 0)
        current = p.get("current_price", 0)
        qty = p.get("qty", 0)
        # Get actual risk from journal, fallback to max
        actual_risk = trade_risks.get(p["symbol"], settings.max_risk_per_trade)
        total_theoretical_risk += actual_risk
        position_risks.append({
            "symbol": p["symbol"],
            "side": p["side"],
            "qty": qty,
            "entry_price": entry,
            "current_price": current,
            "unrealized_pnl": round(pnl, 2),
            "risk_percent": round(actual_risk, 4),
        })

    return {
        "account_equity": balance,
        "total_risk_percent": round(min(total_theoretical_risk, 1.0), 4),
        "max_total_risk": settings.max_total_risk,
        "open_positions": len(positions),
        "max_positions": settings.max_open_positions,
        "positions": position_risks,
        "can_take_new_trade": (
            total_theoretical_risk < settings.max_total_risk
            and len(positions) < settings.max_open_positions
        ),
    }


@router.get("/scanner")
async def scanner_status():
    """Clean scanner status for the dashboard header.

    Returns what the UI needs to tell the user what the scanner is
    doing right now or when it last ran — nothing more.
    """
    if not scanner:
        return {
            "is_scanning": False,
            "progress": 0,
            "total": 0,
            "last_scan_end": None,
            "last_symbols_loaded": 0,
            "last_signals_found": 0,
            "last_scan_duration": 0,
            "scan_count": 0,
            "market_open": scanner_state.get("market_open", False),
            "last_action": scanner_state.get("last_action", "not started"),
        }
    st = scanner.state
    return {
        "is_scanning": bool(st.get("is_scanning")),
        "progress": st.get("progress", 0),
        "total": st.get("last_symbols_loaded", 0),
        "last_scan_end": st.get("last_scan_end"),
        "last_symbols_loaded": st.get("last_symbols_loaded", 0),
        "last_signals_found": st.get("last_signals_found", 0),
        "last_scan_duration": st.get("last_scan_duration", 0),
        "scan_count": st.get("scan_count", 0),
        "market_open": scanner_state.get("market_open", False),
        "last_action": scanner_state.get("last_action", ""),
    }


@router.get("/debug/scanner")
async def debug_scanner():
    """Raw scanner state for debugging."""
    return {
        "scanner_loop": scanner_state,
        "scanner_internal": scanner.state if scanner else "not set",
        "symbols_cached": len(scanner._all_symbols) if scanner else 0,
    }


# NOTE: /api/debug/db, /api/debug/write and DELETE /api/journal/clear
# were removed. They were one-off diagnostics during the initial
# PostgreSQL setup and are no longer needed. Anyone with the URL could
# hit them anonymously to write junk rows or wipe the entire journal,
# so they had to go.
