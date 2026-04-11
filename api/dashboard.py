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
    # Calculate live P&L and equity
    total_unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
    trading_capital = settings.max_capital
    live_equity = trading_capital + total_unrealized_pnl
    if account:
        account["trading_capital"] = trading_capital
        account["live_equity"] = round(live_equity, 2)
        account["buying_power_live"] = round(live_equity * 8, 2)
        account["unrealized_pnl"] = round(total_unrealized_pnl, 2)
        account["pnl_percent"] = round((total_unrealized_pnl / trading_capital) * 100, 2) if trading_capital > 0 else 0
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
):
    """Paginated trade journal entries.

    Query params:
        ticker: case-insensitive substring match (e.g. "TSL" matches "TSLA")
        decision: "EXECUTE" or "REJECT"
        since_hours: only rows from the last N hours
    """
    if not journal:
        return {"trades": [], "page": 1, "total_pages": 0, "total": 0}

    since = None
    if since_hours is not None and since_hours > 0:
        from datetime import timedelta
        from db.database import _utc_naive_now
        # Compare against the tz-naive UTC timestamps stored in trade_log
        since = _utc_naive_now() - timedelta(hours=since_hours)

    # Normalize decision to expected values
    decision_norm = decision.upper() if decision else None
    if decision_norm and decision_norm not in ("EXECUTE", "REJECT"):
        decision_norm = None

    result = await journal.get_paginated_trades(
        page=page,
        per_page=per_page,
        ticker=ticker,
        decision=decision_norm,
        since=since,
    )
    return result


@router.get("/stats")
async def get_stats():
    """Trading performance statistics."""
    if not journal:
        return {"stats": {}}
    stats = await journal.get_stats()
    return {"stats": stats}


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


@router.get("/debug/db")
async def debug_db():
    """Check database connection."""
    from db.database import engine, db_url
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
            # Check if table exists
            tables = await conn.execute(
                __import__("sqlalchemy").text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            )
            table_list = [r[0] for r in tables]
            return {"status": "connected", "db_type": "postgresql" if "postgresql" in db_url else "sqlite", "tables": table_list}
    except Exception as e:
        return {"status": "error", "error": str(e), "url_prefix": db_url[:30] + "..."}


# NOTE: /api/debug/write and DELETE /api/journal/clear were removed.
# They served one-off diagnostics during the initial PostgreSQL debugging
# on Railway and are no longer needed. Anyone with the URL could write
# junk rows or wipe the entire journal, so they had to go.
