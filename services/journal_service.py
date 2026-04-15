import json
from datetime import datetime, timezone
from loguru import logger
from sqlalchemy import select, delete, func, update, case
from db.database import async_session, TradeLog, init_db, _utc_naive_now
from models.journal import TradeJournalEntry
from models.orders import RiskParams

# Rolling journal cap: once we exceed this, the oldest rows are pruned.
MAX_JOURNAL_ROWS = 30_000


def _ibkr_pro_commission_leg(shares: int, price: float, is_sell: bool) -> float:
    """Estimated IBKR Pro Tiered commission for one side of a trade.

    Display-only — used by the dashboard "commissions" KPI so the user sees
    what a real IBKR account would have charged. Does not affect P&L,
    equity, or decision making anywhere in the system.

    Tiered plan (starting tier, before volume discounts):
      - Broker commission: $0.0035/share, NO minimum per order, max 1% of value
      - SEC fee (sells only): 0.00278% of principal
      - FINRA TAF (sells only): $0.000166/share, capped at $8.30
      - Exchange/clearing pass-through: rough ~$0.003/share (taker rate)

    Notes:
      - This is a conservative estimate that assumes all legs take liquidity.
        In practice take-profit (limit) exits earn small rebates, making real
        costs slightly lower.
    """
    if shares <= 0 or price <= 0:
        return 0.0
    value = shares * price
    broker = min(shares * 0.0035, value * 0.01)
    total = broker + (shares * 0.003)
    if is_sell:
        total += value * 0.0000278
        total += min(shares * 0.000166, 8.30)
    return total


class JournalService:
    """Persists every trade decision (executed or rejected) to the database."""

    async def _trim_to_max(self, session) -> None:
        """Delete the oldest rows so total stays <= MAX_JOURNAL_ROWS.

        Best-effort: any failure is logged but does not abort the write.
        """
        try:
            count_result = await session.execute(
                select(func.count()).select_from(TradeLog)
            )
            total = count_result.scalar() or 0
            if total <= MAX_JOURNAL_ROWS:
                return
            excess = total - MAX_JOURNAL_ROWS
            # Oldest rows by timestamp (ties broken by id)
            old_ids_q = await session.execute(
                select(TradeLog.id)
                .order_by(TradeLog.timestamp.asc(), TradeLog.id.asc())
                .limit(excess)
            )
            old_ids = [row[0] for row in old_ids_q.all()]
            if not old_ids:
                return
            await session.execute(
                delete(TradeLog).where(TradeLog.id.in_(old_ids))
            )
            await session.commit()
            logger.info(
                f"Journal trim: removed {len(old_ids)} oldest rows "
                f"(was {total}, now {total - len(old_ids)}, cap {MAX_JOURNAL_ROWS})"
            )
        except Exception as e:
            logger.warning(f"Journal trim failed (non-fatal): {type(e).__name__}: {e}")

    async def initialize(self):
        await init_db()
        logger.info("Trade journal database initialized")

    async def log_trade(self, entry: TradeJournalEntry) -> int:
        try:
            # Serialize signal_data safely
            try:
                signal_json = json.dumps(entry.signal_data, default=str)
            except Exception:
                signal_json = "{}"
            try:
                reasoning_json = json.dumps(entry.reasoning, default=str)
            except Exception:
                reasoning_json = "[]"

            # Make timestamp timezone-naive for PostgreSQL compatibility
            ts = entry.timestamp
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)

            async with async_session() as session:
                record = TradeLog(
                    ticker=entry.ticker,
                    side=entry.side,
                    action_taken=entry.action_taken,
                    entry_price=entry.entry_price,
                    stop_loss=entry.stop_loss,
                    take_profit=entry.take_profit,
                    position_size=entry.position_size,
                    risk_amount=entry.risk_params.risk_amount if entry.risk_params else None,
                    risk_percent=entry.risk_params.risk_percent if entry.risk_params else None,
                    signal_data=signal_json,
                    reasoning=reasoning_json,
                    timestamp=ts,
                    status=entry.status,
                )
                session.add(record)
                await session.commit()
                logger.info(f"Journal: {entry.action_taken} {entry.ticker} {entry.side}")
                new_id = record.id
                await self._trim_to_max(session)
                return new_id
        except Exception as e:
            logger.error(f"Journal write FAILED for {entry.ticker}: {type(e).__name__}: {e}")
            # Try minimal write as fallback
            try:
                async with async_session() as session:
                    record = TradeLog(
                        ticker=entry.ticker,
                        side=str(entry.side),
                        action_taken=str(entry.action_taken),
                        entry_price=float(entry.entry_price) if entry.entry_price else None,
                        stop_loss=float(entry.stop_loss) if entry.stop_loss else None,
                        take_profit=float(entry.take_profit) if entry.take_profit else None,
                        position_size=int(entry.position_size) if entry.position_size else None,
                        risk_amount=float(entry.risk_params.risk_amount) if entry.risk_params else None,
                        risk_percent=float(entry.risk_params.risk_percent) if entry.risk_params else None,
                        signal_data="{}",
                        reasoning=json.dumps([str(r) for r in entry.reasoning], default=str),
                        timestamp=_utc_naive_now(),
                        status=str(entry.status),
                    )
                    session.add(record)
                    await session.commit()
                    logger.info(f"Journal: FALLBACK write succeeded for {entry.ticker}")
                    new_id = record.id
                    await self._trim_to_max(session)
                    return new_id
            except Exception as e2:
                logger.error(f"Journal FALLBACK also failed for {entry.ticker}: {e2}")
            return -1

    def _row_to_dict(self, r) -> dict:
        try:
            reasoning = json.loads(r.reasoning) if r.reasoning else []
        except Exception:
            reasoning = []
        return {
            "id": r.id,
            "ticker": r.ticker,
            "side": r.side,
            "action_taken": r.action_taken,
            "entry_price": r.entry_price,
            "stop_loss": r.stop_loss,
            "take_profit": r.take_profit,
            "position_size": r.position_size,
            "risk_amount": r.risk_amount,
            "risk_percent": r.risk_percent,
            "reasoning": reasoning,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "pnl": r.pnl,
            "status": r.status,
        }

    async def get_recent_trades(self, limit: int = 20) -> list[dict]:
        async with async_session() as session:
            result = await session.execute(
                select(TradeLog).order_by(TradeLog.timestamp.desc()).limit(limit)
            )
            return [self._row_to_dict(r) for r in result.scalars().all()]

    async def get_open_trades(self) -> list:
        """Return ORM rows where status='OPEN' AND action_taken='EXECUTE'.

        Used by the reconciliation service to find trades whose on-Alpaca
        position has disappeared and need to be marked closed.
        """
        async with async_session() as session:
            result = await session.execute(
                select(TradeLog).where(
                    TradeLog.status == "OPEN",
                    TradeLog.action_taken == "EXECUTE",
                )
            )
            return list(result.scalars().all())

    async def mark_trade_closed(
        self,
        trade_id: int,
        exit_price: float,
        exit_timestamp: datetime,
        pnl: float,
    ) -> bool:
        """Idempotent close: conditional UPDATE guarantees only the first
        caller wins. Returns True if the row transitioned OPEN -> CLOSED,
        False if it was already closed or missing.

        Makes exit_timestamp tz-naive to match the log_trade pattern for
        PostgreSQL compatibility.
        """
        try:
            async with async_session() as session:
                if exit_timestamp.tzinfo is not None:
                    exit_timestamp = exit_timestamp.replace(tzinfo=None)
                result = await session.execute(
                    update(TradeLog)
                    .where(TradeLog.id == trade_id, TradeLog.status == "OPEN")
                    .values(
                        exit_price=exit_price,
                        exit_timestamp=exit_timestamp,
                        pnl=pnl,
                        status="CLOSED",
                    )
                )
                await session.commit()
                return result.rowcount == 1
        except Exception as e:
            logger.error(
                f"mark_trade_closed({trade_id}) failed: {type(e).__name__}: {e}"
            )
            return False

    async def get_risk_for_tickers(self, tickers: list[str]) -> dict[str, float]:
        """Return the most recent EXECUTE risk_percent for each given ticker.

        Queries the DB directly by ticker so it is NOT affected by how much
        REJECT noise has been written to the journal in between. Tickers with
        no EXECUTE record (or a null risk_percent) are simply missing from the
        returned dict — caller decides the fallback.
        """
        if not tickers:
            return {}
        async with async_session() as session:
            result = await session.execute(
                select(
                    TradeLog.ticker,
                    TradeLog.risk_percent,
                    TradeLog.timestamp,
                )
                .where(
                    TradeLog.ticker.in_(tickers),
                    TradeLog.action_taken == "EXECUTE",
                    TradeLog.risk_percent.is_not(None),
                )
                .order_by(TradeLog.timestamp.desc())
            )
            risks: dict[str, float] = {}
            for ticker, risk_pct, _ts in result.all():
                # First row per ticker wins (ordered desc by timestamp)
                risks.setdefault(ticker, float(risk_pct))
            return risks

    async def get_paginated_trades(
        self,
        page: int = 1,
        per_page: int = 20,
        ticker: str | None = None,
        decision: str | None = None,
        since: datetime | None = None,
        status: str | None = None,
    ) -> dict:
        """Get paginated trades with optional filters.

        Args:
            page: 1-indexed page number
            per_page: rows per page
            ticker: case-insensitive substring match on ticker (e.g. "TSL")
            decision: exact action_taken match ("EXECUTE" or "REJECT")
            since: only rows with timestamp >= this datetime
            status: exact status match ("OPEN", "CLOSED", "REJECTED")
        """
        async with async_session() as session:
            filters = []
            if ticker:
                filters.append(TradeLog.ticker.ilike(f"%{ticker.strip()}%"))
            if decision:
                filters.append(TradeLog.action_taken == decision)
            if since is not None:
                filters.append(TradeLog.timestamp >= since)
            if status:
                filters.append(TradeLog.status == status)

            count_stmt = select(func.count()).select_from(TradeLog)
            if filters:
                count_stmt = count_stmt.where(*filters)
            count_result = await session.execute(count_stmt)
            total = count_result.scalar() or 0
            total_pages = max(1, (total + per_page - 1) // per_page)
            page = max(1, min(page, total_pages))

            offset = (page - 1) * per_page
            stmt = select(TradeLog).order_by(TradeLog.timestamp.desc())
            if filters:
                stmt = stmt.where(*filters)
            stmt = stmt.offset(offset).limit(per_page)
            result = await session.execute(stmt)
            trades = [self._row_to_dict(r) for r in result.scalars().all()]

            return {
                "trades": trades,
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
            }

    async def get_stats(self) -> dict:
        """Return dashboard stats using a single SQL aggregation query.

        Previously this loaded every TradeLog row into Python and filtered
        in-memory, which became a problem as the journal grew toward its
        30k cap (the dashboard polls this every 30s). Now the DB does all
        the counting and summing in one shot.
        """
        async with async_session() as session:
            # ``closed`` means an EXECUTE row whose pnl has been written.
            # ``wins`` / ``losses`` are subsets of ``closed``.
            closed_cond = (TradeLog.action_taken == "EXECUTE") & TradeLog.pnl.is_not(None)
            result = await session.execute(
                select(
                    func.count().label("total_signals"),
                    func.sum(
                        case((TradeLog.action_taken == "EXECUTE", 1), else_=0)
                    ).label("executed"),
                    func.sum(
                        case((TradeLog.action_taken == "REJECT", 1), else_=0)
                    ).label("rejected"),
                    func.sum(
                        case((closed_cond, 1), else_=0)
                    ).label("closed"),
                    func.sum(
                        case((closed_cond & (TradeLog.pnl > 0), 1), else_=0)
                    ).label("wins"),
                    func.sum(
                        case((closed_cond & (TradeLog.pnl <= 0), 1), else_=0)
                    ).label("losses"),
                    func.sum(
                        case((closed_cond, TradeLog.pnl), else_=0.0)
                    ).label("total_pnl"),
                )
            )
            row = result.one()
            total_signals = int(row.total_signals or 0)
            executed = int(row.executed or 0)
            rejected = int(row.rejected or 0)
            closed = int(row.closed or 0)
            wins = int(row.wins or 0)
            losses = int(row.losses or 0)
            total_pnl = float(row.total_pnl or 0.0)
            return {
                "total_signals": total_signals,
                "executed": executed,
                "rejected": rejected,
                "closed": closed,
                "wins": wins,
                "losses": losses,
                "win_rate": (wins / closed) if closed else 0,
                "total_pnl": round(total_pnl, 2),
            }

    async def get_total_commissions(self) -> float:
        """Sum estimated IBKR Pro commissions over every EXECUTE trade.

        Each executed trade contributes the entry leg always; the exit leg
        is only added if the trade has been closed (has exit_price). For
        longs, the sell leg is the exit; for shorts, the sell leg is the
        entry. Regulatory fees (SEC, FINRA TAF) only hit the sell leg.
        """
        async with async_session() as session:
            result = await session.execute(
                select(
                    TradeLog.side,
                    TradeLog.position_size,
                    TradeLog.entry_price,
                    TradeLog.exit_price,
                    TradeLog.status,
                ).where(TradeLog.action_taken == "EXECUTE")
            )
            total = 0.0
            for side, shares, entry, exit_price, status in result.all():
                shares = int(shares or 0)
                entry = float(entry or 0.0)
                if shares <= 0 or entry <= 0:
                    continue
                is_short = side in ("sell", "short")
                # Entry leg: short entry = sell (regulatory applies)
                total += _ibkr_pro_commission_leg(shares, entry, is_sell=is_short)
                # Exit leg: only if closed
                if exit_price and status == "CLOSED":
                    ep = float(exit_price)
                    if ep > 0:
                        # Long exit = sell; short exit = buy
                        total += _ibkr_pro_commission_leg(shares, ep, is_sell=not is_short)
            return round(total, 2)

    async def get_realized_pnl_today(self) -> float:
        """Sum of PnL from trades closed since midnight ET.

        Used by the intraday dashboard's "daily P&L" KPI. DB timestamps are
        stored UTC-naive, so we compute the current ET date's midnight and
        convert it to UTC-naive before filtering.
        """
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        utc = ZoneInfo("UTC")
        now_et = datetime.now(et)
        start_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc_naive = start_et.astimezone(utc).replace(tzinfo=None)

        async with async_session() as session:
            result = await session.execute(
                select(func.sum(TradeLog.pnl)).where(
                    TradeLog.action_taken == "EXECUTE",
                    TradeLog.pnl.is_not(None),
                    TradeLog.exit_timestamp >= start_utc_naive,
                )
            )
            total = result.scalar()
            return float(total or 0.0)
