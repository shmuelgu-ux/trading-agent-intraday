"""System state management — kill switch + daily circuit breaker.

Two independent safety mechanisms live here:

1. **Kill switch**: a manual user toggle from the dashboard. Flipping it ON
   stops the scanner from opening new trades AND closes every open position
   at market. Stays ON until the user explicitly flips it OFF. Persists
   across restarts via the ``system_state`` table.

2. **Circuit breaker**: automatic daily loss cap. Snapshots account equity
   the first time the market is open each day; if equity later drops 10%
   below that snapshot, the scanner stops for the rest of the day. Open
   positions keep their existing stop/take-profit legs — the breaker only
   prevents NEW entries. The breaker resets automatically the next trading
   day when a fresh snapshot is taken.

Trading is allowed only when BOTH of these are clear.
"""
from datetime import date, datetime
from loguru import logger
from sqlalchemy import select

from db.database import async_session, SystemState, _utc_naive_now

# Daily loss percentage that fires the breaker.
CIRCUIT_BREAKER_THRESHOLD = 0.10  # 10%


class SystemStateService:
    async def _load_or_create(self, session) -> SystemState:
        row = (await session.execute(select(SystemState).where(SystemState.id == 1))).scalar_one_or_none()
        if row is None:
            row = SystemState(id=1)
            session.add(row)
            await session.flush()
        return row

    async def is_trading_enabled(self) -> tuple[bool, str]:
        """Return (enabled, reason). reason is empty when enabled."""
        async with async_session() as session:
            row = await self._load_or_create(session)
            await session.commit()
            if row.kill_switch_active:
                return False, "kill_switch"
            today = date.today()
            if row.circuit_breaker_fired_date == today:
                return False, "circuit_breaker"
            return True, ""

    async def activate_kill_switch(self) -> None:
        async with async_session() as session:
            row = await self._load_or_create(session)
            row.kill_switch_active = True
            row.kill_switch_activated_at = _utc_naive_now()
            await session.commit()
        logger.warning("KILL SWITCH ACTIVATED — scanner will not open new trades")

    async def deactivate_kill_switch(self) -> None:
        async with async_session() as session:
            row = await self._load_or_create(session)
            row.kill_switch_active = False
            row.kill_switch_activated_at = None
            await session.commit()
        logger.info("Kill switch cleared — scanner resumed")

    async def ensure_daily_snapshot(self, current_equity: float) -> None:
        """Take (or refresh) today's equity snapshot.

        Called once per day when the market opens. If today's snapshot is
        missing or stale, writes a fresh one. Also clears yesterday's
        circuit breaker if it was set.

        A non-positive equity (transient API/connection failure returning a
        zero default) is refused: snapshotting 0 would silently disable the
        circuit breaker for the entire day. We log and let the next
        scanner tick try again.
        """
        if current_equity <= 0:
            logger.warning(
                f"ensure_daily_snapshot: refusing to snapshot non-positive "
                f"equity (${current_equity:,.2f}) — will retry next tick"
            )
            return

        async with async_session() as session:
            row = await self._load_or_create(session)
            today = date.today()
            if row.daily_snapshot_date != today:
                row.daily_snapshot_date = today
                row.daily_equity_snapshot = current_equity
                # A new day means yesterday's breaker no longer applies.
                if row.circuit_breaker_fired_date and row.circuit_breaker_fired_date != today:
                    row.circuit_breaker_fired_date = None
                await session.commit()
                logger.info(f"Daily equity snapshot: ${current_equity:,.2f}")

    async def check_circuit_breaker(self, current_equity: float) -> bool:
        """Evaluate whether the 10% daily loss threshold has been crossed.

        Returns True if the breaker is newly-fired by this call. If the
        breaker already fired today, or the threshold hasn't been crossed,
        returns False (caller should just continue).
        """
        async with async_session() as session:
            row = await self._load_or_create(session)
            today = date.today()
            if row.circuit_breaker_fired_date == today:
                return False  # already fired today
            if row.daily_equity_snapshot is None or row.daily_snapshot_date != today:
                return False  # no snapshot yet
            snapshot = row.daily_equity_snapshot
            if snapshot <= 0:
                return False
            loss_pct = (snapshot - current_equity) / snapshot
            if loss_pct >= CIRCUIT_BREAKER_THRESHOLD:
                row.circuit_breaker_fired_date = today
                await session.commit()
                logger.warning(
                    f"CIRCUIT BREAKER FIRED — equity dropped {loss_pct:.1%} "
                    f"(${snapshot:,.2f} → ${current_equity:,.2f})"
                )
                return True
            return False

    async def get_status(self) -> dict:
        """Full state snapshot for the dashboard."""
        async with async_session() as session:
            row = await self._load_or_create(session)
            await session.commit()
            today = date.today()
            breaker_active = row.circuit_breaker_fired_date == today
            return {
                "kill_switch_active": bool(row.kill_switch_active),
                "kill_switch_activated_at": (
                    row.kill_switch_activated_at.isoformat() + "Z"
                    if row.kill_switch_activated_at else None
                ),
                "circuit_breaker_active": breaker_active,
                "circuit_breaker_fired_date": (
                    row.circuit_breaker_fired_date.isoformat() if row.circuit_breaker_fired_date else None
                ),
                "daily_equity_snapshot": row.daily_equity_snapshot,
                "daily_snapshot_date": (
                    row.daily_snapshot_date.isoformat() if row.daily_snapshot_date else None
                ),
                "circuit_breaker_threshold_pct": CIRCUIT_BREAKER_THRESHOLD * 100,
            }
