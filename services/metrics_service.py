"""Performance metrics computed off the equity-snapshot time series.

Four headline numbers + supporting metadata:

- **Sharpe ratio**: annualised mean return / annualised stddev of
  returns. Classic risk-adjusted return. Typical benchmark: >1 good,
  >2 strong, <0 losing.
- **Sortino ratio**: like Sharpe but only downside deviation in the
  denominator. More honest for asymmetric P&L (we don't penalize
  positive volatility).
- **Max Drawdown**: largest peak-to-trough equity drop over the period.
  Not annualised — it's an absolute worst-case, expressed as a
  fraction of the preceding peak.
- **Calmar ratio**: annualised return / max drawdown. Captures "how
  much return per unit of pain" better than Sharpe for strategies with
  skewed return distributions.

The service resamples raw per-tick snapshots to one equity close per
calendar date (last snapshot of each day wins). Metrics are computed
on those daily closes — running them per-tick would drown the signal
in microstructure noise.

``record_snapshot`` is a no-op for non-positive equity (transient API
failure returning a zero default). The HWM / circuit-breaker machinery
has the same defence in ``system_state_service``; we don't want a
corrupted zero point to skew Sharpe either.
"""
import math
import statistics
from datetime import date, datetime, timedelta
from loguru import logger
from sqlalchemy import select, delete, func

from db.database import async_session, EquitySnapshot, _utc_naive_now


# US equity trading days per year — used to annualise daily Sharpe/Sortino.
TRADING_DAYS_PER_YEAR = 252

# Rolling cap so the table doesn't grow unbounded. At one snapshot per
# minute during market hours (~390/day) this gives ~256 days of history.
MAX_SNAPSHOT_ROWS = 100_000


class MetricsService:
    async def record_snapshot(self, equity: float, source: str = "scanner_tick") -> None:
        """Append one snapshot. Non-positive equity is silently dropped."""
        if equity <= 0:
            return
        try:
            async with async_session() as session:
                snap = EquitySnapshot(
                    timestamp=_utc_naive_now(),
                    equity=float(equity),
                    source=source,
                )
                session.add(snap)
                await session.commit()
                # Occasionally trim. Not every call — that would double the
                # DB traffic for the same information.
                if snap.id and (snap.id % 500 == 0):
                    await self._trim_to_max(session)
        except Exception as e:
            logger.warning(f"record_snapshot failed (non-fatal): {type(e).__name__}: {e}")

    async def _trim_to_max(self, session) -> None:
        """Delete oldest rows so total <= MAX_SNAPSHOT_ROWS."""
        try:
            total = (await session.execute(
                select(func.count()).select_from(EquitySnapshot)
            )).scalar() or 0
            if total <= MAX_SNAPSHOT_ROWS:
                return
            excess = total - MAX_SNAPSHOT_ROWS
            old_ids = (await session.execute(
                select(EquitySnapshot.id)
                .order_by(EquitySnapshot.timestamp.asc(), EquitySnapshot.id.asc())
                .limit(excess)
            )).all()
            ids = [row[0] for row in old_ids]
            if not ids:
                return
            await session.execute(
                delete(EquitySnapshot).where(EquitySnapshot.id.in_(ids))
            )
            await session.commit()
        except Exception as e:
            logger.warning(f"EquitySnapshot trim failed (non-fatal): {e}")

    async def _daily_closes(
        self, lookback_days: int | None = None
    ) -> list[tuple[date, float]]:
        """Return ``[(date, equity)]`` — one equity per calendar date,
        taking the LATEST snapshot of each date. Sorted ascending."""
        async with async_session() as session:
            stmt = select(EquitySnapshot.timestamp, EquitySnapshot.equity).order_by(
                EquitySnapshot.timestamp.asc()
            )
            if lookback_days is not None and lookback_days > 0:
                cutoff = _utc_naive_now() - timedelta(days=lookback_days)
                stmt = stmt.where(EquitySnapshot.timestamp >= cutoff)
            rows = (await session.execute(stmt)).all()
        by_date: dict[date, float] = {}
        for ts, equity in rows:
            # Later iterations overwrite earlier — so the last snapshot
            # of each date is the "close" we use for metrics.
            by_date[ts.date()] = float(equity)
        return sorted(by_date.items(), key=lambda x: x[0])

    async def compute(self, lookback_days: int | None = None) -> dict:
        """Compute the four headline metrics plus context. Returns
        ``{"insufficient_data": True, ...}`` when fewer than two daily
        closes are available (need at least one return to compute anything).
        """
        closes = await self._daily_closes(lookback_days=lookback_days)
        n = len(closes)
        base = {
            "sample_size": n,
            "insufficient_data": n < 2,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "max_drawdown_pct": None,
            "calmar": None,
            "annualized_return": None,
            "annualized_return_pct": None,
            "total_return": None,
            "total_return_pct": None,
            "date_range": None,
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        }
        if n < 2:
            return base

        dates = [d for d, _ in closes]
        equities = [e for _, e in closes]
        returns = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, n)
            if equities[i - 1] > 0
        ]
        if len(returns) < 1:
            return base

        mean_ret = statistics.mean(returns)

        # Sharpe — sample stdev, needs at least 2 returns (3 equity points).
        sharpe = None
        if len(returns) >= 2:
            std_ret = statistics.stdev(returns)
            if std_ret > 0:
                sharpe = (mean_ret / std_ret) * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Sortino — downside deviation computed over ALL returns (zeroing
        # positives) per the standard Sortino formulation.
        sortino = None
        downside_var = sum((r * r) for r in returns if r < 0) / len(returns)
        downside_std = math.sqrt(downside_var)
        if downside_std > 0:
            sortino = (mean_ret / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Max drawdown on the raw equity curve (not returns).
        peak = equities[0]
        max_dd = 0.0
        for e in equities:
            if e > peak:
                peak = e
            if peak > 0:
                dd = (peak - e) / peak
                if dd > max_dd:
                    max_dd = dd

        # Annualised total return — calendar days with 365.25 for leap years.
        days = max((dates[-1] - dates[0]).days, 1)
        total_return = (equities[-1] / equities[0]) - 1 if equities[0] > 0 else 0.0
        annualized = None
        if equities[0] > 0 and equities[-1] > 0:
            try:
                annualized = (equities[-1] / equities[0]) ** (365.25 / days) - 1
            except (ValueError, OverflowError):
                annualized = None

        calmar = None
        if annualized is not None and max_dd > 0:
            calmar = annualized / max_dd

        base.update({
            "insufficient_data": False,
            "sharpe": round(sharpe, 3) if sharpe is not None else None,
            "sortino": round(sortino, 3) if sortino is not None else None,
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "calmar": round(calmar, 3) if calmar is not None else None,
            "annualized_return": round(annualized, 4) if annualized is not None else None,
            "annualized_return_pct": (
                round(annualized * 100, 2) if annualized is not None else None
            ),
            "total_return": round(total_return, 4),
            "total_return_pct": round(total_return * 100, 2),
            "date_range": {
                "start": dates[0].isoformat(),
                "end": dates[-1].isoformat(),
                "calendar_days": days,
            },
        })
        return base
