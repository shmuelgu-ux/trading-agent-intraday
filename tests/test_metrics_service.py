"""Tests for MetricsService.

Uses in-memory SQLite the same way test_system_state_service.py does —
a real schema and real inserts, because the math is downstream of a
resampling step we want to verify end-to-end.
"""
import math
from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

import db.database as db_mod
from db.database import Base, EquitySnapshot
import services.metrics_service as metrics_mod
from services.metrics_service import MetricsService, TRADING_DAYS_PER_YEAR


@pytest.fixture
async def isolated_db(monkeypatch):
    """Fresh in-memory SQLite per test; point ``async_session`` at it."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db_mod, "async_session", session_factory)
    monkeypatch.setattr(metrics_mod, "async_session", session_factory)
    yield session_factory
    await engine.dispose()


async def _seed_daily_closes(session_factory, equities: list[float], start_days_ago: int = None):
    """Seed one snapshot per day, oldest first, spaced by 1 calendar day.
    If ``start_days_ago`` is None, last point lands today."""
    n = len(equities)
    if start_days_ago is None:
        start_days_ago = n - 1  # last point = today
    async with session_factory() as session:
        for i, eq in enumerate(equities):
            ts = datetime.utcnow() - timedelta(days=start_days_ago - i, minutes=1)
            session.add(EquitySnapshot(timestamp=ts, equity=eq, source="test"))
        await session.commit()


class TestRecordSnapshot:
    async def test_records_positive_equity(self, isolated_db):
        svc = MetricsService()
        await svc.record_snapshot(2000.0, source="manual")
        async with isolated_db() as session:
            from sqlalchemy import select
            rows = (await session.execute(select(EquitySnapshot))).scalars().all()
        assert len(rows) == 1
        assert rows[0].equity == 2000.0
        assert rows[0].source == "manual"

    async def test_silently_drops_non_positive(self, isolated_db):
        svc = MetricsService()
        await svc.record_snapshot(0)
        await svc.record_snapshot(-100)
        async with isolated_db() as session:
            from sqlalchemy import select
            rows = (await session.execute(select(EquitySnapshot))).scalars().all()
        assert rows == []


class TestInsufficientData:
    async def test_zero_snapshots_returns_insufficient(self, isolated_db):
        svc = MetricsService()
        m = await svc.compute()
        assert m["insufficient_data"] is True
        assert m["sample_size"] == 0
        assert m["sharpe"] is None and m["sortino"] is None
        assert m["max_drawdown"] is None and m["calmar"] is None

    async def test_one_snapshot_returns_insufficient(self, isolated_db):
        await _seed_daily_closes(isolated_db, [2000.0])
        svc = MetricsService()
        m = await svc.compute()
        assert m["insufficient_data"] is True
        assert m["sample_size"] == 1


class TestResampling:
    async def test_last_snapshot_per_day_wins(self, isolated_db):
        """Two snapshots on the same day — only the later one becomes the daily close."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        async with isolated_db() as session:
            # Yesterday morning and evening
            session.add(EquitySnapshot(timestamp=yesterday + timedelta(hours=9), equity=2000.0, source="t"))
            session.add(EquitySnapshot(timestamp=yesterday + timedelta(hours=15), equity=2100.0, source="t"))
            # Today
            session.add(EquitySnapshot(timestamp=today + timedelta(hours=10), equity=2150.0, source="t"))
            await session.commit()
        svc = MetricsService()
        closes = await svc._daily_closes()
        assert len(closes) == 2
        # Return must be (2150/2100)-1, not (2150/2000)-1
        d1_eq = closes[0][1]
        d2_eq = closes[1][1]
        assert d1_eq == 2100.0  # yesterday close = later snapshot
        assert d2_eq == 2150.0


class TestMaxDrawdown:
    async def test_monotonic_up_has_zero_drawdown(self, isolated_db):
        await _seed_daily_closes(isolated_db, [2000, 2050, 2100, 2150, 2200])
        m = await MetricsService().compute()
        assert m["max_drawdown"] == 0.0

    async def test_simple_peak_to_trough(self, isolated_db):
        # Peak 2500, trough 2000 → DD = 500/2500 = 0.20
        await _seed_daily_closes(isolated_db, [2000, 2500, 2200, 2000, 2100])
        m = await MetricsService().compute()
        # Service rounds to 4dp → tolerance 5e-5
        assert abs(m["max_drawdown"] - 0.20) < 5e-5
        assert m["max_drawdown_pct"] == 20.0

    async def test_multiple_drawdowns_reports_largest(self, isolated_db):
        # DDs: 2500->2300 = 0.08, then new peak 2600->2200 = ~0.154
        await _seed_daily_closes(isolated_db, [2000, 2500, 2300, 2600, 2200, 2400])
        m = await MetricsService().compute()
        expected = (2600 - 2200) / 2600  # 0.15384...
        # Service rounds to 4dp → tolerance 5e-5
        assert abs(m["max_drawdown"] - expected) < 5e-5


class TestSharpe:
    async def test_monotonic_positive_returns_finite_sharpe(self, isolated_db):
        # Constant 1% daily gain → stdev > 0 only if there's variance.
        # Use slightly varying positive returns so stdev > 0.
        await _seed_daily_closes(isolated_db, [2000, 2020, 2040.4, 2061.204, 2083.0])
        m = await MetricsService().compute()
        # With small variance this will be a very large Sharpe
        assert m["sharpe"] is not None
        assert m["sharpe"] > 0

    async def test_constant_equity_returns_none_sharpe(self, isolated_db):
        """Zero variance → undefined Sharpe (divide by zero)."""
        await _seed_daily_closes(isolated_db, [2000, 2000, 2000, 2000])
        m = await MetricsService().compute()
        assert m["sharpe"] is None  # stdev is 0
        # Max DD should also be 0
        assert m["max_drawdown"] == 0.0

    async def test_matches_hand_computation(self, isolated_db):
        """Exact arithmetic check against manual computation."""
        # Returns: +10%, -10% → mean = 0, stdev > 0 → Sharpe = 0
        await _seed_daily_closes(isolated_db, [1000, 1100, 990])
        m = await MetricsService().compute()
        import statistics as st
        r = [(1100 - 1000) / 1000, (990 - 1100) / 1100]
        mean = st.mean(r)
        std = st.stdev(r)
        expected = (mean / std) * math.sqrt(TRADING_DAYS_PER_YEAR)
        assert m["sharpe"] == round(expected, 3)


class TestSortino:
    async def test_all_positive_returns_none(self, isolated_db):
        """Sortino is undefined when there are no downside returns."""
        await _seed_daily_closes(isolated_db, [2000, 2050, 2100, 2150])
        m = await MetricsService().compute()
        assert m["sortino"] is None  # no negative returns → downside_std = 0

    async def test_sortino_greater_than_sharpe_for_asymmetric(self, isolated_db):
        """When positive swings are bigger than negative, Sortino > Sharpe."""
        # Big up-days, small down-days → downside_std < full_std → Sortino > Sharpe
        await _seed_daily_closes(isolated_db, [1000, 1100, 1095, 1200, 1190, 1300])
        m = await MetricsService().compute()
        if m["sharpe"] and m["sortino"]:
            assert m["sortino"] > m["sharpe"]


class TestCalmar:
    async def test_calmar_when_drawdown_positive(self, isolated_db):
        """Calmar = annualised_return / max_drawdown. Hand-verify sign."""
        await _seed_daily_closes(isolated_db, [2000, 2100, 1900, 2200])
        m = await MetricsService().compute()
        # Calmar should be finite since we have a drawdown
        assert m["calmar"] is not None

    async def test_none_when_zero_drawdown(self, isolated_db):
        """No drawdown → Calmar is undefined (infinite)."""
        await _seed_daily_closes(isolated_db, [2000, 2050, 2100, 2150])
        m = await MetricsService().compute()
        assert m["calmar"] is None


class TestTotalAndAnnualisedReturn:
    async def test_total_return_percent(self, isolated_db):
        await _seed_daily_closes(isolated_db, [2000, 2200])
        m = await MetricsService().compute()
        assert m["total_return"] == round(0.10, 4)
        assert m["total_return_pct"] == 10.0

    async def test_date_range_populated(self, isolated_db):
        await _seed_daily_closes(isolated_db, [2000, 2100, 2200])
        m = await MetricsService().compute()
        assert m["date_range"] is not None
        assert m["date_range"]["calendar_days"] == 2


class TestLookbackFilter:
    async def test_lookback_excludes_old_snapshots(self, isolated_db):
        """When lookback_days is set, only recent snapshots are considered."""
        now = datetime.utcnow()
        async with isolated_db() as session:
            # 100 days ago
            session.add(EquitySnapshot(timestamp=now - timedelta(days=100), equity=1000, source="t"))
            # 5 days ago
            session.add(EquitySnapshot(timestamp=now - timedelta(days=5), equity=2000, source="t"))
            # today
            session.add(EquitySnapshot(timestamp=now, equity=2100, source="t"))
            await session.commit()
        svc = MetricsService()
        # Full: 3 dates
        m_full = await svc.compute()
        assert m_full["sample_size"] == 3
        # 10-day lookback: only the last 2
        m_recent = await svc.compute(lookback_days=10)
        assert m_recent["sample_size"] == 2
