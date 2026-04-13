import json
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text, Float, Integer, DateTime, Index, text
from config import settings


# Use PostgreSQL if available, otherwise SQLite
db_url = settings.database_url
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

engine = create_async_engine(db_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def _utc_naive_now() -> datetime:
    """Current UTC time as a tz-naive datetime.

    The trade_log table stores timestamps without timezone info (the
    Railway PostgreSQL column is DateTime, not DateTimeTZ) so we
    intentionally strip the tzinfo. Replaces the deprecated
    datetime.utcnow().
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class TradeLog(Base):
    __tablename__ = "trade_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    action_taken: Mapped[str] = mapped_column(Text, nullable=False)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_data: Mapped[str] = mapped_column(Text, default="{}")
    reasoning: Mapped[str] = mapped_column(Text, default="[]")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="OPEN")

    __table_args__ = (
        # Journal pagination + time-window filters (ORDER BY timestamp DESC)
        Index("ix_trade_log_timestamp", "timestamp"),
        # Ticker search filter on the journal
        Index("ix_trade_log_ticker", "ticker"),
        # Reconciliation lookups: WHERE status='OPEN' AND action_taken='EXECUTE'
        Index("ix_trade_log_status_action", "status", "action_taken"),
        # get_risk_for_tickers: WHERE ticker IN (...) AND action_taken='EXECUTE'
        Index("ix_trade_log_ticker_action", "ticker", "action_taken"),
    )


class LearningReport(Base):
    """Stores the output of each learning cycle (every 10 closed trades)."""
    __tablename__ = "learning_report"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Which closed trades were analyzed (comma-separated IDs)
    trade_ids: Mapped[str] = mapped_column(Text, nullable=False)
    # Number of trades in this batch
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Per-trade analysis (JSON array of objects)
    individual_analysis: Mapped[str] = mapped_column(Text, default="[]")
    # Combined insights (JSON object with patterns, recommendations)
    combined_insights: Mapped[str] = mapped_column(Text, default="{}")
    # When this report was generated
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)
    # Stats snapshot at time of report
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)


# SQL statements used to create the indexes on an already-existing table.
# SQLAlchemy's create_all only creates indexes alongside a fresh CREATE TABLE;
# for a table that already exists on Railway's PostgreSQL we need to add the
# indexes explicitly with CREATE INDEX IF NOT EXISTS. Both PostgreSQL and
# SQLite support this syntax.
_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS ix_trade_log_timestamp ON trade_log (timestamp)",
    "CREATE INDEX IF NOT EXISTS ix_trade_log_ticker ON trade_log (ticker)",
    "CREATE INDEX IF NOT EXISTS ix_trade_log_status_action ON trade_log (status, action_taken)",
    "CREATE INDEX IF NOT EXISTS ix_trade_log_ticker_action ON trade_log (ticker, action_taken)",
)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Ensure all indexes exist on an already-created table. Idempotent.
        for stmt in _INDEX_DDL:
            try:
                await conn.execute(text(stmt))
            except Exception as e:
                logger.warning(f"Index create skipped ({stmt[:60]}...): {e}")
