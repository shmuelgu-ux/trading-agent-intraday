"""Historical bar fetching + SQLite cache for the backtester.

The backtester re-runs the scanner's per-day analysis against historical
data, which needs O(90 days × N tickers) bars per trading day × ~250 days
per year. Hitting Alpaca every time would be slow and rate-limited, so
we cache to a local SQLite file keyed by (ticker, timeframe, date).

The cache is OFFLINE FROM PRODUCTION — it lives in ``backtest_cache.db``
by default, totally separate from the ``trade_log`` Postgres used by
the running bot. Corrupt cache can be deleted without touching live.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from backtest.models import Bar


logger = logging.getLogger(__name__)

# Default cache file. Relative to CWD so tests + CLI can override.
DEFAULT_CACHE_PATH = "backtest_cache.db"


class BarCache:
    """SQLite-backed cache of daily/weekly OHLCV bars.

    Schema intentionally boring (one row per (ticker, timeframe, date))
    so a bar can be looked up in O(log N) and bulk-inserted with a single
    INSERT OR REPLACE. No ORM here — sqlite3 is quite enough.
    """

    def __init__(self, path: str = DEFAULT_CACHE_PATH):
        self.path = path
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS bars (
                    ticker    TEXT NOT NULL,
                    timeframe TEXT NOT NULL,      -- 'day' / 'week'
                    trade_date TEXT NOT NULL,     -- ISO yyyy-mm-dd
                    open      REAL NOT NULL,
                    high      REAL NOT NULL,
                    low       REAL NOT NULL,
                    close     REAL NOT NULL,
                    volume    REAL NOT NULL,
                    PRIMARY KEY (ticker, timeframe, trade_date)
                )
                """
            )
            # Read path used by ``get_bars``: WHERE ticker=? AND timeframe=? ORDER BY date.
            # The PK already covers this, so no secondary index is needed.

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def put(self, ticker: str, timeframe: str, bars: Iterable[Bar]) -> int:
        """Bulk-upsert bars. Returns the count written (after dedup)."""
        rows = [
            (ticker, timeframe, b.trade_date.isoformat(), b.open, b.high, b.low, b.close, b.volume)
            for b in bars
        ]
        if not rows:
            return 0
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> list[Bar]:
        """Return cached bars for ``ticker`` in ``[start, end]`` inclusive.
        Dates outside the range are excluded; gaps are NOT filled (caller
        decides whether to fetch-and-merge missing ranges)."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT trade_date, open, high, low, close, volume FROM bars
                WHERE ticker = ? AND timeframe = ?
                  AND trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date ASC
                """,
                (ticker, timeframe, start.isoformat(), end.isoformat()),
            ).fetchall()
        return [
            Bar(
                trade_date=date.fromisoformat(r[0]),
                open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5],
            )
            for r in rows
        ]

    def coverage(self, ticker: str, timeframe: str) -> tuple[date | None, date | None, int]:
        """Return (first_date, last_date, count) for this (ticker, timeframe).
        Useful for deciding whether to fetch or hit cache."""
        with self._conn() as c:
            row = c.execute(
                """
                SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM bars
                WHERE ticker = ? AND timeframe = ?
                """,
                (ticker, timeframe),
            ).fetchone()
        if not row or row[2] == 0:
            return None, None, 0
        return date.fromisoformat(row[0]), date.fromisoformat(row[1]), row[2]

    def clear(self, ticker: str | None = None) -> None:
        """Wipe cache (all or one ticker). Tests use this; users shouldn't
        need to — deleting the file is simpler."""
        with self._conn() as c:
            if ticker:
                c.execute("DELETE FROM bars WHERE ticker = ?", (ticker,))
            else:
                c.execute("DELETE FROM bars")


def alpaca_bars_to_domain(alpaca_bars) -> list[Bar]:
    """Convert Alpaca's ``BarSet`` entries (or a list of Bar-like objects
    with .timestamp / .open / etc.) into our domain ``Bar`` objects.

    Kept separate from the fetching code so tests don't need to stub
    the whole Alpaca client — just hand-built Bar-like objects work.
    """
    out: list[Bar] = []
    for b in alpaca_bars:
        ts = getattr(b, "timestamp", None) or getattr(b, "t", None)
        if ts is None:
            continue
        d = ts.date() if isinstance(ts, datetime) else ts
        out.append(Bar(
            trade_date=d,
            open=float(b.open),
            high=float(b.high),
            low=float(b.low),
            close=float(b.close),
            volume=float(b.volume),
        ))
    return out


class AlpacaBarFetcher:
    """Thin wrapper around Alpaca's historical API.

    Kept small and mockable — the constructor takes a raw
    ``StockHistoricalDataClient`` so tests can inject a fake.
    Separate from the production ``AlpacaClient`` to avoid pulling the
    whole trading-client dependency graph into backtest-only code.
    """

    def __init__(self, data_client, feed=None):
        self._client = data_client
        # DataFeed.IEX by default (matches live scanner); None means
        # the Alpaca library's default for the given account tier.
        self._feed = feed

    def fetch(self, ticker: str, timeframe, start: date, end: date) -> list[Bar]:
        """Fetch bars directly from Alpaca. Does NOT touch the cache —
        callers (e.g. ``populate_cache``) compose fetch + cache.put.
        """
        # Lazy import so the backtest package stays importable without
        # alpaca-py installed (tests can skip this path).
        from alpaca.data.requests import StockBarsRequest

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.max.time()),
        )
        if self._feed is not None:
            request.feed = self._feed
        resp = self._client.get_stock_bars(request)
        bars_list = resp.data.get(ticker, []) if hasattr(resp, "data") else resp[ticker]
        return alpaca_bars_to_domain(bars_list)


def populate_cache(
    fetcher: AlpacaBarFetcher,
    cache: BarCache,
    ticker: str,
    timeframe_str: str,
    timeframe,
    start: date,
    end: date,
    force_refetch: bool = False,
) -> int:
    """Fetch from Alpaca if the cache doesn't already cover [start, end]
    for this (ticker, timeframe). Returns bars_written.

    The coverage check is coarse-grained (first/last dates), so small
    internal gaps can still get missed. Good enough for an MVP; a
    fine-grained merge would complicate the code without much benefit.
    """
    if not force_refetch:
        first, last, _ = cache.coverage(ticker, timeframe_str)
        if first is not None and last is not None and first <= start and last >= end:
            return 0  # cache already covers this range
    try:
        bars = fetcher.fetch(ticker, timeframe, start, end)
    except Exception as e:
        logger.warning("fetch failed for %s %s: %s", ticker, timeframe_str, e)
        return 0
    if not bars:
        return 0
    return cache.put(ticker, timeframe_str, bars)
