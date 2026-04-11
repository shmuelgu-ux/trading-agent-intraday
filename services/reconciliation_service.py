"""Reconciles closed positions: detects when an OPEN trade in the journal
no longer has a matching Alpaca position, finds the closing fill, and
writes back exit_price / exit_timestamp / pnl / status='CLOSED'.

This is what keeps the 'Performance Statistics' panel's `closed`, `wins`,
`losses`, `win_rate`, and `total_pnl` metrics in sync with reality.

Runs in two places:
- Once at startup (catches anything closed while the server was down).
- In the scanner loop whenever the position count drops (handles live
  stop-loss / take-profit hits).

All failures are logged and swallowed — rows stay OPEN and retry on the
next loop, so the mechanism is fully self-healing.
"""
import asyncio
from datetime import timedelta
from loguru import logger

from services.alpaca_client import AlpacaClient
from services.journal_service import JournalService


class ReconciliationService:
    def __init__(self, alpaca: AlpacaClient, journal: JournalService):
        self.alpaca = alpaca
        self.journal = journal

    async def reconcile_closed_trades(self) -> int:
        """Walk every OPEN+EXECUTE journal row whose ticker is no longer
        in Alpaca positions, look up the closing fill, and mark the row
        closed with realized P&L. Returns the number of rows reconciled.
        """
        open_trades = await self.journal.get_open_trades()
        if not open_trades:
            return 0

        try:
            positions = await asyncio.to_thread(self.alpaca.get_open_positions)
        except Exception as e:
            logger.error(f"Reconcile: failed to fetch positions: {e}")
            return 0
        open_symbols = {p["symbol"] for p in positions}

        # Defensive grouping — shouldn't happen given the dedup check in
        # DecisionEngine, but if a stale duplicate ever lands in OPEN state
        # we want each one handled independently instead of crashing.
        by_ticker: dict[str, list] = {}
        for t in open_trades:
            by_ticker.setdefault(t.ticker, []).append(t)

        reconciled = 0
        for ticker, trades in by_ticker.items():
            if ticker in open_symbols:
                # Position still open on Alpaca — nothing to reconcile.
                continue

            for trade in trades:
                # Guard against corrupt rows
                if not trade.position_size or not trade.entry_price:
                    logger.warning(
                        f"Reconcile {ticker} #{trade.id}: missing "
                        f"position_size/entry_price, skipping"
                    )
                    continue

                opposite = "sell" if trade.side == "buy" else "buy"
                # Look back a minute before entry to be safe against clock skew
                after = trade.timestamp - timedelta(minutes=1)

                try:
                    fill = await asyncio.to_thread(
                        self.alpaca.get_last_closing_fill,
                        ticker,
                        opposite,
                        after,
                    )
                except Exception as e:
                    logger.error(
                        f"Reconcile {ticker} #{trade.id}: "
                        f"get_last_closing_fill failed: {e}"
                    )
                    continue

                if not fill:
                    logger.warning(
                        f"Reconcile {ticker} #{trade.id}: no closing fill "
                        f"found (manual close? partial? leaving OPEN)"
                    )
                    continue

                exit_price = float(fill["exit_price"])
                exit_ts = fill["exit_timestamp"]

                # Guard against stale orders returning with a timestamp
                # earlier than our entry (shouldn't happen but belt-and-braces)
                compare_ts = exit_ts
                if compare_ts is not None and compare_ts.tzinfo is not None:
                    compare_ts = compare_ts.replace(tzinfo=None)
                if compare_ts is not None and compare_ts < trade.timestamp:
                    logger.warning(
                        f"Reconcile {ticker} #{trade.id}: exit_ts {exit_ts} "
                        f"precedes entry {trade.timestamp}, skipping"
                    )
                    continue

                qty = trade.position_size
                entry_price = trade.entry_price
                if trade.side == "buy":
                    pnl = (exit_price - entry_price) * qty
                else:
                    pnl = (entry_price - exit_price) * qty

                updated = await self.journal.mark_trade_closed(
                    trade.id, exit_price, exit_ts, round(pnl, 2)
                )
                if updated:
                    reconciled += 1
                    logger.info(
                        f"Reconciled {ticker} #{trade.id}: "
                        f"entry={entry_price} exit={exit_price} "
                        f"qty={qty} side={trade.side} pnl={pnl:+.2f}"
                    )
                else:
                    logger.info(
                        f"Reconcile {ticker} #{trade.id}: already closed "
                        f"(race with another reconciler?)"
                    )

        return reconciled
