from loguru import logger
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.common.enums import Sort
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
from config import settings
from models.orders import RiskParams


class AlpacaClient:
    """Wrapper around Alpaca Trading API for paper trading."""

    def __init__(self):
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            logger.warning("Alpaca API keys not configured - running in dry-run mode")
            self._client = None
            self._data_client = None
        else:
            self._client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=True,
            )
            self._data_client = StockHistoricalDataClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def get_account(self) -> dict:
        """Get account info (balance, buying power, etc.)."""
        if not self._client:
            return self._dry_run_account()

        account = self._client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
        }

    def get_open_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self._client:
            return []

        positions = self._client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value,
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pnl": float(p.unrealized_pl),
                "market_value": float(p.market_value),
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> dict | None:
        """Get position for a specific symbol."""
        if not self._client:
            return None
        try:
            p = self._client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value,
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pnl": float(p.unrealized_pl),
            }
        except Exception:
            return None

    def submit_bracket_order(
        self,
        symbol: str,
        side: str,
        risk_params: RiskParams,
    ) -> dict:
        """Submit a bracket order (entry + SL + TP)."""
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        if not self._client:
            return self._dry_run_order(symbol, side, risk_params)

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=risk_params.position_size,
            side=order_side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=risk_params.take_profit),
            stop_loss=StopLossRequest(stop_price=risk_params.stop_loss),
        )

        order = self._client.submit_order(order_request)
        logger.info(
            f"Order submitted: {symbol} {side} x{risk_params.position_size} "
            f"@ market | SL={risk_params.stop_loss} TP={risk_params.take_profit}"
        )
        return {
            "order_id": str(order.id),
            "symbol": symbol,
            "side": side,
            "qty": risk_params.position_size,
            "status": order.status.value,
            "stop_loss": risk_params.stop_loss,
            "take_profit": risk_params.take_profit,
        }

    def close_all_positions(self, cancel_orders: bool = True) -> dict:
        """Close every open position at market and (optionally) cancel any
        still-open orders. Used by the intraday end-of-day safety net so
        nothing ever rolls into the next trading day.

        Returns a summary dict: how many positions were requested to
        close, how many failed, plus the raw list if available.
        """
        if not self._client:
            logger.info("[DRY RUN] close_all_positions called")
            return {"requested": 0, "failed": 0, "dry_run": True}

        failed = 0
        requested = 0
        results = []
        try:
            # Alpaca's close_all_positions returns a list of per-symbol
            # responses; each has a status code per position.
            response = self._client.close_all_positions(cancel_orders=cancel_orders)
            if response:
                for item in response:
                    requested += 1
                    # Newer alpaca-py returns objects with .status on each leg
                    status_code = getattr(item, "status", None)
                    if status_code is not None and int(status_code) >= 400:
                        failed += 1
                    results.append({
                        "symbol": getattr(item, "symbol", None),
                        "status": status_code,
                    })
            logger.info(
                f"close_all_positions: requested={requested}, failed={failed}"
            )
        except Exception as e:
            logger.error(f"close_all_positions failed: {type(e).__name__}: {e}")
            return {"requested": requested, "failed": requested, "error": str(e)}

        return {"requested": requested, "failed": failed, "results": results}

    def get_last_closing_fill(
        self,
        symbol: str,
        opposite_side: str,
        after: datetime,
    ) -> dict | None:
        """Return the most recent filled order on ``opposite_side`` for ``symbol``
        submitted after ``after``. Used by the reconciliation service to find
        the stop-loss or take-profit leg that closed a bracket order.

        Alpaca returns bracket child orders as independent top-level rows
        when ``nested=False``, sharing the parent's symbol and having the
        OPPOSITE side of the entry. So filtering by symbol + side + after
        finds the exit directly.

        Args:
            symbol: The ticker to query.
            opposite_side: "buy" or "sell" — the opposite of the entry side
                (use "sell" when the entry was a long buy, "buy" when the
                entry was a short sell).
            after: Only consider orders submitted after this timestamp.

        Returns:
            Dict with keys ``exit_price``, ``exit_timestamp``, ``filled_qty``
            or ``None`` if no matching filled order was found.
        """
        if not self._client:
            return None
        side_enum = OrderSide.BUY if opposite_side == "buy" else OrderSide.SELL
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[symbol],
            side=side_enum,
            after=after,
            direction=Sort.DESC,
            limit=50,
            nested=False,
        )
        try:
            orders = self._client.get_orders(filter=req)
        except Exception as e:
            logger.warning(f"get_last_closing_fill({symbol}) failed: {e}")
            return None
        for o in orders:
            if o.filled_at is None or o.filled_qty is None or o.filled_avg_price is None:
                continue
            try:
                filled_qty = float(o.filled_qty)
                filled_avg = float(o.filled_avg_price)
            except (TypeError, ValueError):
                continue
            if filled_qty <= 0 or filled_avg <= 0:
                continue
            return {
                "exit_price": filled_avg,
                "exit_timestamp": o.filled_at,
                "filled_qty": filled_qty,
            }
        return None

    def get_atr(self, symbol: str, period: int = 14) -> float | None:
        """Calculate ATR from Alpaca market data."""
        if not self._data_client:
            return None
        try:
            end = datetime.now()
            start = end - timedelta(days=period * 3)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=period + 1,
            )
            request.feed = DataFeed.IEX
            bars = self._data_client.get_stock_bars(request)
            data = bars[symbol]
            if len(data) < 2:
                return None

            true_ranges = []
            for i in range(1, len(data)):
                high = float(data[i].high)
                low = float(data[i].low)
                prev_close = float(data[i - 1].close)
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)

            atr = sum(true_ranges[-period:]) / min(period, len(true_ranges))
            return round(atr, 4)
        except Exception as e:
            logger.error(f"Failed to calculate ATR for {symbol}: {e}")
            return None

    def get_rsi(self, symbol: str, period: int = 14) -> float | None:
        """Calculate RSI from Alpaca market data."""
        if not self._data_client:
            return None
        try:
            end = datetime.now()
            start = end - timedelta(days=period * 4)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=period + 2,
            )
            request.feed = DataFeed.IEX
            bars = self._data_client.get_stock_bars(request)
            data = bars[symbol]
            if len(data) < period + 1:
                return None

            changes = [float(data[i].close) - float(data[i - 1].close) for i in range(1, len(data))]
            gains = [c if c > 0 else 0 for c in changes]
            losses = [-c if c < 0 else 0 for c in changes]

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return round(rsi, 2)
        except Exception as e:
            logger.error(f"Failed to calculate RSI for {symbol}: {e}")
            return None

    def _dry_run_account(self) -> dict:
        return {
            "equity": 100_000.0,
            "cash": 100_000.0,
            "buying_power": 200_000.0,
            "portfolio_value": 100_000.0,
            "pattern_day_trader": False,
        }

    def _dry_run_order(self, symbol: str, side: str, risk_params: RiskParams) -> dict:
        logger.info(
            f"[DRY RUN] Order: {symbol} {side} x{risk_params.position_size} "
            f"@ {risk_params.entry_price} | SL={risk_params.stop_loss} "
            f"TP={risk_params.take_profit}"
        )
        return {
            "order_id": "dry-run-001",
            "symbol": symbol,
            "side": side,
            "qty": risk_params.position_size,
            "status": "accepted",
            "stop_loss": risk_params.stop_loss,
            "take_profit": risk_params.take_profit,
            "dry_run": True,
        }
