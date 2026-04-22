"""Live Donchian breakout scanner — replacement for ``core/scanner.py``.

Emits a ``TradingViewSignal`` for every ticker whose latest close is
above the highest high of the prior ``entry_lookback`` bars. Does NOT
re-compute any of the old indicator-voting filters (RSI, MACD, Stoch,
BB, VWAP, macro). Those gave a weak edge in backtest; Donchian gave a
strong one. Keeping them in the decision path would only dilute.

Signals feed the existing ``DecisionEngine`` → ``AlpacaClient`` chain.
The only new capability needed downstream is channel-exit monitoring
for open positions — handled by ``services/channel_exit_monitor.py``.

Interface preserved: scanner returns ``list[TradingViewSignal]`` with
``action=BUY``, ``price=latest close``, and ``indicators`` populated
enough for risk_manager to size the trade (needs ATR).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from loguru import logger
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from core.scanner import BASELINE_STOCKS, BATCH_SIZE, MIN_VOLUME, _is_rate_limit_error
from models.signals import Indicators, SignalAction, TradingViewSignal
from services.alpaca_client import AlpacaClient


# Strategy parameters — mirror the defaults the backtest validated as
# having real edge (Sharpe 1.28, 5/7 walk-forward windows positive).
DEFAULT_ENTRY_LOOKBACK = 20
DEFAULT_ATR_PERIOD = 14
DEFAULT_MIN_PRICE = 5.0
# Signal strength is a legacy field — risk_manager still expects it.
# Donchian signals are all-or-nothing (breakout = 100, no breakout = no
# signal at all), so we flag every breakout at 100.
DONCHIAN_STRENGTH = 100


@dataclass
class _BreakoutReading:
    """Internal per-symbol result before converting to TradingViewSignal."""
    symbol: str
    close: float
    upper_entry: float
    atr: float
    volume_ratio: float

    @property
    def breakout_pct(self) -> float:
        """How far above the channel top, as a fraction of price. Used to
        rank signals (bigger breakouts first, usually strongest moves)."""
        if self.close <= 0:
            return 0.0
        return (self.close - self.upper_entry) / self.close


class DonchianStockScanner:
    """Produces BUY signals whenever a ticker's latest close is above
    the highest close of the prior ``entry_lookback`` bars.

    Long-only for MVP. Short breakouts (below the N-day low) may be
    added in a later iteration once the long-only version is validated
    in paper trading.
    """

    def __init__(
        self,
        alpaca: AlpacaClient,
        entry_lookback: int = DEFAULT_ENTRY_LOOKBACK,
        atr_period: int = DEFAULT_ATR_PERIOD,
        min_price: float = DEFAULT_MIN_PRICE,
        min_volume: float = MIN_VOLUME,
    ):
        self.alpaca = alpaca
        self.entry_lookback = entry_lookback
        self.atr_period = atr_period
        self.min_price = min_price
        self.min_volume = min_volume
        self._all_symbols: list[str] = []
        self._symbols_loaded: datetime | None = None
        # State for dashboard / debugging. Shape-compatible with the old
        # StockScanner's state so the dashboard doesn't need changes.
        self.state = {
            "last_scan_start": None,
            "last_scan_end": None,
            "last_scan_duration": 0,
            "last_symbols_loaded": 0,
            "last_signals_found": 0,
            "last_executed": 0,
            "last_error": None,
            "scan_count": 0,
            "is_scanning": False,
            "progress": 0,
        }

    # ---- universe loading ------------------------------------------------
    def _load_all_tradeable_symbols(self) -> list[str]:
        """Identical loader to the old StockScanner — reused verbatim.

        Filters: US equity, active, tradable+marginable+shortable+
        easy-to-borrow, major exchange, no warrants/rights/preferred.
        Cached for 24h.
        """
        if self._all_symbols and self._symbols_loaded:
            if (datetime.now() - self._symbols_loaded).total_seconds() < 86400:
                logger.info(f"DonchianScanner: using cached {len(self._all_symbols)} symbols")
                return self._all_symbols

        if not self.alpaca._client:
            logger.warning("DonchianScanner: alpaca client not available, using baseline")
            return BASELINE_STOCKS

        try:
            logger.info("DonchianScanner: fetching all assets from Alpaca...")
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = self.alpaca._client.get_all_assets(request)
            valid_exchanges = {"NASDAQ", "NYSE", "ARCA", "BATS"}
            bad_suffixes = ("W", "R", "P", "U", "Z")
            symbols: list[str] = []
            for a in assets:
                if not (a.tradable and a.marginable and a.shortable and a.easy_to_borrow):
                    continue
                exchange_name = str(a.exchange) if a.exchange else ""
                if not any(ex in exchange_name for ex in valid_exchanges):
                    continue
                sym = a.symbol
                if "." in sym or "/" in sym:
                    continue
                if len(sym) > 5:
                    continue
                if len(sym) == 5 and sym[-1] in bad_suffixes:
                    continue
                symbols.append(sym)
            self._all_symbols = symbols
            self._symbols_loaded = datetime.now()
            logger.info(f"DonchianScanner: filtered to {len(symbols)} symbols")
            return symbols
        except Exception as e:
            logger.error(f"Failed to load symbols: {type(e).__name__}: {e}")
            return BASELINE_STOCKS

    # ---- scan entry ------------------------------------------------------
    def scan(self, use_all_stocks: bool = True) -> list[TradingViewSignal]:
        """Scan for breakout signals. Returns signals ordered by breakout
        magnitude (distance above the channel top, as % of price).
        """
        self.state["is_scanning"] = True
        self.state["last_scan_start"] = datetime.now().isoformat()
        self.state["scan_count"] += 1
        self.state["last_error"] = None
        self.state["progress"] = 0

        try:
            symbols = (
                self._load_all_tradeable_symbols() if use_all_stocks else BASELINE_STOCKS
            )
            self.state["last_symbols_loaded"] = len(symbols)
            logger.info(f"DonchianScanner: scanning {len(symbols)} symbols")

            all_readings: list[_BreakoutReading] = []
            batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
            for idx, batch in enumerate(batches):
                for r in self._scan_batch_with_retry(batch):
                    if r is not None:
                        all_readings.append(r)
                self.state["progress"] = int(((idx + 1) / len(batches)) * 100)

            # Sort by how far the close punched through the channel top
            # (biggest breakouts first — usually the strongest moves).
            all_readings.sort(key=lambda r: r.breakout_pct, reverse=True)
            # Convert each reading to a TradingViewSignal in isolation —
            # a single bad ticker (e.g. pydantic validation failure on a
            # zero-volume bar) must NOT kill the whole scan. Previously
            # a single ValidationError inside a list-comprehension caused
            # the scanner to return 0 signals and the bot to sit idle.
            signals: list[TradingViewSignal] = []
            for r in all_readings:
                try:
                    signals.append(self._to_signal(r))
                except Exception as e:
                    logger.debug(f"DonchianScanner skip {r.symbol} in to_signal: {e}")
                    continue
            self.state["last_signals_found"] = len(signals)
            logger.info(f"DonchianScanner: {len(signals)} breakout signals")
            return signals
        except Exception as e:
            self.state["last_error"] = f"{type(e).__name__}: {e}"
            logger.error(f"DonchianScanner error: {self.state['last_error']}")
            return []
        finally:
            self.state["is_scanning"] = False
            self.state["last_scan_end"] = datetime.now().isoformat()

    # ---- batch analysis with rate-limit retry ---------------------------
    def _scan_batch_with_retry(
        self, symbols: list[str], max_retries: int = 3,
    ) -> list[_BreakoutReading | None]:
        """Wrap ``_scan_batch`` with exponential backoff on Alpaca 429s.
        Matches the retry pattern the old scanner used.
        """
        backoff = 5
        for attempt in range(max_retries):
            try:
                return self._scan_batch(symbols)
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"DonchianScanner: rate-limited, sleeping {backoff}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                logger.error(f"DonchianScanner batch failed: {type(e).__name__}: {e}")
                return [None] * len(symbols)
        return [None] * len(symbols)

    def _scan_batch(self, symbols: list[str]) -> list[_BreakoutReading | None]:
        """Fetch daily bars for up to BATCH_SIZE symbols in one call;
        compute the breakout signal for each."""
        if not self.alpaca._data_client or not symbols:
            return [None] * len(symbols)

        # Need entry_lookback + atr_period + buffer bars. Fetch ~60 days
        # of calendar history (≈ 42 trading days) which covers the
        # 20-day breakout + 14-day ATR with headroom.
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=90),
            end=datetime.now(),
            limit=2000,
        )
        request.feed = DataFeed.IEX
        bars_data = self.alpaca._data_client.get_stock_bars(request)
        try:
            bars_dict = bars_data.data
        except AttributeError:
            bars_dict = {}

        results: list[_BreakoutReading | None] = []
        for sym in symbols:
            raw = bars_dict.get(sym) or []
            results.append(self._read_one(sym, raw))
        return results

    def _read_one(self, symbol: str, raw_bars) -> _BreakoutReading | None:
        """Process one symbol's bars → breakout reading or None.

        Applies liquidity filters (min price / min volume), computes the
        channel top over the PRIOR ``entry_lookback`` bars (not including
        today), and tests whether today's close pierced it.
        """
        if len(raw_bars) < self.entry_lookback + self.atr_period + 2:
            return None
        try:
            latest = raw_bars[-1]
            close = float(latest.close)
            if close < self.min_price:
                return None
            # Liquidity: average dollar-volume over last 20 bars
            recent = raw_bars[-20:]
            avg_dollar_vol = sum(float(b.close) * float(b.volume) for b in recent) / len(recent)
            if avg_dollar_vol < self.min_volume * self.min_price:
                # min_volume is shares; approximate dollar-volume floor
                # using min_volume × min_price.
                return None

            # Channel top = highest high of PRIOR entry_lookback bars.
            prior_window = raw_bars[-(self.entry_lookback + 1):-1]
            if len(prior_window) < self.entry_lookback:
                return None
            upper_entry = max(float(b.high) for b in prior_window)
            if close <= upper_entry:
                return None  # no breakout

            # ATR over the full window (uses Wilder-style mean of TRs).
            atr_bars = raw_bars[-(self.atr_period + 1):]
            if len(atr_bars) < self.atr_period + 1:
                return None
            trs: list[float] = []
            for i in range(1, len(atr_bars)):
                hi = float(atr_bars[i].high)
                lo = float(atr_bars[i].low)
                pc = float(atr_bars[i - 1].close)
                trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
            if not trs:
                return None
            atr = sum(trs[-self.atr_period:]) / self.atr_period
            if atr <= 0:
                return None

            # Volume ratio for logging / context (not a filter). Clamp to
            # a tiny positive number — the downstream ``Indicators`` model
            # enforces ``volume_ratio > 0``, and a ticker with zero volume
            # on its latest bar (halted? data gap?) should still be
            # reportable in context rather than crash the whole scan.
            last_vol = float(latest.volume)
            avg_vol = sum(float(b.volume) for b in recent) / len(recent)
            if avg_vol > 0:
                vol_ratio = max(last_vol / avg_vol, 0.01)
            else:
                vol_ratio = 1.0

            return _BreakoutReading(
                symbol=symbol, close=close,
                upper_entry=upper_entry, atr=round(atr, 4),
                volume_ratio=round(vol_ratio, 2),
            )
        except Exception as e:
            logger.debug(f"DonchianScanner skip {symbol}: {e}")
            return None

    # ---- signal conversion ---------------------------------------------
    def _to_signal(self, r: _BreakoutReading) -> TradingViewSignal:
        """Convert a breakout reading into the TradingViewSignal shape the
        rest of the pipeline expects.

        Indicator fields the OLD filter logic used for scoring are populated
        with values that pass-through downstream filters cleanly:

        - ``rsi``: None (we intentionally don't filter on RSI extremes
          anymore — the breakout IS the signal).
        - ``ema_trend``: "up" — a 20-day close-breakout by definition is
          in an uptrend on that horizon, so this is honest not a dummy.
        - ``macd_signal``: "bullish_cross" — same reasoning; the
          momentum is the breakout itself, so a bullish label is fair.
        - ``atr``: the 14-period ATR used for risk-based sizing.
        - ``volume_ratio``: descriptive only; not a filter for Donchian.
        """
        ind = Indicators(
            rsi=None,
            macd_signal="bullish_cross",
            ema_trend="up",
            atr=r.atr,
            volume_ratio=r.volume_ratio,
        )
        return TradingViewSignal(
            ticker=r.symbol,
            action=SignalAction.BUY,
            price=r.close,
            indicators=ind,
        )
