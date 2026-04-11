import time
from datetime import datetime, timedelta
from loguru import logger
from config import settings
from models.signals import TradingViewSignal, SignalAction, Indicators
from services.alpaca_client import AlpacaClient
from core.technical_analysis import analyze, Bar, AnalysisResult
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus


# Detects Alpaca rate-limit errors across the different ways they surface
# from the SDK (message text varies by transport / tier).
def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "ratelimit" in msg
    )


# Max symbols per Alpaca get_stock_bars request. 200 is the documented
# upper bound; going higher returns a 400. We send batches of this size
# instead of one call per symbol so a full market scan turns into ~24
# API calls instead of ~4,700.
BATCH_SIZE = 200

# ===== REAL INDEX CONSTITUENTS (updated April 2026) =====

# S&P 500 - full list from stockanalysis.com
SP500 = [
    "NVDA","GOOGL","GOOG","AAPL","MSFT","AMZN","AVGO","META","TSLA","BRK.B",
    "WMT","LLY","JPM","XOM","V","JNJ","MU","COST","MA","NFLX","ORCL","CVX",
    "AMD","ABBV","BAC","CAT","PG","PLTR","HD","KO","CSCO","GE","LRCX","AMAT",
    "MRK","INTC","MS","UNH","GS","RTX","WFC","GEV","PM","LIN","IBM","KLAC",
    "MCD","TMUS","AXP","C","PEP","VZ","NEE","T","TXN","AMGN","TMO","ANET",
    "ABT","TJX","DIS","GILD","BA","ADI","SCHW","APH","DE","ISRG","BLK","CRM",
    "PFE","COP","ETN","UNP","HON","UBER","LMT","BX","BKNG","WELL","GLW",
    "PANW","DHR","LOW","QCOM","APP","PLD","SYK","SPGI","CB","NEM","PH","BMY",
    "DELL","COF","ACN","PGR","WDC","MDT","HCA","VRTX","MO","STX","SBUX",
    "CME","SO","CRWD","INTU","MCK","CEG","DUK","NOW","CVS","CMCSA","EQIX",
    "HWM","TT","NOC","ADBE","ICE","GD","FCX","WM","BSX","MAR","WMB","FDX",
    "PNC","BK","PWR","USB","UPS","JCI","KKR","SHW","CMI","AMT","ADP","EMR",
    "CDNS","MCO","REGN","ABNB","SNPS","CSX","MMM","SLB","ORLY","ITW","ECL",
    "CRH","RCL","MDLZ","CVNA","EOG","MSI","SPG","MNST","KMI","CI","AEP",
    "HLT","ROST","VLO","AON","ELV","CTAS","DASH","GM","TDG","CL","WBD",
    "MPC","LHX","PSX","RSG","APD","NSC","TEL","PCAR","DLR","HOOD","SRE",
    "MPWR","TRV","NKE","COR","BKR","APO","FTNT","TFC","OXY","O","AFL","AZO",
    "AJG","CTVA","TER","TGT","D","FAST","ALL","OKE","KEYS","GWW","FIX",
    "AME","VST","ETR","FANG","TRGP","NXPI","EA","PSA","XEL","ADSK","CAH",
    "ZTS","EXC","NDAQ","F","CARR","GRMN","MET","URI","IDXX","EW","COIN",
    "WAB","BDX","FITB","YUM","DAL","CMG","ROK","KR","EBAY","ODFL","HSY",
    "NUE","AIG","DHI","PYPL","PEG","DDOG","CBRE","AMP","ED","MSCI","PCG",
    "VTR","CCL","HIG","MCHP","VMC","WEC","LYV","STT","MLM","EQT","TTWO",
    "CCI","ROP","LVS","SYY","EME","KDP","ACGL","ADM","ARES","NRG","GEHC",
    "PRU","RMD","HBAN","KVUE","IR","HPE","A","MTB","IBKR","PAYX","KMB",
    "CPRT","IRM","HAL","AXON","ATO","CHTR","UAL","WAT","AEE","XYL","CBOE",
    "DTE","OTIS","TPL","WDAY","TDY","TPR","EXR","JBL","FISV","DVN","FE",
    "VICI","PPL","DOV","CTSH","RJF","EXPE","IQV","EIX","CNP","DOW","NTRS",
    "HUBB","WTW","KHC","STLD","CFG","DG","ON","MTD","AWK","BIIB","ROL",
    "ES","STZ","FICO","EL","FOXA","CINF","CTRA","DXCM","WRB","SYF","VRSN",
    "PPG","BG","FOX","CMS","FIS","LYB","HUM","TSCO","AVB","EQR","RF",
    "ULTA","SBAC","PHM","NI","BRO","VRSK","KEY","TSN","RL","LH","L","DRI",
    "WSM","CHD","EFX","VLTO","STE","DGX","LEN","OMC","FSLR","JBHT","ALB",
    "DLTR","CPAY","MRNA","PFG","CHRW","LDOS","TROW","LUV","SNA","NTAP",
    "IP","AMCR","DD","GIS","EXPD","CF","EVRG","WST","INCY","LNT","IFF",
    "BR","NVR","PKG","LULU","CNC","FTV","ZBH","WY","GPN","HPQ","PTC",
    "AKAM","ESS","LII","BALL","CSGP","CDW","HII","TXT","FFIV","VTRS",
    "TRMB","INVH","J","NDSN","KIM","DECK","MAA","GPC","IEX","PNR","REG",
    "NWS","PODD","TYL","SMCI","COO","HST","APA","NWSA","MKC","AVY","BBY",
    "EG","ERIE","HAS","BEN","APTV","CLX","MAS","ALGN","DPZ","UDR","PNW",
    "ALLE","GNRC","GL","GEN","UHS","JKHY","SOLV","AIZ","SWK","WYNN","CPT",
    "GDDY","IVZ","ZBRA","IT","AES","RVTY","SJM","DVA","TTD","MGM","FRT",
    "BXP","AOS","NCLH","BLDR","BAX","HSIC","TECH","CRL","SWKS","MOS","TAP",
    "FDS","ARE","POOL","MOH","MTCH","CAG","EPAM","CPB","LW","PAYC","HRL",
]

# ARKK - ARK Innovation ETF holdings
ARKK = [
    "CRSP","TEM","SHOP","CRCL","BEAM","TWST","TXG","NTLA","ACHR",
    "ILMN","ROKU","RBLX",
]

# IWM notable (Russell 2000 high-volume names not in S&P 500)
IWM_EXTRA = [
    "MARA","RIOT","PLUG","FCEL","BLNK","CHPT","QS","LAZR","IONQ","RGTI",
    "SOUN","BBAI","AFRM","UPST","OPEN","AI","NET","OKTA","CFLT","GTLB",
    "ESTC","MNDY","CELH","HIMS","PTON","CHWY","ETSY","W","PINS","SNAP",
    "MGNI","CARG","JOBY","ASTS","SOFI","SQ","PATH","DKNG","U","ZM",
    "DNA","IOVA","VERV","RIVN","NIO","BABA","JD","LCID",
]

# ETFs
ETFS = ["SPY", "QQQ", "IWM", "DIA", "ARKK"]

# Combine hardcoded lists as baseline
BASELINE_STOCKS = sorted(set(SP500 + ARKK + IWM_EXTRA + ETFS))

# Minimum average daily volume to consider a stock tradeable
MIN_VOLUME = 500_000


class StockScanner:
    """Scans all tradeable US stocks for swing trading opportunities."""

    def __init__(self, alpaca: AlpacaClient):
        self.alpaca = alpaca
        self._all_symbols: list[str] = []
        self._symbols_loaded: datetime | None = None
        # State tracking for debugging
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

    def _load_all_tradeable_symbols(self) -> list[str]:
        """Load all HIGH QUALITY tradeable US equities from Alpaca, cached 24h.

        Strict filtering:
        - US Equity only
        - Active
        - Tradable
        - Marginable (indicates liquidity + institutional acceptance)
        - Shortable (required for our SELL signals)
        - Easy to borrow (indicates deep liquidity)
        - NASDAQ or NYSE only (no OTC)
        - Max 5 chars, no special suffixes (W=warrant, R=right, etc.)
        """
        if self._all_symbols and self._symbols_loaded:
            if (datetime.now() - self._symbols_loaded).total_seconds() < 86400:
                logger.info(f"Scanner: using cached {len(self._all_symbols)} symbols")
                return self._all_symbols

        if not self.alpaca._client:
            logger.warning("Scanner: alpaca client not available, using baseline")
            return BASELINE_STOCKS

        try:
            logger.info("Scanner: fetching all assets from Alpaca...")
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = self.alpaca._client.get_all_assets(request)
            logger.info(f"Scanner: received {len(assets)} active US equity assets")

            # Valid exchanges
            valid_exchanges = {"NASDAQ", "NYSE", "ARCA", "BATS"}
            # Suffixes to exclude (warrants, rights, preferred, etc.)
            bad_suffixes = ("W", "R", "P", "U", "Z")

            symbols = []
            for a in assets:
                # Must be tradable + marginable + shortable + easy to borrow
                if not (a.tradable and a.marginable and a.shortable and a.easy_to_borrow):
                    continue
                # Valid exchange
                exchange_name = str(a.exchange) if a.exchange else ""
                if not any(ex in exchange_name for ex in valid_exchanges):
                    continue
                # Clean ticker symbol
                sym = a.symbol
                if "." in sym or "/" in sym:
                    continue
                if len(sym) > 5:
                    continue
                # Exclude warrants, rights, preferred
                if len(sym) == 5 and sym[-1] in bad_suffixes:
                    continue
                symbols.append(sym)

            self._all_symbols = symbols
            self._symbols_loaded = datetime.now()
            logger.info(f"Scanner: filtered to {len(symbols)} high-quality symbols")
            return symbols
        except Exception as e:
            logger.error(f"Failed to load symbols: {type(e).__name__}: {e}")
            logger.warning("Falling back to baseline stocks")
            return BASELINE_STOCKS

    def scan(self, use_all_stocks: bool = True) -> list[TradingViewSignal]:
        """Scan stocks for opportunities, sorted by strength (best first).

        Args:
            use_all_stocks: If True, scans ALL tradeable US equities (~4700).
                           If False, scans baseline 556 index stocks.
        """
        self.state["is_scanning"] = True
        self.state["last_scan_start"] = datetime.now().isoformat()
        self.state["scan_count"] += 1
        self.state["last_error"] = None
        self.state["progress"] = 0

        try:
            if use_all_stocks:
                logger.info("Scanner: loading all tradeable symbols...")
                symbols = self._load_all_tradeable_symbols()
            else:
                symbols = BASELINE_STOCKS

            self.state["last_symbols_loaded"] = len(symbols)
            logger.info(f"Scanner: starting scan of {len(symbols)} stocks...")

            if not symbols:
                self.state["last_error"] = "No symbols loaded"
                logger.error("Scanner: no symbols to scan!")
                return []

            results: list[AnalysisResult] = []
            scanned = 0
            errors = 0
            rate_limit_hits = 0
            start_time = datetime.now()

            # Batched scanning: one Alpaca call per BATCH_SIZE symbols.
            # At BATCH_SIZE=200 a full 4,700-symbol universe becomes 24
            # API calls instead of 4,700 — ~20x faster end to end.
            for i in range(0, len(symbols), BATCH_SIZE):
                batch = symbols[i:i + BATCH_SIZE]

                # Retry the whole batch on rate-limit errors
                attempt = 0
                batch_results: dict[str, AnalysisResult | None] = {}
                while True:
                    try:
                        batch_results = self._full_analysis_batch(batch)
                        break
                    except Exception as e:
                        if _is_rate_limit_error(e) and attempt < 3:
                            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                            rate_limit_hits += 1
                            logger.warning(
                                f"Scanner batch rate limit hit (symbols {i}-{i+len(batch)}) "
                                f"attempt {attempt + 1}/3, backing off {wait}s"
                            )
                            time.sleep(wait)
                            attempt += 1
                            continue
                        # Non-rate-limit failure: log once and skip the whole batch
                        errors += len(batch)
                        if errors <= len(batch) * 3:
                            logger.warning(
                                f"Scanner batch failed (symbols {i}-{i+len(batch)}): "
                                f"{type(e).__name__}: {e}"
                            )
                        batch_results = {}
                        break

                # Accumulate results + count
                scanned += len(batch)
                for result in batch_results.values():
                    if result:
                        results.append(result)

                # Progress log after every batch
                self.state["progress"] = scanned
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Scanner progress: {scanned}/{len(symbols)} "
                    f"({elapsed:.0f}s elapsed, {len(results)} valid so far, "
                    f"batch {i // BATCH_SIZE + 1}/{(len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE})"
                )

            # Sort by strength score (best signals first)
            scored = [(r, self._to_signal(r)) for r in results if r.signal != "NONE"]
            scored.sort(key=lambda x: x[0].strength, reverse=True)
            signals = [s for _, s in scored]

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Scanner DONE: {scanned}/{len(symbols)} scanned in {elapsed:.0f}s, "
                f"{len(signals)} signals found, {errors} errors, "
                f"{rate_limit_hits} rate-limit retries"
            )
            for r, s in scored[:15]:
                logger.info(f"  -> [{r.strength}/100] {s.action.value} {s.ticker} @ {s.price}")

            self.state["last_signals_found"] = len(signals)
            self.state["last_scan_duration"] = elapsed
            self.state["last_scan_end"] = datetime.now().isoformat()
            return signals

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.state["last_error"] = error_msg
            logger.error(f"Scanner CRASHED: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            self.state["is_scanning"] = False

    def _full_analysis(self, symbol: str) -> AnalysisResult | None:
        """Run full technical analysis on a symbol using intraday bars.

        Filters low-quality names (price < $5, thin volume). The bar
        timeframe and lookback window are both driven by config so the
        intraday timeframe (default: 15m over 10 days) can be tuned
        without touching this method.
        """
        if not self.alpaca._data_client:
            return None

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(settings.scanner_bar_minutes, TimeFrameUnit.Minute),
                start=datetime.now() - timedelta(days=settings.scanner_lookback_days),
                end=datetime.now(),
                limit=1000,
            )
            request.feed = DataFeed.IEX
            bars_data = self.alpaca._data_client.get_stock_bars(request)
            raw_bars = bars_data[symbol]

            # Need enough bars for EMA(50) + context; 60 gives a comfortable margin
            if len(raw_bars) < 60:
                return None

            # Quality filters on recent data
            current_price = float(raw_bars[-1].close)
            if current_price < 5.0:  # No penny stocks
                return None

            # Volume check — intraday bars have much lower per-bar volume than
            # daily bars, so use the sum over the last full trading day
            # (~26 bars of 15 minutes) and require decent liquidity.
            bars_per_day = max(1, (6 * 60 + 30) // settings.scanner_bar_minutes)
            recent = raw_bars[-bars_per_day:] if len(raw_bars) >= bars_per_day else raw_bars
            day_volume = sum(float(b.volume) for b in recent)
            if day_volume < 500_000:  # ~500k shares per day minimum
                return None

            from core.technical_analysis import Bar
            bars = [
                Bar(
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume),
                )
                for b in raw_bars
            ]

            return analyze(symbol, bars)

        except Exception as e:
            logger.debug(f"Scanner skip {symbol}: {e}")
            return None

    def _full_analysis_batch(
        self, symbols: list[str]
    ) -> dict[str, AnalysisResult | None]:
        """Run full technical analysis on a BATCH of symbols in a single
        Alpaca API call. Returns a dict keyed by symbol — values are
        either a full AnalysisResult or None for symbols that didn't
        pass the quality filters / had insufficient data / errored.

        Raises on transport-level or rate-limit errors so the caller
        can retry the whole batch.
        """
        if not self.alpaca._data_client or not symbols:
            return {s: None for s in symbols}

        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame(settings.scanner_bar_minutes, TimeFrameUnit.Minute),
            start=datetime.now() - timedelta(days=settings.scanner_lookback_days),
            end=datetime.now(),
            limit=10000,  # per-symbol ceiling, generous for safety
        )
        request.feed = DataFeed.IEX

        # NOTE: any exception here propagates up — scan() handles
        # rate-limit retry at the batch level.
        bars_data = self.alpaca._data_client.get_stock_bars(request)

        # The alpaca-py BarSet supports both `.data[symbol]` (dict) and
        # `bars_data[symbol]` (via __getitem__). Some symbols in the
        # requested batch may be missing entirely if Alpaca had no bars.
        try:
            bars_dict = bars_data.data  # dict[str, list[Bar]]
        except AttributeError:
            bars_dict = {}

        results: dict[str, AnalysisResult | None] = {}
        bars_per_day = max(1, (6 * 60 + 30) // settings.scanner_bar_minutes)

        for symbol in symbols:
            raw_bars = bars_dict.get(symbol) or []
            try:
                # Need enough bars for EMA(50) + context
                if len(raw_bars) < 60:
                    results[symbol] = None
                    continue

                current_price = float(raw_bars[-1].close)
                if current_price < 5.0:  # No penny stocks
                    results[symbol] = None
                    continue

                # Volume check — last full trading day worth of bars
                recent = (
                    raw_bars[-bars_per_day:]
                    if len(raw_bars) >= bars_per_day
                    else raw_bars
                )
                day_volume = sum(float(b.volume) for b in recent)
                if day_volume < 500_000:
                    results[symbol] = None
                    continue

                from core.technical_analysis import Bar
                bars = [
                    Bar(
                        open=float(b.open),
                        high=float(b.high),
                        low=float(b.low),
                        close=float(b.close),
                        volume=float(b.volume),
                    )
                    for b in raw_bars
                ]
                results[symbol] = analyze(symbol, bars)
            except Exception as e:
                # Per-symbol analysis failure — don't fail the whole batch
                logger.debug(f"Batch skip {symbol}: {e}")
                results[symbol] = None

        return results

    def _to_signal(self, result: AnalysisResult) -> TradingViewSignal:
        """Convert analysis result to a trading signal."""
        action = SignalAction.BUY if result.signal == "BUY" else SignalAction.SELL

        return TradingViewSignal(
            ticker=result.ticker,
            action=action,
            price=result.price,
            timeframe=f"{settings.scanner_bar_minutes}m",
            indicators=Indicators(
                rsi=result.rsi,
                atr=result.atr,
                ema_trend=result.ema_trend,
                macd_signal=result.macd_signal,
                volume_ratio=result.volume_ratio,
            ),
        )
