"""Backtest CLI.

Two modes:

  # Populate the cache (one-time, or to extend the range)
  python -m backtest.cli fetch \
      --tickers AAPL,MSFT,NVDA,... \
      --start 2022-01-01 --end 2024-12-31

  # Run a single backtest on cached data
  python -m backtest.cli run \
      --tickers AAPL,MSFT,NVDA \
      --start 2023-01-01 --end 2024-12-31 \
      --capital 2000

  # Walk-forward (train=365, test=90 days, non-overlapping)
  python -m backtest.cli walkforward \
      --tickers AAPL,MSFT \
      --start 2021-01-01 --end 2024-12-31 \
      --train-days 365 --test-days 90

The ``fetch`` command requires live Alpaca creds (``ALPACA_API_KEY`` +
``ALPACA_SECRET_KEY`` in the env — same as the live bot). ``run`` and
``walkforward`` are pure offline once the cache is populated.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime

from backtest.data_loader import (
    AlpacaBarFetcher,
    BarCache,
    DEFAULT_CACHE_PATH,
    populate_cache,
)
from backtest.models import BacktestConfig, BacktestResult
from backtest.simulator import Backtester
from backtest.walk_forward import WalkForwardResult, run_walk_forward, split_windows


# ---------- helpers -----------------------------------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def _format_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:+.2f}%"


def _format_num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _print_result(title: str, result: BacktestResult) -> None:
    print(f"\n== {title} ==")
    print(f"  window       : {result.config.start} → {result.config.end}")
    print(f"  capital      : ${result.config.initial_capital:,.2f}")
    print(f"  trades closed: {len(result.closed_trades)}  "
          f"(wins={result.win_count}, losses={result.loss_count}, "
          f"win_rate={result.win_rate:.0%})")
    print(f"  total PnL    : ${result.total_pnl:,.2f}")
    print(f"  total return : {_format_pct(result.total_return_pct)}")
    print(f"  annualised   : {_format_pct(result.annualized_return_pct)}")
    print(f"  Sharpe       : {_format_num(result.sharpe)}")
    print(f"  Sortino      : {_format_num(result.sortino)}")
    print(f"  max drawdown : {_format_pct(result.max_drawdown_pct)}")
    print(f"  Calmar       : {_format_num(result.calmar)}")
    print(f"  sample size  : {result.sample_size} daily closes")


def _print_walk_forward(wf: WalkForwardResult) -> None:
    if not wf.windows:
        print("No windows fit the given date range + train/test sizes.")
        return
    print(f"\nWalk-Forward: {len(wf.windows)} test windows\n")
    print(f"  {'test window':<28} {'return':>10} {'sharpe':>8} {'maxDD':>10} {'trades':>8}")
    print("  " + "-" * 66)
    for w, r in zip(wf.windows, wf.per_window):
        ret = _format_pct(r.total_return_pct)
        sh = _format_num(r.sharpe)
        dd = _format_pct(-r.max_drawdown_pct) if r.max_drawdown_pct is not None else "—"
        print(f"  {w.test_start.isoformat()} → {w.test_end.isoformat()}  "
              f"{ret:>10} {sh:>8} {dd:>10} {len(r.closed_trades):>8}")
    print()
    print(f"  avg Sharpe       : {_format_num(wf.avg_sharpe)}")
    print(f"  avg Sortino      : {_format_num(wf.avg_sortino)}")
    print(f"  worst drawdown   : {_format_pct(wf.worst_drawdown_pct)}")
    print(f"  consistently +ve : {wf.consistent}")


# ---------- subcommands -------------------------------------------------

def _cmd_fetch(args) -> int:
    """Populate the cache with historical bars for the given tickers."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        print("ERROR: set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars", file=sys.stderr)
        return 2

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret)
    fetcher = AlpacaBarFetcher(client, feed=DataFeed.IEX)
    cache = BarCache(args.cache)

    tickers = _parse_tickers(args.tickers)
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    print(f"Fetching {len(tickers)} tickers from {start} to {end} → {args.cache}")
    total_written = 0
    for t in tickers:
        written = populate_cache(
            fetcher, cache, t, "day", TimeFrame.Day, start, end,
            force_refetch=args.force,
        )
        first, last, count = cache.coverage(t, "day")
        print(f"  {t}: +{written} new, total {count} bars "
              f"({first} → {last})" if count else f"  {t}: NO DATA (delisted or bad ticker)")
        total_written += written
    print(f"\nDone. Wrote {total_written} new bars.")
    return 0


def _cmd_run(args) -> int:
    """Single-window backtest."""
    cache = BarCache(args.cache)
    tickers = _parse_tickers(args.tickers)
    config = BacktestConfig(
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        initial_capital=args.capital,
        universe=tickers,
        max_risk_per_trade=args.risk_per_trade,
        max_total_risk=args.total_risk,
        max_open_positions=args.max_positions,
        default_rr_ratio=args.rr,
        atr_sl_multiplier=args.atr_mult,
        min_signal_strength=args.min_strength,
        commission_per_share=args.commission,
        max_hold_days=args.max_hold,
    )
    result = Backtester(cache, config).run()
    _print_result("Backtest Result", result)
    return 0


def _cmd_walkforward(args) -> int:
    """Rolling-window walk-forward run."""
    cache = BarCache(args.cache)
    tickers = _parse_tickers(args.tickers)
    total_start = _parse_date(args.start)
    total_end = _parse_date(args.end)
    windows = split_windows(
        total_start=total_start,
        total_end=total_end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    if not windows:
        print("No windows fit. Try shorter train/test periods or a wider date range.")
        return 1
    base = BacktestConfig(
        start=total_start,  # overridden per-window
        end=total_end,
        initial_capital=args.capital,
        universe=tickers,
        max_risk_per_trade=args.risk_per_trade,
        max_total_risk=args.total_risk,
        max_open_positions=args.max_positions,
        default_rr_ratio=args.rr,
        atr_sl_multiplier=args.atr_mult,
        min_signal_strength=args.min_strength,
        commission_per_share=args.commission,
        max_hold_days=args.max_hold,
    )
    wf = run_walk_forward(cache, base, windows)
    _print_walk_forward(wf)
    return 0


# ---------- arg parser --------------------------------------------------

def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tickers", required=True, help="Comma-separated symbols, e.g. AAPL,MSFT,NVDA")
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--capital", type=float, default=2000.0, help="Initial capital (default: 2000)")
    p.add_argument("--cache", default=DEFAULT_CACHE_PATH, help="SQLite cache path")
    p.add_argument("--risk-per-trade", type=float, default=0.03)
    p.add_argument("--total-risk", type=float, default=0.20)
    p.add_argument("--max-positions", type=int, default=20)
    p.add_argument("--rr", type=float, default=2.0, help="Risk:reward ratio")
    p.add_argument("--atr-mult", type=float, default=1.5)
    p.add_argument("--min-strength", type=int, default=45, help="Minimum signal strength 0-100")
    p.add_argument("--commission", type=float, default=0.0035)
    p.add_argument("--max-hold", type=int, default=60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Historical backtest engine for the swing trading bot",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Populate the bar cache from Alpaca")
    p_fetch.add_argument("--tickers", required=True)
    p_fetch.add_argument("--start", required=True)
    p_fetch.add_argument("--end", required=True)
    p_fetch.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    p_fetch.add_argument("--force", action="store_true", help="Re-fetch even if cache already covers the range")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_run = sub.add_parser("run", help="Run a single-window backtest")
    _add_common_run_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_wf = sub.add_parser("walkforward", help="Run a walk-forward validation")
    _add_common_run_args(p_wf)
    p_wf.add_argument("--train-days", type=int, default=365)
    p_wf.add_argument("--test-days", type=int, default=90)
    p_wf.add_argument("--step-days", type=int, default=None,
                      help="Days between consecutive test windows (defaults to test-days)")
    p_wf.set_defaults(func=_cmd_walkforward)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
