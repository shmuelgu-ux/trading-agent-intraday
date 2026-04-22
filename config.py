from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Intraday day-trading agent settings.

    Values are tuned for the day-trading pipeline: higher frequency of
    trades per day, tighter stops (ATR * 1.0), smaller per-trade risk
    (1%) so a losing streak of several trades doesn't blow through the
    20% daily budget, and a mandatory close-all cutoff before the bell.
    """

    # Alpaca (new paper account, different from the swing agent)
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Capital Management
    max_capital: float = Field(default=2000.0, gt=0)

    # Risk Parameters — intraday Donchian breakout.
    # 1% per-trade risk — the 20% daily cap allows ~20 losing trades
    # before the risk budget closes the day.
    max_risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.05)
    max_total_risk: float = Field(default=0.20, ge=0.01, le=0.50)
    max_open_positions: int = Field(default=20, ge=1, le=30)
    # High default so the bracket order's take-profit leg is effectively
    # out of reach — the Donchian intraday strategy doesn't use a fixed
    # RR target; the real exit is EOD force-close at 15:55 ET.
    # NOTE: this line was supposed to ship in PR #5 (live wiring) but
    # was left out of the commit by mistake. That's why the intraday bot
    # was running the new scanner with the OLD tight 1.5:1 RR for its
    # first two days, causing every position to fall into either a
    # close TP hit or an EOD noise exit — 18% win rate on ~22 closed
    # trades vs. the ~45% seen in backtest. This PR corrects that.
    default_rr_ratio: float = Field(default=20.0, ge=1.0, le=100.0)

    # ATR-based Stop Loss multiplier — 1.5x ATR matches the backtest-
    # validated Donchian intraday sweet spot. (le bumped from 3.0 so
    # env-overrides can experiment with wider stops if needed.)
    atr_sl_multiplier: float = Field(default=1.5, ge=0.5, le=5.0)

    # Scanner — 15-minute bars, scan every few minutes during market hours
    scanner_enabled: bool = True
    scanner_bar_minutes: int = Field(default=15, ge=1, le=60)  # timeframe of each candle
    scanner_lookback_days: int = Field(default=10, ge=3, le=60)  # how much history to fetch
    scanner_interval_seconds: int = Field(default=300, ge=60, le=900)  # time between scans

    # Intraday session guardrails (all times in Eastern Time)
    # No new trades after this time — too close to the bell to be worth it.
    intraday_no_new_trades_hour: int = Field(default=15, ge=9, le=16)
    intraday_no_new_trades_minute: int = Field(default=0, ge=0, le=59)
    # Force-close every open position at this time — safety net so nothing
    # ever rolls over to the next day.
    intraday_force_close_hour: int = Field(default=15, ge=9, le=16)
    intraday_force_close_minute: int = Field(default=55, ge=0, le=59)

    # Database (PostgreSQL on Railway, SQLite locally)
    database_url: str = "sqlite+aiosqlite:///./trading_agent_intraday.db"
    database_private_url: str = ""  # Railway sets this for PostgreSQL

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
