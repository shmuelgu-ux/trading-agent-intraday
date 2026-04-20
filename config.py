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

    # Risk Parameters — intraday
    # Smaller per-trade risk because intraday runs many more trades than
    # swing; a 20% daily loss cap allows ~20 losing trades before we stop.
    max_risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.05)
    max_total_risk: float = Field(default=0.20, ge=0.01, le=0.50)
    max_open_positions: int = Field(default=20, ge=1, le=30)
    default_rr_ratio: float = Field(default=1.5, ge=1.0, le=5.0)

    # ATR-based Stop Loss multiplier — 1.5x ATR gives the trade room
    # to breathe past normal intraday noise. At 1.0x, 80% of trades
    # were getting stopped out by single-candle noise before reaching TP.
    atr_sl_multiplier: float = Field(default=1.5, ge=0.5, le=3.0)

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
