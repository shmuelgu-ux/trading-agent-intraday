from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Indicators(BaseModel):
    rsi: Optional[float] = Field(None, ge=0, le=100)
    macd_signal: Optional[str] = None  # "bullish_cross", "bearish_cross", "neutral"
    ema_trend: Optional[str] = None  # "up", "down", "flat"
    atr: Optional[float] = Field(None, gt=0)
    volume_ratio: Optional[float] = Field(None, gt=0)


class TradingViewSignal(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    action: SignalAction
    price: float = Field(..., gt=0)
    timeframe: str = Field(default="1H")  # "5m", "15m", "1H", "4H", "1D"
    indicators: Indicators = Field(default_factory=Indicators)
