from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional
from .orders import RiskParams


class TradeJournalEntry(BaseModel):
    id: Optional[int] = None
    ticker: str
    side: str
    action_taken: str  # "EXECUTED" or "REJECTED"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[int] = None
    risk_params: Optional[RiskParams] = None
    signal_data: dict = Field(default_factory=dict)
    reasoning: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
