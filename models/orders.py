from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional


class DecisionAction(str, Enum):
    EXECUTE = "EXECUTE"
    REJECT = "REJECT"


class RiskParams(BaseModel):
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int  # number of shares
    risk_amount: float  # dollar amount at risk
    risk_percent: float  # % of account at risk
    reward_risk_ratio: float


class TradeDecision(BaseModel):
    action: DecisionAction
    ticker: str
    side: Optional[str] = None  # "buy" or "sell"
    risk_params: Optional[RiskParams] = None
    reasoning: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_reason(self, reason: str) -> None:
        self.reasoning.append(reason)
