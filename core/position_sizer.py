import math


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly Criterion for optimal position sizing.

    Returns the fraction of account to risk.
    Capped at 0.25 (25%) to prevent over-leverage.
    """
    if avg_loss == 0:
        return 0.0

    b = avg_win / avg_loss  # win/loss ratio
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b
    # Half-Kelly is safer in practice
    half_kelly = kelly / 2
    return max(0.0, min(half_kelly, 0.25))


def volatility_adjusted_size(
    account_balance: float,
    entry_price: float,
    atr: float,
    risk_pct: float = 0.02,
    atr_multiplier: float = 1.5,
) -> int:
    """Position size adjusted for current volatility via ATR."""
    risk_amount = account_balance * risk_pct
    risk_per_share = atr * atr_multiplier

    if risk_per_share <= 0:
        return 0

    shares = math.floor(risk_amount / risk_per_share)
    max_affordable = math.floor(account_balance / entry_price)
    return min(shares, max_affordable)
