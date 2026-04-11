"""Advanced technical analysis for swing trading.

Indicators used:
- EMA 9/21/50 (trend direction + crossovers)
- RSI 14 (momentum + divergences)
- MACD 12/26/9 (momentum confirmation)
- ATR 14 (volatility for SL/TP sizing)
- Volume ratio (confirmation of moves)
- Bollinger Bands 20/2 (mean reversion + squeeze)
- VWAP (institutional price level)
- Stochastic RSI (refined overbought/oversold)
"""

from dataclasses import dataclass
from loguru import logger


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class AnalysisResult:
    """Full technical analysis output."""
    ticker: str
    price: float
    signal: str  # "BUY", "SELL", or "NONE"
    strength: int  # 0-100 confidence score
    reasons: list[str]

    # Indicators
    rsi: float
    atr: float
    ema_trend: str  # "up", "down", "flat"
    macd_signal: str  # "bullish_cross", "bearish_cross", "neutral"
    volume_ratio: float
    bb_position: str  # "above_upper", "below_lower", "middle"
    stoch_rsi: float


def calculate_ema(closes: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    if len(closes) < period:
        return []
    multiplier = 2 / (period + 1)
    ema = [sum(closes[:period]) / period]
    for price in closes[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema


def calculate_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [c if c > 0 else 0 for c in changes[-period:]]
    losses = [-c if c < 0 else 0 for c in changes[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def calculate_atr(bars: list[Bar], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        trs.append(tr)
    return round(sum(trs[-period:]) / period, 4)


def calculate_macd(closes: list[float]) -> tuple[float, float, str]:
    """Returns (macd_line, signal_line, signal_type)."""
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)
    if not ema12 or not ema26:
        return 0, 0, "neutral"

    # Align lengths
    diff = len(ema12) - len(ema26)
    macd_line = [ema12[diff + i] - ema26[i] for i in range(len(ema26))]
    signal_line = calculate_ema(macd_line, 9)
    if not signal_line:
        return 0, 0, "neutral"

    ml = macd_line[-1]
    sl = signal_line[-1]
    prev_ml = macd_line[-2] if len(macd_line) > 1 else ml
    prev_sl = signal_line[-2] if len(signal_line) > 1 else sl

    if prev_ml <= prev_sl and ml > sl:
        signal = "bullish_cross"
    elif prev_ml >= prev_sl and ml < sl:
        signal = "bearish_cross"
    elif ml > sl:
        signal = "bullish"
    elif ml < sl:
        signal = "bearish"
    else:
        signal = "neutral"

    return round(ml, 4), round(sl, 4), signal


def calculate_bollinger(closes: list[float], period: int = 20, std_dev: float = 2.0) -> tuple[float, float, float, str]:
    """Returns (upper, middle, lower, position)."""
    if len(closes) < period:
        return 0, 0, 0, "middle"

    sma = sum(closes[-period:]) / period
    variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
    std = variance ** 0.5
    upper = sma + std_dev * std
    lower = sma - std_dev * std

    price = closes[-1]
    if price > upper:
        position = "above_upper"
    elif price < lower:
        position = "below_lower"
    elif price > sma:
        position = "upper_half"
    else:
        position = "lower_half"

    return round(upper, 2), round(sma, 2), round(lower, 2), position


def calculate_stoch_rsi(closes: list[float], period: int = 14) -> float | None:
    """Stochastic RSI - more sensitive than regular RSI."""
    if len(closes) < period * 2 + 1:
        return None

    # Calculate RSI series
    rsi_values = []
    for i in range(period + 1, len(closes) + 1):
        r = calculate_rsi(closes[:i], period)
        if r is not None:
            rsi_values.append(r)

    if len(rsi_values) < period:
        return None

    recent = rsi_values[-period:]
    rsi_min = min(recent)
    rsi_max = max(recent)
    if rsi_max == rsi_min:
        return 50.0

    stoch = ((rsi_values[-1] - rsi_min) / (rsi_max - rsi_min)) * 100
    return round(stoch, 2)


def calculate_volume_ratio(volumes: list[float], period: int = 20) -> float:
    """Current volume vs average volume."""
    if len(volumes) < period + 1:
        return 1.0
    avg = sum(volumes[-period - 1:-1]) / period
    if avg == 0:
        return 1.0
    return round(volumes[-1] / avg, 2)


def calculate_vwap(bars: list[Bar]) -> float | None:
    """Volume Weighted Average Price."""
    if not bars:
        return None
    total_vol = sum(b.volume for b in bars)
    if total_vol == 0:
        return None
    vwap = sum(((b.high + b.low + b.close) / 3) * b.volume for b in bars) / total_vol
    return round(vwap, 2)


def analyze(ticker: str, bars: list[Bar]) -> AnalysisResult | None:
    """Full swing trading analysis on a stock.

    Scoring system (0-100):
    - EMA trend alignment: +20
    - MACD confirmation: +20
    - RSI in favorable zone: +15
    - Stochastic RSI confirmation: +10
    - Bollinger Band position: +15
    - Volume confirmation: +10
    - Price above/below VWAP: +10
    """
    if len(bars) < 50:
        return None

    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    price = closes[-1]

    # Calculate all indicators
    rsi = calculate_rsi(closes)
    atr = calculate_atr(bars)
    stoch_rsi = calculate_stoch_rsi(closes)
    macd_val, macd_sig, macd_signal = calculate_macd(closes)
    bb_upper, bb_mid, bb_lower, bb_pos = calculate_bollinger(closes)
    vol_ratio = calculate_volume_ratio(volumes)
    vwap = calculate_vwap(bars[-20:])

    if rsi is None or atr is None:
        return None

    # EMA trend
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    ema50 = calculate_ema(closes, 50)

    if not ema9 or not ema21 or not ema50:
        return None

    ema9_val = ema9[-1]
    ema21_val = ema21[-1]
    ema50_val = ema50[-1]

    if ema9_val > ema21_val > ema50_val:
        ema_trend = "up"
    elif ema9_val < ema21_val < ema50_val:
        ema_trend = "down"
    else:
        ema_trend = "flat"

    # === SCORING ===
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []

    # 1. EMA Trend (+20)
    if ema_trend == "up":
        buy_score += 20
        buy_reasons.append(f"EMA uptrend (9>{ema9_val:.0f} > 21>{ema21_val:.0f} > 50>{ema50_val:.0f})")
    elif ema_trend == "down":
        sell_score += 20
        sell_reasons.append(f"EMA downtrend (9<{ema9_val:.0f} < 21<{ema21_val:.0f} < 50<{ema50_val:.0f})")

    # 2. MACD (+20)
    if macd_signal == "bullish_cross":
        buy_score += 20
        buy_reasons.append("MACD bullish crossover")
    elif macd_signal == "bullish":
        buy_score += 10
        buy_reasons.append("MACD bullish")
    elif macd_signal == "bearish_cross":
        sell_score += 20
        sell_reasons.append("MACD bearish crossover")
    elif macd_signal == "bearish":
        sell_score += 10
        sell_reasons.append("MACD bearish")

    # 3. RSI (+15)
    if rsi < 35:
        buy_score += 15
        buy_reasons.append(f"RSI oversold ({rsi})")
    elif rsi < 50:
        buy_score += 8
        buy_reasons.append(f"RSI favorable ({rsi})")
    elif rsi > 65:
        sell_score += 15
        sell_reasons.append(f"RSI overbought ({rsi})")
    elif rsi > 50:
        sell_score += 8
        sell_reasons.append(f"RSI bearish ({rsi})")

    # 4. Stochastic RSI (+10)
    if stoch_rsi is not None:
        if stoch_rsi < 20:
            buy_score += 10
            buy_reasons.append(f"StochRSI oversold ({stoch_rsi})")
        elif stoch_rsi > 80:
            sell_score += 10
            sell_reasons.append(f"StochRSI overbought ({stoch_rsi})")

    # 5. Bollinger Bands (+15)
    if bb_pos == "below_lower":
        buy_score += 15
        buy_reasons.append(f"Price below Bollinger lower band ({bb_lower})")
    elif bb_pos == "lower_half" and ema_trend == "up":
        buy_score += 8
        buy_reasons.append("Price in lower Bollinger half with uptrend")
    elif bb_pos == "above_upper":
        sell_score += 15
        sell_reasons.append(f"Price above Bollinger upper band ({bb_upper})")
    elif bb_pos == "upper_half" and ema_trend == "down":
        sell_score += 8
        sell_reasons.append("Price in upper Bollinger half with downtrend")

    # 6. Volume (+10)
    if vol_ratio > 1.5:
        if buy_score > sell_score:
            buy_score += 10
            buy_reasons.append(f"High volume confirmation ({vol_ratio}x avg)")
        elif sell_score > buy_score:
            sell_score += 10
            sell_reasons.append(f"High volume confirmation ({vol_ratio}x avg)")
    elif vol_ratio > 1.2:
        if buy_score > sell_score:
            buy_score += 5
        elif sell_score > buy_score:
            sell_score += 5

    # 7. VWAP (+10)
    if vwap:
        if price > vwap and ema_trend == "up":
            buy_score += 10
            buy_reasons.append(f"Price above VWAP ({vwap})")
        elif price < vwap and ema_trend == "down":
            sell_score += 10
            sell_reasons.append(f"Price below VWAP ({vwap})")

    # === DECISION ===
    min_score = 45  # Minimum confidence to generate a signal

    if buy_score >= min_score and buy_score > sell_score:
        signal = "BUY"
        strength = buy_score
        reasons = buy_reasons
    elif sell_score >= min_score and sell_score > buy_score:
        signal = "SELL"
        strength = sell_score
        reasons = sell_reasons
    else:
        signal = "NONE"
        strength = max(buy_score, sell_score)
        reasons = []

    return AnalysisResult(
        ticker=ticker,
        price=price,
        signal=signal,
        strength=strength,
        reasons=reasons,
        rsi=rsi,
        atr=atr,
        ema_trend=ema_trend,
        macd_signal=macd_signal,
        volume_ratio=vol_ratio,
        bb_position=bb_pos,
        stoch_rsi=stoch_rsi or 50.0,
    )
