"""Sentiment analysis using keyword-based scoring.

Analyzes news headlines to determine if sentiment is bullish or bearish.
Score range: -100 (very bearish) to +100 (very bullish).
"""

from dataclasses import dataclass

# Weighted keyword lists
BULLISH_KEYWORDS = {
    # Strong bullish (+3)
    "upgrade": 3, "upgrades": 3, "upgraded": 3,
    "beat": 3, "beats": 3, "exceeded": 3, "surpassed": 3,
    "record revenue": 3, "record profit": 3, "record earnings": 3,
    "strong growth": 3, "accelerating": 3,
    "fda approval": 3, "fda approved": 3,
    "buyback": 3, "share repurchase": 3,
    "dividend increase": 3, "raised dividend": 3,
    "breakout": 3, "all-time high": 3,

    # Moderate bullish (+2)
    "buy rating": 2, "outperform": 2, "overweight": 2,
    "positive": 2, "optimistic": 2, "bullish": 2,
    "growth": 2, "revenue growth": 2, "profit growth": 2,
    "strong demand": 2, "expanding": 2,
    "partnership": 2, "contract": 2, "deal": 2,
    "innovation": 2, "launch": 2, "new product": 2,
    "raised guidance": 2, "raised forecast": 2,
    "above expectations": 2, "better than expected": 2,

    # Mild bullish (+1)
    "gains": 1, "rises": 1, "climbs": 1, "jumps": 1, "surges": 1,
    "recovery": 1, "rebound": 1, "rally": 1,
    "momentum": 1, "upside": 1, "opportunity": 1,
    "analyst": 1, "target raised": 1,
}

BEARISH_KEYWORDS = {
    # Strong bearish (-3)
    "downgrade": 3, "downgrades": 3, "downgraded": 3,
    "miss": 3, "misses": 3, "missed": 3, "disappointing": 3,
    "lawsuit": 3, "sued": 3, "fraud": 3, "investigation": 3,
    "bankruptcy": 3, "default": 3, "insolvency": 3,
    "recall": 3, "fda reject": 3, "fda warning": 3,
    "sec investigation": 3, "accounting fraud": 3,
    "dividend cut": 3, "suspended dividend": 3,
    "guidance cut": 3, "lowered guidance": 3,

    # Moderate bearish (-2)
    "sell rating": 2, "underperform": 2, "underweight": 2,
    "negative": 2, "pessimistic": 2, "bearish": 2,
    "decline": 2, "declining": 2, "slowing": 2,
    "layoffs": 2, "restructuring": 2, "cost cutting": 2,
    "loss": 2, "losses": 2, "deficit": 2,
    "below expectations": 2, "worse than expected": 2,
    "warning": 2, "risk": 2, "concern": 2,
    "downside": 2, "target cut": 2, "price target lowered": 2,

    # Mild bearish (-1)
    "falls": 1, "drops": 1, "slides": 1, "tumbles": 1, "plunges": 1,
    "pressure": 1, "headwinds": 1, "uncertainty": 1,
    "volatility": 1, "pullback": 1, "correction": 1,
    "weak": 1, "soft": 1, "flat": 1,
}


@dataclass
class SentimentResult:
    score: int  # -100 to +100
    label: str  # "bullish", "bearish", "neutral"
    key_factors: list[str]  # Headlines that influenced the score
    headline_count: int


def analyze_sentiment(articles: list[dict]) -> SentimentResult:
    """Analyze sentiment from news articles.

    Returns a score from -100 to +100.
    """
    if not articles:
        return SentimentResult(score=0, label="neutral", key_factors=["אין חדשות אחרונות"], headline_count=0)

    total_score = 0
    key_factors = []

    for article in articles:
        text = (article.get("headline", "") + " " + article.get("summary", "")).lower()
        article_score = 0

        for keyword, weight in BULLISH_KEYWORDS.items():
            if keyword in text:
                article_score += weight

        for keyword, weight in BEARISH_KEYWORDS.items():
            if keyword in text:
                article_score -= weight

        total_score += article_score

        if abs(article_score) >= 2:
            direction = "חיובי" if article_score > 0 else "שלילי"
            key_factors.append(f"{direction}: {article['headline'][:80]}")

    # Normalize to -100 to +100
    max_possible = len(articles) * 6  # Rough max
    if max_possible > 0:
        normalized = int((total_score / max_possible) * 100)
        normalized = max(-100, min(100, normalized))
    else:
        normalized = 0

    if normalized > 15:
        label = "bullish"
    elif normalized < -15:
        label = "bearish"
    else:
        label = "neutral"

    return SentimentResult(
        score=normalized,
        label=label,
        key_factors=key_factors[:5],
        headline_count=len(articles),
    )
