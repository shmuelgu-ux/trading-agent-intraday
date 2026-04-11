"""News service using Alpaca News API."""

from datetime import datetime, timedelta
from loguru import logger
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from config import settings


class NewsService:
    """Fetches recent news for stocks via Alpaca."""

    def __init__(self):
        if settings.alpaca_api_key and settings.alpaca_secret_key:
            self._client = NewsClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )
        else:
            self._client = None

    def get_news(self, ticker: str, days: int = 3, limit: int = 10) -> list[dict]:
        """Get recent news headlines for a ticker."""
        if not self._client:
            return []

        try:
            request = NewsRequest(
                symbols=ticker,
                start=datetime.now() - timedelta(days=days),
                end=datetime.now(),
                limit=limit,
                sort="DESC",
            )
            news = self._client.get_news(request)

            # Extract news list from NewsSet
            articles = []
            raw_news = []
            for key, val in news:
                if key == "data" and isinstance(val, dict) and "news" in val:
                    raw_news = val["news"]
                    break

            for item in raw_news:
                articles.append({
                    "headline": item.get("headline", "") if isinstance(item, dict) else getattr(item, "headline", ""),
                    "summary": (item.get("summary", "") if isinstance(item, dict) else getattr(item, "summary", "")) or "",
                    "source": item.get("source", "") if isinstance(item, dict) else getattr(item, "source", ""),
                    "created_at": "",
                    "url": item.get("url", "") if isinstance(item, dict) else getattr(item, "url", ""),
                })

            logger.debug(f"News: {len(articles)} articles for {ticker}")
            return articles

        except Exception as e:
            logger.debug(f"News fetch failed for {ticker}: {e}")
            return []
