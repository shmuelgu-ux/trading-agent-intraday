"""Fundamental data service using Financial Modeling Prep API."""

from dataclasses import dataclass
from loguru import logger
import httpx
from config import settings


@dataclass
class FundamentalData:
    ticker: str
    market_cap: float | None  # in billions
    pe_ratio: float | None
    eps: float | None
    revenue_growth: float | None  # quarterly YoY %
    earnings_growth: float | None  # quarterly YoY %
    dividend_yield: float | None
    sector: str
    verdict: str  # "strong", "moderate", "weak", "unknown"
    reasons: list[str]


class FundamentalsService:
    """Fetches fundamental data from Financial Modeling Prep."""

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self):
        self.api_key = settings.fmp_api_key

    def get_fundamentals(self, ticker: str) -> FundamentalData:
        """Get fundamental analysis for a ticker."""
        if not self.api_key:
            return FundamentalData(
                ticker=ticker, market_cap=None, pe_ratio=None, eps=None,
                revenue_growth=None, earnings_growth=None, dividend_yield=None,
                sector="", verdict="unknown", reasons=["אין FMP API key"],
            )

        try:
            profile = self._fetch_profile(ticker)
            ratios = self._fetch_ratios(ticker)
            growth = self._fetch_growth(ticker)

            pe = profile.get("pe") or ratios.get("peRatioTTM")
            eps = profile.get("eps")
            market_cap = profile.get("mktCap", 0) / 1e9 if profile.get("mktCap") else None
            div_yield = profile.get("lastDiv", 0) / profile.get("price", 1) if profile.get("price") else None
            sector = profile.get("sector", "")
            rev_growth = growth.get("revenueGrowth")
            earn_growth = growth.get("epsgrowth")

            # Analyze
            reasons = []
            score = 0

            if pe is not None:
                if 0 < pe < 25:
                    score += 2
                    reasons.append(f"P/E = {pe:.1f} - מכפיל סביר")
                elif 25 <= pe < 50:
                    score += 1
                    reasons.append(f"P/E = {pe:.1f} - מכפיל בינוני")
                elif pe >= 50:
                    score -= 1
                    reasons.append(f"P/E = {pe:.1f} - מכפיל גבוה")
                elif pe < 0:
                    score -= 2
                    reasons.append(f"P/E שלילי - החברה מפסידה כסף")

            if eps is not None:
                if eps > 0:
                    score += 1
                    reasons.append(f"EPS = ${eps:.2f} - רווחית")
                else:
                    score -= 1
                    reasons.append(f"EPS = ${eps:.2f} - לא רווחית")

            if rev_growth is not None:
                pct = rev_growth * 100
                if pct > 10:
                    score += 2
                    reasons.append(f"צמיחת הכנסות {pct:.1f}% - צמיחה חזקה")
                elif pct > 0:
                    score += 1
                    reasons.append(f"צמיחת הכנסות {pct:.1f}%")
                else:
                    score -= 1
                    reasons.append(f"הכנסות בירידה {pct:.1f}%")

            if earn_growth is not None:
                pct = earn_growth * 100
                if pct > 15:
                    score += 2
                    reasons.append(f"צמיחת רווחים {pct:.1f}% - חזק")
                elif pct > 0:
                    score += 1
                    reasons.append(f"צמיחת רווחים {pct:.1f}%")
                elif pct < -20:
                    score -= 2
                    reasons.append(f"רווחים בנפילה {pct:.1f}%")

            if market_cap is not None:
                if market_cap > 10:
                    reasons.append(f"שווי שוק ${market_cap:.1f}B - Large Cap")
                elif market_cap > 2:
                    reasons.append(f"שווי שוק ${market_cap:.1f}B - Mid Cap")
                else:
                    reasons.append(f"שווי שוק ${market_cap:.1f}B - Small Cap")

            if score >= 4:
                verdict = "strong"
            elif score >= 2:
                verdict = "moderate"
            elif score >= 0:
                verdict = "weak"
            else:
                verdict = "negative"

            return FundamentalData(
                ticker=ticker, market_cap=market_cap, pe_ratio=pe, eps=eps,
                revenue_growth=rev_growth, earnings_growth=earn_growth,
                dividend_yield=div_yield, sector=sector,
                verdict=verdict, reasons=reasons,
            )

        except Exception as e:
            logger.debug(f"Fundamentals failed for {ticker}: {e}")
            return FundamentalData(
                ticker=ticker, market_cap=None, pe_ratio=None, eps=None,
                revenue_growth=None, earnings_growth=None, dividend_yield=None,
                sector="", verdict="unknown", reasons=[f"שגיאה בטעינת נתונים"],
            )

    def _fetch_profile(self, ticker: str) -> dict:
        url = f"{self.BASE_URL}/profile?symbol={ticker}&apikey={self.api_key}"
        resp = httpx.get(url, timeout=10)
        data = resp.json()
        return data[0] if isinstance(data, list) and data else {}

    def _fetch_ratios(self, ticker: str) -> dict:
        url = f"{self.BASE_URL}/ratios-ttm?symbol={ticker}&apikey={self.api_key}"
        resp = httpx.get(url, timeout=10)
        data = resp.json()
        return data[0] if isinstance(data, list) and data else {}

    def _fetch_growth(self, ticker: str) -> dict:
        url = f"{self.BASE_URL}/financial-growth?symbol={ticker}&period=quarter&limit=1&apikey={self.api_key}"
        resp = httpx.get(url, timeout=10)
        data = resp.json()
        return data[0] if isinstance(data, list) and data else {}
