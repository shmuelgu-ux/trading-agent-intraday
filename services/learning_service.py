"""Learning service — analyzes batches of 10 closed trades to extract
patterns and insights. Does NOT build a fixed strategy; instead it
maintains a rolling window of observations that evolve with the market.

Flow:
1. After each reconciliation, check how many trades closed since the
   last learning cycle.
2. When 10 new closed trades accumulate, trigger a learning cycle.
3. Analyze each trade individually (why it won/lost).
4. Combine all 10 into pattern detection + recommendations.
5. Store the report in the learning_report table.
6. The latest report's insights feed back into signal scoring as
   soft adjustments (future enhancement).
"""
import json
from datetime import datetime, timezone
from loguru import logger
from sqlalchemy import select, func
from db.database import async_session, TradeLog, LearningReport, _utc_naive_now


BATCH_SIZE = 10  # Trades per learning cycle


class LearningService:

    async def check_and_learn(self) -> dict | None:
        """Check if we have enough new closed trades for a learning cycle.
        If yes, run the analysis and store the report. Returns the report
        dict or None if not enough trades yet.
        """
        async with async_session() as session:
            # How many closed trades exist total?
            total_closed = (await session.execute(
                select(func.count()).select_from(TradeLog).where(
                    TradeLog.status == "CLOSED",
                    TradeLog.action_taken == "EXECUTE",
                )
            )).scalar() or 0

            # How many trades have we already analyzed?
            already_analyzed = 0
            last_report = (await session.execute(
                select(LearningReport)
                .order_by(LearningReport.id.desc())
                .limit(1)
            )).scalar_one_or_none()

            if last_report:
                # Count IDs in the last report's trade_ids
                prev_ids = set(last_report.trade_ids.split(",")) if last_report.trade_ids else set()
                # Total analyzed = sum of all reports' trade counts
                total_analyzed_q = await session.execute(
                    select(func.sum(LearningReport.trade_count))
                )
                already_analyzed = total_analyzed_q.scalar() or 0

            new_closed = total_closed - already_analyzed
            if new_closed < BATCH_SIZE:
                return None

            logger.info(
                f"Learning cycle triggered: {new_closed} new closed trades "
                f"(total={total_closed}, analyzed={already_analyzed})"
            )

            # Fetch the BATCH_SIZE most recent unanalyzed closed trades
            all_closed = (await session.execute(
                select(TradeLog)
                .where(
                    TradeLog.status == "CLOSED",
                    TradeLog.action_taken == "EXECUTE",
                )
                .order_by(TradeLog.id.desc())
                .limit(BATCH_SIZE)
            )).scalars().all()

            if len(all_closed) < BATCH_SIZE:
                return None

            # === INDIVIDUAL ANALYSIS ===
            individual = []
            wins = []
            losses = []
            for trade in all_closed:
                analysis = self._analyze_single_trade(trade)
                individual.append(analysis)
                if trade.pnl and trade.pnl > 0:
                    wins.append(analysis)
                else:
                    losses.append(analysis)

            # === COMBINED INSIGHTS ===
            combined = self._combine_insights(individual, wins, losses)

            # === STORE REPORT ===
            trade_ids = ",".join(str(t.id) for t in all_closed)
            win_count = len(wins)
            loss_count = len(losses)
            total_pnl = sum(t.pnl or 0 for t in all_closed)
            wr = win_count / BATCH_SIZE if BATCH_SIZE > 0 else 0

            report = LearningReport(
                trade_ids=trade_ids,
                trade_count=BATCH_SIZE,
                individual_analysis=json.dumps(individual, ensure_ascii=False),
                combined_insights=json.dumps(combined, ensure_ascii=False),
                timestamp=_utc_naive_now(),
                win_rate=round(wr, 4),
                total_pnl=round(total_pnl, 2),
            )
            session.add(report)
            await session.commit()

            logger.info(
                f"Learning report #{report.id}: "
                f"{win_count}W/{loss_count}L, PnL=${total_pnl:.2f}, "
                f"{len(combined.get('patterns', []))} patterns found"
            )

            return {
                "id": report.id,
                "individual": individual,
                "combined": combined,
                "win_rate": wr,
                "total_pnl": total_pnl,
            }

    def _analyze_single_trade(self, trade: TradeLog) -> dict:
        """Analyze a single closed trade — why did it win or lose?"""
        pnl = trade.pnl or 0
        is_win = pnl > 0
        entry = trade.entry_price or 0
        sl = trade.stop_loss or 0
        tp = trade.take_profit or 0
        exit_p = trade.exit_price or 0

        # Parse reasoning from journal
        try:
            reasons = json.loads(trade.reasoning) if trade.reasoning else []
        except Exception:
            reasons = []

        # Extract indicators from reasoning text
        indicators = {}
        for r in reasons:
            if "RSI" in r:
                indicators["rsi_note"] = r
            if "EMA" in r or "טרנד" in r:
                indicators["trend_note"] = r
            if "MACD" in r:
                indicators["macd_note"] = r
            if "ווליום" in r:
                indicators["volume_note"] = r
            if "מאקרו" in r:
                indicators["macro_note"] = r

        # Determine exit type
        if is_win:
            exit_type = "take_profit"
        else:
            exit_type = "stop_loss"

        # Calculate how far price moved
        if entry > 0 and exit_p > 0:
            move_pct = ((exit_p - entry) / entry) * 100
            if trade.side in ("sell", "short"):
                move_pct = -move_pct  # Flip for shorts
        else:
            move_pct = 0

        # Risk/reward analysis
        if entry > 0 and sl > 0 and tp > 0:
            risk_dist = abs(entry - sl)
            reward_dist = abs(tp - entry)
            actual_rr = abs(exit_p - entry) / risk_dist if risk_dist > 0 else 0
        else:
            risk_dist = 0
            reward_dist = 0
            actual_rr = 0

        analysis = {
            "ticker": trade.ticker,
            "side": trade.side,
            "pnl": round(pnl, 2),
            "result": "win" if is_win else "loss",
            "exit_type": exit_type,
            "move_pct": round(move_pct, 2),
            "actual_rr": round(actual_rr, 2),
            "indicators": indicators,
            "entry": entry,
            "exit": exit_p,
            "stop_loss": sl,
            "take_profit": tp,
        }

        # Generate human-readable explanation
        if is_win:
            analysis["explanation"] = (
                f"{trade.ticker}: רווח של ${pnl:.2f}. "
                f"{'קנייה' if trade.side in ('buy','long') else 'מכירה'} "
                f"ב-${entry:.2f}, הגיע לטייק פרופיט ב-${exit_p:.2f} "
                f"(תנועה של {move_pct:.1f}%)."
            )
        else:
            analysis["explanation"] = (
                f"{trade.ticker}: הפסד של ${abs(pnl):.2f}. "
                f"{'קנייה' if trade.side in ('buy','long') else 'מכירה'} "
                f"ב-${entry:.2f}, נפגע בסטופ ב-${exit_p:.2f} "
                f"(תנועה של {move_pct:.1f}%)."
            )

        return analysis

    def _combine_insights(
        self,
        all_trades: list[dict],
        wins: list[dict],
        losses: list[dict],
    ) -> dict:
        """Combine individual analyses into patterns and recommendations."""
        patterns = []
        recommendations = []
        strengths = []

        total = len(all_trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0

        # --- Pattern: Side bias ---
        longs = [t for t in all_trades if t["side"] in ("buy", "long")]
        shorts = [t for t in all_trades if t["side"] in ("sell", "short")]
        long_wins = [t for t in longs if t["result"] == "win"]
        short_wins = [t for t in shorts if t["result"] == "win"]

        if longs and shorts:
            long_wr = len(long_wins) / len(longs) if longs else 0
            short_wr = len(short_wins) / len(shorts) if shorts else 0

            if long_wr > short_wr + 0.3:
                patterns.append({
                    "type": "side_bias",
                    "detail": f"קניות הצליחו הרבה יותר ממכירות ({long_wr:.0%} מול {short_wr:.0%})",
                    "direction": "favor_long",
                })
                recommendations.append("לשקול להעדיף קניות על פני מכירות בתנאי השוק הנוכחיים")
            elif short_wr > long_wr + 0.3:
                patterns.append({
                    "type": "side_bias",
                    "detail": f"מכירות הצליחו הרבה יותר מקניות ({short_wr:.0%} מול {long_wr:.0%})",
                    "direction": "favor_short",
                })
                recommendations.append("לשקול להעדיף מכירות על פני קניות בתנאי השוק הנוכחיים")

        # --- Pattern: Repeat losers ---
        ticker_results = {}
        for t in all_trades:
            tk = t["ticker"]
            if tk not in ticker_results:
                ticker_results[tk] = {"wins": 0, "losses": 0}
            if t["result"] == "win":
                ticker_results[tk]["wins"] += 1
            else:
                ticker_results[tk]["losses"] += 1

        repeat_losers = [
            tk for tk, r in ticker_results.items()
            if r["losses"] >= 2 and r["wins"] == 0
        ]
        if repeat_losers:
            patterns.append({
                "type": "repeat_losers",
                "detail": f"מניות שהפסידו שוב ושוב: {', '.join(repeat_losers)}",
                "tickers": repeat_losers,
            })
            recommendations.append(
                f"המניות {', '.join(repeat_losers)} הפסידו כמה פעמים — "
                f"לשקול להימנע מהן בטווח הקרוב"
            )

        # --- Pattern: Macro alignment ---
        macro_aligned_wins = sum(
            1 for t in wins if "macro_note" in t.get("indicators", {})
            and "תומך" in t["indicators"].get("macro_note", "")
        )
        macro_against_losses = sum(
            1 for t in losses if "macro_note" in t.get("indicators", {})
            and "נגד" in t["indicators"].get("macro_note", "")
        )
        if macro_against_losses >= 2:
            patterns.append({
                "type": "macro_conflict",
                "detail": f"{macro_against_losses} הפסדים היו נגד המגמה הגדולה (מאקרו)",
            })
            recommendations.append("עסקאות נגד המגמה הגדולה מפסידות — להגביר את משקל המאקרו בציון")

        if macro_aligned_wins >= 2:
            strengths.append(f"{macro_aligned_wins} רווחים היו בכיוון המגמה הגדולה — המאקרו עובד")

        # --- Pattern: Average win vs average loss size ---
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else 0
        if avg_win > 0 and avg_loss > 0:
            if avg_win > avg_loss * 1.5:
                strengths.append(
                    f"הרווח הממוצע (${avg_win:.2f}) גדול פי "
                    f"{avg_win/avg_loss:.1f} מההפסד הממוצע (${avg_loss:.2f}) — ניהול סיכונים טוב"
                )
            elif avg_loss > avg_win * 1.5:
                patterns.append({
                    "type": "bad_rr",
                    "detail": f"ההפסד הממוצע (${avg_loss:.2f}) גדול מהרווח הממוצע (${avg_win:.2f})",
                })
                recommendations.append("לבדוק אם הסטופ לוס רחוק מדי או הטייק פרופיט קרוב מדי")

        # --- Pattern: Volume confirmation ---
        vol_wins = sum(1 for t in wins if "volume_note" in t.get("indicators", {}))
        vol_losses = sum(1 for t in losses if "volume_note" in t.get("indicators", {}))
        if vol_wins > vol_losses and vol_wins >= 2:
            strengths.append("עסקאות עם אישור ווליום גבוה הצליחו יותר")
        elif vol_losses > vol_wins and vol_losses >= 2:
            patterns.append({
                "type": "volume_misleading",
                "detail": "ווליום גבוה לא סייע — רוב ההפסדים היו עם ווליום גבוה",
            })

        # --- Summary ---
        if win_rate >= 0.5:
            summary = f"ביצועים חיוביים: {win_rate:.0%} הצלחה מתוך {total} עסקאות."
        elif win_rate >= 0.3:
            summary = f"ביצועים בינוניים: {win_rate:.0%} הצלחה מתוך {total} עסקאות. יש מה לשפר."
        else:
            summary = f"ביצועים חלשים: {win_rate:.0%} הצלחה מתוך {total} עסקאות. צריך שיפור משמעותי."

        return {
            "summary": summary,
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "patterns": patterns,
            "strengths": strengths,
            "recommendations": recommendations,
            "total_trades": total,
            "wins": win_count,
            "losses": loss_count,
        }

    async def get_latest_report(self) -> dict | None:
        """Return the most recent learning report for the dashboard."""
        async with async_session() as session:
            report = (await session.execute(
                select(LearningReport)
                .order_by(LearningReport.id.desc())
                .limit(1)
            )).scalar_one_or_none()

            if not report:
                return None

            return {
                "id": report.id,
                "timestamp": report.timestamp.isoformat() if report.timestamp else None,
                "trade_count": report.trade_count,
                "win_rate": report.win_rate,
                "total_pnl": report.total_pnl,
                "individual": json.loads(report.individual_analysis) if report.individual_analysis else [],
                "combined": json.loads(report.combined_insights) if report.combined_insights else {},
            }
