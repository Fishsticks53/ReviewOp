from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from math import sqrt
from typing import Optional

from sqlalchemy import case, func, text
from sqlalchemy.orm import Session, aliased, joinedload

from models.tables import EvidenceSpan, Prediction, Review
from services.analytics_common import infer_origin, parse_dt


def aspect_leaderboard(db: Session, limit: int = 25, domain: Optional[str] = None) -> list[dict]:
    now = datetime.utcnow()
    current_start = now - timedelta(days=7)
    previous_start = now - timedelta(days=14)

    def _wilson_ci_95(p_hat: float, n: int) -> tuple[float, float]:
        if n <= 0:
            return (0.0, 0.0)
        z = 1.96
        denom = 1.0 + (z * z) / n
        center = (p_hat + (z * z) / (2.0 * n)) / denom
        half = (z * sqrt((p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n))) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    def _pct(numer: float, denom: float) -> float:
        if denom <= 0:
            return 0.0
        return (numer / denom) * 100.0

    def _base_query(start: datetime, end: datetime):
        q = (
            db.query(
                Prediction.aspect_raw.label("aspect"),
                func.count(Prediction.id).label("frequency"),
                func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("positive"),
                func.sum(case((Prediction.sentiment == "neutral", 1), else_=0)).label("neutral"),
                func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("negative"),
            )
            .join(Review, Review.id == Prediction.review_id)
            .filter(Review.created_at >= start, Review.created_at < end)
        )
        if domain:
            q = q.filter(Review.domain == domain)
        return q.group_by(Prediction.aspect_raw)

    current_rows = {row.aspect: row for row in _base_query(current_start, now).all()}
    previous_rows = {row.aspect: row for row in _base_query(previous_start, current_start).all()}

    # Global sample size for normalization (total reviews in current window).
    sample_size_q = db.query(func.count(Review.id)).filter(Review.created_at >= current_start, Review.created_at < now)
    if domain:
        sample_size_q = sample_size_q.filter(Review.domain == domain)
    sample_size = int(sample_size_q.scalar() or 0)

    # Keep the leaderboard focused on aspects that appear in the current window.
    aspects_sorted = sorted(current_rows.keys(), key=lambda a: (-int(current_rows[a].frequency or 0), str(a)))
    aspects = aspects_sorted[: max(1, int(limit))]
    if not aspects:
        return []

    # Compute implicit/explicit ratio for returned aspects only (small, bounded scope).
    evidence_q = (
        db.query(Prediction.id, Prediction.aspect_raw, EvidenceSpan.snippet)
        .join(Review, Review.id == Prediction.review_id)
        .outerjoin(EvidenceSpan, EvidenceSpan.prediction_id == Prediction.id)
        .filter(Review.created_at >= current_start, Review.created_at < now)
        .filter(Prediction.aspect_raw.in_(aspects))
    )
    if domain:
        evidence_q = evidence_q.filter(Review.domain == domain)

    pred_aspect: dict[int, str] = {}
    pred_is_explicit: dict[int, bool] = {}
    for pred_id, aspect, snippet in evidence_q.all():
        pred_id_int = int(pred_id)
        aspect_str = str(aspect)
        pred_aspect[pred_id_int] = aspect_str
        if pred_is_explicit.get(pred_id_int):
            continue
        if infer_origin(aspect_str, str(snippet) if snippet is not None else None) == "explicit":
            pred_is_explicit[pred_id_int] = True

    origin_counts: dict[str, dict[str, int]] = {a: {"implicit": 0, "explicit": 0} for a in aspects}
    for pred_id_int, aspect_str in pred_aspect.items():
        if aspect_str not in origin_counts:
            continue
        if pred_is_explicit.get(pred_id_int, False):
            origin_counts[aspect_str]["explicit"] += 1
        else:
            origin_counts[aspect_str]["implicit"] += 1

    out: list[dict] = []
    for aspect in aspects:
        current = current_rows.get(aspect)
        previous = previous_rows.get(aspect)
        current_freq = int(current.frequency or 0) if current else 0
        previous_freq = int(previous.frequency or 0) if previous else 0
        current_positive = int(current.positive or 0) if current else 0
        current_neutral = int(current.neutral or 0) if current else 0
        current_negative = int(current.negative or 0) if current else 0

        total = max(0, current_positive + current_neutral + current_negative)
        pos_pct = _pct(current_positive, total)
        neu_pct = _pct(current_neutral, total)
        neg_pct = _pct(current_negative, total)
        net_sentiment = pos_pct - neg_pct

        if previous_freq <= 0:
            change_vs_previous_period = 100.0 if current_freq else 0.0
        else:
            change_vs_previous_period = ((current_freq - previous_freq) / previous_freq) * 100.0

        p_hat = (current_negative / total) if total > 0 else 0.0
        ci_lo, ci_hi = _wilson_ci_95(p_hat, total)

        oc = origin_counts.get(str(aspect)) or {"implicit": 0, "explicit": 0}
        implicit = int(oc.get("implicit", 0))
        explicit = int(oc.get("explicit", 0))
        implicit_pct = _pct(implicit, implicit + explicit)

        out.append(
            {
                "aspect": str(aspect),
                "frequency": current_freq,
                "sample_size": sample_size,
                "mentions_per_100_reviews": round(_pct(current_freq, sample_size), 4),
                "positive_pct": round(pos_pct, 2),
                "neutral_pct": round(neu_pct, 2),
                "negative_pct": round(neg_pct, 2),
                "net_sentiment": round(net_sentiment, 2),
                "change_vs_previous_period": round(change_vs_previous_period, 2),
                "change_7d_vs_prev_7d": round(change_vs_previous_period, 2),
                "negative_ci_low": round(ci_lo * 100.0, 2),
                "negative_ci_high": round(ci_hi * 100.0, 2),
                "implicit_pct": round(implicit_pct, 2),
            }
        )

    out.sort(key=lambda row: (-row["frequency"], -row["change_7d_vs_prev_7d"], row["aspect"]))
    return out[: max(1, int(limit))]


def top_aspects(db: Session, limit: int, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    f = parse_dt(dt_from)
    t = parse_dt(dt_to)
    q = db.query(
        Prediction.aspect_raw.label("aspect"),
        func.count(Prediction.id).label("count"),
    ).join(Review, Review.id == Prediction.review_id)
    if domain:
        q = q.filter(Review.domain == domain)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)
    q = q.group_by(Prediction.aspect_raw).order_by(text("count DESC")).limit(limit)
    return [{"aspect": r.aspect, "count": int(r.count)} for r in q.all()]


def aspect_sentiment_distribution(db: Session, limit: int, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    f = parse_dt(dt_from)
    t = parse_dt(dt_to)
    q = db.query(
        Prediction.aspect_raw.label("aspect"),
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("positive"),
        func.sum(case((Prediction.sentiment == "neutral", 1), else_=0)).label("neutral"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("negative"),
        func.count(Prediction.id).label("total"),
    ).join(Review, Review.id == Prediction.review_id)
    if domain:
        q = q.filter(Review.domain == domain)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)
    q = q.group_by(Prediction.aspect_raw).order_by(text("total DESC")).limit(limit)
    out = []
    for r in q.all():
        out.append(
            {
                "aspect": r.aspect,
                "positive": int(r.positive or 0),
                "neutral": int(r.neutral or 0),
                "negative": int(r.negative or 0),
            }
        )
    return out


def trends(db: Session, interval: str, aspect: Optional[str], dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    f = parse_dt(dt_from)
    t = parse_dt(dt_to)
    if interval == "week":
        bucket_expr = func.date_format(Review.created_at, "%x-W%v")
    else:
        bucket_expr = func.date_format(Review.created_at, "%Y-%m-%d")
    q = db.query(
        bucket_expr.label("bucket"),
        func.count(Prediction.id).label("mentions"),
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("pos"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg"),
    ).join(Review, Review.id == Prediction.review_id)
    if domain:
        q = q.filter(Review.domain == domain)
    if aspect:
        q = q.filter(Prediction.aspect_raw == aspect)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)
    q = q.group_by(bucket_expr).order_by(bucket_expr.asc())
    out = []
    for r in q.all():
        mentions = int(r.mentions or 0)
        pos = int(r.pos or 0)
        neg = int(r.neg or 0)
        neg_pct = (neg / mentions) if mentions > 0 else 0.0
        pos_pct = (pos / mentions) if mentions > 0 else 0.0
        score = pos_pct - neg_pct
        out.append({"bucket": str(r.bucket), "mentions": mentions, "negative_pct": round(neg_pct, 4), "sentiment_score": round(score, 4)})
    return out


def evidence_drilldown(db: Session, aspect: Optional[str] = None, sentiment: Optional[str] = None, limit: int = 50, domain: Optional[str] = None) -> list[dict]:
    q = db.query(Prediction, Review).options(joinedload(Prediction.evidence_spans)).join(Review, Review.id == Prediction.review_id)
    if aspect:
        q = q.filter(Prediction.aspect_raw == aspect)
    if sentiment:
        q = q.filter(Prediction.sentiment == sentiment)
    if domain:
        q = q.filter(Review.domain == domain)
    q = q.order_by(Review.created_at.desc()).limit(max(1, min(limit, 200)))
    rows = []
    for pred, review in q.all():
        span = pred.evidence_spans[0] if pred.evidence_spans else None
        snippet = span.snippet if span else None
        rows.append({"review_id": review.id, "review_text": review.text, "aspect": pred.aspect_raw, "sentiment": pred.sentiment, "origin": infer_origin(pred.aspect_raw, snippet), "evidence": snippet, "evidence_start": span.start_char if span else None, "evidence_end": span.end_char if span else None, "created_at": review.created_at.isoformat() if review.created_at else None})
    return rows


def aspect_trends(db: Session, interval: str = "day", domain: Optional[str] = None, limit: int = 200) -> list[dict]:
    out = []
    top = top_aspects(db, 12, None, None, domain)
    for row in top:
        aspect = row["aspect"]
        points = trends(db, interval, aspect, None, None, domain)
        for point in points:
            out.append({"bucket": point["bucket"], "aspect": aspect, "mentions": point["mentions"], "negative_pct": point["negative_pct"]})
    return out[: max(1, limit)]


def emerging_aspects(db: Session, interval: str = "day", lookback_buckets: int = 7, domain: Optional[str] = None) -> list[dict]:
    tr = aspect_trends(db, interval=interval, domain=domain, limit=5000)
    grouped = defaultdict(list)
    for item in tr:
        grouped[item["aspect"]].append(item)
    out = []
    for aspect, points in grouped.items():
        points_sorted = sorted(points, key=lambda item: item["bucket"])
        if len(points_sorted) < 2:
            continue
        recent = points_sorted[-1]["mentions"]
        baseline_points = points_sorted[-(lookback_buckets + 1):-1]
        if not baseline_points:
            continue
        baseline = sum(p["mentions"] for p in baseline_points) / len(baseline_points)
        if recent >= 3 and recent > baseline * 1.5:
            out.append({"aspect": aspect, "recent_mentions": recent, "baseline_mentions": round(baseline, 2)})
    out.sort(key=lambda x: (-(x["recent_mentions"] - x["baseline_mentions"]), x["aspect"]))
    return out


def aspect_detail(db: Session, aspect: str, interval: str = "day", domain: Optional[str] = None) -> dict:
    dist_rows = aspect_sentiment_distribution(db, 500, None, None, domain)
    dist = next((item for item in dist_rows if item["aspect"] == aspect), None)
    if dist is None:
        return {"aspect": aspect, "frequency": 0, "positive": 0, "neutral": 0, "negative": 0, "explicit_count": 0, "implicit_count": 0, "connected_aspects": [], "trend": [], "examples": []}
    examples = evidence_drilldown(db, aspect=aspect, limit=8, domain=domain)
    origin_samples = db.query(Prediction.aspect_raw.label("aspect"), func.count(Prediction.id).label("count"), func.max(EvidenceSpan.snippet).label("snippet")).outerjoin(EvidenceSpan, EvidenceSpan.prediction_id == Prediction.id).join(Review, Review.id == Prediction.review_id).filter(Prediction.aspect_raw == aspect)
    if domain:
        origin_samples = origin_samples.filter(Review.domain == domain)
    origin_samples = origin_samples.group_by(Prediction.aspect_raw).all()
    explicit_count = 0
    implicit_count = 0
    for sample in origin_samples:
        origin = infer_origin(sample.aspect, sample.snippet)
        count = int(sample.count or 0)
        if origin == "implicit":
            implicit_count += count
        else:
            explicit_count += count
    pred_anchor = aliased(Prediction)
    pred_peer = aliased(Prediction)
    connected_q = db.query(pred_peer.aspect_raw.label("aspect"), func.count(func.distinct(pred_anchor.review_id)).label("weight")).join(pred_peer, pred_peer.review_id == pred_anchor.review_id).join(Review, Review.id == pred_anchor.review_id).filter(pred_anchor.aspect_raw == aspect, pred_peer.aspect_raw != aspect)
    if domain:
        connected_q = connected_q.filter(Review.domain == domain)
    connected_rows = connected_q.group_by(pred_peer.aspect_raw).order_by(text("weight DESC"), pred_peer.aspect_raw.asc()).limit(12).all()
    connected = [{"aspect": row.aspect, "weight": int(row.weight or 0)} for row in connected_rows]
    trend = [t for t in aspect_trends(db, interval=interval, domain=domain, limit=5000) if t["aspect"] == aspect]
    return {"aspect": aspect, "frequency": int(dist["positive"] + dist["neutral"] + dist["negative"]), "positive": int(dist["positive"]), "neutral": int(dist["neutral"]), "negative": int(dist["negative"]), "explicit_count": explicit_count, "implicit_count": implicit_count, "connected_aspects": connected, "trend": trend, "examples": examples}
