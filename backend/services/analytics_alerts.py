from __future__ import annotations

import hashlib
from collections import defaultdict
from statistics import pstdev
from typing import Optional

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from models.tables import Alert, DismissedAlert, Review
from services.analytics_aspects import aspect_leaderboard, emerging_aspects


def _generate_alert_candidates(db: Session, domain: Optional[str] = None) -> list[dict]:
    out = []
    leaderboard = aspect_leaderboard(db, limit=20, domain=domain)
    if not leaderboard:
        return out
    change_series = [float(row["change_7d_vs_prev_7d"]) for row in leaderboard]
    sigma = pstdev(change_series) if len(change_series) > 1 else 0.0
    mean_change = sum(change_series) / len(change_series) if change_series else 0.0
    for row in leaderboard:
        z_score = (row["change_7d_vs_prev_7d"] - mean_change) / sigma if sigma > 0 else 0.0
        if (row["change_vs_previous_period"] >= 50 or z_score >= 2.0) and row["frequency"] >= 5:
            out.append({"type": "frequency_spike", "aspect": row["aspect"], "severity": "high", "message": f"Frequency spike detected for {row['aspect']}", "value": round(max(row["change_vs_previous_period"], z_score), 2), "threshold": 2.0 if z_score >= 2.0 else 50.0})
        if row["negative_pct"] >= 45 and row["frequency"] >= 5:
            out.append({"type": "negative_threshold", "aspect": row["aspect"], "severity": "medium", "message": f"Negative sentiment threshold breached for {row['aspect']}", "value": row["negative_pct"], "threshold": 45.0})
    for item in emerging_aspects(db, interval="day", domain=domain):
        out.append({"type": "emerging_aspect", "aspect": item["aspect"], "severity": "medium", "message": f"Emerging issue detected: {item['aspect']}", "value": float(item["recent_mentions"]), "threshold": float(item["baseline_mentions"])})
    return out[:50]


def _alert_signature(alert_type: str, aspect: str, message: str, domain: Optional[str]) -> tuple[str, str, str, Optional[str]]:
    return (alert_type, aspect, message, domain)


def _alert_signature_hash(alert_type: str, aspect: str, message: str, domain: Optional[str]) -> str:
    raw = f"{alert_type}|{aspect}|{message}|{domain or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sync_alerts(db: Session, domain: Optional[str] = None) -> list[Alert]:
    generated = _generate_alert_candidates(db, domain=domain)
    all_signatures = {_alert_signature(item["type"], item["aspect"], item["message"], domain) for item in generated}
    dismissed_q = db.query(DismissedAlert)
    dismissed_q = dismissed_q.filter(DismissedAlert.domain.is_(None) if domain is None else DismissedAlert.domain == domain)
    dismissed_rows = dismissed_q.all()
    dismissed_signatures = {row.signature for row in dismissed_rows}
    for row in dismissed_rows:
        sig_hash = _alert_signature_hash(row.type, row.aspect, row.message, row.domain)
        if _alert_signature(row.type, row.aspect, row.message, row.domain) not in all_signatures:
            db.delete(row)
            dismissed_signatures.discard(sig_hash)
    generated = [item for item in generated if _alert_signature_hash(item["type"], item["aspect"], item["message"], domain) not in dismissed_signatures]
    signatures = {_alert_signature(item["type"], item["aspect"], item["message"], domain): item for item in generated}
    existing_q = db.query(Alert)
    existing_q = existing_q.filter(Alert.domain.is_(None) if domain is None else Alert.domain == domain)
    existing = existing_q.all()
    existing_by_signature = {(row.type, row.aspect, row.message, row.domain): row for row in existing}
    for signature, payload in signatures.items():
        row = existing_by_signature.get(signature)
        if row:
            row.severity = payload["severity"]
            row.value = float(payload["value"])
            row.threshold = float(payload["threshold"])
        else:
            db.add(Alert(type=payload["type"], aspect=payload["aspect"], severity=payload["severity"], message=payload["message"], value=float(payload["value"]), threshold=float(payload["threshold"]), domain=domain, signature=_alert_signature_hash(payload["type"], payload["aspect"], payload["message"], domain)))
    stale_signatures = set(existing_by_signature.keys()) - set(signatures.keys())
    for signature in stale_signatures:
        db.delete(existing_by_signature[signature])
    db.commit()
    q = db.query(Alert)
    q = q.filter(Alert.domain.is_(None) if domain is None else Alert.domain == domain)
    return q.order_by(Alert.created_at.desc(), Alert.id.desc()).all()


def alerts(db: Session, domain: Optional[str] = None) -> list[dict]:
    if domain is None:
        active_domains = [row[0] for row in db.query(Review.domain).filter(Review.domain.isnot(None)).distinct().all()]
        if active_domains:
            for active_domain in active_domains:
                sync_alerts(db, domain=active_domain)
        has_global_reviews = db.query(Review.id).filter(Review.domain.is_(None)).filter(Review.predictions.any()).first() is not None
        if has_global_reviews or not active_domains:
            rows = sync_alerts(db, domain=None)
        else:
            rows = db.query(Alert).order_by(Alert.created_at.desc(), Alert.id.desc()).all()
    else:
        rows = sync_alerts(db, domain=domain)
    return [{"id": row.id, "type": row.type, "aspect": row.aspect, "severity": row.severity, "message": row.message, "value": float(row.value), "threshold": float(row.threshold), "status": "open", "detected_at": row.created_at.isoformat() if row.created_at else None, "priority_score": float(row.value), "domain": row.domain} for row in rows]


def clear_alert(db: Session, alert_id: int) -> bool:
    row = db.query(Alert).filter(Alert.id == alert_id).first()
    if not row:
        return False
    signature = _alert_signature_hash(row.type, row.aspect, row.message, row.domain)
    exists = db.query(DismissedAlert).filter(DismissedAlert.signature == signature).first()
    if not exists:
        db.add(DismissedAlert(type=row.type, aspect=row.aspect, message=row.message, domain=row.domain, signature=signature))
    db.delete(row)
    db.commit()
    return True

